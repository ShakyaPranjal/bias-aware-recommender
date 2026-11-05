"""Build a static, server-free version of the web app.

Trains the engine in Python, exports the model (embeddings, CF factors, NLP
features, bias population, viewing histories) to compact files, and copies
the dashboard next to them. ``web/engine.js`` then runs the same inference
in the browser, so the app can be hosted anywhere static files are served
(GitHub Pages, Netlify, an S3 bucket).

Usage:
    python scripts/build_static.py               # writes dist/
    python -m http.server -d dist 8080           # preview at http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biasrec.bias import BIAS_KEYS, desc_order  # noqa: E402
from biasrec.engine import RecommenderEngine  # noqa: E402
from biasrec.nlp import EMOTIONS  # noqa: E402


class BinWriter:
    """Concatenates typed arrays into one buffer, 8-byte aligned."""

    def __init__(self):
        self.parts, self.offset, self.manifest = [], 0, {}

    def add(self, name: str, arr: np.ndarray, dtype: str):
        arr = np.ascontiguousarray(arr, dtype=dtype)
        pad = (-self.offset) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.offset += pad
        data = arr.tobytes()
        self.manifest[name] = {"offset": self.offset, "dtype": dtype, "shape": list(arr.shape)}
        self.parts.append(data)
        self.offset += len(data)

    def bytes(self) -> bytes:
        return b"".join(self.parts)


def export(engine: RecommenderEngine, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    f, mv = engine.features, engine.movies
    w = BinWriter()
    w.add("embedding", f["embedding"], "float32")
    w.add("cf_V", engine.cf.V, "float32")
    w.add("themes", f["themes"], "float32")
    w.add("emotions", f["emotions"], "float32")
    w.add("valence", f["valence"], "float32")
    w.add("hero", f["hero"], "float32")
    w.add("map", engine._map, "float32")
    w.add("genre_matrix", f["genre_matrix"], "uint8")

    users = sorted(engine._user_groups)
    offsets, idx, rating, ts = [0], [], [], []
    for u in users:
        i, r, t = engine.user_history(u)
        idx.append(i)
        rating.append(np.round(r * 2))
        ts.append(t)
        offsets.append(offsets[-1] + len(i))
    w.add("user_ids", np.array(users), "int32")
    w.add("user_offsets", np.array(offsets), "int32")
    w.add("r_idx", np.concatenate(idx), "uint16")
    w.add("r_half_stars", np.concatenate(rating), "uint8")
    w.add("r_ts", np.concatenate(ts), "uint32")
    (out / "model.bin").write_bytes(w.bytes())

    year = mv["year"].to_numpy()
    movies = {
        "movieId": mv["movieId"].astype(int).tolist(),
        "title": mv["clean_title"].tolist(),
        "year": [None if np.isnan(y) else int(y) for y in year],
        "yearFilled": engine.bias.years.tolist(),
        "genres": [[f["genres"].index(g) for g in gs] for gs in mv["genre_list"]],
        "overview": mv["overview"].fillna("").tolist(),
        "meanRating": mv["mean_rating"].tolist(),
        "nRatings": mv["n_ratings"].astype(int).tolist(),
        "quality": mv["quality"].tolist(),
    }
    (out / "movies.json").write_text(json.dumps(movies, separators=(",", ":")))

    catalog_share = f["genre_matrix"][engine.ratings["idx"].to_numpy()].sum(axis=0)
    meta = {
        "n_movies": int(len(mv)), "n_ratings": int(len(engine.ratings)), "n_users": int(engine.n_users),
        "genres": f["genres"], "themes": f["theme_names"], "emotions": EMOTIONS,
        "featured_users": engine.featured_users(),
        "with_overview": int((mv["overview"] != "").sum()),
        "min_support": int(engine.min_support),
        "decade_values": [int(d) for d in engine.bias.decade_values],
        "population": {k: engine.bias.population[k].tolist() for k in BIAS_KEYS},
        "catalog_share": (catalog_share / catalog_share.sum()).tolist(),
        "cf_s": engine.cf.s.tolist(),
        "catalog_map": [int(i) for i in desc_order(mv["n_ratings"].to_numpy())[:5000]],
        "binary": w.manifest,
    }
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return out


def copy_site(out: Path):
    web = ROOT / "web"
    for name in ("app.js", "engine.js", "styles.css"):
        shutil.copy(web / name, out / name)
    (out / "vendor").mkdir(exist_ok=True)
    shutil.copy(web / "vendor" / "chart.umd.min.js", out / "vendor" / "chart.umd.min.js")
    html = (web / "index.html").read_text()
    html = html.replace('href="/static/styles.css"', 'href="styles.css"')
    html = html.replace('<script src="/static/vendor/chart.umd.min.js"></script>',
                        '<script src="vendor/chart.umd.min.js"></script>\n'
                        '  <script src="engine.js" data-model="data/"></script>')
    html = html.replace('src="/static/app.js"', 'src="app.js"')
    (out / "index.html").write_text(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "dist"))
    args = ap.parse_args()
    out = Path(args.out)
    engine = RecommenderEngine.from_dir(args.data)
    export(engine, out / "data")
    copy_site(out)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Static site written to {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
