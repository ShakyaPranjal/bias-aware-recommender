"""Offline evaluation of the Bias-Aware Recommender.

Protocol: temporal hold-out. For every user with 20+ ratings, the most
recent 20% of their ratings form the test set and everything earlier is
training data. All models are trained on the training ratings only. A test
movie rated 4.0+ counts as relevant.

Metrics @K: Precision, Recall, catalogue Coverage, Intra-List Similarity,
Novelty (self-information, bits), Genre-entropy gain and the Bias Reduction
Index (see biasrec.engine.RecommenderEngine.bias_reduction).

Usage:
    python scripts/evaluate.py                 # K=10, writes reports/
    python scripts/evaluate.py --k 20 --data data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biasrec.data import load_dataset  # noqa: E402
from biasrec.engine import RecommenderEngine, Weights  # noqa: E402


def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2, min_ratings: int = 20):
    ratings = ratings.sort_values(["userId", "timestamp"])
    train_parts, test_parts = [], []
    for _, g in ratings.groupby("userId"):
        if len(g) < min_ratings:
            train_parts.append(g)
            continue
        cut = int(round(len(g) * (1 - test_frac)))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()
    k = args.k

    t0 = time.time()
    raw = pd.read_csv(Path(args.data) / "ratings.csv")
    train, test = temporal_split(raw)
    ds = load_dataset(args.data, ratings=train)
    engine = RecommenderEngine(ds, build_map=False)
    id_to_idx = dict(zip(engine.movies["movieId"], engine.movies["idx"]))
    test = test[test["movieId"].isin(id_to_idx)].copy()
    test["idx"] = test["movieId"].map(id_to_idx)
    relevant = {u: set(g.loc[g["rating"] >= 4.0, "idx"]) for u, g in test.groupby("userId")}
    users = [u for u, rel in relevant.items() if rel and u in engine._user_groups]
    print(f"engine trained in {time.time() - t0:.1f}s, evaluating {len(users)} users @K={k}")

    pop_order = np.argsort(engine.movies["n_ratings"].to_numpy())[::-1]

    def popularity(profile):
        seen = set(profile.idx.tolist())
        return [int(i) for i in pop_order if i not in seen][:k]

    def top_k(score, profile):
        score = score.astype(float).copy()
        score[~engine._eligible(profile)] = -np.inf
        return [int(i) for i in np.argsort(score)[::-1][:k]]

    def content_only(profile):
        return top_k(engine.content.score(profile.idx, profile.ratings), profile)

    def hybrid(profile):
        S = engine._scores(profile, Weights(0.4, 0.6, 0, 0), 0.0)
        return top_k(S["score"], profile)

    models = {
        "Popularity": popularity,
        "Content-based": content_only,
        "Collaborative (PureSVD)": lambda p: engine.baseline(p, k),
        "Hybrid (content + CF)": hybrid,
        "Bias-aware, exploration 0.3": lambda p: engine.recommend(p, k, 0.3)[0],
        "Bias-aware, exploration 0.6": lambda p: engine.recommend(p, k, 0.6)[0],
        "Bias-aware, exploration 0.9": lambda p: engine.recommend(p, k, 0.9)[0],
    }

    results = {}
    for name, fn in models.items():
        t1 = time.time()
        prec, rec, ils, nov, gain, bri, items = [], [], [], [], [], [], set()
        for u in users:
            profile = engine.profile_for_user(u)
            recs = fn(profile)
            hits = len(set(recs) & relevant[u])
            prec.append(hits / k)
            rec.append(hits / len(relevant[u]))
            items.update(recs)
            m = engine.list_metrics(profile, recs)
            ils.append(m["intra_list_similarity"])
            nov.append(m["novelty"])
            gain.append(m["genre_entropy_after"] - m["genre_entropy_before"])
            bri.append(m["bias_reduction_index"])
        results[name] = {
            f"precision@{k}": float(np.mean(prec)), f"recall@{k}": float(np.mean(rec)),
            "coverage": len(items) / int(engine.support.sum()),
            "intra_list_similarity": float(np.mean(ils)), "novelty_bits": float(np.mean(nov)),
            "genre_entropy_gain": float(np.mean(gain)), "bias_reduction_index": float(np.mean(bri)),
        }
        print(f"  {name:30s} done in {time.time() - t1:.1f}s  "
              f"P@{k}={results[name][f'precision@{k}']:.4f}  BRI={results[name]['bias_reduction_index']:+.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(
        {"k": k, "n_users": len(users), "protocol": "temporal 80/20 per user, relevant = rating >= 4",
         "results": results}, indent=2))

    cols = [f"precision@{k}", f"recall@{k}", "coverage", "intra_list_similarity",
            "novelty_bits", "genre_entropy_gain", "bias_reduction_index"]
    header = ["Model", f"Precision@{k}", f"Recall@{k}", "Coverage", "ILS ↓", "Novelty (bits)",
              "Genre entropy gain", "Bias Reduction Index"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, r in results.items():
        cells = [f"{r[c]:+.3f}" if c in ("bias_reduction_index", "genre_entropy_gain") else f"{r[c]:.3f}"
                 for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    md = (f"# Offline evaluation\n\nMovieLens ml-latest-small, temporal 80/20 split per user, "
          f"{len(users)} test users, K = {k}. A held-out rating of 4.0 or more counts as relevant.\n\n"
          + "\n".join(lines) + "\n\nGenerated by `python scripts/evaluate.py`.\n")
    (out / "evaluation.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
