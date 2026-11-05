"""Download the datasets used by the Bias-Aware Recommender.

* MovieLens ml-latest-small (ratings, movies, tags, links) from GroupLens
* TMDB 5000 movie metadata (plot overviews + keywords), optional but
  recommended: it powers sentiment analysis and topic modelling.

Usage:
    python scripts/download_data.py               # both datasets
    python scripts/download_data.py --skip-tmdb   # MovieLens only

If the TMDB mirror is unavailable, download ``tmdb_5000_movies.csv`` from
Kaggle (tmdb/tmdb-movie-metadata) and place it in data/raw/.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
TMDB_URL = "https://raw.githubusercontent.com/ELAREMSaber/tmdb_5000_movies/main/tmdb_5000_movies.csv"
MOVIELENS_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "bias-aware-recommender/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def download_movielens(force: bool):
    if not force and all((RAW / f).exists() for f in MOVIELENS_FILES):
        print("MovieLens already present, skipping (use --force to re-download)")
        return
    print(f"Downloading MovieLens from {MOVIELENS_URL} ...")
    with zipfile.ZipFile(io.BytesIO(fetch(MOVIELENS_URL))) as z:
        for name in z.namelist():
            base = Path(name).name
            if base in MOVIELENS_FILES + ["README.txt"]:
                with z.open(name) as src, open(RAW / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print("  saved", ", ".join(MOVIELENS_FILES))


def download_tmdb(url: str, force: bool):
    target = RAW / "tmdb_5000_movies.csv"
    if target.exists() and not force:
        print("TMDB metadata already present, skipping")
        return
    print(f"Downloading TMDB overviews from {url} ...")
    try:
        target.write_bytes(fetch(url))
        print("  saved tmdb_5000_movies.csv")
    except Exception as err:  # noqa: BLE001
        print(f"  could not download TMDB data ({err}). The app still runs without it,"
              " using tags and genres for the NLP features.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tmdb", action="store_true")
    ap.add_argument("--tmdb-url", default=TMDB_URL)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    download_movielens(args.force)
    if not args.skip_tmdb:
        download_tmdb(args.tmdb_url, args.force)
    print("Done. Start the app with: uvicorn biasrec.api:app --reload")


if __name__ == "__main__":
    main()
