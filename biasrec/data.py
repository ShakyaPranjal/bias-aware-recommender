"""Loading and cleaning MovieLens (+ optional TMDB overviews)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
ARTICLE_RE = re.compile(r"^(.*), (The|A|An|Les|La|Le|Il|Der|Die|Das|El)(\s*\(.*\))?$")


@dataclass
class Dataset:
    movies: pd.DataFrame   # one row per movie, indexed 0..n-1 by `idx`
    ratings: pd.DataFrame  # userId, movieId, rating, timestamp, idx


def _clean_title(raw: str) -> str:
    title = YEAR_RE.sub("", raw).strip()
    # "Matrix, The" -> "The Matrix" (keeps alternate titles intact)
    m = ARTICLE_RE.match(title)
    if m:
        title = f"{m.group(2)} {m.group(1)}{m.group(3) or ''}"
    return title


def _parse_json_names(value) -> list[str]:
    if not isinstance(value, str) or not value.startswith("["):
        return []
    try:
        return [d["name"] for d in json.loads(value)]
    except (ValueError, KeyError, TypeError):
        return []


def load_dataset(data_dir: str | Path, min_movie_ratings: int = 1,
                 ratings: pd.DataFrame | None = None) -> Dataset:
    """Load MovieLens CSVs from ``data_dir`` and build a clean movie table.

    Expected files: movies.csv, ratings.csv (required), tags.csv, links.csv,
    tmdb_5000_movies.csv (optional, adds plot overviews + keywords).
    ``ratings`` overrides ratings.csv (used for train/test evaluation).
    """
    data_dir = Path(data_dir)
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv") if ratings is None else ratings.copy()

    # --- cleaning ---------------------------------------------------------
    ratings = ratings.dropna().drop_duplicates(["userId", "movieId"], keep="last")
    ratings = ratings[ratings["rating"].between(0.5, 5.0)]
    counts = ratings.groupby("movieId").size()
    keep = counts[counts >= min_movie_ratings].index
    movies = movies[movies["movieId"].isin(keep)].copy()

    movies["year"] = movies["title"].str.extract(YEAR_RE)[0].astype(float)
    movies["clean_title"] = movies["title"].map(_clean_title)
    movies["genre_list"] = movies["genres"].map(
        lambda g: [x for x in str(g).split("|") if x and x != "(no genres listed)"]
    )

    # --- tags (user generated) -------------------------------------------
    tags_path = data_dir / "tags.csv"
    if tags_path.exists():
        tags = pd.read_csv(tags_path).dropna(subset=["tag"])
        tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
        tag_text = tags.groupby("movieId")["tag"].apply(lambda s: " ".join(sorted(set(s))))
        movies["tags"] = movies["movieId"].map(tag_text).fillna("")
    else:
        movies["tags"] = ""

    # --- TMDB enrichment: overview + keywords -----------------------------
    movies["overview"] = ""
    movies["keywords"] = ""
    links_path, tmdb_path = data_dir / "links.csv", data_dir / "tmdb_5000_movies.csv"
    if links_path.exists() and tmdb_path.exists():
        links = pd.read_csv(links_path).dropna(subset=["tmdbId"])
        tmdb = pd.read_csv(tmdb_path, usecols=["id", "overview", "keywords"])
        tmdb["keywords"] = tmdb["keywords"].map(lambda v: " | ".join(_parse_json_names(v)).lower())
        enriched = links.merge(tmdb, left_on="tmdbId", right_on="id", how="inner")
        enriched = enriched.drop_duplicates("movieId").set_index("movieId")
        movies["overview"] = movies["movieId"].map(enriched["overview"]).fillna("")
        movies["keywords"] = movies["movieId"].map(enriched["keywords"]).fillna("")

    # --- popularity & quality ---------------------------------------------
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    movies["n_ratings"] = movies["movieId"].map(stats["count"]).fillna(0).astype(int)
    movies["mean_rating"] = movies["movieId"].map(stats["mean"]).fillna(0.0)
    # Bayesian average so a single 5-star rating does not dominate
    global_mean, m = ratings["rating"].mean(), 10
    movies["quality"] = (
        movies["n_ratings"] * movies["mean_rating"] + m * global_mean
    ) / (movies["n_ratings"] + m)
    movies["popularity_pct"] = movies["n_ratings"].rank(pct=True)

    movies = movies.reset_index(drop=True)
    movies["idx"] = np.arange(len(movies))
    id_to_idx = dict(zip(movies["movieId"], movies["idx"]))
    ratings = ratings[ratings["movieId"].isin(id_to_idx)].copy()
    ratings["idx"] = ratings["movieId"].map(id_to_idx).astype(int)
    return Dataset(movies=movies, ratings=ratings.reset_index(drop=True))
