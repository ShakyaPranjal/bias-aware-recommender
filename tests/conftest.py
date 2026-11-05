"""Synthetic MovieLens-shaped dataset so tests run without downloading data."""

import numpy as np
import pandas as pd
import pytest

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation",
          "Documentary", "War"]
WORDS = ["love", "murder", "family", "war", "space", "friendship", "revenge", "ghost", "hope",
         "heist", "school", "music", "alien", "prison", "wedding", "storm", "detective", "dream"]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("ml") / "raw"
    root.mkdir()
    movies = []
    for i in range(1, 121):
        gs = rng.choice(GENRES, size=rng.integers(1, 4), replace=False)
        movies.append({"movieId": i, "title": f"Movie {i} ({rng.integers(1950, 2018)})",
                       "genres": "|".join(gs)})
    movies = pd.DataFrame(movies)
    movies.to_csv(root / "movies.csv", index=False)

    rows, tags = [], []
    for u in range(1, 61):
        # each user prefers a couple of genres -> realistic, biased histories
        fav = set(rng.choice(GENRES, size=2, replace=False))
        weights = movies["genres"].map(lambda g: 4.0 if fav & set(g.split("|")) else 1.0).to_numpy()
        n = int(rng.integers(20, 45))
        picks = rng.choice(movies["movieId"], size=n, replace=False, p=weights / weights.sum())
        ts = np.sort(rng.integers(1_100_000_000, 1_500_000_000, size=n))
        for m, t in zip(picks, ts):
            liked = fav & set(movies.loc[movies.movieId == m, "genres"].iloc[0].split("|"))
            r = float(np.clip(rng.normal(4.0 if liked else 2.8, 0.7), 0.5, 5.0))
            rows.append({"userId": u, "movieId": int(m), "rating": round(r * 2) / 2, "timestamp": int(t)})
    pd.DataFrame(rows).to_csv(root / "ratings.csv", index=False)

    for m in movies["movieId"]:
        for w in rng.choice(WORDS, size=3, replace=False):
            tags.append({"userId": 1, "movieId": int(m), "tag": w, "timestamp": 1_400_000_000})
    pd.DataFrame(tags).to_csv(root / "tags.csv", index=False)
    return root


@pytest.fixture(scope="session")
def engine(data_dir):
    from biasrec.engine import RecommenderEngine
    return RecommenderEngine.from_dir(data_dir, min_support=2)
