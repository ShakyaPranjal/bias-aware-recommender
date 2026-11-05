"""Content-based and collaborative filtering models.

Both models work from a sparse "profile" (movie index -> rating), so the same
code serves MovieLens users and brand-new users who rate a few movies in the
web app (no retraining needed: new users are folded into the latent space).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class ContentModel:
    """Cosine similarity between candidate movies and a taste vector built from
    the user's ratings (ratings above the user's mean pull the vector toward a
    movie, ratings below push it away)."""

    def __init__(self, embedding: np.ndarray):
        self.E = embedding

    def taste_vector(self, idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        w = ratings - ratings.mean()
        if np.allclose(w, 0):
            w = np.ones_like(ratings)
        # liked movies count more than disliked ones push away
        w = np.where(w > 0, w, 0.35 * w)
        v = (w[:, None] * self.E[idx]).sum(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def score(self, idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        sims = self.E @ self.taste_vector(idx, ratings)
        return _minmax(sims)


class CollaborativeModel:
    """PureSVD matrix factorisation (Cremonesi et al., 2010).

    R ~ U S V^T.  A user's score vector is r_u V V^T, which is the
    projection of their ratings onto the item latent space. This makes
    fold-in for new users exact and free.
    """

    def __init__(self, n_factors: int = 60, seed: int = 42):
        self.n_factors = n_factors
        self.seed = seed

    def fit(self, ratings: pd.DataFrame, n_items: int) -> "CollaborativeModel":
        users = ratings["userId"].astype("category")
        self.user_codes = dict(zip(users.cat.categories, range(len(users.cat.categories))))
        R = sparse.csr_matrix(
            (ratings["rating"].to_numpy(float), (users.cat.codes.to_numpy(), ratings["idx"].to_numpy())),
            shape=(len(self.user_codes), n_items),
        )
        k = max(2, min(self.n_factors, min(R.shape) - 1))
        _, s, vt = svds(R, k=k, random_state=self.seed)
        order = np.argsort(s)[::-1]
        self.s = s[order]
        self.V = vt[order].T.astype(np.float32)              # items x k
        # item-item similarity in latent space (scaled by singular values)
        F = self.V * np.sqrt(self.s)[None, :]
        norms = np.linalg.norm(F, axis=1, keepdims=True)
        self.item_factors = F / np.where(norms == 0, 1, norms)
        return self

    def raw_scores(self, idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        r = np.zeros(self.V.shape[0], dtype=np.float32)
        r[idx] = ratings
        return (r @ self.V) @ self.V.T

    def score(self, idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        return _minmax(self.raw_scores(idx, ratings))

    def most_similar_rated(self, target: int, idx: np.ndarray, ratings: np.ndarray,
                           min_rating: float = 3.5) -> tuple[int, float] | None:
        liked = idx[ratings >= min_rating]
        if len(liked) == 0:
            return None
        sims = self.item_factors[liked] @ self.item_factors[target]
        j = int(np.argmax(sims))
        return int(liked[j]), float(sims[j])
