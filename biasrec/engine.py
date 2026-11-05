"""Recommendation engine: wires features, models, bias engine and explainer.

    Score(m, u) = α·Content(m,u) + β·CF(m,u) + γ·BiasAdj(m,u) + δ·Serendipity(m,u)

followed by a diversity-aware re-ranking (Maximal Marginal Relevance) so the
final list is not ten variations of the same film.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .bias import BIAS_INFO, BIAS_KEYS, BiasEngine, Profile, desc_order, normalized_entropy
from .data import Dataset, load_dataset
from .explain import Explainer
from .models import CollaborativeModel, ContentModel, _minmax
from .nlp import EMOTIONS, FeatureBuilder

log = logging.getLogger("biasrec")


@dataclass
class Weights:
    alpha: float  # content
    beta: float   # collaborative
    gamma: float  # bias adjustment
    delta: float  # serendipity

    @classmethod
    def from_exploration(cls, x: float) -> "Weights":
        """One slider controls the trade-off between comfort and growth."""
        x = float(np.clip(x, 0, 1))
        # Collaborative filtering is the strongest relevance signal on MovieLens,
        # so content similarity gets a smaller share (tuned in scripts/evaluate.py).
        return cls(alpha=0.12 * (1 - x) + 0.05, beta=0.68 * (1 - x) + 0.15,
                   gamma=0.45 * x + 0.05, delta=0.25 * x + 0.05)

    def as_dict(self):
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "delta": self.delta}


class RecommenderEngine:
    def __init__(self, dataset: Dataset, n_factors: int = 60, min_support: int = 8,
                 cache_dir: str | Path | None = None, build_map: bool = True):
        t0 = time.time()
        self.data = dataset
        self.movies = dataset.movies
        self.ratings = dataset.ratings
        self.min_support = min_support
        self.features = FeatureBuilder().fit(self.movies)
        # float32-rounded features, float64 maths: identical results in the
        # Python API and the in-browser build (scripts/build_static.py)
        for key in ("embedding", "themes", "genre_matrix", "valence", "emotions", "hero"):
            self.features[key] = self.features[key].astype(np.float64)
        self.content = ContentModel(self.features["embedding"])
        self.cf = CollaborativeModel(n_factors=n_factors).fit(self.ratings, len(self.movies))
        self.bias = BiasEngine(self.features, self.movies).fit_population(self.ratings)
        self.explainer = Explainer(self)
        self.quality_norm = _minmax(self.movies["quality"].to_numpy())
        self.support = self.movies["n_ratings"].to_numpy() >= min_support
        self._map = (self._build_map(cache_dir) if build_map else
                     PCA(n_components=2, random_state=0).fit_transform(self.features["embedding"]))
        self._user_groups = {u: g for u, g in self.ratings.groupby("userId")}
        self.n_users = max(len(self._user_groups), 1)
        self.title_index = self.movies["clean_title"].str.lower().to_numpy()
        log.info("engine ready in %.1fs (%d movies, %d ratings)",
                 time.time() - t0, len(self.movies), len(self.ratings))

    @classmethod
    def from_dir(cls, data_dir: str | Path, **kw) -> "RecommenderEngine":
        kw.setdefault("cache_dir", Path(data_dir).parent / "cache")
        return cls(load_dataset(data_dir), **kw)

    def _build_map(self, cache_dir) -> np.ndarray:
        """2-D 'story space' for the comfort-zone map.

        t-SNE keeps similar films together, which is what the map needs to
        show. It runs on films with 3+ ratings; rarer films are placed at the
        mean position of their nearest embedded neighbours. Cached on disk.
        """
        E = self.features["embedding"]
        n = self.movies["n_ratings"].to_numpy()
        key = f"map_{len(E)}_{int(abs(E.sum()) * 1000) % 10**8}.npy"
        path = Path(cache_dir) / key if cache_dir else None
        if path is not None and path.exists():
            return np.load(path)
        core = np.where(n >= 3)[0]
        if len(core) < 60:
            Y = PCA(n_components=2, random_state=0).fit_transform(E)
        else:
            from sklearn.manifold import TSNE
            t0 = time.time()
            Yc = TSNE(n_components=2, perplexity=min(35, (len(core) - 1) // 3), init="pca",
                      metric="cosine", random_state=0).fit_transform(E[core])
            Y = np.zeros((len(E), 2))
            Y[core] = Yc
            rest = np.setdiff1d(np.arange(len(E)), core)
            for start in range(0, len(rest), 2048):
                chunk = rest[start:start + 2048]
                sims = E[chunk] @ E[core].T
                nn = np.argsort(-sims, axis=1)[:, :5]
                Y[chunk] = Yc[nn].mean(axis=1)
            log.info("story-space map built in %.1fs", time.time() - t0)
        Y = (Y - Y.mean(axis=0)) / (np.abs(Y).max() + 1e-9)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, Y)
        return Y

    # ----------------------------------------------------------- profiles
    def user_history(self, user_id: int):
        g = self._user_groups.get(user_id)
        if g is None:
            raise KeyError(user_id)
        g = g.sort_values("timestamp", kind="stable")
        return g["idx"].to_numpy(), g["rating"].to_numpy(float), g["timestamp"].to_numpy(float)

    def profile_for_user(self, user_id: int) -> Profile:
        return self.bias.analyze(*self.user_history(user_id))

    def profile_from_ratings(self, items: list[dict]) -> Profile:
        id_to_idx = dict(zip(self.movies["movieId"], self.movies["idx"]))
        pairs = [(id_to_idx[int(i["movieId"])], float(i["rating"]))
                 for i in items if int(i["movieId"]) in id_to_idx]
        if not pairs:
            raise ValueError("none of the rated movies are in the catalogue")
        idx = np.array([p[0] for p in pairs])
        r = np.array([p[1] for p in pairs])
        return self.bias.analyze(idx, r, None)

    # -------------------------------------------------------------- core
    @staticmethod
    def _rank_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Percentile-rank among eligible movies, so components with very
        different raw scales (cosine vs. latent dot products) compete fairly."""
        out = np.zeros_like(x, dtype=float)
        if mask.any():
            order = np.argsort(np.argsort(x[mask], kind="stable"), kind="stable")
            out[mask] = (order + 1) / mask.sum()
        return out

    def _scores(self, profile: Profile, weights: Weights, exploration: float):
        elig = self._eligible(profile)
        content = self._rank_norm(self.content.score(profile.idx, profile.ratings), elig)
        cf = self._rank_norm(self.cf.raw_scores(profile.idx, profile.ratings), elig)
        strengths = self.bias.strengths(profile)
        counters = self.bias.counter_vectors(profile)
        total = sum(strengths.values())
        bias_adj = (sum(strengths[k] * counters[k] for k in BIAS_KEYS) / total
                    if total > 1e-9 else np.zeros(len(self.movies)))
        stretch = counters["comfort_zone"]
        target = 0.30 + 0.30 * exploration     # sweet spot just outside the comfort zone
        serendipity = self.quality_norm * np.exp(-((stretch - target) ** 2) / (2 * 0.14 ** 2))
        score = (weights.alpha * content + weights.beta * cf +
                 weights.gamma * bias_adj + weights.delta * serendipity)
        return {"score": score, "content": content, "cf": cf, "bias_adj": bias_adj,
                "serendipity": serendipity, "stretch": stretch,
                "strengths": strengths, "counters": counters}

    def _eligible(self, profile: Profile) -> np.ndarray:
        mask = self.support.copy()
        mask[profile.idx] = False
        return mask

    def _mmr(self, score: np.ndarray, candidates: np.ndarray, k: int, lam: float) -> list[int]:
        E = self.features["embedding"]
        pool = candidates[desc_order(score[candidates])[:400]]
        if len(pool) == 0:
            return []
        s = _minmax(score[pool])
        chosen, max_sim = [], np.zeros(len(pool))
        available = np.ones(len(pool), dtype=bool)
        for _ in range(min(k, len(pool))):
            mmr = (1 - lam) * s - lam * max_sim
            mmr[~available] = -np.inf
            j = int(np.argmax(mmr))
            chosen.append(int(pool[j]))
            available[j] = False
            max_sim = np.maximum(max_sim, E[pool] @ E[pool[j]])
        return chosen

    def recommend(self, profile: Profile, k: int = 12, exploration: float = 0.5,
                  weights: Weights | None = None, diversify: bool = True):
        weights = weights or Weights.from_exploration(exploration)
        S = self._scores(profile, weights, exploration)
        elig = self._eligible(profile)
        # Relevance floor: bias-countering picks must still be plausible for this user
        relevance = S["cf"]
        keep_share = 0.05 + 0.40 * exploration
        thresh = np.quantile(relevance[elig], 1 - keep_share) if elig.any() else 0
        candidates = np.where(elig & (relevance >= thresh))[0]
        lam = 0.30 + 0.25 * exploration if diversify else 0.0
        picks = self._mmr(S["score"], candidates, k, lam)
        return picks, S, weights

    def baseline(self, profile: Profile, k: int = 12) -> list[int]:
        """What a traditional similarity-driven recommender would show."""
        cf = self.cf.raw_scores(profile.idx, profile.ratings)
        elig = self._eligible(profile)
        cf[~elig] = -np.inf
        return [int(i) for i in desc_order(cf)[:k]]

    # ------------------------------------------------------------ metrics
    def list_metrics(self, profile: Profile, recs: list[int]) -> dict:
        E = self.features["embedding"]
        G = self.features["genre_matrix"]
        n_g = G.shape[1]
        before = normalized_entropy(G[profile.idx].sum(axis=0), n_g)
        after = normalized_entropy(G[np.concatenate([profile.idx, recs])].sum(axis=0), n_g)
        if len(recs) > 1:
            sims = E[recs] @ E[recs].T
            ils = float(sims[np.triu_indices(len(recs), 1)].mean())
        else:
            ils = 0.0
        # Novelty = mean self-information, -log2 P(a user has rated the film)
        p = np.clip(self.movies["n_ratings"].to_numpy()[recs] / self.n_users, 1e-6, 1)
        novelty = float((-np.log2(p)).mean()) if len(recs) else 0.0
        bri, list_scores = self.bias_reduction(profile, recs)
        return {"genre_entropy_before": before, "genre_entropy_after": after,
                "intra_list_similarity": ils, "novelty": novelty,
                "bias_reduction_index": bri, "list_bias_scores": list_scores}

    def bias_reduction(self, profile: Profile, recs: list[int]) -> tuple[float, dict]:
        """Bias Reduction Index (BRI).

        Analyse the recommendation list as if it were a viewing history and
        compare its bias percentiles with the user's. BRI is the
        strength-weighted mean drop: > 0 means the list leans against the
        user's biases, < 0 means it reinforces them. Recency is skipped
        because a list has no watch timestamps.
        """
        if len(recs) < 3:
            return 0.0, {}
        lst = self.bias.analyze(np.asarray(recs), np.full(len(recs), 4.0), None).details["scores"]
        old = profile.details["scores"]
        w = self.bias.strengths(profile)
        keys = [k for k in BIAS_KEYS if k != "recency"]
        total = sum(w[k] for k in keys)
        if total < 1e-9:
            w, total = {k: 1.0 for k in keys}, float(len(keys))
        bri = sum(w[k] * (old[k] - lst[k]) for k in keys) / total
        return float(bri), lst

    # ------------------------------------------------------------- output
    def movie_card(self, m: int) -> dict:
        mv = self.movies.iloc[m]
        return {
            "movieId": int(mv["movieId"]), "title": mv["clean_title"],
            "year": None if np.isnan(mv["year"]) else int(mv["year"]),
            "genres": list(mv["genre_list"]), "overview": mv["overview"] or "",
            "rating": round(float(mv["mean_rating"]), 2), "n_ratings": int(mv["n_ratings"]),
            "valence": round(float(self.features["valence"][m]), 2),
            "emotion": EMOTIONS[int(np.argmax(self.features["emotions"][m]))],
        }

    def analyze(self, profile: Profile, k: int = 12, exploration: float = 0.5,
                weights: Weights | None = None) -> dict:
        picks, S, weights = self.recommend(profile, k, exploration, weights)
        base = self.baseline(profile, k)
        strengths = S["strengths"]
        scores = profile.details["scores"]

        recs, used = [], {b: 0 for b in BIAS_KEYS}
        for m in picks:
            # each bias gets credit less often as the list goes on, so the
            # explanations cover the whole profile instead of repeating one
            contrib = {b: strengths[b] * float(S["counters"][b][m]) * 0.6 ** used[b] for b in BIAS_KEYS}
            comp = {c: float(S[c][m]) for c in ("content", "cf", "bias_adj", "serendipity", "stretch")}
            card = self.movie_card(m)
            expl = self.explainer.recommendation(m, profile, comp, contrib, used)
            if expl["counters"]:
                used[expl["counters"]] += 1
            card.update(score=float(S["score"][m]), components=comp, explanation=expl)
            recs.append(card)

        biases = []
        for b in BIAS_KEYS:
            name, desc = BIAS_INFO[b]
            biases.append({"key": b, "name": name, "description": desc,
                           "score": scores[b], "raw": profile.raw[b],
                           "level": "high" if scores[b] >= 0.7 else "moderate" if scores[b] >= 0.4 else "low",
                           "evidence": self.explainer.bias_evidence(b, profile)})

        d = profile.details
        genres = self.features["genres"]
        catalog_share = self.features["genre_matrix"][self.ratings["idx"].to_numpy()].sum(axis=0)
        catalog_share = catalog_share / catalog_share.sum()
        user_share = d["genre_counts"] / max(d["genre_counts"].sum(), 1)
        rec_counts = self.features["genre_matrix"][picks].sum(axis=0) if picks else np.zeros(len(genres))
        rec_share = rec_counts / max(rec_counts.sum(), 1)

        liked = profile.idx[desc_order(profile.ratings)[:8]]
        return {
            "profile": {
                "n_ratings": int(len(profile.idx)),
                "mean_rating": round(float(profile.ratings.mean()), 2),
                "top_genres": self.explainer.top_genres(profile, 3),
                "favourites": [self.movie_card(int(i)) for i in liked],
                "emotions": dict(zip(EMOTIONS, map(float, d["emotions"]))),
                "genre_fingerprint": [
                    {"genre": g, "you": float(user_share[i]), "typical": float(catalog_share[i]),
                     "recommended": float(rec_share[i])} for i, g in enumerate(genres)],
            },
            "biases": biases,
            "weights": weights.as_dict(),
            "exploration": exploration,
            "recommendations": recs,
            "baseline": [self.movie_card(m) for m in base],
            "metrics": {"bias_aware": self.list_metrics(profile, picks),
                        "traditional": self.list_metrics(profile, base)},
            "map": self.map_points(profile, picks, base),
        }

    # ---------------------------------------------------------------- map
    def map_points(self, profile: Profile, picks, base) -> dict:
        P = self._map

        def pts(ids):
            return [[round(float(P[i, 0]), 4), round(float(P[i, 1]), 4)] for i in ids]
        # the centre of the comfort zone is the film closest to the user's
        # embedding centroid (a 2-D mean could land between clusters)
        E = self.features["embedding"]
        c = E[profile.idx].mean(axis=0)
        centre = int(np.argmax(E[profile.idx] @ c))
        centre = int(profile.idx[centre])
        return {"watched": pts(profile.idx[:400]), "recommended": pts(picks),
                "traditional": pts(base), "centroid": pts([centre])[0],
                "centre_title": self.movies.iloc[centre]["clean_title"]}

    def catalog_map(self, n: int = 1500) -> list:
        top = desc_order(self.movies["n_ratings"].to_numpy())[:n]
        return [[round(float(self._map[i, 0]), 4), round(float(self._map[i, 1]), 4)] for i in top]

    # ------------------------------------------------------------- helpers
    def search(self, q: str, limit: int = 12) -> list[dict]:
        q = q.lower().strip()
        if not q:
            return []
        hits = np.where(np.char.find(self.title_index.astype(str), q) >= 0)[0]
        hits = hits[desc_order(self.movies["n_ratings"].to_numpy()[hits])][:limit]
        return [self.movie_card(int(i)) for i in hits]

    def featured_users(self) -> list[dict]:
        """A few MovieLens users with a clearly dominant bias, for the demo."""
        out, used = [], set()
        rows = []
        for u, g in self._user_groups.items():
            if 40 <= len(g) <= 600:
                rows.append(u)
        cache = {u: self.profile_for_user(u).details["scores"] for u in rows}
        for b in BIAS_KEYS:
            ranked = sorted(cache, key=lambda u: cache[u][b], reverse=True)
            for u in ranked:
                if u not in used:
                    used.add(u)
                    out.append({"userId": int(u), "dominant": b, "label": BIAS_INFO[b][0],
                                "n_ratings": int(len(self._user_groups[u]))})
                    break
        balanced = min((u for u in cache if u not in used),
                       key=lambda u: np.mean(list(cache[u].values())), default=None)
        if balanced is not None:
            out.append({"userId": int(balanced), "dominant": None, "label": "Balanced viewer",
                        "n_ratings": int(len(self._user_groups[balanced]))})
        return out

