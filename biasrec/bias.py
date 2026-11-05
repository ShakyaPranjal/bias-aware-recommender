"""Bias Detection Engine.

Quantifies six psychological tendencies from a viewing history. Each bias has
a *raw* metric (interpretable on its own) and a *score* in [0, 1], which is
the viewer's percentile among all MovieLens users (0.9 = stronger than 90%
of viewers). Scores drive the bias adjustment; raw metrics drive the
explanations.

Formulas follow the project proposal:
  Negativity  N_b = #{movies with negative sentiment} / #{all watched movies}
  Genre entropy  H = -sum_i p_i log p_i     (low H -> tunnel vision)
  Comfort zone   C_z = 1/N sum_i ||v_i - c||  with c the embedding centroid
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NEGATIVE_VALENCE = -0.15

BIAS_INFO = {
    "negativity": ("Negativity bias", "Prefers dark, tragic or violent stories over hopeful ones."),
    "novelty_avoidance": ("Novelty avoidance", "Stays in the same genres, eras and mainstream titles."),
    "confirmation": ("Confirmation bias", "Keeps returning to the same themes and rates them highest."),
    "recency": ("Recency bias", "Favours new releases and rates recent watches more generously."),
    "hero_projection": ("Hero projection", "Gravitates to hero / saviour arcs that mirror an ideal self."),
    "comfort_zone": ("Comfort zone", "History sits in a tight cluster of very similar films."),
}
BIAS_KEYS = list(BIAS_INFO)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def normalized_entropy(counts: np.ndarray, n_categories: int) -> float:
    if n_categories <= 1:
        return 0.0
    return entropy(counts) / np.log(n_categories)


def comfort_zone_index(vectors: np.ndarray) -> float:
    if len(vectors) == 0:
        return 0.0
    c = vectors.mean(axis=0)
    return float(np.linalg.norm(vectors - c, axis=1).mean())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class Profile:
    idx: np.ndarray
    ratings: np.ndarray
    timestamps: np.ndarray | None = None
    raw: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)


class BiasEngine:
    def __init__(self, features: dict, movies):
        self.f = features
        self.movies = movies
        self.years = movies["year"].fillna(movies["year"].median()).to_numpy()
        self.decades = (self.years // 10 * 10).astype(int)
        self.decade_values = np.unique(self.decades)
        # log-scaled popularity: 0 = unseen, 1 = the most-rated film
        n = movies["n_ratings"].to_numpy(float)
        self.pop = np.log1p(n) / np.log1p(max(n.max(), 1))
        self.population: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ raw
    def raw_metrics(self, idx: np.ndarray, ratings: np.ndarray, ts: np.ndarray | None = None):
        f = self.f
        G = f["genre_matrix"][idx]
        genre_counts = G.sum(axis=0)
        n_genres = G.shape[1]
        details = {}

        # 1. negativity  N_b
        val = f["valence"][idx]
        nb = float((val < NEGATIVE_VALENCE).mean())
        details["negative_share"] = nb
        details["mean_valence"] = float(val.mean())
        details["emotions"] = f["emotions"][idx].mean(axis=0)

        # 2. novelty avoidance = genre concentration + era concentration + mainstreamness
        h_genre = normalized_entropy(genre_counts, n_genres)
        dec_counts = np.array([(self.decades[idx] == d).sum() for d in self.decade_values])
        h_era = normalized_entropy(dec_counts, min(len(self.decade_values), 8))
        mainstream = float(self.pop[idx].mean())
        novelty_avoid = (1 - h_genre) * 0.4 + (1 - min(h_era, 1)) * 0.25 + mainstream * 0.35
        details.update(genre_entropy=h_genre, era_entropy=min(h_era, 1.0), mainstream=mainstream,
                       genre_counts=genre_counts, decade_counts=dec_counts)

        # 3. confirmation bias = theme concentration + in-theme rating polarisation
        T = f["themes"][idx]
        dominant = T.argmax(axis=1)
        theme_counts = np.bincount(dominant, minlength=T.shape[1])
        theme_conc = 1 - normalized_entropy(theme_counts, T.shape[1])
        top_themes = np.argsort(theme_counts)[::-1][:3]
        inside = np.isin(dominant, top_themes)
        gap = 0.0
        if inside.any() and (~inside).any():
            gap = float(ratings[inside].mean() - ratings[~inside].mean())
        confirmation = 0.6 * theme_conc + 0.4 * float(np.clip(gap / 1.5, 0, 1))
        details.update(theme_concentration=theme_conc, theme_rating_gap=gap,
                       top_themes=top_themes.tolist(), theme_counts=theme_counts)

        # 4. recency bias = rating drift over time + appetite for new releases
        drift, fresh = 0.0, 0.0
        if ts is not None and len(ts) == len(idx):
            drift = max(0.0, _spearman(ts, ratings))
            watch_year = 1970 + ts / (365.25 * 86400)
            fresh = float((watch_year - self.years[idx] <= 2.5).mean())
        recency = 0.45 * drift + 0.55 * fresh
        details.update(rating_drift=drift, fresh_release_share=fresh,
                       median_year=float(np.median(self.years[idx])))

        # 5. hero projection = rating-weighted hero archetype share
        w = np.clip(ratings, 0.5, None)
        hero = float((f["hero"][idx] * w).sum() / w.sum())
        details["hero_share"] = float((f["hero"][idx] >= 0.5).mean())

        # 6. comfort zone  C_z (low = tight cluster)
        cz = comfort_zone_index(f["embedding"][idx])
        details["comfort_index"] = cz

        raw = {"negativity": nb, "novelty_avoidance": novelty_avoid,
               "confirmation": confirmation, "recency": recency,
               "hero_projection": hero, "comfort_zone": cz}
        return raw, details

    # ----------------------------------------------------------- population
    def fit_population(self, ratings_df, min_ratings: int = 15):
        rows = {k: [] for k in BIAS_KEYS}
        for _, g in ratings_df.groupby("userId"):
            if len(g) < min_ratings:
                continue
            raw, _ = self.raw_metrics(g["idx"].to_numpy(), g["rating"].to_numpy(float),
                                      g["timestamp"].to_numpy(float) if "timestamp" in g else None)
            for k, v in raw.items():
                rows[k].append(v)
        self.population = {k: np.sort(np.array(v)) for k, v in rows.items()}
        return self

    def percentile(self, key: str, value: float) -> float:
        pop = self.population.get(key)
        if pop is None or len(pop) == 0:
            return 0.5
        p = np.searchsorted(pop, value, side="right") / len(pop)
        return float(1 - p) if key == "comfort_zone" else float(p)

    def typical(self, key: str) -> float:
        pop = self.population.get(key)
        return float(np.median(pop)) if pop is not None and len(pop) else 0.0

    # --------------------------------------------------------------- public
    def analyze(self, idx, ratings, ts=None) -> Profile:
        raw, details = self.raw_metrics(idx, ratings, ts)
        prof = Profile(idx=idx, ratings=ratings, timestamps=ts, raw=raw, details=details)
        prof.details["scores"] = {k: self.percentile(k, raw[k]) for k in BIAS_KEYS}
        return prof

    def strengths(self, profile: Profile) -> dict[str, float]:
        """Turn percentiles into adjustment strengths. Below the 35th
        percentile a tendency is ordinary and is left alone."""
        return {k: float(np.clip((s - 0.35) / 0.65, 0, 1)) ** 1.3
                for k, s in profile.details["scores"].items()}

    def counter_vectors(self, profile: Profile) -> dict[str, np.ndarray]:
        """For every bias, how much each movie in the catalogue would
        counterbalance it (0 = reinforces, 1 = strongly counters)."""
        f, d = self.f, profile.details
        idx = profile.idx

        # only films brighter than the viewer's usual tone counter negativity
        positivity = np.clip((f["valence"] - d["mean_valence"]) / 0.9, 0, 1)

        gshare = d["genre_counts"] / max(d["genre_counts"].sum(), 1)
        gfam = gshare / max(gshare.max(), 1e-9)
        Gm = f["genre_matrix"]
        n_g = np.maximum(Gm.sum(axis=1), 1)
        genre_novelty = 1 - (Gm @ gfam) / n_g
        dshare = d["decade_counts"] / max(d["decade_counts"].sum(), 1)
        dec_lookup = dict(zip(self.decade_values, dshare / max(dshare.max(), 1e-9)))
        era_novelty = 1 - np.array([dec_lookup.get(x, 0.0) for x in self.decades])
        novelty = 0.5 * genre_novelty + 0.2 * era_novelty + 0.3 * (1 - self.pop)

        theme_profile = f["themes"][idx].mean(axis=0)
        theme_profile /= np.linalg.norm(theme_profile) + 1e-9
        T = f["themes"] / (np.linalg.norm(f["themes"], axis=1, keepdims=True) + 1e-9)
        theme_distance = 1 - T @ theme_profile

        older = np.clip((d["median_year"] - self.years) / 30.0, 0, 1)

        c = f["embedding"][idx].mean(axis=0)
        dist = np.linalg.norm(f["embedding"] - c, axis=1)
        comfort = (dist - dist.min()) / (np.ptp(dist) + 1e-9)

        return {
            "negativity": positivity,
            "novelty_avoidance": np.clip(novelty, 0, 1),
            "confirmation": np.clip(theme_distance / (theme_distance.max() + 1e-9), 0, 1),
            "recency": older,
            "hero_projection": 1 - f["hero"],
            "comfort_zone": comfort,
        }
