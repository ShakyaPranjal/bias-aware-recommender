import numpy as np

from biasrec.engine import Weights


def test_weights_follow_exploration():
    low, high = Weights.from_exploration(0.0), Weights.from_exploration(1.0)
    assert low.beta > high.beta and low.alpha > high.alpha
    assert high.gamma > low.gamma and high.delta > low.delta


def test_recommendations_exclude_watched(engine):
    profile = engine.profile_for_user(3)
    picks, _, _ = engine.recommend(profile, k=8)
    assert len(picks) == 8
    assert len(set(picks)) == 8
    assert not set(picks) & set(profile.idx.tolist())


def test_more_exploration_counters_bias_more(engine):
    gains = []
    for u in range(1, 21):
        p = engine.profile_for_user(u)
        lo = engine.bias_reduction(p, engine.recommend(p, k=8, exploration=0.0)[0])[0]
        hi = engine.bias_reduction(p, engine.recommend(p, k=8, exploration=1.0)[0])[0]
        gains.append(hi - lo)
    assert np.mean(gains) > 0


def test_analyze_payload_shape(engine):
    out = engine.analyze(engine.profile_for_user(4), k=6, exploration=0.5)
    assert len(out["recommendations"]) == 6
    assert len(out["biases"]) == 6
    for rec in out["recommendations"]:
        assert rec["explanation"]["reason"]
        assert rec["explanation"]["role"] in {"Bridge", "Expand", "Stretch"}
    assert {"bias_aware", "traditional"} <= set(out["metrics"])
    assert len(out["map"]["recommended"]) == 6


def test_cold_start_profile(engine):
    ids = engine.movies["movieId"].head(6).tolist()
    profile = engine.profile_from_ratings([{"movieId": i, "rating": 4.0} for i in ids])
    out = engine.analyze(profile, k=5)
    assert len(out["recommendations"]) == 5


def test_search_finds_titles(engine):
    hits = engine.search("movie 1")
    assert hits and all("movie 1" in h["title"].lower() for h in hits)
