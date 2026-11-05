import math

import numpy as np

from biasrec.bias import BIAS_KEYS, comfort_zone_index, entropy, normalized_entropy


def test_entropy_matches_formula():
    p = np.array([0.5, 0.25, 0.25])
    assert math.isclose(entropy(p), -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25)))


def test_entropy_ignores_zeros_and_normalises_counts():
    assert math.isclose(entropy(np.array([2, 2, 0])), math.log(2))
    assert entropy(np.array([0, 0])) == 0.0


def test_normalized_entropy_bounds():
    assert math.isclose(normalized_entropy(np.ones(8), 8), 1.0)
    assert normalized_entropy(np.array([5, 0, 0, 0]), 4) == 0.0


def test_comfort_zone_index():
    tight = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    spread = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    assert comfort_zone_index(tight) == 0.0
    assert comfort_zone_index(spread) > 0.5


def test_profile_scores_are_percentiles(engine):
    profile = engine.profile_for_user(1)
    scores = profile.details["scores"]
    assert set(scores) == set(BIAS_KEYS)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_negativity_bias_detects_dark_history(engine):
    val = engine.features["valence"]
    dark = np.argsort(val)[:15]
    bright = np.argsort(val)[-15:]
    r = np.full(15, 4.5)
    p_dark = engine.bias.analyze(dark, r)
    p_bright = engine.bias.analyze(bright, r)
    assert p_dark.raw["negativity"] > p_bright.raw["negativity"]
    assert p_dark.details["scores"]["negativity"] >= p_bright.details["scores"]["negativity"]


def test_counter_vectors_in_unit_range(engine):
    profile = engine.profile_for_user(2)
    for key, vec in engine.bias.counter_vectors(profile).items():
        assert vec.shape == (len(engine.movies),), key
        assert np.all(vec >= -1e-9) and np.all(vec <= 1 + 1e-9), key
