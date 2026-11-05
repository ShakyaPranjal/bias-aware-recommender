"""The in-browser engine (web/engine.js) must match the Python engine.

Exports the trained model with scripts/build_static.py, runs the same API
requests through FastAPI and through engine.js in Node, and compares the
results: recommendations, bias scores, explanations and metrics.
"""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")


def assert_close(a, b, path="$"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys differ {set(a) ^ set(b)}"
        for k in a:
            assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list), f"{path}: expected a list, got {b!r}"
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def run_parity(engine, model_dir: Path, requests: list, tmp_path: Path):
    from build_static import export

    from biasrec import api

    export(engine, model_dir)
    api.state.clear()
    api.state["engine"] = engine
    api.state["featured"] = engine.featured_users()
    with TestClient(api.app) as client:
        expected = [client.post(r["path"], json=r["body"]).json() if r.get("body") else client.get(r["path"]).json()
                    for r in requests]
    req_file = tmp_path / "requests.json"
    req_file.write_text(json.dumps(requests))
    proc = subprocess.run(["node", str(ROOT / "tests" / "static_runner.js"), str(model_dir), str(req_file)],
                          capture_output=True, text=True, check=True)
    actual = json.loads(proc.stdout)
    for req, exp, act in zip(requests, expected, actual):
        assert_close(exp, act, req["path"])


def test_static_engine_matches_python(engine, tmp_path):
    requests = [{"path": f"/api/users/{u}/analysis?exploration={x}&k=6"}
                for u in (1, 7, 23, 42) for x in (0.0, 0.55, 1.0)]
    requests += [
        {"path": "/api/meta"},
        {"path": "/api/map?n=100"},
        {"path": "/api/movies/search?q=movie 1&limit=5"},
        {"path": "/api/analyze", "body": {"ratings": [{"movieId": i, "rating": r} for i, r in
                                                      [(3, 5), (8, 4.5), (15, 2), (21, 4), (40, 3.5), (77, 1)]],
                                          "exploration": 0.7, "k": 6}},
    ]
    run_parity(engine, tmp_path / "model", requests, tmp_path)
