import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(data_dir, engine):
    from biasrec import api
    api.DATA_DIR = data_dir
    api.state.clear()
    api.state["engine"] = engine
    api.state["featured"] = engine.featured_users()
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_meta_lists_featured_users(client):
    meta = client.get("/api/meta").json()
    assert meta["n_movies"] == 120
    assert meta["featured_users"]


def test_user_analysis(client):
    r = client.get("/api/users/5/analysis", params={"exploration": 0.7, "k": 5})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 5


def test_unknown_user_is_404(client):
    assert client.get("/api/users/99999/analysis").status_code == 404


def test_custom_profile(client):
    body = {"ratings": [{"movieId": i, "rating": 4.5} for i in range(1, 7)], "exploration": 0.4, "k": 4}
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 4


def test_custom_profile_validation(client):
    r = client.post("/api/analyze", json={"ratings": [{"movieId": 1, "rating": 9}]})
    assert r.status_code == 422


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Bias-Aware Recommender" in r.text
