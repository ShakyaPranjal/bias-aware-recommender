"""FastAPI web application: JSON API + the psychological dashboard UI."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .engine import RecommenderEngine, Weights

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = Path(os.environ.get("BIASREC_DATA_DIR", ROOT / "data" / "raw"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
state: dict = {}


def get_engine() -> RecommenderEngine:
    if "engine" not in state:
        if not (DATA_DIR / "ratings.csv").exists():
            raise HTTPException(503, f"Dataset not found in {DATA_DIR}. Run: python scripts/download_data.py")
        state["engine"] = RecommenderEngine.from_dir(DATA_DIR)
        state["featured"] = state["engine"].featured_users()
    return state["engine"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if (DATA_DIR / "ratings.csv").exists() and os.environ.get("BIASREC_LAZY") != "1":
        get_engine()
    yield


app = FastAPI(title="Bias-Aware Recommender", version=__version__, lifespan=lifespan,
              description="Psychological bias-aware movie recommendations with explanations.")
app.add_middleware(GZipMiddleware, minimum_size=1000)


class RatedMovie(BaseModel):
    movieId: int
    rating: float = Field(ge=0.5, le=5.0)


class WeightsIn(BaseModel):
    alpha: float = Field(ge=0, le=1)
    beta: float = Field(ge=0, le=1)
    gamma: float = Field(ge=0, le=1)
    delta: float = Field(ge=0, le=1)


class AnalyzeRequest(BaseModel):
    ratings: list[RatedMovie] = Field(min_length=3)
    exploration: float = Field(0.5, ge=0, le=1)
    k: int = Field(12, ge=3, le=30)
    weights: WeightsIn | None = None


def _weights(w: WeightsIn | None) -> Weights | None:
    return Weights(**w.model_dump()) if w else None


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "loaded": "engine" in state}


@app.get("/api/meta")
def meta():
    e = get_engine()
    return {"n_movies": int(len(e.movies)), "n_ratings": int(len(e.ratings)),
            "n_users": int(e.n_users), "genres": e.features["genres"],
            "themes": e.features["theme_names"], "featured_users": state["featured"],
            "with_overview": int((e.movies["overview"] != "").sum())}


@app.get("/api/map")
def catalog_map(n: int = Query(1500, ge=100, le=5000)):
    return {"points": get_engine().catalog_map(n)}


@app.get("/api/movies/search")
def search(q: str = Query(..., min_length=1, max_length=80), limit: int = Query(12, ge=1, le=40)):
    return {"results": get_engine().search(q, limit)}


@app.get("/api/movies/popular")
def popular(limit: int = Query(24, ge=1, le=100)):
    e = get_engine()
    top = e.movies.sort_values("n_ratings", ascending=False).head(limit)["idx"]
    return {"results": [e.movie_card(int(i)) for i in top]}


@app.get("/api/users/{user_id}/analysis")
def user_analysis(user_id: int, exploration: float = Query(0.5, ge=0, le=1),
                  k: int = Query(12, ge=3, le=30)):
    e = get_engine()
    try:
        profile = e.profile_for_user(user_id)
    except KeyError:
        raise HTTPException(404, f"user {user_id} not found") from None
    out = e.analyze(profile, k=k, exploration=exploration)
    out["userId"] = user_id
    return out


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    e = get_engine()
    try:
        profile = e.profile_from_ratings([r.model_dump() for r in req.ratings])
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    return e.analyze(profile, k=req.k, exploration=req.exploration, weights=_weights(req.weights))


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")
