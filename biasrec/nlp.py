"""NLP feature pipeline: sentiment, emotions, themes and movie embeddings.

Every movie gets
  * ``valence``   in [-1, 1]  (VADER on plot/tags blended with a genre prior)
  * ``emotions``  distribution over joy/fear/sadness/anger/surprise/trust
  * ``hero``      in [0, 1]   how strongly the story follows a hero archetype
  * ``themes``    NMF topic distribution (topic modelling)
  * ``embedding`` dense L2-normalised vector (TF-IDF+SVD, or Sentence-BERT)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

EMOTIONS = ["joy", "fear", "sadness", "anger", "surprise", "trust"]

# How each genre tends to feel. Used as a prior and for movies with no text.
GENRE_VALENCE = {
    "Comedy": 0.55, "Animation": 0.5, "Children": 0.6, "Musical": 0.55,
    "Romance": 0.35, "Fantasy": 0.2, "Adventure": 0.2, "Documentary": 0.0,
    "Sci-Fi": -0.05, "Action": -0.1, "Western": -0.15, "IMAX": 0.1,
    "Mystery": -0.3, "Drama": -0.2, "Crime": -0.5, "Thriller": -0.5,
    "War": -0.65, "Horror": -0.75, "Film-Noir": -0.55,
}

GENRE_EMOTION = {
    "Comedy": {"joy": 1.0, "surprise": 0.3}, "Animation": {"joy": 0.8, "trust": 0.4},
    "Children": {"joy": 0.9, "trust": 0.6}, "Musical": {"joy": 1.0},
    "Romance": {"joy": 0.6, "trust": 0.8}, "Fantasy": {"surprise": 0.6, "joy": 0.4},
    "Adventure": {"surprise": 0.6, "joy": 0.4}, "Sci-Fi": {"surprise": 0.8, "fear": 0.3},
    "Action": {"anger": 0.6, "surprise": 0.4}, "Western": {"anger": 0.5, "trust": 0.3},
    "Mystery": {"surprise": 0.7, "fear": 0.4}, "Drama": {"sadness": 0.7, "trust": 0.4},
    "Crime": {"anger": 0.8, "fear": 0.4}, "Thriller": {"fear": 0.9, "surprise": 0.5},
    "War": {"sadness": 0.8, "anger": 0.7, "fear": 0.5}, "Horror": {"fear": 1.0},
    "Film-Noir": {"fear": 0.5, "sadness": 0.5}, "Documentary": {"trust": 0.6},
}

# Compact emotion lexicon (stems) applied to overviews, keywords and tags.
EMOTION_LEXICON = {
    "joy": "love laugh fun happ joy friend celebrat wedding dream hope comedy funny "
           "heartwarm delight magic family feel-good triumph win party cute charming",
    "fear": "fear terror horror haunt ghost monster kill murder serial dark creepy "
            "nightmare demon possess threat danger stalk scary trap zombie paranoi "
            "psycho killer slasher blood gore disturb sinister",
    "sadness": "death die dying grief loss lonely tragic tragedy cancer mourn depress "
               "sad suicide war widow orphan funeral broken lost tearjerker melancholy",
    "anger": "revenge war fight battle violen brutal gang rage betray crime assault "
             "kill vengeance prison corrupt bloody brutality torture violence drug "
            "mafia mob gangster massacre",
    "surprise": "twist mystery secret discover strange unexpected mind-bending surreal "
                "puzzle reveal alien time travel heist conspiracy",
    "trust": "friendship loyal together team family faith redemption mentor honest "
             "true story inspiring courage hope community bond",
}

HERO_TERMS = [
    "superhero", "super hero", "hero", "based on comic", "marvel", "dc comics",
    "vigilante", "saving the world", "save the world", "chosen one", "one man army",
    "secret agent", "super power", "superpower", "mutant", "special forces",
    "masked", "crime fighter", "warrior", "knight", "rescue",
]


def _text_fields(movies: pd.DataFrame) -> pd.Series:
    return (
        movies["overview"].fillna("") + " " +
        movies["keywords"].fillna("").str.replace("|", " ", regex=False) + " " +
        movies["tags"].fillna("")
    ).str.strip()


def compute_valence(movies: pd.DataFrame, emotions: np.ndarray | None = None) -> np.ndarray:
    """Emotional tone of a film in [-1, 1].

    Plot synopses describe conflict, so raw VADER polarity alone is noisy.
    Three signals are blended: VADER on the text, the genre tone prior, and
    the balance of positive vs negative emotions from the emotion lexicon.
    """
    analyzer = SentimentIntensityAnalyzer()
    prior = movies["genre_list"].map(
        lambda gs: np.mean([GENRE_VALENCE.get(g, 0.0) for g in gs]) if gs else 0.0
    ).to_numpy()
    if emotions is None:
        emotions = compute_emotions(movies)
    pos = emotions[:, [EMOTIONS.index("joy"), EMOTIONS.index("trust")]].sum(axis=1)
    neg = emotions[:, [EMOTIONS.index(e) for e in ("fear", "sadness", "anger")]].sum(axis=1)
    balance = np.clip((pos - neg) * 1.5, -1, 1)
    text = _text_fields(movies)
    out = 0.6 * prior + 0.4 * balance
    for i, t in enumerate(text):
        if len(t) > 20:
            compound = analyzer.polarity_scores(t)["compound"]
            out[i] = 0.3 * compound + 0.4 * prior[i] + 0.3 * balance[i]
    return np.clip(out, -1, 1)


def compute_emotions(movies: pd.DataFrame) -> np.ndarray:
    stems = {e: words.split() for e, words in EMOTION_LEXICON.items()}
    text = _text_fields(movies).str.lower()
    mat = np.zeros((len(movies), len(EMOTIONS)))
    for i, (gs, t) in enumerate(zip(movies["genre_list"], text)):
        for g in gs:
            for e, w in GENRE_EMOTION.get(g, {}).items():
                mat[i, EMOTIONS.index(e)] += w
        if t:
            for j, e in enumerate(EMOTIONS):
                mat[i, j] += 0.5 * sum(t.count(s) for s in stems[e])
    mat += 1e-6
    return mat / mat.sum(axis=1, keepdims=True)


def compute_hero(movies: pd.DataFrame) -> np.ndarray:
    text = (movies["keywords"].fillna("") + " " + movies["tags"].fillna("") + " " +
            movies["overview"].fillna("")).str.lower()
    hits = text.map(lambda t: sum(1 for term in HERO_TERMS if term in t)).to_numpy(float)
    genre_boost = movies["genre_list"].map(
        lambda gs: 0.25 * len({"Action", "Adventure", "Fantasy", "Sci-Fi"} & set(gs))
    ).to_numpy(float)
    return np.clip(0.3 * hits + genre_boost * 0.6, 0, 1)


class FeatureBuilder:
    """Builds the semantic space shared by content model and bias engine."""

    def __init__(self, n_components: int = 64, n_themes: int = 14, seed: int = 42):
        self.n_components = n_components
        self.n_themes = n_themes
        self.seed = seed

    def fit(self, movies: pd.DataFrame) -> dict:
        text = _text_fields(movies)
        genre_text = movies["genre_list"].map(lambda gs: " ".join(gs))
        decade = (movies["year"].fillna(1990) // 10 * 10).astype(int)

        tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5,
            sublinear_tf=True, max_features=30000,
        )
        docs = (text + " " + genre_text).str.lower()
        X_text = tfidf.fit_transform(docs)

        mlb = MultiLabelBinarizer()
        X_genre = sparse.csr_matrix(mlb.fit_transform(movies["genre_list"]).astype(float))
        decades = sorted(decade.unique())
        X_dec = sparse.csr_matrix(
            (decade.to_numpy()[:, None] == np.array(decades)[None, :]).astype(float)
        )
        # genres carry most signal for movies without plots; weight blocks
        X = sparse.hstack([X_text, 1.6 * X_genre, 0.6 * X_dec]).tocsr()
        k = max(2, min(self.n_components, X.shape[1] - 1, X.shape[0] - 1))
        svd = TruncatedSVD(n_components=k, random_state=self.seed)
        emb = normalize(svd.fit_transform(X))

        if os.environ.get("BIASREC_EMBEDDINGS", "").lower() == "sbert":
            emb = self._sbert(movies, text, genre_text, fallback=emb)

        # Topic modelling: NMF over the text TF-IDF -> human readable themes
        n_themes = max(2, min(self.n_themes, X_text.shape[1] - 1, X_text.shape[0] - 1))
        nmf = NMF(n_components=n_themes, init="nndsvda", random_state=self.seed, max_iter=400)
        W = nmf.fit_transform(X_text)
        vocab = np.array(tfidf.get_feature_names_out())
        theme_names = []
        for comp in nmf.components_:
            words = [w for w in vocab[np.argsort(comp)[::-1]] if " " not in w][:3]
            theme_names.append(" / ".join(words))
        W = W + 1e-6
        themes = W / W.sum(axis=1, keepdims=True)
        emotions = compute_emotions(movies)

        return {
            "embedding": emb.astype(np.float32),
            "themes": themes.astype(np.float32),
            "theme_names": theme_names,
            "genres": list(mlb.classes_),
            "genre_matrix": X_genre.toarray().astype(np.float32),
            "valence": compute_valence(movies, emotions).astype(np.float32),
            "emotions": emotions.astype(np.float32),
            "hero": compute_hero(movies).astype(np.float32),
        }

    @staticmethod
    def _sbert(movies, text, genre_text, fallback):  # pragma: no cover - optional
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return fallback
        model = SentenceTransformer(os.environ.get("BIASREC_SBERT_MODEL", "all-MiniLM-L6-v2"))
        docs = (movies["clean_title"] + ". " + genre_text + ". " + text).tolist()
        vecs = model.encode(docs, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(vecs)

