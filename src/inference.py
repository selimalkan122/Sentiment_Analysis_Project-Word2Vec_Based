from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from json import JSONDecodeError

import numpy as np
from gensim.models import Word2Vec

from .preprocessing import cleaning_text
from .features import (
    transform_tokens_to_w2v,
    build_vader_features,
    build_hybrid_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class ArtifactLoadError(RuntimeError):
    pass


def _load_w2v_model() -> Word2Vec:
    path = ARTIFACTS_DIR / "w2v.model"
    if not path.exists():
        raise ArtifactLoadError(f"Word2Vec modeli bulunamadı: {path}")
    return Word2Vec.load(str(path))


def _load_tfidf():
    path = ARTIFACTS_DIR / "tfidf.pkl"
    if not path.exists():
        raise ArtifactLoadError(f"TF-IDF vektörleştirici bulunamadı: {path}")
    with path.open("rb") as f:
        tfidf = pickle.load(f)
    return tfidf


def _load_pipeline():
    path = ARTIFACTS_DIR / "best_pipeline.pkl"
    if not path.exists():
        raise ArtifactLoadError(f"Eğitilmiş model pipeline'ı bulunamadı: {path}")
    with path.open("rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def _load_metadata() -> Dict[str, Any]:
    path = ARTIFACTS_DIR / "metadata.json"
    if not path.exists():
        raise ArtifactLoadError(f"Metadata dosyası bulunamadı: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except JSONDecodeError as e:
        raise ArtifactLoadError(
            f"Metadata JSON bozuk/eksik görünüyor: {path} ({e})"
        ) from e
    return meta


@lru_cache(maxsize=1)
def get_artifacts():
    """
    Load all necessary artifacts once and cache them.
    """
    w2v_model = _load_w2v_model()
    tfidf = _load_tfidf()
    pipeline = _load_pipeline()
    metadata = _load_metadata()

    # Rebuild TF-IDF dictionary for weighted Word2Vec
    tfidf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    return {
        "w2v_model": w2v_model,
        "tfidf": tfidf,
        "tfidf_dict": tfidf_dict,
        "pipeline": pipeline,
        "metadata": metadata,
    }


def _build_features(text: str):
    """
    Given a raw text, build the feature vector that matches the
    configuration used during training (Word2Vec only vs Hybrid).
    """
    artifacts = get_artifacts()
    w2v_model = artifacts["w2v_model"]
    tfidf_dict = artifacts["tfidf_dict"]
    metadata = artifacts["metadata"]

    use_hybrid = bool(metadata.get("use_hybrid", False))

    # Cleaning + tokenization
    tokens = cleaning_text(text)

    # Word2Vec representation
    X_w2v = transform_tokens_to_w2v([tokens], w2v_model, tfidf_dict)

    if use_hybrid:
        # VADER features on raw text
        X_vader = build_vader_features([text])
        X = build_hybrid_features(X_w2v, X_vader)
    else:
        X = X_w2v

    return X


def predict_sentiment(text: str) -> Dict[str, Any]:
    """
    Run a single-text sentiment prediction.

    Returns a dictionary with:
      - label_id: 0/1/2
      - label: 'negative' / 'neutral' / 'positive'
      - confidence: max probability (0-1)
      - probabilities: dict per class (if available)
      - model_name, comb_name: for debugging
    """
    artifacts = get_artifacts()
    pipeline = artifacts["pipeline"]
    metadata = artifacts["metadata"]

    X = _build_features(text)

    pred = pipeline.predict(X)[0]

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    label_id = int(pred)
    label = label_map.get(label_id, str(label_id))

    confidence = None
    probabilities = None

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)[0]
            confidence = float(np.max(proba))
            probabilities = {
                label_map.get(int(i), str(i)): float(p)
                for i, p in enumerate(proba)
            }
        except Exception:
            confidence = None
            probabilities = None

    result: Dict[str, Any] = {
        "label_id": label_id,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "model_name": metadata.get("model_name"),
        "comb_name": metadata.get("comb_name"),
    }

    return result


__all__ = ["get_artifacts", "predict_sentiment"]

