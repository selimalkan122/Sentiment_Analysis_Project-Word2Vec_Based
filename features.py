import multiprocessing
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from gensim.models import Word2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = [
    "train_word2vec",
    "fit_tfidf",
    "get_weighted_sentence_vector",
    "transform_tokens_to_w2v",
    "get_vader_scores",
    "build_vader_features",
    "build_hybrid_features",
]


def train_word2vec(
    sentences: Sequence[Sequence[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    sg: int = 1,
    epochs: int = 10,
    workers: int | None = None,
) -> Word2Vec:
    """
    Train a Word2Vec model on tokenized sentences.
    Default settings mirror the notebook configuration.
    """
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs,
    )
    return model


def fit_tfidf(tokenized_texts: Iterable[Sequence[str]]) -> Tuple[TfidfVectorizer, dict]:
    """
    Fit a TF‑IDF vectorizer on tokenized texts and return both
    the vectorizer and a {token: idf} dictionary.
    """
    tfidf = TfidfVectorizer()
    tfidf.fit(" ".join(tokens) for tokens in tokenized_texts)

    tfidf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    return tfidf, tfidf_dict


def get_weighted_sentence_vector(
    token_list: Sequence[str],
    model: Word2Vec,
    tfidf_dict: dict,
) -> np.ndarray:
    """
    Compute a TF‑IDF weighted average Word2Vec vector for a token list.
    """
    vectors: List[np.ndarray] = []
    weights: List[float] = []

    for word in token_list:
        if word in model.wv and word in tfidf_dict:
            word_vec = model.wv[word]
            weight = tfidf_dict[word]
            vectors.append(word_vec * weight)
            weights.append(weight)

    if not vectors:
        return np.zeros(model.vector_size)

    return np.sum(vectors, axis=0) / np.sum(weights)


def transform_tokens_to_w2v(
    tokenized_texts: Iterable[Sequence[str]],
    model: Word2Vec,
    tfidf_dict: dict,
) -> np.ndarray:
    """
    Convenience helper: convert an iterable of token lists into
    a 2D numpy array of weighted Word2Vec sentence vectors.
    """
    return np.array(
        [
            get_weighted_sentence_vector(tokens, model, tfidf_dict)
            for tokens in tokenized_texts
        ]
    )


_sid = SentimentIntensityAnalyzer()


def get_vader_scores(text: str) -> List[float]:
    """
    Compute VADER sentiment scores for a raw text string.
    Returns [neg, neu, pos, compound].
    """
    scores = _sid.polarity_scores(str(text))
    return [scores["neg"], scores["neu"], scores["pos"], scores["compound"]]


def build_vader_features(
    raw_texts: Iterable[str],
) -> np.ndarray:
    """
    Convert an iterable of raw texts into a VADER feature matrix.
    """
    return np.array([get_vader_scores(t) for t in raw_texts])


def build_hybrid_features(
    X_w2v: np.ndarray,
    X_vader: np.ndarray,
) -> np.ndarray:
    """
    Horizontally stack Word2Vec and VADER features to form a hybrid matrix.
    """
    return np.hstack((X_w2v, X_vader))

