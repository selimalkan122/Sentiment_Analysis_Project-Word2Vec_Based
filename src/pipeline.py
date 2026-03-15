"""
High-level experiment pipeline for the Yelp sentiment analysis project.

You can run the full workflow from a notebook or a script by importing
and calling `run_full_experiment()`.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .data_utils import load_dataset, train_test_split_sentiment
from .preprocessing import cleaning_text
from .features import (
    train_word2vec,
    fit_tfidf,
    transform_tokens_to_w2v,
    build_vader_features,
    build_hybrid_features,
)
from .models import (
    get_models,
    build_combinations,
    run_experiments,
    tune_best_model,
    ExperimentResult,
)


def run_full_experiment(
    csv_path: str = "YelpData.csv",
) -> Dict[str, Any]:
    """
    Run the same end-to-end pipeline as in the notebook:
    - load data
    - map ratings to sentiments and split
    - clean text & tokenize
    - train Word2Vec + TF‑IDF
    - build Word2Vec, VADER and hybrid features
    - train & evaluate models on all combinations

    Returns a dictionary with:
      - df
      - X_train_raw, X_test_raw, y_train, y_test
      - X_train_w2v, X_test_w2v, X_train_vader, X_test_vader
      - X_train_hybrid, X_test_hybrid
      - best_info, best_score, results (list[ExperimentResult])
    """
    # Load and split data
    df = load_dataset(csv_path)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split_sentiment(df)

    # Text cleaning / tokenization
    X_train_tokens = X_train_raw.apply(cleaning_text)
    X_test_tokens = X_test_raw.apply(cleaning_text)

    # Train embeddings + TF‑IDF
    w2v_model = train_word2vec(X_train_tokens)
    tfidf, tfidf_dict = fit_tfidf(X_train_tokens)

    # Build Word2Vec features
    X_train_w2v = transform_tokens_to_w2v(X_train_tokens, w2v_model, tfidf_dict)
    X_test_w2v = transform_tokens_to_w2v(X_test_tokens, w2v_model, tfidf_dict)

    # Build VADER features
    X_train_vader = build_vader_features(X_train_raw)
    X_test_vader = build_vader_features(X_test_raw)

    # Hybrid
    X_train_hybrid = build_hybrid_features(X_train_w2v, X_train_vader)
    X_test_hybrid = build_hybrid_features(X_test_w2v, X_test_vader)

    # Models and combinations
    models = get_models()
    combinations = build_combinations(
        X_train_w2v, X_test_w2v, X_train_hybrid, X_test_hybrid
    )

    best_info, best_score, results = run_experiments(
        combinations, models, y_train, y_test
    )

    # Optional: hyperparameter tuning + CV for the best base model
    tuned_search = tune_best_model(
        best_info,
        X_train_w2v=X_train_w2v,
        X_train_hybrid=X_train_hybrid,
        y_train=y_train,
    )

    return {
        "df": df,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "w2v_model": w2v_model,
        "tfidf": tfidf,
        "tfidf_dict": tfidf_dict,
        "X_train_w2v": X_train_w2v,
        "X_test_w2v": X_test_w2v,
        "X_train_vader": X_train_vader,
        "X_test_vader": X_test_vader,
        "X_train_hybrid": X_train_hybrid,
        "X_test_hybrid": X_test_hybrid,
        "best_info": best_info,
        "best_score": best_score,
        "results": results,
        "tuned_search": tuned_search,
    }


__all__ = ["run_full_experiment"]

