from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import time

__all__ = [
    "get_models",
    "build_combinations",
    "run_experiments",
    "tune_best_xgboost_hybrid_smote",
    "ExperimentResult",
]


@dataclass
class ExperimentResult:
    combination: str
    model_name: str
    f1_score: float
    f1_macro: float
    accuracy: float
    elapsed_sec: float


def get_models():
    """
    Return the list of (name, sklearn/xgboost model) tuples
    used in the notebook.
    """
    models = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
        (
            "SGD (Faster SVM)",
            SGDClassifier(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            xgb.XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
            ),
        ),
    ]
    return models


def build_combinations(
    X_train_w2v: np.ndarray,
    X_test_w2v: np.ndarray,
    X_train_hybrid: np.ndarray,
    X_test_hybrid: np.ndarray,
) -> List[Tuple[str, np.ndarray, np.ndarray, bool]]:
    """
    Mirror the 'comb_list' structure from the notebook.
    """
    return [
        ("1. Only Word2Vec", X_train_w2v, X_test_w2v, False),
        ("2. Word2Vec + SMOTE ", X_train_w2v, X_test_w2v, True),
        ("3. Word2Vec + VADER", X_train_hybrid, X_test_hybrid, False),
        ("4. Hybrid + SMOTE", X_train_hybrid, X_test_hybrid, True),
    ]


def run_experiments(
    combinations: Sequence[Tuple[str, np.ndarray, np.ndarray, bool]],
    models: Sequence[Tuple[str, Any]],
    y_train,
    y_test,
):
    """
    Run all model/feature-set combinations and return:
    - best_info: dict with best configuration and predictions
    - best_score: highest weighted F1 score
    - results: list[ExperimentResult]
      (includes both weighted and macro F1)
    """
    best_score = -1.0
    best_info: Dict[str, Any] = {}
    results: List[ExperimentResult] = []

    for comb_name, X_tra, X_tes, use_smote in combinations:
        for name, model in models:
            start_time = time.time()

            steps = [("scaler", StandardScaler())]
            if use_smote:
                steps.append(("smote", SMOTE(random_state=42)))
            steps.append(("model", model))

            pipeline = ImbPipeline(steps)

            try:
                pipeline.fit(X_tra, y_train)

                preds = pipeline.predict(X_tes)

                f1_weighted = f1_score(y_test, preds, average="weighted")
                f1_macro = f1_score(y_test, preds, average="macro")
                acc = accuracy_score(y_test, preds)
                elapsed = time.time() - start_time

                if f1_weighted > best_score:
                    best_score = f1_weighted
                    best_info = {
                        "comb_name": comb_name,
                        "model_name": name,
                        "pipeline": pipeline,
                        "y_test": y_test,
                        "preds": preds,
                        "classification_report": classification_report(
                            y_test, preds, digits=4
                        ),
                        "confusion_matrix": confusion_matrix(y_test, preds),
                    }

                results.append(
                    ExperimentResult(
                        combination=comb_name,
                        model_name=name,
                        f1_score=f1_weighted,
                        f1_macro=f1_macro,
                        accuracy=acc,
                        elapsed_sec=elapsed,
                    )
                )
            except Exception as e:
                # You can inspect this in the notebook if needed
                results.append(
                    ExperimentResult(
                        combination=comb_name,
                        model_name=f"{name} (ERROR: {e})",
                        f1_score=float("nan"),
                        f1_macro=float("nan"),
                        accuracy=float("nan"),
                        elapsed_sec=0.0,
                    )
                )

    return best_info, best_score, results


def tune_best_xgboost_hybrid_smote(
    X_train_hybrid: np.ndarray,
    y_train,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
):
    """
    Hyperparameter search for the best XGBoost model on the
    Hybrid + SMOTE setup, using cross‑validation.

    Returns:
      randomized_search: fitted RandomizedSearchCV instance
    """
    base_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
    )

    pipeline = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("model", base_model),
        ]
    )

    param_distributions = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_weighted",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=random_state,
    )

    search.fit(X_train_hybrid, y_train)
    return search


