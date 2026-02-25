from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import time

__all__ = [
    "get_models",
    "build_combinations",
    "run_experiments",
    "ExperimentResult",
]


@dataclass
class ExperimentResult:
    combination: str
    model_name: str
    f1_score: float
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

                score = f1_score(y_test, preds, average="weighted")
                acc = accuracy_score(y_test, preds)
                elapsed = time.time() - start_time

                if score > best_score:
                    best_score = score
                    best_info = {
                        "comb_name": comb_name,
                        "model_name": name,
                        "pipeline": pipeline,
                        "y_test": y_test,
                        "preds": preds,
                    }

                results.append(
                    ExperimentResult(
                        combination=comb_name,
                        model_name=name,
                        f1_score=score,
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
                        accuracy=float("nan"),
                        elapsed_sec=0.0,
                    )
                )

    return best_info, best_score, results

