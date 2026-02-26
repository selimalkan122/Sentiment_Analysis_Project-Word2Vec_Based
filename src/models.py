from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import time

__all__ = [
    "get_models",
    "build_combinations",
    "run_experiments",
    "tune_best_model",
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


"""
First, we will evaluate the base models,
and then tune the hyperparameters of the best-performing model
"""

def get_models():
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


def tune_best_model(
    best_info: Dict[str, Any],
    X_train_w2v: np.ndarray,
    X_train_hybrid: np.ndarray,
    y_train,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
):
    """
    Hyperparameter search for the best base model discovered by run_experiments.

    This function:
      - Detects which feature set was used (Word2Vec vs Hybrid)
      - Detects whether SMOTE was used (from combination name)
      - Builds an appropriate ImbPipeline with that model
      - Runs RandomizedSearchCV with cross‑validation

    Supported models: XGBoost, Random Forest, Logistic Regression, SGD (Faster SVM)

    Returns:
      randomized_search (fitted RandomizedSearchCV) or None if model not supported.
    """
    comb_name = best_info.get("comb_name", "")
    model_name = best_info.get("model_name", "")

    # Decide which feature matrix to use
    use_hybrid = "Hybrid" in comb_name or "VADER" in comb_name
    use_smote = "SMOTE" in comb_name

    X_train = X_train_hybrid if use_hybrid else X_train_w2v

    # Select base model and its hyperparameter search space
    if model_name == "XGBoost":
        base_model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
        )
        param_distributions = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7, 9],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
        }
    elif model_name == "Random Forest":
        base_model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        param_distributions = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
    elif model_name == "Logistic Regression":
        base_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
        param_distributions = {
            "model__C": np.logspace(-2, 2, 10),
            "model__penalty": ["l2"],
        }
    elif model_name == "SGD (Faster SVM)":
        base_model = SGDClassifier(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
        )
        param_distributions = {
            "model__alpha": np.logspace(-5, -1, 10),
            "model__loss": ["hinge", "log_loss"],
        }
    else:
        # Unsupported model type for tuning
        return None

    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", base_model))

    pipeline = ImbPipeline(steps)

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

    search.fit(X_train, y_train)
    return search


