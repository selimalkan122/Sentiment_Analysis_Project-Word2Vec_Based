from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

from .pipeline import run_full_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "YelpData.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _select_best_pipeline(results: Dict[str, Any]):
    """
    Decide which trained pipeline to persist:
    - if tuned_search is available, use tuned_search.best_estimator_
    - otherwise fall back to best_info['pipeline']
    """
    best_info = results["best_info"]
    tuned_search = results.get("tuned_search")

    if tuned_search is not None:
        best_pipeline = tuned_search.best_estimator_
        source = "tuned_search"
    else:
        best_pipeline = best_info["pipeline"]
        source = "base_experiment"

    comb_name = best_info.get("comb_name", "")
    model_name = best_info.get("model_name", "")

    # classes_ genellikle numpy tipleri (int32, int64) olabilir; JSON için int'e çeviriyoruz.
    raw_classes = list(getattr(best_pipeline, "classes_", []))
    classes = [int(c) for c in raw_classes]

    metadata = {
        "comb_name": comb_name,
        "model_name": model_name,
        "source": source,
        "use_hybrid": bool("Hybrid" in comb_name or "VADER" in comb_name),
        "use_smote": bool("SMOTE" in comb_name),
        "classes": classes,
    }

    return best_pipeline, metadata


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_full_experiment(str(DATA_PATH))

    w2v_model = results["w2v_model"]
    tfidf = results["tfidf"]

    best_pipeline, metadata = _select_best_pipeline(results)

    # Save artifacts
    w2v_path = ARTIFACTS_DIR / "w2v.model"
    tfidf_path = ARTIFACTS_DIR / "tfidf.pkl"
    pipeline_path = ARTIFACTS_DIR / "best_pipeline.pkl"
    meta_path = ARTIFACTS_DIR / "metadata.json"

    w2v_model.save(str(w2v_path))

    with tfidf_path.open("wb") as f:
        pickle.dump(tfidf, f)

    with pipeline_path.open("wb") as f:
        pickle.dump(best_pipeline, f)

    # Write metadata atomically to avoid partial/corrupt JSON on interruption
    meta_tmp_path = ARTIFACTS_DIR / "metadata.json.tmp"
    with meta_tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    meta_tmp_path.replace(meta_path)

    print(f"Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"- Word2Vec model: {w2v_path}")
    print(f"- TF-IDF vectorizer: {tfidf_path}")
    print(f"- Best pipeline: {pipeline_path}")
    print(f"- Metadata: {meta_path}")


if __name__ == "__main__":
    main()

