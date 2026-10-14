from pathlib import Path
import json
import time

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

LABELS = [
    "bug_report",
    "improvement_request",
    "rating",
    "others",
]

# Model choice was made on validation BEFORE reading test results.
selection = {
    "selected_model": "hybrid",
    "selection_split": "validation",
    "selection_metric": "macro_f1",
    "validation_macro_f1": 0.6719191276206409,
    "embedding_weight": 0.5,
    "C": 10.0,
}

baseline = joblib.load(MODELS / "baseline_classifier.joblib")
embedding_classifier = joblib.load(
    MODELS / "embedding_classifier.joblib"
)
hybrid = joblib.load(MODELS / "hybrid_classifier.joblib")

if (
    hybrid["embedding_weight"] != selection["embedding_weight"]
    or hybrid["classifier"].C != selection["C"]
):
    raise ValueError("Saved hybrid model differs from selected settings.")

with open(
    REPORTS / "final_model_selection.json", "w", encoding="utf-8"
) as file:
    json.dump(selection, file, indent=2)

train = pd.read_csv(DATA / "classification_train.csv")
validation = pd.read_csv(DATA / "classification_validation.csv")
test = pd.read_csv(DATA / "classification_test.csv")

for name, df in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    required = ["review", "label", "text_key"]
    if df.empty or df[required].isna().any().any():
        raise ValueError(f"Missing or empty data in {name}.")
    if not set(df["label"]).issubset(LABELS):
        raise ValueError(f"Unexpected labels in {name}.")

test_keys = set(test["text_key"])
if test_keys & (
    set(train["text_key"]) | set(validation["text_key"])
):
    raise ValueError("Test text overlaps with train or validation!")

print(f"Test rows: {len(test)}")
print("Loading saved encoder...")

encoder = SentenceTransformer(
    str(MODELS / "multilingual_encoder"),
    device="cpu",
)

# Warm up the encoder; this text is not used for training.
encoder.encode(["اختبار"], normalize_embeddings=True)

texts = test["review"].astype(str).tolist()

start = time.perf_counter()
baseline_predictions = baseline.predict(texts)
baseline_seconds = time.perf_counter() - start

start = time.perf_counter()
vectors = encoder.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
encoding_seconds = time.perf_counter() - start

start = time.perf_counter()
embedding_predictions = embedding_classifier.predict(vectors)
embedding_seconds = (
    encoding_seconds + time.perf_counter() - start
)

start = time.perf_counter()
word_vectors = hybrid["vectorizer"].transform(texts)
features = hstack(
    [
        word_vectors,
        csr_matrix(vectors * hybrid["embedding_weight"]),
    ],
    format="csr",
)
hybrid_predictions = hybrid["classifier"].predict(features)
hybrid_seconds = (
    encoding_seconds + time.perf_counter() - start
)

results = [
    ("baseline", baseline_predictions, baseline_seconds),
    ("embeddings", embedding_predictions, embedding_seconds),
    ("hybrid", hybrid_predictions, hybrid_seconds),
]

comparison = []
slices = []

for name, predictions, seconds in results:
    comparison.append({
        "model": name,
        "split": "test",
        "rows": len(test),
        "accuracy": accuracy_score(test["label"], predictions),
        "macro_f1": f1_score(
            test["label"],
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "batch_seconds": seconds,
        "average_ms_per_review": seconds * 1000 / len(test),
    })

    report = classification_report(
        test["label"],
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    with open(
        REPORTS / f"{name}_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    matrix = confusion_matrix(
        test["label"], predictions, labels=LABELS
    )
    pd.DataFrame(
        matrix,
        index=pd.Index(LABELS, name="actual"),
        columns=pd.Index(LABELS, name="predicted"),
    ).to_csv(
        REPORTS / f"{name}_test_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    output = test.copy()
    output["prediction"] = predictions
    output["correct"] = output["label"] == output["prediction"]
    output.to_csv(
        REPORTS / f"{name}_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Platform slices provide another view of model performance.
    if "platform" in output.columns:
        for platform, group in output.groupby(
            "platform", dropna=False
        ):
            slices.append({
                "model": name,
                "platform": str(platform),
                "rows": len(group),
                "accuracy": accuracy_score(
                    group["label"], group["prediction"]
                ),
                "macro_f1_fixed_four_labels": f1_score(
                    group["label"],
                    group["prediction"],
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "reference_classes_present": group["label"].nunique(),
            })

    if name == "hybrid":
        errors_path = REPORTS / "hybrid_test_errors_to_review.csv"

        # Preserve annotations if this script is run again.
        if not errors_path.exists():
            errors = output.loc[~output["correct"]].copy()
            errors = errors.sample(
                n=min(20, len(errors)),
                random_state=42,
            )
            errors["error_category"] = ""
            errors["notes"] = ""
            errors.to_csv(
                errors_path,
                index=False,
                encoding="utf-8-sig",
            )

comparison = pd.DataFrame(comparison)
comparison.to_csv(
    REPORTS / "test_model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

if slices:
    pd.DataFrame(slices).to_csv(
        REPORTS / "test_platform_slices.csv",
        index=False,
        encoding="utf-8-sig",
    )

print("\nFINAL TEST RESULTS")
print(
    comparison[
        ["model", "rows", "accuracy", "macro_f1"]
    ].to_string(index=False)
)
print("\nSelected model: hybrid (chosen using validation).")
print(
    "Timing excludes model loading; averaged batch times "
    "are not single-request UI latency."
)
print("DONE: Test reports saved.")