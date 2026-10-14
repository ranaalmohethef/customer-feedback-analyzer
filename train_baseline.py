from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "classification_train.csv")
validation = pd.read_csv(DATA / "classification_validation.csv")

labels = ["bug_report", "improvement_request", "rating", "others"]

if set(train["label"]) != set(labels):
    raise ValueError("Training data does not contain the expected four classes")

assert set(train["text_key"]).isdisjoint(validation["text_key"])

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=40000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    ),
])

print("Training baseline...")
model.fit(train["review"], train["label"])

predictions = model.predict(validation["review"])

metrics = {
    "evaluation_split": "validation",
    "training_rows": len(train),
    "validation_rows": len(validation),
    "accuracy": accuracy_score(validation["label"], predictions),
    "macro_f1": f1_score(
        validation["label"],
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    ),
}

report = classification_report(
    validation["label"],
    predictions,
    labels=labels,
    output_dict=True,
    zero_division=0,
)

with open(REPORTS / "baseline_metrics.json", "w", encoding="utf-8") as file:
    json.dump(
        {"summary": metrics, "per_class": report},
        file,
        ensure_ascii=False,
        indent=2,
    )

matrix = confusion_matrix(
    validation["label"],
    predictions,
    labels=labels,
)

pd.DataFrame(
    matrix,
    index=labels,
    columns=labels,
).to_csv(
    REPORTS / "baseline_confusion_matrix.csv",
    encoding="utf-8-sig",
)

results = validation[
    ["review", "appName", "platform", "label"]
].copy()
results["prediction"] = predictions
results["correct"] = results["label"].eq(results["prediction"])

results.to_csv(
    REPORTS / "baseline_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

errors = results.loc[~results["correct"]].copy()
error_sample = errors.sample(n=min(20, len(errors)), random_state=42)
error_sample["error_category"] = ""
error_sample["notes"] = ""

error_sample.to_csv(
    REPORTS / "errors_to_review.csv",
    index=False,
    encoding="utf-8-sig",
)

joblib.dump(model, MODELS / "baseline_classifier.joblib")

print("\nValidation results:")
print(json.dumps(metrics, indent=2))
print("\nPer-class results:")
print(classification_report(
    validation["label"],
    predictions,
    labels=labels,
    zero_division=0,
))
print(f"\nMisclassified validation reviews: {len(errors)}")
print("DONE: model and reports saved")