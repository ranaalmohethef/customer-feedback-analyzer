from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "positive"]


def read_split(split):
    df = pd.read_csv(DATA / f"sentiment_{split}.csv")
    required = ["review", "sentiment", "text_key"]

    if df.empty or df[required].isna().any().any():
        raise ValueError(f"Empty or missing data in {split}.")

    if not set(df["sentiment"]).issubset(LABELS):
        raise ValueError(
            f"Unexpected sentiment labels: "
            f"{df['sentiment'].unique().tolist()}"
        )

    return df


train = read_split("train")
validation = read_split("validation")

if set(train["sentiment"]) != set(LABELS):
    raise ValueError("Training data must contain both sentiment classes.")

if set(train["text_key"]) & set(validation["text_key"]):
    raise ValueError("Train/validation text overlap!")

print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")

# Fit vocabulary and IDF on training data only.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=60000,
    sublinear_tf=True,
)

x_train = vectorizer.fit_transform(train["review"])
x_validation = vectorizer.transform(validation["review"])

experiments = []
best_model = None
best_score = -1.0
best_c = None

for c in [1.0, 10.0]:
    classifier = LogisticRegression(
        C=c,
        max_iter=2000,
        random_state=42,
    )
    classifier.fit(x_train, train["sentiment"])
    predictions = classifier.predict(x_validation)

    score = f1_score(
        validation["sentiment"],
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(
        validation["sentiment"], predictions
    )

    experiments.append({
        "C": c,
        "validation_accuracy": float(accuracy),
        "validation_macro_f1": float(score),
    })

    print(
        f"C={c}: Validation Accuracy={accuracy:.4f}, "
        f"Macro-F1={score:.4f}"
    )

    if score > best_score:
        best_score = score
        best_model = classifier
        best_c = c

pd.DataFrame(experiments).to_csv(
    REPORTS / "sentiment_experiments.csv",
    index=False,
    encoding="utf-8-sig",
)

# Both steps are already fitted. Do not call fit again.
pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", best_model),
])

joblib.dump(
    pipeline,
    MODELS / "sentiment_classifier.joblib",
)

selection = {
    "selection_split": "validation",
    "selection_metric": "macro_f1",
    "selected_C": best_c,
    "validation_macro_f1": float(best_score),
    "labels": LABELS,
}

with open(
    REPORTS / "sentiment_selection.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(selection, file, indent=2)

# Read test only after selecting and saving the model.
test = read_split("test")

if set(test["text_key"]) & (
    set(train["text_key"]) | set(validation["text_key"])
):
    raise ValueError("Test text overlaps with train or validation!")

predictions = pipeline.predict(test["review"])

summary = {
    "evaluation_split": "test",
    "test_rows": len(test),
    "accuracy": float(
        accuracy_score(test["sentiment"], predictions)
    ),
    "macro_f1": float(
        f1_score(
            test["sentiment"],
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    ),
}

report = classification_report(
    test["sentiment"],
    predictions,
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)

with open(
    REPORTS / "sentiment_test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {"summary": summary, "per_class": report},
        file,
        ensure_ascii=False,
        indent=2,
    )

matrix = confusion_matrix(
    test["sentiment"], predictions, labels=LABELS
)
pd.DataFrame(
    matrix,
    index=pd.Index(LABELS, name="actual"),
    columns=pd.Index(LABELS, name="predicted"),
).to_csv(
    REPORTS / "sentiment_test_confusion_matrix.csv",
    encoding="utf-8-sig",
)

output = test.copy()
output["prediction"] = predictions
output["correct"] = output["sentiment"] == output["prediction"]
output.to_csv(
    REPORTS / "sentiment_test_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

errors_path = REPORTS / "sentiment_test_errors_to_review.csv"
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

print("\nSENTIMENT TEST RESULTS")
print(json.dumps(summary, indent=2))
print(
    classification_report(
        test["sentiment"],
        predictions,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )
)
print("DONE: Sentiment model and reports saved.")