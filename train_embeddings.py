from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
CACHE = ROOT / "Data" / "embedding_cache"

for folder in [REPORTS, MODELS, CACHE]:
    folder.mkdir(parents=True, exist_ok=True)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
ENCODER_PATH = MODELS / "multilingual_encoder"

LABELS = [
    "bug_report",
    "improvement_request",
    "rating",
    "others",
]


def read_split(split):
    path = DATA / f"classification_{split}.csv"
    df = pd.read_csv(path)

    required = {"review", "label", "text_key"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path.name}")

    if df.empty or df[list(required)].isna().any().any():
        raise ValueError(f"Empty or missing data in {path.name}")

    if not set(df["label"]).issubset(LABELS):
        raise ValueError(f"Unexpected labels in {path.name}")

    return df


train = read_split("train")
validation = read_split("validation")

if set(train["label"]) != set(LABELS):
    raise ValueError("Training data must contain all four classes.")

if set(train["text_key"]) & set(validation["text_key"]):
    raise ValueError("Text overlap between train and validation!")

print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print("Loading embedding model...")

# First run downloads the encoder; later runs load the local copy.
encoder = SentenceTransformer(
    str(ENCODER_PATH) if ENCODER_PATH.exists() else MODEL_NAME,
    device="cpu",
)

if not ENCODER_PATH.exists():
    encoder.save(str(ENCODER_PATH))


def encode_split(df, split):
    texts = df["review"].astype(str).tolist()

    # Cache depends on text content, order, and encoder settings.
    signature = json.dumps(
        {
            "model": MODEL_NAME,
            "max_seq_length": encoder.max_seq_length,
            "normalize_embeddings": True,
            "texts": texts,
        },
        ensure_ascii=False,
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"classification_{split}_{digest}.npy"

    if path.exists():
        print(f"Loading cached embeddings: {split}")
        return np.load(path, allow_pickle=False), None

    print(f"Encoding {split}...")
    start = time.perf_counter()
    vectors = encoder.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    seconds = time.perf_counter() - start
    np.save(path, vectors)

    return vectors, seconds


x_train, train_encoding_seconds = encode_split(train, "train")
x_validation, val_encoding_seconds = encode_split(
    validation, "validation"
)

# Select settings using validation only.
# The test set remains untouched.
experiments = []
best_score = -1.0
best_model = None
best_settings = None

for c in [0.1, 1.0, 10.0]:
    for weight in [None, "balanced"]:
        classifier = LogisticRegression(
            C=c,
            class_weight=weight,
            max_iter=2000,
            random_state=42,
        )

        start = time.perf_counter()
        classifier.fit(x_train, train["label"])
        fit_seconds = time.perf_counter() - start

        predictions = classifier.predict(x_validation)
        score = f1_score(
            validation["label"],
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        settings = {
            "C": c,
            "class_weight": weight,
        }
        experiments.append(
            {
                **settings,
                "validation_macro_f1": float(score),
                "validation_accuracy": float(
                    accuracy_score(validation["label"], predictions)
                ),
                "classifier_fit_seconds": fit_seconds,
            }
        )

        print(
            f"C={c}, class_weight={weight}: "
            f"validation Macro-F1={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_model = classifier
            best_settings = settings

pd.DataFrame(experiments).to_csv(
    REPORTS / "embedding_experiments.csv",
    index=False,
    encoding="utf-8-sig",
)

predictions = best_model.predict(x_validation)

summary = {
    "evaluation_split": "validation",
    "encoder": MODEL_NAME,
    "encoder_fine_tuned": False,
    "max_seq_length": encoder.max_seq_length,
    "training_rows": len(train),
    "validation_rows": len(validation),
    "accuracy": float(
        accuracy_score(validation["label"], predictions)
    ),
    "macro_f1": float(best_score),
    "selected_settings": best_settings,
    "train_encoding_seconds": train_encoding_seconds,
    "validation_encoding_seconds": val_encoding_seconds,
    "timing_note": (
        "Encoding times exclude model loading. "
        "Null means cached embeddings were used."
    ),
}

report = classification_report(
    validation["label"],
    predictions,
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)

with open(
    REPORTS / "embedding_validation_metrics.json",
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
    validation["label"],
    predictions,
    labels=LABELS,
)

pd.DataFrame(
    matrix,
    index=pd.Index(LABELS, name="actual"),
    columns=pd.Index(LABELS, name="predicted"),
).to_csv(
    REPORTS / "embedding_validation_confusion_matrix.csv",
    encoding="utf-8-sig",
)

output = validation.copy()
output["prediction"] = predictions
output["correct"] = output["label"] == output["prediction"]
output.to_csv(
    REPORTS / "embedding_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

joblib.dump(
    best_model,
    MODELS / "embedding_classifier.joblib",
)

print("\nSelected settings:", best_settings)
print(f"Validation Accuracy: {summary['accuracy']:.4f}")
print(f"Validation Macro-F1: {summary['macro_f1']:.4f}")

print(
    classification_report(
        validation["label"],
        predictions,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )
)

print("DONE: Model and validation reports saved.")
print("Test data and previous error annotations were not changed.")


# Hybrid experiment: TF-IDF + sentence embeddings
from scipy.sparse import csr_matrix, hstack

print("\nTraining hybrid models: TF-IDF + embeddings...")

# Reuse the TF-IDF vocabulary learned from training data only.
baseline = joblib.load(MODELS / "baseline_classifier.joblib")
vectorizer = baseline.steps[0][1]

word_train = vectorizer.transform(train["review"])
word_validation = vectorizer.transform(validation["review"])

hybrid_experiments = []
best_hybrid_score = -1.0
best_hybrid_model = None
best_hybrid_settings = None

# Both feature groups are normalized; alpha controls their relative weight.
for alpha in [0.5, 1.0, 2.0]:
    hybrid_train = hstack(
        [word_train, csr_matrix(x_train * alpha)],
        format="csr",
    )
    hybrid_validation = hstack(
        [word_validation, csr_matrix(x_validation * alpha)],
        format="csr",
    )

    for c in [1.0, 10.0]:
        classifier = LogisticRegression(
            C=c,
            max_iter=2000,
            random_state=42,
        )
        classifier.fit(hybrid_train, train["label"])
        pred = classifier.predict(hybrid_validation)

        score = f1_score(
            validation["label"],
            pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        accuracy = accuracy_score(validation["label"], pred)

        hybrid_experiments.append({
            "embedding_weight": alpha,
            "C": c,
            "validation_macro_f1": float(score),
            "validation_accuracy": float(accuracy),
        })

        print(
            f"embedding_weight={alpha}, C={c}: "
            f"Macro-F1={score:.4f}, Accuracy={accuracy:.4f}"
        )

        if score > best_hybrid_score:
            best_hybrid_score = score
            best_hybrid_model = classifier
            best_hybrid_settings = {
                "embedding_weight": alpha,
                "C": c,
            }

pd.DataFrame(hybrid_experiments).to_csv(
    REPORTS / "hybrid_experiments.csv",
    index=False,
    encoding="utf-8-sig",
)

alpha = best_hybrid_settings["embedding_weight"]
hybrid_validation = hstack(
    [word_validation, csr_matrix(x_validation * alpha)],
    format="csr",
)
hybrid_predictions = best_hybrid_model.predict(hybrid_validation)

hybrid_summary = {
    "evaluation_split": "validation",
    "selected_settings": best_hybrid_settings,
    "accuracy": float(
        accuracy_score(validation["label"], hybrid_predictions)
    ),
    "macro_f1": float(best_hybrid_score),
}

hybrid_report = classification_report(
    validation["label"],
    hybrid_predictions,
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)

with open(
    REPORTS / "hybrid_validation_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {"summary": hybrid_summary, "per_class": hybrid_report},
        file,
        ensure_ascii=False,
        indent=2,
    )

hybrid_output = validation.copy()
hybrid_output["prediction"] = hybrid_predictions
hybrid_output["correct"] = (
    hybrid_output["label"] == hybrid_output["prediction"]
)
hybrid_output.to_csv(
    REPORTS / "hybrid_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "classifier": best_hybrid_model,
        "embedding_weight": alpha,
        "encoder": MODEL_NAME,
        "normalize_embeddings": True,
    },
    MODELS / "hybrid_classifier.joblib",
)

print("\nHYBRID RESULTS")
print(json.dumps(hybrid_summary, indent=2))
print(
    classification_report(
        validation["label"],
        hybrid_predictions,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )
)
print("DONE: Hybrid model saved. Test set remains untouched.")