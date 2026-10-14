"""Choose "not sure" thresholds on VALIDATION data only.

Rule (fixed before looking at results): the app abstains on the ~10% of
validation reviews where the model is least confident. The threshold is the
10th percentile of the top predicted probability on validation.
Test data is not used.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ABSTAIN_SHARE = 0.10

encoder = SentenceTransformer(str(MODELS / "multilingual_encoder"), device="cpu")


def encode(texts):
    return encoder.encode(texts, batch_size=16, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=True)


def summarize(name, y_true, proba, classes):
    top = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    tau = float(np.quantile(top, ABSTAIN_SHARE))
    keep = top >= tau
    correct = pred == np.asarray(y_true)
    stats = {
        "threshold": round(tau, 4),
        "validation_rows": int(len(top)),
        "abstained_share": round(float(1 - keep.mean()), 4),
        "accuracy_all": round(float(correct.mean()), 4),
        "accuracy_when_answering": round(float(correct[keep].mean()), 4),
        "accuracy_on_abstained": round(float(correct[~keep].mean()), 4),
    }
    print(name, stats)
    return stats


# Arabic topic (hybrid: TF-IDF + embeddings).
topic = pd.read_csv(DATA / "classification_validation.csv")
hybrid = joblib.load(MODELS / "hybrid_classifier.joblib")
texts = topic["review"].astype(str).tolist()
x = hstack([hybrid["vectorizer"].transform(texts),
            csr_matrix(encode(texts) * hybrid["embedding_weight"])], format="csr")
topic_stats = summarize("topic", topic["label"], hybrid["classifier"].predict_proba(x),
                        hybrid["classifier"].classes_)

# Arabic sentiment (v2 if present, otherwise v1).
senti = pd.read_csv(DATA / "sentiment_validation.csv")
texts = senti["review"].astype(str).tolist()
v2_path = MODELS / "sentiment_ar_v2.joblib"
bundle = joblib.load(v2_path) if v2_path.exists() else None
if bundle and bundle["kind"] != "baseline":
    vectors = encode(texts) if bundle["embedding_weight"] else np.zeros((len(texts), 384))
    x = hstack([bundle["word"].transform(texts), bundle["char"].transform(texts) * 0.5,
                csr_matrix(vectors * bundle["embedding_weight"])], format="csr")
    clf = bundle["classifier"]
    proba, classes = clf.predict_proba(x), clf.classes_
else:
    model = bundle["pipeline"] if bundle else joblib.load(MODELS / "sentiment_classifier.joblib")
    proba, classes = model.predict_proba(texts), model.classes_
senti_stats = summarize("sentiment", senti["sentiment"], proba, classes)

out = {"rule": f"abstain on the least confident {ABSTAIN_SHARE:.0%} of validation reviews",
       "selection_split": "validation", "topic": topic_stats, "sentiment_ar": senti_stats,
       "note": "English sentiment has no threshold; it stays experimental."}
(REPORTS / "abstention_thresholds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("DONE: reports/abstention_thresholds.json")
