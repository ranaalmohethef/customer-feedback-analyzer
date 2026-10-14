from pathlib import Path
import hashlib
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "Data" / "raw"
OUT = ROOT / "Data" / "processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def find_one(pattern):
    files = list(RAW.glob(pattern))
    if len(files) != 1:
        raise ValueError(
            f"Expected one file matching {pattern}, found: {files}"
        )
    return files[0]


def text_key(text):
    # Used for duplicate detection and splitting.
    # The original review text remains unchanged.
    return re.sub(r"\s+", " ", str(text)).strip()


def choose_split(key):
    # The same text always receives the same split across both AURA files.
    value = int(
        hashlib.sha256(("feedback-v1|" + key).encode("utf-8")).hexdigest(),
        16,
    ) % 10000

    if value < 7000:
        return "train"
    if value < 8500:
        return "validation"
    return "test"


audit = {}


def prepare_labeled(df, name, label_column):
    original_rows = len(df)

    empty = (
        df["review"].isna()
        | df["review"].fillna("").astype(str).str.strip().eq("")
        | df[label_column].isna()
    )

    df = df.loc[~empty].copy()
    df["text_key"] = df["review"].map(text_key)

    # Conflicting labels are saved for review, not silently corrected.
    label_counts = df.groupby("text_key")[label_column].nunique()
    conflict_keys = label_counts[label_counts > 1].index
    conflict_mask = df["text_key"].isin(conflict_keys)

    df.loc[conflict_mask].to_csv(
        REPORTS / f"{name}_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    conflict_rows = int(conflict_mask.sum())
    df = df.loc[~conflict_mask].copy()

    duplicates_removed = int(df.duplicated("text_key").sum())
    df = df.drop_duplicates("text_key").copy()
    df["split"] = df["text_key"].map(choose_split)

    for split in ["train", "validation", "test"]:
        part = df.loc[df["split"].eq(split)].copy()
        if part.empty:
            raise ValueError(f"{name}: empty {split} split")

        part.to_csv(
            OUT / f"{name}_{split}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    distribution = pd.crosstab(df[label_column], df["split"])
    distribution.to_csv(
        REPORTS / f"{name}_distribution.csv",
        encoding="utf-8-sig",
    )

    audit[name] = {
        "original_rows": original_rows,
        "empty_or_missing_label_removed": int(empty.sum()),
        "conflicting_text_groups": len(conflict_keys),
        "conflict_rows_excluded": conflict_rows,
        "duplicate_text_rows_removed": duplicates_removed,
        "remaining_rows": len(df),
        "split_counts": df["split"].value_counts().to_dict(),
    }

    print(f"\n{name}")
    print(distribution)
    return df


classification = pd.read_csv(find_one("AURA-Classification*.csv"))
sentiment = pd.read_parquet(find_one("*.parquet"))
english = pd.read_csv(RAW / "user_quality.csv")

classification = prepare_labeled(
    classification, "classification", "label"
)
sentiment = prepare_labeled(
    sentiment, "sentiment", "sentiment"
)

# Verify that shared AURA texts never cross split boundaries.
shared = classification[["text_key", "split"]].merge(
    sentiment[["text_key", "split"]],
    on="text_key",
    suffixes=("_classification", "_sentiment"),
)

assert (
    shared["split_classification"] == shared["split_sentiment"]
).all(), "Shared AURA texts crossed split boundaries"

audit["shared_aura_texts_after_cleaning"] = len(shared)

# English data: create an exploratory sample, not a labeled test set.
english_original_rows = len(english)
empty = (
    english["review"].isna()
    | english["review"].fillna("").astype(str).str.strip().eq("")
)
english = english.loc[~empty].copy()
english["text_key"] = english["review"].map(text_key)

english_duplicates = int(english.duplicated("text_key").sum())
english = english.drop_duplicates("text_key").copy()

english_sample = english.sample(
    n=min(3000, len(english)),
    random_state=42,
)

english_sample.to_csv(
    OUT / "english_exploration_sample.csv",
    index=False,
    encoding="utf-8-sig",
)

audit["english"] = {
    "original_rows": english_original_rows,
    "empty_rows_removed": int(empty.sum()),
    "duplicate_text_rows_removed": english_duplicates,
    "exploratory_sample_rows": len(english_sample),
    "note": "Star ratings are not topic labels. Language not verified per row.",
}

with open(REPORTS / "data_audit.json", "w", encoding="utf-8") as file:
    json.dump(audit, file, ensure_ascii=False, indent=2)

print("\nDONE: prepared data saved in Data/processed")
print("Audit and conflicts saved in reports")