from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "processed"
REPORTS = ROOT / "reports"
CACHE = ROOT / "Data" / "embedding_cache"
MODELS = ROOT / "models"

REPORTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

path = DATA / "english_exploration_sample.csv"
df = pd.read_csv(path)

if "review" not in df.columns:
    raise ValueError("The file must contain a review column.")

df = df.dropna(subset=["review"]).copy()
df["review"] = df["review"].astype(str).str.strip()
df = df.loc[df["review"].ne("")].copy()

# Remove duplicates after whitespace normalization.
df["exploration_key"] = df["review"].str.replace(
    r"\s+", " ", regex=True
)
df = df.drop_duplicates("exploration_key").reset_index(drop=True)

if len(df) < 20:
    raise ValueError("Not enough reviews for this exploration.")

texts = df["review"].tolist()
print(f"Reviews to explore: {len(texts)}")

encoder = SentenceTransformer(
    str(MODELS / "multilingual_encoder"),
    device="cpu",
)

signature = json.dumps(
    {
        "encoder": "paraphrase-multilingual-MiniLM-L12-v2",
        "max_seq_length": encoder.max_seq_length,
        "normalized": True,
        "texts": texts,
    },
    ensure_ascii=False,
)
digest = hashlib.sha256(
    signature.encode("utf-8")
).hexdigest()[:16]
cache_path = CACHE / f"english_{digest}.npy"

if cache_path.exists():
    vectors = np.load(cache_path, allow_pickle=False)
    print("Loaded cached embeddings.")
else:
    vectors = encoder.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, vectors)

# Choose among a small, fixed set of cluster counts.
experiments = []
best_score = -np.inf
best_model = None

for k in [5, 8, 10]:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    assignments = model.fit_predict(vectors)

    if len(np.unique(assignments)) < 2:
        continue

    score = silhouette_score(
        vectors,
        assignments,
        metric="cosine",
        sample_size=min(1500, len(df)),
        random_state=42,
    )

    experiments.append({
        "clusters": k,
        "silhouette_cosine": float(score),
    })
    print(f"k={k}: silhouette={score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model

if best_model is None:
    raise ValueError("Could not obtain valid clusters.")

df["cluster"] = best_model.labels_ + 1

# Keywords describe frequent weighted terms within each cluster.
# They are hints for interpretation, not verified topic names.
word_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=20000,
)
word_vectors = word_vectorizer.fit_transform(texts)
terms = word_vectorizer.get_feature_names_out()

summary_rows = []
example_rows = []

for cluster_index in range(best_model.n_clusters):
    indices = np.flatnonzero(
        best_model.labels_ == cluster_index
    )

    mean_weights = np.asarray(
        word_vectors[indices].mean(axis=0)
    ).ravel()

    ranked_terms = np.argsort(mean_weights)[::-1]
    top_terms = [
        terms[i]
        for i in ranked_terms
        if mean_weights[i] > 0
    ][:8]

    # Nearest reviews to the cluster center under KMeans distance.
    distances = np.linalg.norm(
        vectors[indices]
        - best_model.cluster_centers_[cluster_index],
        axis=1,
    )
    representative_indices = indices[
        np.argsort(distances)[:3]
    ]

    summary_rows.append({
        "cluster": cluster_index + 1,
        "review_count": len(indices),
        "keywords": " | ".join(top_terms),
        "suggested_topic_manual": "",
        "review_notes": "",
    })

    for rank, row_index in enumerate(
        representative_indices, start=1
    ):
        example_rows.append({
            "cluster": cluster_index + 1,
            "example_type": "nearest_to_center",
            "rank": rank,
            "review": texts[row_index],
        })

    # Random examples help expose variety beyond central examples.
    rng = np.random.default_rng(42 + cluster_index)
    remaining = np.setdiff1d(
        indices, representative_indices
    )
    random_indices = rng.choice(
        remaining,
        size=min(2, len(remaining)),
        replace=False,
    )

    for rank, row_index in enumerate(random_indices, start=1):
        example_rows.append({
            "cluster": cluster_index + 1,
            "example_type": "random",
            "rank": rank,
            "review": texts[row_index],
        })

summary = pd.DataFrame(summary_rows)
examples = pd.DataFrame(example_rows)

summary.to_csv(
    REPORTS / "english_cluster_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
examples.to_csv(
    REPORTS / "english_cluster_examples.csv",
    index=False,
    encoding="utf-8-sig",
)
df.to_csv(
    REPORTS / "english_clustered_reviews.csv",
    index=False,
    encoding="utf-8-sig",
)
pd.DataFrame(experiments).to_csv(
    REPORTS / "english_cluster_experiments.csv",
    index=False,
    encoding="utf-8-sig",
)

metadata = {
    "purpose": "Exploratory clustering, not supervised evaluation",
    "rows": len(df),
    "selected_k": best_model.n_clusters,
    "silhouette_cosine": float(best_score),
    "selection_candidates": [5, 8, 10],
    "language_verified_per_row": False,
    "notes": [
        "Silhouette is not classification accuracy.",
        "Cluster numbers are arbitrary identifiers.",
        "Topics require inspection of actual reviews.",
        "Star ratings were not used as topic labels.",
        "No Arabic training or test data were changed.",
    ],
}

with open(
    REPORTS / "english_exploration_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

print("\nENGLISH EXPLORATION RESULTS")
print(f"Selected clusters: {best_model.n_clusters}")
print(f"Silhouette: {best_score:.4f}")
print(summary[["cluster", "review_count", "keywords"]].to_string(
    index=False
))
print("\nDONE: English exploration reports saved.")