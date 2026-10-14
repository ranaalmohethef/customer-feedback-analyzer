# صوت العميل — Customer Feedback Analyzer

An applied NLP prototype for Arabic app-review classification and sentiment analysis, with exploratory clustering of an additional multilingual review source.

**Course:** Applied NLP capstone, [SDAIA Academy](https://github.com/SDAIAAcademy) · **Use case:** 3, Voice of Customer Analyzer

## Problem and scope

App reviews mix bugs, feature requests, praise, and unrelated content. This project helps reviewers organize feedback and inspect aggregate predictions. Outputs support human review; they are not guaranteed decisions.

- Arabic topic classification: `bug_report`, `improvement_request`, `rating`, `others`.
- Arabic binary sentiment: `negative`, `positive` (no neutral or mixed class).
- Local Gradio demo with an Arabic / English interface: one review or up to 100 newline-separated reviews. When the model is not confident, it answers "Not sure" instead of guessing.
- Exploration tab: shows clusters of a 3,000-review sample from user_quality, with example reviews, and semantic search over that sample. This is exploratory, not an English classification benchmark.
- Experimental English sentiment, trained on star-derived labels. Topic classification is Arabic only.

## Data

Three datasets from **two independent projects**, not three independent providers:

| Dataset | Original rows | Use |
|---|---:|---|
| [AURA Classification V2](https://huggingface.co/datasets/irfan-ahmad/AURA-Classification) | 2,900 | Arabic topic classification |
| [AURA Sentiment](https://huggingface.co/datasets/irfan-ahmad/AURA-Sentiment) | 29,700 | Arabic sentiment |
| [user_quality reviews](https://github.com/sealuzh/user_quality) | 288,065 | 3,000-review exploratory sample |

AURA Classification and Sentiment share some review texts. They are separate tasks, not independent replications. The third source is predominantly English but inspected examples include other languages (e.g. Portuguese), names, and unclear text. Language has not been verified per row. Star ratings are not used as topic labels.

Raw data are stored locally in `Data/raw/`. Download datasets from their original sources; raw data are not bundled by default. Consult source terms before redistribution. This repository does not grant rights over third-party data or models.

### Preparation and splits

`prepare_data.py` removes missing/empty records, builds a whitespace-normalized text key, excludes conflicting labels for identical keys, and deduplicates by that key. Original review text is retained. Conflict exclusions are documented and provisional; we do not claim their labels have been corrected.

Splits use a deterministic SHA-256 hash of `feedback-v1|` plus the normalized text, with nominal 70%/15%/15% thresholds. Shared text across the two AURA tasks receives the same split. Actual counts vary because this is not a stratified split.

| Task | Train | Validation | Test |
|---|---:|---:|---:|
| Topic | 2,050 | 390 | 451 |
| Sentiment | 20,094 | 4,239 | 4,356 |

This prevents exact normalized-text overlap; it does not guarantee removal of near duplicates or separation by application, author, or time. Test results represent held-out reviews under this split, not performance on entirely unseen apps.

## Approach

### Topic classification

1. **Baseline:** word TF-IDF (unigrams/bigrams) and Logistic Regression.
2. **Embeddings:** frozen [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), normalized 384-dimensional vectors, and Logistic Regression.
3. **Hybrid:** concatenate training-fitted TF-IDF features with normalized embeddings multiplied by 0.5; Logistic Regression with `C=10`.

Embedding-only selection tested C in {0.1, 1, 10}, with and without balanced class weights. Hybrid selection tested embedding weights {0.5, 1, 2} and C in {1, 10}. All choices used validation Macro-F1. The selected models were trained on train only; no train-plus-validation refit was performed before test evaluation.

The encoder is frozen: training the classifier **is not encoder fine-tuning**. See [ADAPTATION.md](ADAPTATION.md).

### Sentiment

**Version 1:** word TF-IDF plus Logistic Regression. C in {1, 10} was compared on validation; C=1 was selected. This is a separate model and dataset.

**Version 2 (`upgrade_models.py`):** candidates combine word TF-IDF, character TF-IDF (3–5 char n-grams, weight 0.5), and optionally normalized embeddings. Embedding weight in {0, 1} and C in {1, 10} were compared on validation, together with the version 1 model. The selected Arabic model uses **embedding weight 0 and C=1**, so it is word + character TF-IDF only; it does not use the encoder. The report file labels it `hybrid` because of the code path, not because embeddings contribute.

### Experimental English sentiment

Trained on a 12,000-review sample of user_quality with labels derived from stars (1–2 negative, 4–5 positive, 3 excluded). Rows were kept only if they contained Latin letters and no Arabic letters; this is not verified English. 1,454 text groups with conflicting star labels were excluded. Split 8,375 / 1,783 / 1,842 by hash. Selected on validation: embedding weight 1, C=1.

### Low-confidence abstention (`choose_thresholds.py`)

The app answers "Not sure — needs human review" when the top predicted probability is below a threshold. The rule was fixed before looking at results: abstain on the least confident 10% of **validation** reviews. Test data was not used.

| Task | Threshold | Accuracy (all validation) | Accuracy when answering | Accuracy on abstained |
|---|---:|---:|---:|---:|
| Arabic topic | 0.493 | 72.05% | 74.93% | 46.15% |
| Arabic sentiment | 0.583 | 87.21% | 90.46% | 58.02% |

On abstained sentiment reviews the model was close to a coin flip (58%), so abstaining removes mostly unreliable answers. Abstention does not catch confident errors: `التطبيق خيالي` is still predicted negative. English sentiment has no threshold.

### Exploratory clustering

Normalized embeddings and KMeans were tested with k in {5, 8, 10}. The script selects by cosine silhouette on a reproducible sample of up to 1,500 reviews. The submitted summary contains five clusters. Silhouette is an internal clustering measure, not classification accuracy; no unprovided silhouette score is claimed here.

Three nearest-center and two random examples were inspected per cluster (25 total), with AI-assisted topic names. Groups include general praise (633), operational/update problems (685), unclear and varied short texts (571), usability/performance with mixed content (666), and short praise (445). Praise groups overlap; not all clusters form distinct useful topics. These are provisional interpretations, not labels verified for all 3,000 reviews.

## Results

### Topic classification: same 451-review held-out test

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF baseline | 68.29% | 57.01% |
| Embeddings only | 63.86% | 56.15% |
| Hybrid | **69.18%** | **62.64%** |

Hybrid was chosen before test evaluation. Relative to baseline, it adds **5.63 percentage points of Macro-F1** and **0.89 percentage points of accuracy**. Embeddings alone did not outperform baseline. The paired bootstrap interval excludes zero; it is still a single test set and a single split.

95% bootstrap CIs (1,000 resamples) for test Macro-F1: baseline [51.21%, 62.31%], embeddings [50.95%, 61.07%], hybrid [57.67%, 67.21%]. Paired bootstrap of the hybrid minus baseline difference: [+0.77, +10.81] points; the hybrid was better in 99.1% of resamples.

Validation Macro-F1: baseline 59.57%, embeddings 60.20%, hybrid 67.19%.

### Topic slices by platform

| Platform | Test rows | Baseline accuracy | Hybrid accuracy | Baseline Macro-F1 | Hybrid Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Android | 212 | 60.38% | 63.21% | 49.66% | 56.60% |
| iOS | 239 | 75.31% | 74.48% | 61.32% | 66.82% |

All four reference classes appear in both slices. Macro-F1 improved in both; iOS accuracy declined slightly. Platform differences do not establish a causal effect of platform.

### Sentiment: 4,356-review held-out test

Accuracy **85.95%**, Macro-F1 **85.92%**.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.844 | 0.889 | 0.866 | 2,224 |
| Positive | 0.878 | 0.828 | 0.852 | 2,132 |

Topic and sentiment were evaluated on different test sets. These numbers do not measure joint correctness of both outputs for the same review.

### Sentiment version 2

| Model | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Macro-F1 95% CI |
|---|---:|---:|---:|---:|
| Arabic v1 (word TF-IDF) | 86.19% | 85.95% | 85.92% | — |
| Arabic v2 (word + char TF-IDF) | 87.20% | **86.52%** | **86.49%** | [85.43%, 87.51%] |
| English experimental (star proxy) | 82.23% | 87.68% | 82.33% | [80.38%, 84.31%] |

CIs are percentile bootstrap (1,000 resamples) over test predictions. The Arabic v2 choice was made on validation. The Arabic test set had already been inspected during error analysis, so the v2 test result is descriptive, not a fresh independent evaluation. The +0.57 point Macro-F1 gain is small and within the v2 confidence interval.

Arabic v2 by platform: Android 2,188 reviews, Macro-F1 85.24%; iOS 2,168 reviews, Macro-F1 87.70%.

English scores measure agreement with star-derived labels, not human-verified sentiment. The English test set is imbalanced (1,398 positive, 444 negative), so accuracy is higher than Macro-F1.

## Error analysis and limitations

The reports include 20 sampled topic disagreements and 20 sampled sentiment disagreements, each categorized and explained with AI assistance. Annotations explicitly identify AI assistance and still need human confirmation. Reference labels and test data were not edited. A disagreement is not automatically a model mistake.

Observed patterns:

- Multiple intents: praise plus a bug or feature request.
- Confusion between missing features, product policy, and actual defects.
- Dialect, negation, spelling variation, and unclear text.
- Mixed sentiment and neutral questions forced into binary classes.
- Potentially questionable reference labels: e.g. positive wording labeled negative. These are review flags, not confirmed relabelings.

The UI example `التطبيق خيالي` was predicted as negative although its apparent intent is praise. `ما اعجبتني الواجهه بس التطبيق مفيد` was predicted positive, but contains mixed sentiment. No keyword exception was added to hide these failures.

Error samples are selected from disagreements, so they cannot estimate overall annotation-noise prevalence. The model may truncate long text at the encoder's input limit. Arabic-character checking in the UI is only a basic input check, not reliable language identification. The prototype is not production-ready.

## Environment, latency, and cost

Development used Windows PowerShell, Python 3.11, and CPU inference; `torch.cuda.is_available()` returned False. The user-provided logs show approximately 14 seconds to encode 2,050 training reviews and 2 seconds for 390 validation reviews. UI screenshots show roughly 0.25–0.35 seconds for three single-review requests; these are observations, not a controlled latency benchmark.

`reports/test_model_comparison.csv` contains batch timings excluding model load. Batch-average milliseconds are not single-request UI latency. Initial model download and startup are excluded. The pipeline uses local models rather than paid inference APIs; hardware, electricity, storage, and download costs were not measured. Exact machine specifications and library versions should accompany any reproducibility claim.

## Setup and run (Windows PowerShell)

Run from the project root. Keep filenames and directory capitalization as shown.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place these inputs in `Data/raw/`:

- Exactly one `AURA-Classification*.csv` (use V2).
- Exactly one `*.parquet` (AURA Sentiment).
- `user_quality.csv` (rename the downloaded user_quality `reviews.csv`).

For a fresh reproduction:

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe train_baseline.py
.\.venv\Scripts\python.exe train_embeddings.py
.\.venv\Scripts\python.exe evaluate_test.py
.\.venv\Scripts\python.exe train_sentiment.py
.\.venv\Scripts\python.exe explore_english.py
.\.venv\Scripts\python.exe upgrade_models.py
.\.venv\Scripts\python.exe choose_thresholds.py
.\.venv\Scripts\python.exe check_app.py
.\.venv\Scripts\python.exe app.py
```

`check_app.py` runs automated checks of the app functions and writes `reports/upgrade_app_checks.json` and `reports/upgrade_ui_examples.csv`. A pass confirms the code runs, not that predictions are correct.

The first embedding run requires internet to download the model. Subsequent runs use the local encoder and cached vectors. `train_embeddings.py` includes the appended hybrid experiment. Open the local address printed by Gradio and keep the terminal running. `share=False` does not create a public demo URL.

**For the existing trained project, run only `app.py` to open the demo.** Back up reviewed CSVs before reproduction: `train_baseline.py` overwrites its error-review file, and `explore_english.py` overwrites the annotated cluster summary. Topic/sentiment test-review files are preserved by the supplied evaluation scripts when they already exist; if intentionally regenerating different models, archive stale reviews first rather than mixing runs.

Freeze the actual working environment after installation:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 requirements.txt
```

Model binaries, raw data, processed data, embeddings, and `.venv` are excluded from Git by the supplied `.gitignore`. They remain on the local computer. Reviewers can reproduce them using the original data sources and scripts. Preserve the local encoder for an offline presentation.

## Main files

| File | Purpose |
|---|---|
| `prepare_data.py` | Clean, audit, split, and sample data |
| `train_baseline.py` | Topic baseline |
| `train_embeddings.py` | Embedding and hybrid topic models |
| `evaluate_test.py` | Topic test comparison and platform slices |
| `train_sentiment.py` | Sentiment selection and test evaluation |
| `explore_english.py` | Additional-source clustering |
| `upgrade_models.py` | Sentiment v2 (Arabic) and experimental English sentiment |
| `choose_thresholds.py` | "Not sure" thresholds chosen on validation |
| `check_app.py` | Automated app checks and sample outputs |
| `app.py` | Local Gradio demo |
| `reports/` | Metrics, predictions, error analysis, clustering summaries |

## Observed sample inputs and outputs

These are actual UI observations, including failures, not expected labels substituted for predictions.

| Input | Predicted topic | Predicted sentiment | Interpretation |
|---|---|---|---|
| التطبيق خيالي | rating | negative | Sentiment error under apparent praise intent |
| التطبيق مفيد ولكن يحتاج بعض التحسينات | improvement_request | positive | Praise plus improvement request |
| ما اعجبتني الواجهه بس التطبيق مفيد | rating | positive | Mixed sentiment reduced to a binary output |

After the version 2 upgrade, before the English topic output was removed:

| Input | Predicted topic | Predicted sentiment | Interpretation |
|---|---|---|---|
| التطبيق خيالي | rating | negative | Still a sentiment error |
| مشكور والله ما قصرتو | rating | negative | Sentiment error on dialect thanks |
| ليس جميل نيمو أفضل منه | rating | positive | Negation missed |
| ما اعجبتني الواجهه بس التطبيق مفيد | rating | positive | Mixed sentiment, binary output |
| The app is amazing | bug_report (experimental) | positive (experimental) | English topic error |
| The app crashes every time I open it | bug_report (experimental) | negative (experimental) | Plausible |
| Please add dark mode | others (experimental) | positive (experimental) | English topic error; expected improvement_request |

An earlier version also guessed English topics by similarity to category descriptions. It was never evaluated and failed on simple examples (above), so it was removed from the demo. English input now gets sentiment only.

Current app, with abstention (`reports/upgrade_ui_examples.csv`):

| Input | Predicted topic | Predicted sentiment |
|---|---|---|
| التطبيق خيالي | rating | negative (confident error) |
| مشكور والله ما قصرتو | rating | not sure |
| ليس جميل نيمو أفضل منه | rating | positive (error) |
| ما اعجبتني الواجهه بس التطبيق مفيد | rating | positive |
| The app is amazing | — | positive (experimental) |
| The app crashes every time I open it | — | negative (experimental) |
| Please add dark mode | — | positive (experimental) |

The same meaning can get different results: `اكل لذيذ` is positive, but `لذيذ الاكل` is "not sure", because the word model treats `الاكل` and `اكل` as different tokens and learned `الاكل` as negative from delivery complaints.

## Presentation and remaining release checks

Prepare a 10-minute talk/live demo and allow 5 minutes for questions. Record a backup demo. Confirm the final UI, reviewed annotations, requirements file, source scripts, and reports are in place, publish the intended repository, and submit its link before the course deadline. The documentation does not claim that repository publication, a recording, or slides are already complete.
