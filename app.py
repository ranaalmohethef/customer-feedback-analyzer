"""صوت العميل — Gradio demo for Arabic app-review analysis."""
from pathlib import Path
from functools import lru_cache
import hashlib
import html
import re
import time

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
CACHE = ROOT / "Data" / "embedding_cache"

TOPICS = {
    "bug_report": "بلاغ خلل",
    "improvement_request": "طلب تحسين",
    "rating": "تقييم عام",
    "others": "أخرى",
}
SENTIMENTS = {"positive": "إيجابي", "negative": "سلبي"}

# ---------------------------------------------------------------- models
required = ["hybrid_classifier.joblib", "sentiment_classifier.joblib", "multilingual_encoder"]
missing = [name for name in required if not (MODELS / name).exists()]
if missing:
    raise FileNotFoundError("Missing local models: " + ", ".join(missing))

print("Loading saved models...")
hybrid = joblib.load(MODELS / "hybrid_classifier.joblib")
sentiment_v1 = joblib.load(MODELS / "sentiment_classifier.joblib")
encoder = SentenceTransformer(str(MODELS / "multilingual_encoder"), device="cpu")


def load_optional(name):
    path = MODELS / name
    return joblib.load(path) if path.exists() else None


ar_v2 = load_optional("sentiment_ar_v2.joblib")
en_v2 = load_optional("sentiment_en_v2.joblib")

ARABIC = re.compile(r"[ء-غف-ي]")
LATIN = re.compile(r"[A-Za-z]")
MAX_CHARS = 3000
MAX_BATCH = 100


def sentiment_predict(bundle, texts, vectors):
    if bundle["kind"] == "baseline":
        return bundle["pipeline"].predict(texts)
    features = hstack([
        bundle["word"].transform(texts),
        bundle["char"].transform(texts) * 0.5,
        csr_matrix(vectors * bundle["embedding_weight"]),
    ], format="csr")
    return bundle["classifier"].predict(features)


def predict_reviews(texts):
    """Arabic: evaluated topic + sentiment. English: experimental sentiment only."""
    vectors = encoder.encode(texts, batch_size=16, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
    rows = []
    for i, text in enumerate(texts):
        v = vectors[i:i + 1]
        if ARABIC.search(text):
            features = hstack([hybrid["vectorizer"].transform([text]),
                               csr_matrix(v * hybrid["embedding_weight"])], format="csr")
            topic = TOPICS[hybrid["classifier"].predict(features)[0]]
            label = sentiment_predict(ar_v2, [text], v)[0] if ar_v2 else sentiment_v1.predict([text])[0]
            feeling = SENTIMENTS[label]
            note = "نص مختلط؛ حُلّل كعربي." if LATIN.search(text) else ""
        elif LATIN.search(text) and en_v2:
            topic = "—"
            feeling = SENTIMENTS[sentiment_predict(en_v2, [text], v)[0]] + " (تجريبي)"
            note = "الإنجليزي: المشاعر فقط، وهي تجريبية. نوع التعليق للعربي فقط."
        else:
            topic, feeling = "—", "—"
            note = "اكتبي التعليق بالعربي."
        rows.append({"التعليق": text, "نوع التعليق": topic, "المشاعر": feeling, "ملاحظات": note})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analysis
def analyze_one(text):
    text = (text or "").strip()
    if not text:
        return "—", "—", "اكتبي تعليقًا أولًا."
    if len(text) > MAX_CHARS:
        return "—", "—", f"الحد الأقصى {MAX_CHARS} حرف."
    start = time.perf_counter()
    row = predict_reviews([text]).iloc[0]
    note = f"{time.perf_counter() - start:.2f} ثانية"
    if row["ملاحظات"]:
        note += " · " + row["ملاحظات"]
    return row["نوع التعليق"], row["المشاعر"], note


def analyze_batch(text):
    texts = [t.strip() for t in (text or "").splitlines() if t.strip()]
    empty = pd.DataFrame(columns=["التعليق", "نوع التعليق", "المشاعر", "ملاحظات"])
    if not texts:
        return empty, "ضعي كل تعليق في سطر مستقل."
    if len(texts) > MAX_BATCH or any(len(t) > MAX_CHARS for t in texts):
        return empty, f"الحد {MAX_BATCH} تعليق، وكل تعليق لا يتجاوز {MAX_CHARS} حرف."
    result = predict_reviews(texts)
    return result, summary_html(result)


# ---------------------------------------------------------------- HTML views
def tone(sentiment):
    if sentiment.startswith("إيجابي"):
        return "pos"
    if sentiment.startswith("سلبي"):
        return "neg"
    return "muted"


def result_card(text):
    topic, feeling, note = analyze_one(text)
    if topic == "—" and feeling == "—":
        return f'<div class="card empty">{html.escape(note)}</div>'
    return f"""
    <div class="card">
      <div class="stat"><span class="label">نوع التعليق</span>
        <span class="value">{html.escape(topic)}</span></div>
      <div class="stat"><span class="label">المشاعر</span>
        <span class="pill {tone(feeling)}">{html.escape(feeling)}</span></div>
      <div class="note">{html.escape(note)}</div>
    </div>"""


def bars(counts, total, css=""):
    rows = []
    for label, count in counts.items():
        pct = 100 * count / total
        rows.append(f"""
        <div class="bar-row">
          <span class="bar-label">{html.escape(str(label))}</span>
          <span class="bar"><span class="fill {css or tone(str(label))}" style="width:{pct:.0f}%"></span></span>
          <span class="bar-count">{count}</span>
        </div>""")
    return "".join(rows)


def summary_html(result):
    total = len(result)
    feelings = result["المشاعر"].value_counts()
    topics = result.loc[result["نوع التعليق"] != "—", "نوع التعليق"].value_counts()
    topic_block = bars(topics, total, "accent") if len(topics) else '<p class="muted-text">لا يوجد تعليق عربي.</p>'
    return f"""
    <div class="summary">
      <div class="card"><h4>المشاعر</h4>{bars(feelings, total)}</div>
      <div class="card"><h4>أنواع التعليقات</h4>{topic_block}</div>
    </div>
    <p class="muted-text">{total} تعليق · نتائج متوقعة من النموذج وليست مؤكدة.</p>"""


def batch_ui(text):
    result, summary = analyze_batch(text)
    if result.empty:
        return result, f'<div class="card empty">{html.escape(summary)}</div>'
    return result, summary


# ---------------------------------------------------------------- exploration
def read_report(name, columns):
    path = REPORTS / name
    if not path.exists():
        raise gr.Error(f"الملف غير موجود: reports/{name}")
    frame = pd.read_csv(path).fillna("")
    if not set(columns).issubset(frame.columns):
        raise gr.Error(f"أعمدة غير متوقعة في {name}")
    return frame


def load_exploration():
    summary = read_report("english_cluster_summary.csv",
                          ["cluster", "review_count", "keywords", "suggested_topic_manual"])
    examples = read_report("english_cluster_examples.csv", ["cluster", "review"])
    shown = summary[["cluster", "suggested_topic_manual", "review_count", "keywords"]].copy()
    shown.columns = ["المجموعة", "الموضوع", "العدد", "الكلمات البارزة"]
    samples = examples[["cluster", "review"]].copy()
    samples.columns = ["المجموعة", "مثال"]
    return shown, samples


@lru_cache(maxsize=1)
def search_index(file_signature):
    frame = read_report("english_clustered_reviews.csv", ["review", "cluster"])
    frame = frame[frame["review"].astype(str).str.strip().ne("")].reset_index(drop=True)
    texts = frame["review"].astype(str).tolist()
    identity = str(MODELS / "multilingual_encoder") + str(encoder.max_seq_length)
    digest = hashlib.sha256((identity + "\0" + "\0".join(texts)).encode()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"ui_search_{digest}.npy"
    if path.exists():
        vectors = np.load(path, allow_pickle=False)
    else:
        vectors = encoder.encode(texts, batch_size=16, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)
        np.save(path, vectors)
    return frame, vectors


def search_reviews(query):
    query = (query or "").strip()
    if not query:
        raise gr.Error("اكتبي موضوع البحث أولًا.")
    if len(query) > MAX_CHARS:
        raise gr.Error(f"الحد الأقصى {MAX_CHARS} حرف.")
    path = REPORTS / "english_clustered_reviews.csv"
    if not path.exists():
        raise gr.Error("شغّلي explore_english.py أولًا.")
    start = time.perf_counter()
    frame, vectors = search_index((path.stat().st_mtime_ns, path.stat().st_size))
    query_vector = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = vectors @ query_vector
    order = np.argsort(-scores)[:5]
    result = frame.iloc[order][["review", "cluster"]].copy()
    result["similarity"] = np.round(scores[order], 3)
    result.columns = ["تعليق مشابه", "المجموعة", "التشابه"]
    return result, f"{time.perf_counter() - start:.2f} ثانية · الدرجة للتشابه وليست احتمال صحة."


# ---------------------------------------------------------------- layout
ABOUT = """
### نتائج الاختبار

| المهمة | النموذج | Macro-F1 | عدد التعليقات |
|---|---|---:|---:|
| نوع التعليق | TF-IDF (الأساس) | 57.01% | 451 |
| نوع التعليق | TF-IDF + Embeddings | **62.64%** | 451 |
| المشاعر | TF-IDF كلمات (v1) | 85.92% | 4,356 |
| المشاعر | TF-IDF كلمات وحروف (v2) | **86.49%** | 4,356 |

### حدود النموذج
- المشاعر إيجابي أو سلبي فقط؛ لا توجد فئة محايد أو مختلط.
- يختار نوعًا واحدًا حتى لو جمع التعليق أكثر من غرض.
- قد يخطئ مع اللهجات والسخرية والنفي.
- الإنجليزي: المشاعر فقط، وهي تجريبية ومتعلمة من النجوم.
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap');

:root {
  --ink: #1f2933; --muted: #6b7682; --line: #e6e9ee; --bg: #f7f8fa;
  --accent: #0f766e; --accent-soft: #e7f3f1;
  --pos: #15803d; --pos-soft: #e8f5ec; --neg: #b42318; --neg-soft: #fdecea;
}
.gradio-container {
  max-width: 920px !important; margin: auto !important; direction: rtl;
  font-family: 'IBM Plex Sans Arabic', Tahoma, Arial, sans-serif !important;
  color: var(--ink);
}
.gradio-container * { font-family: inherit; }
textarea, input { direction: rtl !important; text-align: right !important; font-size: 16px !important; line-height: 1.8 !important; }
#search-input textarea { direction: ltr !important; text-align: left !important; }

.header { padding: 28px 4px 12px; }
.header h1 { font-size: 28px !important; font-weight: 600; margin: 0 0 6px !important; color: var(--ink) !important; }
.header p { color: var(--muted) !important; margin: 0 !important; font-size: 15px; }

.card {
  background: #fff; border: 1px solid var(--line); border-radius: 14px;
  padding: 18px 20px; display: flex; flex-direction: column; gap: 14px;
}
.card.empty { color: var(--muted); align-items: center; padding: 28px; }
.card h4 { margin: 0 !important; font-size: 15px; font-weight: 600; color: var(--ink); }
.stat { display: flex; justify-content: space-between; align-items: center; }
.stat .label { color: var(--muted); font-size: 14px; }
.stat .value { font-size: 18px; font-weight: 600; }
.pill { padding: 4px 14px; border-radius: 999px; font-weight: 600; font-size: 15px; }
.pill.pos { background: var(--pos-soft); color: var(--pos); }
.pill.neg { background: var(--neg-soft); color: var(--neg); }
.pill.muted { background: var(--bg); color: var(--muted); }
.note { color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); padding-top: 10px; }

.summary { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 700px) { .summary { grid-template-columns: 1fr; } }
.bar-row { display: grid; grid-template-columns: 110px 1fr 36px; align-items: center; gap: 10px; font-size: 14px; }
.bar { height: 8px; background: var(--bg); border-radius: 99px; overflow: hidden; }
.fill { display: block; height: 100%; border-radius: 99px; background: var(--muted); }
.fill.pos { background: var(--pos); } .fill.neg { background: var(--neg); } .fill.accent { background: var(--accent); }
.bar-count { text-align: left; color: var(--muted); }
.muted-text { color: var(--muted) !important; font-size: 13px; }

button.primary { border-radius: 10px !important; min-height: 44px; font-size: 15px !important; }

.gradio-container table, .gradio-container table td, .gradio-container table th,
.gradio-container [role="gridcell"], .gradio-container [role="columnheader"] {
  background: #fff !important; color: var(--ink) !important; border-color: var(--line) !important;
}
.gradio-container table th, .gradio-container [role="columnheader"] { background: var(--bg) !important; }
.gradio-container table td * , .gradio-container table th * { color: inherit !important; }
"""

theme = gr.themes.Base(primary_hue="teal", neutral_hue="slate").set(
    body_background_fill="#f7f8fa", body_background_fill_dark="#f7f8fa",
    body_text_color="#1f2933", body_text_color_dark="#1f2933",
    block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
    block_border_color="#e6e9ee", block_border_color_dark="#e6e9ee",
    block_label_text_color="#6b7682", block_label_text_color_dark="#6b7682",
    block_title_text_color="#1f2933", block_title_text_color_dark="#1f2933",
    input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
    input_border_color="#e6e9ee", input_border_color_dark="#e6e9ee",
    button_primary_background_fill="#0f766e", button_primary_background_fill_dark="#0f766e",
    button_primary_background_fill_hover="#115e59", button_primary_background_fill_hover_dark="#115e59",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    border_color_primary="#e6e9ee", border_color_primary_dark="#e6e9ee",
)

gradio_major = int(gr.__version__.split(".")[0])
style = {"theme": theme, "css": CSS}

with gr.Blocks(title="صوت العميل", **(style if gradio_major < 6 else {})) as demo:
    gr.HTML("""<div class="header"><h1>صوت العميل</h1>
        <p>اعرفي نوع تعليق العميل ومشاعره في ثوانٍ.</p></div>""")

    with gr.Tab("تعليق واحد"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                review = gr.Textbox(label="التعليق", lines=4, placeholder="اكتبي تعليق العميل هنا...")
                analyze_button = gr.Button("حلّلي", variant="primary")
                gr.Examples(label="أمثلة", inputs=[review], examples=[
                    ["التطبيق يقفل كل ما حاولت أسجل الدخول"],
                    ["أتمنى تضيفون الوضع الليلي"],
                    ["تطبيق جميل وسهل الاستخدام"],
                ])
            with gr.Column(scale=2):
                result = gr.HTML('<div class="card empty">تظهر النتيجة هنا</div>')
        analyze_button.click(result_card, inputs=[review], outputs=[result])
        review.submit(result_card, inputs=[review], outputs=[result])

    with gr.Tab("مجموعة تعليقات"):
        batch_input = gr.Textbox(label=f"كل تعليق في سطر — حتى {MAX_BATCH}", lines=6,
                                 placeholder="التطبيق يعلق عند الدفع\nأتمنى إضافة الوضع الليلي\nتطبيق ممتاز وسهل")
        batch_button = gr.Button("حلّلي المجموعة", variant="primary")
        batch_summary = gr.HTML()
        batch_table = gr.Dataframe(headers=["التعليق", "نوع التعليق", "المشاعر", "ملاحظات"],
                                   interactive=False, wrap=True, label="النتائج")
        batch_button.click(batch_ui, inputs=[batch_input], outputs=[batch_table, batch_summary])

    with gr.Tab("استكشاف"):
        gr.Markdown("مجموعات من 3,000 تعليق (أغلبها إنجليزي)، وبحث عن تعليقات مشابهة.")
        load_button = gr.Button("اعرضي المجموعات", variant="primary")
        clusters = gr.Dataframe(interactive=False, wrap=True, label="المجموعات")
        examples = gr.Dataframe(interactive=False, wrap=True, label="أمثلة")
        load_button.click(load_exploration, outputs=[clusters, examples])
        query = gr.Textbox(label="ابحثي عن موضوع", elem_id="search-input",
                           placeholder="The app crashes after the update")
        search_button = gr.Button("ابحثي", variant="primary")
        search_table = gr.Dataframe(interactive=False, wrap=True, label="أقرب 5 تعليقات")
        search_note = gr.Markdown(elem_classes=["muted-text"])
        search_button.click(search_reviews, inputs=[query], outputs=[search_table, search_note])
        query.submit(search_reviews, inputs=[query], outputs=[search_table, search_note])

    with gr.Tab("عن النموذج"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", share=False, inbrowser=True,
                        **(style if gradio_major >= 6 else {}))
