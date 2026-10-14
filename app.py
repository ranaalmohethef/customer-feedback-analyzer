"""صوت العميل — Customer Feedback Analyzer (Gradio demo, Arabic / English UI)."""
from pathlib import Path
from functools import lru_cache
import hashlib
import html
import json
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
MAX_CHARS = 3000
MAX_BATCH = 100

# ---------------------------------------------------------------- text
TOPIC_NAMES = {
    "ar": {"bug_report": "بلاغ خلل", "improvement_request": "طلب تحسين", "rating": "تقييم عام", "others": "أخرى",
           "unsure": "غير متأكد"},
    "en": {"bug_report": "Bug report", "improvement_request": "Improvement request", "rating": "General rating",
           "others": "Other", "unsure": "Not sure"},
}
SENTIMENT_NAMES = {
    "ar": {"positive": "إيجابي", "negative": "سلبي", "unsure": "غير متأكد"},
    "en": {"positive": "Positive", "negative": "Negative", "unsure": "Not sure"},
}
# Kept for check_app.py and older code.
TOPICS = TOPIC_NAMES["ar"]
SENTIMENTS = SENTIMENT_NAMES["ar"]

CLUSTER_NAMES_AR = {
    "General praise": "مدح عام",
    "Operational / update problems": "مشاكل التشغيل والتحديث",
    "Unclear and varied short texts": "نصوص قصيرة متنوعة وغير واضحة",
    "Usability / performance, mixed content": "سهولة الاستخدام والأداء",
    "Short praise": "مدح قصير",
}

T = {
    "ar": {
        "dir": "rtl", "title": "صوت العميل", "subtitle": "اعرف نوع تعليق العميل ومشاعره في ثوانٍ.",
        "nav": [("تعليق واحد", "one"), ("مجموعة تعليقات", "batch"), ("استكشاف", "explore"), ("عن النموذج", "about")],
        "review_ph": "اكتب تعليق العميل هنا...", "analyze": "حلّل",
        "batch_ph": "كل تعليق في سطر — حتى 100\nالتطبيق يعلق عند الدفع\nأتمنى إضافة الوضع الليلي",
        "analyze_batch": "حلّل المجموعة", "search_ph": "ابحث عن موضوع، مثل: The app crashes after the update",
        "search": "ابحث", "show_clusters": "اعرض المجموعات", "examples": "جرّب مثالًا:",
        "placeholder": "تظهر النتيجة هنا", "topic": "نوع التعليق", "sentiment": "المشاعر",
        "seconds": "ثانية", "reviews": "تعليق",
        "summary_note": "نتائج متوقعة من النموذج وليست مؤكدة.", "no_arabic": "لا يوجد تعليق عربي.",
        "col_review": "التعليق", "col_note": "ملاحظات",
        "explore_intro": "مجموعات من 3,000 تعليق أغلبها إنجليزي، وبحث عن تعليقات مشابهة.",
        "cluster": "المجموعة", "keywords": "كلمات بارزة", "similarity": "التشابه",
        "search_note": "الدرجة تقيس التشابه، وليست احتمال صحة.",
        "empty": "اكتب تعليقًا أولًا.", "too_long": f"الحد الأقصى {MAX_CHARS} حرف.",
        "batch_empty": "ضع كل تعليق في سطر مستقل.",
        "batch_limit": f"الحد {MAX_BATCH} تعليق، وكل تعليق لا يتجاوز {MAX_CHARS} حرف.",
        "search_empty": "اكتب موضوع البحث أولًا.",
        "notes": {"mixed": "نص مختلط؛ حُلّل كعربي.",
                  "unsupported": "التحليل للتعليقات العربية فقط.",
                  "unsure": "ثقة النموذج منخفضة؛ يحتاج مراجعة بشرية."},
        "about": """
<h3>نتائج الاختبار</h3>
<table class="t"><tr><th>المهمة</th><th>النموذج</th><th>Macro-F1</th><th>التعليقات</th></tr>
<tr><td>نوع التعليق</td><td>TF-IDF (الأساس)</td><td>57.01%</td><td>451</td></tr>
<tr><td>نوع التعليق</td><td>TF-IDF + Embeddings</td><td><b>62.64%</b></td><td>451</td></tr>
<tr><td>المشاعر</td><td>كلمات (v1)</td><td>85.92%</td><td>4,356</td></tr>
<tr><td>المشاعر</td><td>كلمات وحروف (v2)</td><td><b>86.49%</b></td><td>4,356</td></tr></table>
<h3>حدود النموذج</h3>
<ul><li>المشاعر إيجابي أو سلبي فقط؛ لا توجد فئة محايد أو مختلط.</li>
<li>يختار نوعًا واحدًا حتى لو جمع التعليق أكثر من غرض.</li>
<li>قد يخطئ مع اللهجات والسخرية والنفي.</li>
<li>إذا كانت ثقة النموذج منخفضة يعرض "غير متأكد" بدل التخمين. الحد مختار من بيانات التحقق.</li>
<li>تحليل التعليقات للعربي فقط. زر English يغيّر لغة الواجهة.</li></ul>""",
    },
    "en": {
        "dir": "ltr", "title": "Voice of the Customer", "subtitle": "See the type and sentiment of a customer review in seconds.",
        "nav": [("Single review", "one"), ("Batch", "batch"), ("Explore", "explore"), ("About", "about")],
        "review_ph": "Type a customer review here...", "analyze": "Analyze",
        "batch_ph": "One review per line — up to 100\nThe app crashes at checkout\nPlease add dark mode",
        "analyze_batch": "Analyze batch", "search_ph": "Search a topic, e.g. The app crashes after the update",
        "search": "Search", "show_clusters": "Show clusters", "examples": "Try an example:",
        "placeholder": "The result appears here", "topic": "Review type", "sentiment": "Sentiment",
        "seconds": "s", "reviews": "reviews",
        "summary_note": "Model predictions, not confirmed labels.", "no_arabic": "No Arabic reviews.",
        "col_review": "Review", "col_note": "Notes",
        "explore_intro": "Clusters of 3,000 mostly English reviews, plus search for similar reviews.",
        "cluster": "Cluster", "keywords": "Keywords", "similarity": "Similarity",
        "search_note": "The score measures similarity, not the probability of being correct.",
        "empty": "Type a review first.", "too_long": f"Maximum {MAX_CHARS} characters.",
        "batch_empty": "Put each review on its own line.",
        "batch_limit": f"Up to {MAX_BATCH} reviews, {MAX_CHARS} characters each.",
        "search_empty": "Type a search topic first.",
        "notes": {"mixed": "Mixed text; analyzed as Arabic.",
                  "unsupported": "Review analysis supports Arabic text only.",
                  "unsure": "Low model confidence; needs human review."},
        "about": """
<h3>Test results</h3>
<table class="t"><tr><th>Task</th><th>Model</th><th>Macro-F1</th><th>Reviews</th></tr>
<tr><td>Review type</td><td>TF-IDF (baseline)</td><td>57.01%</td><td>451</td></tr>
<tr><td>Review type</td><td>TF-IDF + Embeddings</td><td><b>62.64%</b></td><td>451</td></tr>
<tr><td>Sentiment</td><td>Words (v1)</td><td>85.92%</td><td>4,356</td></tr>
<tr><td>Sentiment</td><td>Words + characters (v2)</td><td><b>86.49%</b></td><td>4,356</td></tr></table>
<h3>Limitations</h3>
<ul><li>Sentiment is positive or negative only; no neutral or mixed class.</li>
<li>One review type is chosen even when a review has several purposes.</li>
<li>Dialect, sarcasm and negation can cause errors.</li>
<li>When confidence is low the app shows "Not sure" instead of guessing. The threshold was chosen on validation data.</li>
<li>Review analysis is Arabic only; English changes the interface language.</li></ul>""",
    },
}
EXAMPLES = ["التطبيق يقفل كل ما حاولت أسجل الدخول", "أتمنى تضيفون الوضع الليلي", "تطبيق جميل وسهل الاستخدام"]

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


def load_thresholds():
    """Thresholds from choose_thresholds.py (validation only). Missing file = no abstention."""
    path = REPORTS / "abstention_thresholds.json"
    if not path.exists():
        return 0.0, 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["topic"]["threshold"], data["sentiment_ar"]["threshold"]


TOPIC_TAU, SENTIMENT_TAU = load_thresholds()

ARABIC = re.compile(r"[ء-غف-ي]")
LATIN = re.compile(r"[A-Za-z]")


def sentiment_proba(bundle, texts, vectors):
    if bundle is None:
        return sentiment_v1.classes_, sentiment_v1.predict_proba(texts)
    if bundle["kind"] == "baseline":
        return bundle["pipeline"].classes_, bundle["pipeline"].predict_proba(texts)
    features = hstack([
        bundle["word"].transform(texts),
        bundle["char"].transform(texts) * 0.5,
        csr_matrix(vectors * bundle["embedding_weight"]),
    ], format="csr")
    return bundle["classifier"].classes_, bundle["classifier"].predict_proba(features)


def confident(classes, proba, tau):
    """Top class, or 'unsure' when its probability is below the validation threshold."""
    i = int(np.argmax(proba))
    return classes[i] if proba[i] >= tau else "unsure"


def predict_codes(texts):
    """Return language-neutral predictions: topic code, sentiment code, note code."""
    vectors = encoder.encode(texts, batch_size=16, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
    out = []
    for i, text in enumerate(texts):
        v = vectors[i:i + 1]
        if ARABIC.search(text):
            features = hstack([hybrid["vectorizer"].transform([text]),
                               csr_matrix(v * hybrid["embedding_weight"])], format="csr")
            clf = hybrid["classifier"]
            topic = confident(clf.classes_, clf.predict_proba(features)[0], TOPIC_TAU)
            classes, proba = sentiment_proba(ar_v2, [text], v)
            senti = confident(classes, proba[0], SENTIMENT_TAU)
            note = "unsure" if "unsure" in (topic, senti) else ("mixed" if LATIN.search(text) else "")
            out.append((topic, senti, note))
        else:
            out.append((None, None, "unsupported"))
    return out


def label_row(code, lang):
    topic, senti, note = code
    t = T[lang]
    topic_txt = TOPIC_NAMES[lang][topic] if topic else "—"
    senti_txt = SENTIMENT_NAMES[lang][senti] if senti else "—"
    return topic_txt, senti_txt, t["notes"].get(note, ""), senti


def predict_reviews(texts, lang="ar"):
    t = T[lang]
    rows = []
    for text, code in zip(texts, predict_codes(texts)):
        topic, senti, note, _ = label_row(code, lang)
        rows.append({t["col_review"]: text, t["topic"]: topic, t["sentiment"]: senti, t["col_note"]: note})
    return pd.DataFrame(rows)


def analyze_one(text, lang="ar"):
    t = T[lang]
    text = (text or "").strip()
    if not text:
        return "—", "—", t["empty"]
    if len(text) > MAX_CHARS:
        return "—", "—", t["too_long"]
    start = time.perf_counter()
    topic, senti, note, _ = label_row(predict_codes([text])[0], lang)
    timing = f"{time.perf_counter() - start:.2f} {t['seconds']}"
    return topic, senti, timing + (" · " + note if note else "")


def analyze_batch(text, lang="ar"):
    t = T[lang]
    texts = [x.strip() for x in (text or "").splitlines() if x.strip()]
    empty = pd.DataFrame(columns=[t["col_review"], t["topic"], t["sentiment"], t["col_note"]])
    if not texts:
        return empty, t["batch_empty"]
    if len(texts) > MAX_BATCH or any(len(x) > MAX_CHARS for x in texts):
        return empty, t["batch_limit"]
    result = predict_reviews(texts, lang)
    return result, summary_html(result, lang)


# ---------------------------------------------------------------- HTML views
def esc(x):
    return html.escape(str(x))


def tone(code):
    return {"positive": "pos", "negative": "neg"}.get(code, "muted")


def message(text, lang):
    return f'<div class="card empty" dir="{T[lang]["dir"]}">{esc(text)}</div>'


def result_card(text, lang="ar"):
    t = T[lang]
    text = (text or "").strip()
    if not text:
        return message(t["empty"], lang)
    if len(text) > MAX_CHARS:
        return message(t["too_long"], lang)
    start = time.perf_counter()
    topic, senti, note, code = label_row(predict_codes([text])[0], lang)
    timing = f"{time.perf_counter() - start:.2f} {t['seconds']}"
    return f"""<div class="card" dir="{t['dir']}">
      <div class="stat"><span class="label">{t['topic']}</span><span class="value">{esc(topic)}</span></div>
      <div class="stat"><span class="label">{t['sentiment']}</span><span class="pill {tone(code)}">{esc(senti)}</span></div>
      <div class="note">{esc(timing + (' · ' + note if note else ''))}</div></div>"""


def bars(counts, total, css=None):
    rows = []
    for label, count in counts.items():
        cls = css or ("pos" if str(label).startswith(("إيجابي", "Positive")) else
                      "neg" if str(label).startswith(("سلبي", "Negative")) else "muted")
        rows.append(f"""<div class="bar-row"><span>{esc(label)}</span>
          <span class="bar"><span class="fill {cls}" style="width:{100 * count / total:.0f}%"></span></span>
          <span class="bar-count">{count}</span></div>""")
    return "".join(rows)


def summary_html(result, lang="ar"):
    t = T[lang]
    total = len(result)
    feelings = result[t["sentiment"]].value_counts()
    topics = result.loc[result[t["topic"]] != "—", t["topic"]].value_counts()
    topic_block = bars(topics, total, "accent") if len(topics) else f'<p class="muted-text">{t["no_arabic"]}</p>'
    return f"""<div dir="{t['dir']}"><div class="summary">
      <div class="card"><h4>{t['sentiment']}</h4>{bars(feelings, total)}</div>
      <div class="card"><h4>{t['topic']}</h4>{topic_block}</div></div>
      <p class="muted-text">{total} {t['reviews']} · {t['summary_note']}</p></div>"""


def table_html(frame, lang):
    head = "".join(f"<th>{esc(c)}</th>" for c in frame.columns)
    body = "".join("<tr>" + "".join(f'<td dir="auto">{esc(v)}</td>' for v in row) + "</tr>"
                   for row in frame.itertuples(index=False))
    return f'<div class="table-wrap" dir="{T[lang]["dir"]}"><table class="t"><tr>{head}</tr>{body}</table></div>'


def batch_ui(text, lang="ar"):
    result, summary = analyze_batch(text, lang)
    if result.empty:
        return message(summary, lang)
    return summary + table_html(result, lang)


# ---------------------------------------------------------------- exploration
def read_report(name, columns):
    path = REPORTS / name
    if not path.exists():
        raise gr.Error(f"Missing file: reports/{name}")
    frame = pd.read_csv(path).fillna("")
    if not set(columns).issubset(frame.columns):
        raise gr.Error(f"Unexpected columns in {name}")
    return frame


def load_exploration():
    summary = read_report("english_cluster_summary.csv",
                          ["cluster", "review_count", "keywords", "suggested_topic_manual"])
    examples = read_report("english_cluster_examples.csv", ["cluster", "review"])
    return summary, examples


def clusters_ui(lang="ar"):
    t = T[lang]
    summary, examples = load_exploration()
    cards = []
    for row in summary.itertuples(index=False):
        name = str(row.suggested_topic_manual) or f"{t['cluster']} {row.cluster}"
        if lang == "ar":
            name = CLUSTER_NAMES_AR.get(name, name)
        samples = examples.loc[examples["cluster"] == row.cluster, "review"].head(2)
        quotes = "".join(f'<li dir="auto">{esc(s)}</li>' for s in samples)
        cards.append(f"""<div class="card cluster">
          <div class="stat"><span class="value">{esc(name)}</span><span class="count">{row.review_count}</span></div>
          <div class="keywords" dir="ltr">{esc(str(row.keywords).replace(' | ', ' · '))}</div>
          <ul class="quotes">{quotes}</ul></div>""")
    return f'<div class="clusters" dir="{t["dir"]}">{"".join(cards)}</div>'


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


def search_reviews(query, lang="ar"):
    t = T[lang]
    query = (query or "").strip()
    if not query:
        raise gr.Error(t["search_empty"])
    if len(query) > MAX_CHARS:
        raise gr.Error(t["too_long"])
    start = time.perf_counter()
    path = REPORTS / "english_clustered_reviews.csv"
    frame, vectors = search_index((path.stat().st_mtime_ns, path.stat().st_size))
    query_vector = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = vectors @ query_vector
    order = np.argsort(-scores)[:5]
    result = frame.iloc[order][["review", "cluster"]].copy()
    result["similarity"] = np.round(scores[order], 3)
    result.columns = [t["col_review"], t["cluster"], t["similarity"]]
    return result, f"{time.perf_counter() - start:.2f} {t['seconds']} · {t['search_note']}"


def search_ui(query, lang="ar"):
    try:
        result, note = search_reviews(query, lang)
    except gr.Error as e:
        return message(str(e), lang)
    return table_html(result, lang) + f'<p class="muted-text" dir="{T[lang]["dir"]}">{esc(note)}</p>'


# ---------------------------------------------------------------- layout
def header_html(lang):
    t = T[lang]
    return f'<div class="header" dir="{t["dir"]}"><h1>{t["title"]}</h1><p>{t["subtitle"]}</p></div>'


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap');
:root { --ink:#1f2933; --muted:#6b7682; --line:#e6e9ee; --bg:#f7f8fa; --accent:#0f766e;
        --pos:#15803d; --pos-soft:#e8f5ec; --neg:#b42318; --neg-soft:#fdecea; }
.gradio-container { max-width: 940px !important; margin: auto !important; color: var(--ink);
  font-family: 'IBM Plex Sans Arabic', Tahoma, Arial, sans-serif !important; }
.gradio-container * { font-family: inherit; }
textarea, input { unicode-bidi: plaintext; text-align: start !important; font-size: 16px !important; line-height: 1.8 !important; }

.header { padding: 20px 4px 4px; }
.header h1 { font-size: 28px !important; font-weight: 600; margin: 0 0 4px !important; color: var(--ink) !important; }
.header p { color: var(--muted) !important; margin: 0 !important; }

#lang { max-width: 220px; margin-inline-start: auto; }
#nav, #lang { background: transparent !important; border: none !important; box-shadow: none !important; }
#nav .wrap, #lang .wrap { gap: 6px !important; }
#nav label, #lang label { border: 1px solid var(--line) !important; border-radius: 999px !important;
  background: #fff !important; padding: 6px 14px !important; cursor: pointer; }
#nav label.selected, #lang label.selected { background: var(--accent) !important; border-color: var(--accent) !important; }
#nav label.selected span, #lang label.selected span { color: #fff !important; }
#nav input[type=radio], #lang input[type=radio] { display: none; }

.card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px 20px;
  display:flex; flex-direction:column; gap:12px; text-align:start; }
.card.empty { color: var(--muted); align-items: center; padding: 28px; }
.card h4 { margin:0 !important; font-size:15px; font-weight:600; }
.stat { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.stat .label { color: var(--muted); font-size: 14px; }
.stat .value { font-size: 17px; font-weight: 600; }
.pill { padding:4px 14px; border-radius:999px; font-weight:600; font-size:15px; }
.pill.pos { background:var(--pos-soft); color:var(--pos); } .pill.neg { background:var(--neg-soft); color:var(--neg); }
.pill.muted { background:var(--bg); color:var(--muted); }
.note { color:var(--muted); font-size:13px; border-top:1px solid var(--line); padding-top:10px; }

.summary { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:6px; }
.bar-row { display:grid; grid-template-columns:130px 1fr 36px; align-items:center; gap:10px; font-size:14px; }
.bar { height:8px; background:var(--bg); border-radius:99px; overflow:hidden; }
.fill { display:block; height:100%; border-radius:99px; background:var(--muted); }
.fill.pos { background:var(--pos); } .fill.neg { background:var(--neg); } .fill.accent { background:var(--accent); }
.bar-count { color: var(--muted); text-align: end; }
.muted-text { color: var(--muted) !important; font-size: 13px; }

.table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:#fff; margin-top:12px; }
table.t { width:100%; border-collapse:collapse; font-size:14px; }
table.t th { background:var(--bg); color:var(--muted); font-weight:500; text-align:start; padding:10px 14px; }
table.t td { border-top:1px solid var(--line); padding:10px 14px; text-align:start; vertical-align:top; }

.clusters { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.cluster .count { background:var(--bg); color:var(--muted); border-radius:999px; padding:2px 10px; font-size:13px; }
.keywords { color: var(--accent); font-size: 13px; text-align: start; }
.quotes { margin:0; padding-inline-start:18px; color:var(--muted); font-size:13px; line-height:1.7; }
.about h3 { font-size:16px; margin:18px 0 10px; } .about ul { color: var(--ink); line-height: 1.9; }
.ex-label { color: var(--muted); font-size: 13px; }
@media (max-width: 700px) { .summary, .clusters { grid-template-columns: 1fr; } }
"""

theme = gr.themes.Base(primary_hue="teal", neutral_hue="slate").set(
    body_background_fill="#f7f8fa", body_background_fill_dark="#f7f8fa",
    body_text_color="#1f2933", body_text_color_dark="#1f2933",
    block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
    block_border_color="#e6e9ee", block_border_color_dark="#e6e9ee",
    input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
    input_border_color="#e6e9ee", input_border_color_dark="#e6e9ee",
    button_primary_background_fill="#0f766e", button_primary_background_fill_dark="#0f766e",
    button_primary_background_fill_hover="#115e59", button_primary_background_fill_hover_dark="#115e59",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#ffffff",
    button_secondary_text_color="#1f2933", button_secondary_text_color_dark="#1f2933",
)
gradio_major = int(gr.__version__.split(".")[0])
style = {"theme": theme, "css": CSS}
LANGS = [("العربية", "ar"), ("English", "en")]

with gr.Blocks(title="صوت العميل · Voice of the Customer", **(style if gradio_major < 6 else {})) as demo:
    with gr.Row(equal_height=True):
        header = gr.HTML(header_html("ar"))
        lang = gr.Radio(LANGS, value="ar", show_label=False, container=False, elem_id="lang")
    nav = gr.Radio(T["ar"]["nav"], value="one", show_label=False, container=False, elem_id="nav")

    with gr.Column(visible=True) as page_one:
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                review = gr.Textbox(show_label=False, lines=4, placeholder=T["ar"]["review_ph"])
                analyze_btn = gr.Button(T["ar"]["analyze"], variant="primary")
                ex_label = gr.HTML(f'<span class="ex-label">{T["ar"]["examples"]}</span>')
                with gr.Row():
                    ex_buttons = [gr.Button(e, size="sm", variant="secondary") for e in EXAMPLES]
            with gr.Column(scale=2):
                result = gr.HTML(message(T["ar"]["placeholder"], "ar"))

    with gr.Column(visible=False) as page_batch:
        batch_in = gr.Textbox(show_label=False, lines=6, placeholder=T["ar"]["batch_ph"])
        batch_btn = gr.Button(T["ar"]["analyze_batch"], variant="primary")
        batch_out = gr.HTML()

    with gr.Column(visible=False) as page_explore:
        explore_intro = gr.HTML(f'<p class="muted-text" dir="rtl">{T["ar"]["explore_intro"]}</p>')
        clusters_btn = gr.Button(T["ar"]["show_clusters"], variant="primary")
        clusters_out = gr.HTML()
        with gr.Row(equal_height=True):
            query = gr.Textbox(show_label=False, scale=4, placeholder=T["ar"]["search_ph"])
            search_btn = gr.Button(T["ar"]["search"], variant="primary", scale=1)
        search_out = gr.HTML()

    with gr.Column(visible=False) as page_about:
        about = gr.HTML(f'<div class="about" dir="rtl">{T["ar"]["about"]}</div>')

    pages = {"one": page_one, "batch": page_batch, "explore": page_explore, "about": page_about}

    def show_page(key):
        return [gr.Column(visible=(k == key)) for k in pages]

    nav.change(show_page, inputs=[nav], outputs=list(pages.values()))

    def switch_language(code, current_page):
        t = T[code]
        return [
            gr.HTML(header_html(code)),
            gr.Radio(choices=t["nav"], value=current_page),
            gr.Textbox(placeholder=t["review_ph"]),
            gr.Button(value=t["analyze"]),
            gr.HTML(f'<span class="ex-label">{t["examples"]}</span>'),
            gr.HTML(message(t["placeholder"], code)),
            gr.Textbox(placeholder=t["batch_ph"]),
            gr.Button(value=t["analyze_batch"]),
            gr.HTML(""),
            gr.HTML(f'<p class="muted-text" dir="{t["dir"]}">{t["explore_intro"]}</p>'),
            gr.Button(value=t["show_clusters"]),
            gr.HTML(""),
            gr.Textbox(placeholder=t["search_ph"]),
            gr.Button(value=t["search"]),
            gr.HTML(""),
            gr.HTML(f'<div class="about" dir="{t["dir"]}">{t["about"]}</div>'),
        ]

    lang.change(switch_language, inputs=[lang, nav], outputs=[
        header, nav, review, analyze_btn, ex_label, result, batch_in, batch_btn, batch_out,
        explore_intro, clusters_btn, clusters_out, query, search_btn, search_out, about])

    analyze_btn.click(result_card, inputs=[review, lang], outputs=[result])
    review.submit(result_card, inputs=[review, lang], outputs=[result])
    for button, text in zip(ex_buttons, EXAMPLES):
        button.click(lambda code, text=text: (text, result_card(text, code)),
                     inputs=[lang], outputs=[review, result])
    batch_btn.click(batch_ui, inputs=[batch_in, lang], outputs=[batch_out])
    clusters_btn.click(clusters_ui, inputs=[lang], outputs=[clusters_out])
    search_btn.click(search_ui, inputs=[query, lang], outputs=[search_out])
    query.submit(search_ui, inputs=[query, lang], outputs=[search_out])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", share=False, inbrowser=True,
                        **(style if gradio_major >= 6 else {}))
