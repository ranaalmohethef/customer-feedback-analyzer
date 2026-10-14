from pathlib import Path

import re

import time
import hashlib
import json
from functools import lru_cache

import numpy as np

import gradio as gr

import joblib

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

SENTIMENTS = {

    "positive": "إيجابي",

    "negative": "سلبي",

}

required_models = ["hybrid_classifier.joblib", "sentiment_classifier.joblib", "multilingual_encoder"]
missing = [name for name in required_models if not (MODELS / name).exists()]
if missing:
    raise FileNotFoundError("Missing local models: " + ", ".join(missing))

print("Loading saved models...")

hybrid = joblib.load(MODELS / "hybrid_classifier.joblib")

sentiment = joblib.load(MODELS / "sentiment_classifier.joblib")

encoder = SentenceTransformer(

    str(MODELS / "multilingual_encoder"),

    device="cpu",

)

ARABIC_LETTERS = re.compile(r"[\u0621-\u063A\u0641-\u064A]")

def load_optional(name):
    path = MODELS / name
    return joblib.load(path) if path.exists() else None

ar_v2 = load_optional("sentiment_ar_v2.joblib")
en_v2 = load_optional("sentiment_en_v2.joblib")


def language_route(text):
    arabic = len(ARABIC_LETTERS.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if arabic:
        return "ar", "نص مختلط؛ استُخدم مسار العربي." if latin else ""
    if latin:
        return "en", "المسار الإنجليزي تجريبي؛ فحص الحروف لا يتحقق من اللغة."
    return "unsupported", "لم نحدد نصًا عربيًا أو لاتينيًا قابلًا للتحليل."


def sentiment_predict(bundle, texts, vectors):
    if bundle["kind"] == "baseline":
        return bundle["pipeline"].predict(texts)
    features = hstack([bundle["word"].transform(texts),
                      bundle["char"].transform(texts)*0.5,
                      csr_matrix(vectors*bundle["embedding_weight"])], format="csr")
    return bundle["classifier"].predict(features)


@lru_cache(maxsize=1)
def topic_prototypes():
    descriptions = [
        "A user reports a technical bug: the app crashes, freezes, fails to load or cannot log in.",
        "A user requests a new feature, an improvement, or a change to the application.",
        "A user gives a general rating, praise, or criticism of the application.",
        "An unrelated message, a greeting, or a question without a clear app review.",
    ]
    return encoder.encode(descriptions, normalize_embeddings=True, convert_to_numpy=True)


def predict_reviews(texts):
    vectors = encoder.encode(texts, batch_size=16, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
    records=[]
    for i,text in enumerate(texts):
        lang, note=language_route(text)
        if lang == "unsupported":
            records.append({"التعليق":text,"نوع التعليق":"غير مدعوم","المشاعر":"غير متاحة","ملاحظات":note})
            continue
        v=vectors[i:i+1]
        if lang == "ar":
            features=hstack([hybrid["vectorizer"].transform([text]),
                csr_matrix(v*hybrid["embedding_weight"])],format="csr")
            topic=TOPICS[hybrid["classifier"].predict(features)[0]]
            label=sentiment_predict(ar_v2,[text],v)[0] if ar_v2 else sentiment.predict([text])[0]
            feeling=SENTIMENTS[label]
            note += " المشاعر ثنائية؛ التعليق المختلط أو المحايد قد يحتاج مراجعة."
        else:
            # Transparent zero-shot semantic matching; no claim of English topic accuracy.
            index=int(np.argmax(v @ topic_prototypes().T))
            topic=TOPICS[list(TOPICS)[index]] + " (تجريبي)"
            feeling=SENTIMENTS[sentiment_predict(en_v2,[text],v)[0]] + " (تجريبي)" if en_v2 else "غير جاهز: شغّلي upgrade_models.py"
            note += " النوع تشابه مع أوصاف الفئات ولم يُقيّم؛ المشاعر تعلمت من النجوم لا من وسوم نصية بشرية."
        tokens=encoder.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        if len(tokens)>encoder.max_seq_length:
            note += " النص يتجاوز حد Encoder؛ التمثيل الدلالي يستخدم بدايته فقط."
        records.append({"التعليق":text,"نوع التعليق":topic,"المشاعر":feeling,"ملاحظات":note.strip()})
    return pd.DataFrame(records)


def analyze_one(text):
    text=(text or "").strip()
    if not text:
        return "—", "—", "اكتبي تعليقًا أولًا."
    if len(text)>3000:
        return "—", "—", "الحد الأقصى 3000 حرف؛ اختصري النص ثم أعيدي التحليل."
    start=time.perf_counter()
    result=predict_reviews([text]).iloc[0]
    return result["نوع التعليق"],result["المشاعر"],f"{time.perf_counter()-start:.2f} ثانية. " + result["ملاحظات"]


def analyze_batch(text):
    texts=[t.strip() for t in (text or "").splitlines() if t.strip()]
    empty=pd.DataFrame(columns=["التعليق","نوع التعليق","المشاعر","ملاحظات"])
    if not texts:
        return empty,"ضعي كل تعليق في سطر مستقل."
    if len(texts)>100 or any(len(t)>3000 for t in texts):
        return empty,"الحد 100 تعليق، وكل تعليق لا يتجاوز 3000 حرف."
    result=predict_reviews(texts)
    counts=result["المشاعر"].value_counts()
    summary=f"**عدد التعليقات: {len(result)}**\n\n"
    summary+="\n".join(f"- {label}: {count}" for label,count in counts.items())
    summary+="\n\nنتائج متوقعة وليست حقائق مؤكدة. النتائج الإنجليزية التجريبية مفصولة بالاسم عن العربية."
    return result,summary


def active_metrics():
    sections=["### النسخة المستخدمة وحدود التقييم",
        "نوع التعليق العربي: Hybrid الأصلي؛ Accuracy 69.18% وMacro-F1 62.64% على 451 تعليقًا."]
    for prefix,title in [("sentiment_ar","المشاعر العربية"),("sentiment_en","المشاعر الإنجليزية التجريبية")]:
        path=REPORTS/f"{prefix}_v2_metrics.json"
        if path.exists():
            report=json.loads(path.read_text(encoding="utf-8"))
            metric=report["test_metrics"]
            sections.append(f"**{title}**: Validation Macro-F1={report['validation_macro_f1']:.3f}؛ "
                f"الاختبار المسجل ({report['test_rows']}): Accuracy={metric['accuracy']:.3f}، Macro-F1={metric['macro_f1']:.3f}.")
        else:
            sections.append(f"**{title}**: تقرير التحديث غير موجود.")
    sections.extend(["العربي: أعدنا استخدام اختبار سبق فحص أخطائه؛ النتيجة وصفية وليست اختبارًا مستقلًا جديدًا.",
        "الإنجليزي: المقاييس تقيس التوافق مع وسوم مشتقة من النجوم، وليست دقة مثبتة لمشاعر النص. اللغة غير متحققة لكل صف.",
        "نوع التعليق الإنجليزي يعتمد تشابه النص مع أوصاف الفئات؛ لم نقس دقته. لا توجد فئة مشاعر مختلط أو محايد.",
        "إذا لم تُدرّب النسخة الجديدة فالعربي يستخدم نموذج المشاعر القديم: Accuracy 85.95% وMacro-F1 85.92%."])
    return "\n\n".join(sections)


def read_report(name, columns):
    path = REPORTS / name
    if not path.exists():
        raise gr.Error(f"ملف مطلوب غير موجود: reports/{name}. شغّلي explore_english.py أولًا ثم أعيدي ملف الملخص المراجع.")
    frame = pd.read_csv(path).fillna("")
    if not set(columns).issubset(frame.columns):
        raise gr.Error(f"أعمدة غير متوقعة في {name}")
    return frame


def load_exploration():
    summary = read_report("english_cluster_summary.csv", ["cluster", "review_count", "keywords", "suggested_topic_manual", "review_notes"])
    examples = read_report("english_cluster_examples.csv", ["cluster", "example_type", "review"])
    shown = summary[["cluster", "review_count", "suggested_topic_manual", "keywords", "review_notes"]].copy()
    shown.columns = ["المجموعة", "العدد", "الموضوع المقترح", "الكلمات البارزة", "ملاحظات المراجعة"]
    samples = examples[["cluster", "example_type", "review"]].copy()
    samples.columns = ["المجموعة", "اختيار المثال", "التعليق"]
    return shown, samples


@lru_cache(maxsize=1)
def search_index(file_signature):
    # Signature changes invalidate in-memory caching when source changes.
    frame = read_report("english_clustered_reviews.csv", ["review", "cluster"])
    frame = frame[frame["review"].astype(str).str.strip().ne("")].reset_index(drop=True)
    if frame.empty:
        raise gr.Error("ملف التعليقات فارغ.")
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
    if len(vectors) != len(frame):
        raise gr.Error("حجم فهرس البحث لا يطابق البيانات.")
    return frame, vectors


def search_reviews(query):
    query = (query or "").strip()
    if not query:
        raise gr.Error("اكتبي موضوع البحث أولًا، بالعربي أو الإنجليزي.")
    if len(query) > 3000:
        raise gr.Error("الحد الأقصى 3000 حرف.")
    path = REPORTS / "english_clustered_reviews.csv"
    if not path.exists():
        raise gr.Error("ملف english_clustered_reviews.csv غير موجود؛ شغّلي سكربت الاستكشاف أولًا.")
    start = time.perf_counter()
    frame, vectors = search_index((path.stat().st_mtime_ns, path.stat().st_size))
    query_vector = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = vectors @ query_vector
    order = np.argsort(-scores)[:5]
    result = frame.iloc[order][["review", "cluster"]].copy()
    result["similarity"] = np.round(scores[order], 3)
    result.columns = ["تعليق مشابه", "المجموعة", "تشابه جيب التمام"]
    return result, (f"زمن البحث: {time.perf_counter()-start:.2f} ثانية. "
                    "الدرجة للتشابه وليست احتمال صحة؛ النتائج الأقرب قد تظل غير مناسبة. "
                    "البحث عابر اللغات استكشافي ولم يُقيّم بمقياس استرجاع مستقل.")


CSS = """
.gradio-container table,
.gradio-container table thead,
.gradio-container table tbody,
.gradio-container table tr,
.gradio-container table td,
.gradio-container table th,
.gradio-container [role="grid"],
.gradio-container [role="gridcell"],
.gradio-container [role="columnheader"] {
    background: #ffffff !important;
    color: #172b42 !important;
    border-color: #d4dfe8 !important;
}
.gradio-container table th,
.gradio-container [role="columnheader"] {
    background: #e6f2ef !important;
    color: #153e3a !important;
}
.gradio-container table td *,
.gradio-container table th *,
.gradio-container [role="gridcell"] * {
    color: inherit !important;
}
.gradio-container table td {line-height: 1.8 !important;}

.gradio-container {font-size: 15px !important; padding: 20px !important;}
.toast-body, .toast-body *, .toast-wrap, .toast-wrap * {color: #f8fafc !important;}
textarea {font-size: 16px !important;}

#search-input textarea {direction: ltr !important; text-align: left !important;}

.gradio-container {

    max-width: 1120px !important;

    margin: auto !important;

    direction: rtl;

    font-family: Tahoma, Arial, sans-serif !important;

}

.hero {

    background: linear-gradient(120deg, #102b46, #155e63);

    padding: 32px;

    border-radius: 22px;

    margin: 12px 0 24px;

    color: white;

}

.hero h1 {

    color: white !important;

    font-size: 30px !important;

    margin: 8px 0 12px !important;

}

.hero p {

    color: #d7eeee !important;

    font-size: 16px;

    line-height: 1.9;

}

.hero .eyebrow {

    color: #8fe3d5;

    font-size: 13px;

}

.panel {

    border: 1px solid #dce5ed !important;

    border-radius: 18px !important;

    padding: 22px !important;

    background: white !important;

    box-shadow: 0 6px 20px rgba(16, 43, 70, 0.04);

}

textarea, input {

    direction: rtl !important;

    text-align: right !important;

}

textarea {

    line-height: 1.9 !important;

}

#analyze-button, #batch-button {

    border-radius: 12px !important;

    min-height: 48px;

    font-size: 16px !important;

}

.note {

    background: #edf7f5;

    border-right: 4px solid #158577;

    border-radius: 10px;

    padding: 14px 18px;

    color: #244b49;

    line-height: 1.9;

    margin-top: 16px;

}

.footer-note {

    text-align: center;

    color: #64748b;

    padding: 18px;

    font-size: 12px;

}

@media (max-width: 700px) {

    .hero { padding: 22px; }

    .hero h1 { font-size: 24px !important; }

    .panel { padding: 14px !important; }

}

"""

# Set matching light/dark values so the interface stays readable

# even when the browser prefers dark mode.

theme = gr.themes.Soft(

    primary_hue="teal",

    secondary_hue="slate",

    neutral_hue="slate",

).set(

    body_background_fill="#f4f7fb",

    body_background_fill_dark="#f4f7fb",

    body_text_color="#172b42",

    body_text_color_dark="#172b42",

    block_background_fill="#ffffff",

    block_background_fill_dark="#ffffff",

    block_border_color="#dce5ed",

    block_border_color_dark="#dce5ed",

    block_label_text_color="#334155",

    block_label_text_color_dark="#334155",

    block_title_text_color="#172b42",

    block_title_text_color_dark="#172b42",

    input_background_fill="#f8fafc",

    input_background_fill_dark="#f8fafc",

    input_border_color="#dce5ed",

    input_border_color_dark="#dce5ed",

    input_placeholder_color="#64748b",

    input_placeholder_color_dark="#64748b",

    button_primary_background_fill="#117c72",

    button_primary_background_fill_dark="#117c72",

    button_primary_background_fill_hover="#0f625b",

    button_primary_background_fill_hover_dark="#0f625b",

    button_primary_text_color="#ffffff",

    button_primary_text_color_dark="#ffffff",

    button_secondary_background_fill="#e8eef4",

    button_secondary_background_fill_dark="#e8eef4",

    button_secondary_text_color="#172b42",

    button_secondary_text_color_dark="#172b42",

)

# Gradio 6 accepts theme/CSS in launch; older versions in Blocks.

gradio_major = int(gr.__version__.split(".")[0])

style_options = {"theme": theme, "css": CSS}

blocks_options = style_options if gradio_major < 6 else {}

with gr.Blocks(

    title="صوت العميل | محلّل الآراء",

    **blocks_options,

) as demo:

    gr.HTML("""

    <div class="hero">

        <div class="eyebrow">تحليل آراء مستخدمي التطبيقات</div>

        <h1>صوت العميل</h1>

        <p>

            افهم نوع التعليق والمشاعر المتوقعة،

            واستعرض ملخصًا لآراء العملاء في مكان واحد.

        </p>

    </div>

    """)

    with gr.Tab("تحليل تعليق"):

        with gr.Row():

            with gr.Column(scale=3, elem_classes=["panel"]):

                gr.Markdown("### وش يقول العميل؟")

                review = gr.Textbox(

                    label="التعليق بالعربي أو الإنجليزي",

                    placeholder="اكتبي تعليق العميل هنا...",

                    lines=5,

                )

                analyze_button = gr.Button(

                    "تحليل التعليق",

                    variant="primary",

                    elem_id="analyze-button",

                )

                gr.Examples(

                    label="أمثلة للتجربة",

                    examples=[

                        ["التطبيق يقفل كل ما حاولت أسجل الدخول"],

                        ["أتمنى تضيفون الوضع الليلي"],

                        ["تطبيق جميل وسهل الاستخدام"],

                    ],

                    inputs=[review],

                )

            with gr.Column(scale=2, elem_classes=["panel"]):

                gr.Markdown("### نتيجة التحليل")

                topic = gr.Textbox(

                    label="نوع التعليق المتوقع",

                    placeholder="تظهر النتيجة بعد التحليل",

                    interactive=False,

                )

                feeling = gr.Textbox(

                    label="المشاعر المتوقعة",

                    placeholder="إيجابي أو سلبي",

                    interactive=False,

                )

                timing = gr.Textbox(

                    label="زمن المعالجة وملاحظات النتيجة",
                    lines=4,

                    interactive=False,

                )

        gr.HTML("""

        <div class="note">

            <strong>كيف تقرأ النتيجة؟</strong> العربي مُقيّم، والإنجليزي تجريبي وحدوده مذكورة مع النتيجة.

            طلب التحسين قد يكون مصحوبًا بمدح أو نقد.

            النموذج يختار إيجابيًا أو سلبيًا فقط؛

            التعليقات المختلطة والساخرة قد تحتاج مراجعة بشرية.

        </div>

        """)

        analyze_button.click(

            fn=analyze_one,

            inputs=[review],

            outputs=[topic, feeling, timing],

        )

    with gr.Tab("تحليل مجموعة"):

        with gr.Column(elem_classes=["panel"]):

            gr.Markdown(

                "### نظرة على آراء العملاء\n"

                "ضعي كل تعليق في سطر مستقل — حتى 100 تعليق، بالعربي أو الإنجليزي."

            )

            batch_input = gr.Textbox(

                label="التعليقات",

                lines=7,

                placeholder=(

                    "التطبيق يعلق عند الدفع\n"

                    "أتمنى إضافة الوضع الليلي\n"

                    "تطبيق ممتاز وسهل"

                ),

            )

            batch_button = gr.Button(

                "تحليل التعليقات",

                variant="primary",

                elem_id="batch-button",

            )

        batch_summary = gr.Markdown()

        batch_result = gr.Dataframe(

            headers=["التعليق", "نوع التعليق", "المشاعر", "ملاحظات"],

            interactive=False,

            label="التوقعات لكل تعليق",

        )

        batch_button.click(

            fn=analyze_batch,

            inputs=[batch_input],

            outputs=[batch_result, batch_summary],

        )

    with gr.Tab("استكشاف المصدر الثالث"):
        gr.Markdown("### المواضيع المتكررة والبحث عن تعليقات مشابهة\n"
                    "عينة من 3,000 تعليق يغلب عليها الإنجليزي وتتضمن لغات أخرى. "
                    "هذا استكشاف وبحث دلالي؛ لا ينتج تصنيف نوع أو مشاعر إنجليزيًا مُقيّمًا.")
        load_button = gr.Button("عرض المجموعات وأمثلتها", variant="primary")
        summary_table = gr.Dataframe(interactive=False, label="ملخص المجموعات")
        examples_table = gr.Dataframe(interactive=False, label="أمثلة المراجعة")
        load_button.click(load_exploration, inputs=[], outputs=[summary_table, examples_table])
        query = gr.Textbox(label="ابحثي عن موضوع بالعربي أو الإنجليزي",
                           placeholder="The app crashes after the update", elem_id="search-input")
        search_button = gr.Button("البحث عن تعليقات مشابهة", variant="primary")
        search_table = gr.Dataframe(interactive=False, label="أقرب خمسة تعليقات")
        search_note = gr.Textbox(interactive=False, label="ملاحظة البحث")
        search_button.click(search_reviews, inputs=[query], outputs=[search_table, search_note])
        gr.Markdown("أول بحث يجهّز الفهرس وقد يستغرق وقتًا أطول. أرقام المجموعات ليست ترتيبًا للجودة، "
                    "وتسمياتها تفسيرات أولية مبنية على أمثلة محدودة.")

    with gr.Tab("الأداء وحدود الاستخدام"):
        with gr.Column(elem_classes=["panel"]):
            gr.Markdown(active_metrics())

    gr.HTML("""

    <div class="footer-note">

        صوت العميل · مشروع تطبيقي في معالجة اللغة الطبيعية

    </div>

    """)

if __name__ == "__main__":

    launch_options = style_options if gradio_major >= 6 else {}

    demo.queue().launch(

        server_name="127.0.0.1",

        share=False,

        inbrowser=True,

        **launch_options,

    )
