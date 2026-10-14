from pathlib import Path
import json
import pandas as pd
import app

root = Path(__file__).resolve().parent
reports = root / "reports"
reports.mkdir(exist_ok=True)
results = []


def check(name, fn):
    try:
        fn()
        results.append({"check": name, "passed": True})
        print("PASS:", name)
    except Exception as e:
        results.append({"check": name, "passed": False, "detail": str(e)})
        print("FAIL:", name, str(e))


def empty():
    t, s, n = app.analyze_one(" ")
    assert t == "—" and s == "—" and n


def long():
    t, s, n = app.analyze_one("ا" * 3001)
    assert t == "—" and s == "—"


def arabic():
    t, s, n = app.analyze_one("التطبيق خيالي")
    assert t in app.TOPICS.values() and s in app.SENTIMENTS.values()


def english():
    t, s, n = app.analyze_one("The app crashes every time I log in")
    assert t == "—" and s == "—" and "العربية" in n


def batch():
    df, summary = app.analyze_batch("تطبيق مفيد\nGreat app")
    assert len(df) == 2 and "ملاحظات" in df.columns and summary


def english_ui():
    t, s, n = app.analyze_one("تطبيق مفيد", "en")
    assert t in app.TOPIC_NAMES["en"].values() and s in app.SENTIMENT_NAMES["en"].values()
    assert "Review type" in app.result_card("تطبيق مفيد", "en")
    df, summary = app.analyze_batch("تطبيق مفيد\nGreat app", "en")
    assert len(df) == 2 and "Notes" in df.columns


def views():
    for code in ["ar", "en"]:
        assert "card" in app.result_card("تطبيق مفيد", code)
        assert "<table" in app.batch_ui("تطبيق مفيد\nGreat app", code)
        assert "cluster" in app.clusters_ui(code)


def search():
    a, b = app.search_reviews("The app crashes after the update")
    assert len(a) == 5 and b
    assert "<table" in app.search_ui("The app crashes after the update", "en")


for name, fn in [
    ("Empty input", empty), ("Long input", long), ("Arabic inference", arabic),
    ("English input gets Arabic-only message", english), ("Mixed batch", batch), ("English interface", english_ui),
    ("Result views (both languages)", views), ("Semantic search", search),
]:
    check(name, fn)

examples = ["التطبيق خيالي", "مشكور والله ما قصرتو", "ليس جميل نيمو أفضل منه",
            "ما اعجبتني الواجهه بس التطبيق مفيد", "اكل لذيذ", "لذيذ الاكل", "The app is amazing"]
rows = []
for text in examples:
    a, b, c = app.analyze_one(text)
    rows.append({"review": text, "topic_prediction": a, "sentiment_prediction": b, "notes": c})
pd.DataFrame(rows).to_csv(reports / "upgrade_ui_examples.csv", index=False, encoding="utf-8-sig")
(reports / "upgrade_app_checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("PASS confirms execution, not accuracy. See reports/upgrade_ui_examples.csv.")
if not all(x["passed"] for x in results):
    raise SystemExit(1)
