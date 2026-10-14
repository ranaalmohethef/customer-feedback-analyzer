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
    assert app.en_v2 is not None, "Run upgrade_models.py first"
    t, s, n = app.analyze_one("The app crashes every time I log in")
    assert t == "—" and "تجريبي" in s


def batch():
    df, summary = app.analyze_batch("تطبيق مفيد\nGreat app")
    assert len(df) == 2 and "ملاحظات" in df.columns and summary


def card():
    assert "card" in app.result_card("تطبيق مفيد")


def exploration():
    a, b = app.load_exploration()
    assert len(a) > 0 and len(b) > 0


def search():
    a, b = app.search_reviews("The app crashes after the update")
    assert len(a) == 5 and b


for name, fn in [
    ("Empty input", empty), ("Long input", long), ("Arabic inference", arabic),
    ("English sentiment only", english), ("Mixed batch", batch), ("Result card", card),
    ("Cluster reports", exploration), ("Semantic search", search),
]:
    check(name, fn)

examples = ["التطبيق خيالي", "مشكور والله ما قصرتو", "ليس جميل نيمو أفضل منه",
            "ما اعجبتني الواجهه بس التطبيق مفيد", "The app is amazing",
            "The app crashes every time I open it", "Please add dark mode"]
rows = []
for text in examples:
    a, b, c = app.analyze_one(text)
    rows.append({"review": text, "topic_prediction": a, "sentiment_prediction": b, "notes": c})
pd.DataFrame(rows).to_csv(reports / "upgrade_ui_examples.csv", index=False, encoding="utf-8-sig")
(reports / "upgrade_app_checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("PASS confirms execution, not accuracy. See reports/upgrade_ui_examples.csv.")
if not all(x["passed"] for x in results):
    raise SystemExit(1)
