from pathlib import Path
import json
import pandas as pd
import app

root=Path(__file__).resolve().parent
reports=root/'reports';reports.mkdir(exist_ok=True)
results=[]
def check(name,fn):
    try: fn(); results.append({'check':name,'passed':True});print('PASS:',name)
    except Exception as e: results.append({'check':name,'passed':False,'detail':str(e)});print('FAIL:',name,str(e))
def empty():
    t,s,n=app.analyze_one(' ');assert t=='—' and s=='—' and n

def long():
    t,s,n=app.analyze_one('ا'*3001);assert t=='—' and s=='—'
def arabic():
    assert len(app.analyze_one('التطبيق خيالي'))==3

def english():
    assert app.en_v2 is not None,'Run upgrade_models.py first'
    topic,sentiment,note=app.analyze_one('The app crashes every time I log in')
    assert 'تجريبي' in topic and 'تجريبي' in sentiment

def batch():
    df,summary=app.analyze_batch('تطبيق مفيد\nGreat app')
    assert len(df)==2 and 'ملاحظات' in df.columns and summary

def exploration():
    a,b=app.load_exploration();assert len(a)>0 and len(b)>0

def search():
    a,b=app.search_reviews('The app crashes after the update');assert len(a)==5 and b
for name,fn in [('Empty input clears output',empty),('Long input handled',long),('Arabic inference',arabic),('English experimental inference',english),('Mixed batch',batch),('Cluster reports',exploration),('Semantic search',search)]:check(name,fn)
examples=['التطبيق خيالي','مشكور والله ما قصرتو','ليس جميل نيمو أفضل منه','ما اعجبتني الواجهه بس التطبيق مفيد','The app is amazing','The app crashes every time I open it','Please add dark mode']
rows=[]
for text in examples:
    try:
        a,b,c=app.analyze_one(text);rows.append({'review':text,'topic_prediction':a,'sentiment_prediction':b,'notes':c})
    except Exception as e:rows.append({'review':text,'topic_prediction':'ERROR','sentiment_prediction':'ERROR','notes':str(e)})
pd.DataFrame(rows).to_csv(reports/'upgrade_ui_examples.csv',index=False,encoding='utf-8-sig')
(reports/'upgrade_app_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS confirms execution, not semantic accuracy. Inspect upgrade_ui_examples.csv.')
if not all(x['passed'] for x in results):raise SystemExit(1)
