"""Arabic sentiment v2: train candidates on train, select on validation. Keep original models/reports."""
from pathlib import Path
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODELS = ROOT/'models'
REPORTS = ROOT/'reports'
CACHE = ROOT/'Data'/'embedding_cache'
for p in [MODELS, REPORTS, CACHE]: p.mkdir(parents=True, exist_ok=True)
encoder = SentenceTransformer(str(MODELS/'multilingual_encoder'), device='cpu')
LABELS = ['negative', 'positive']

def encode(texts, name):
    digest = hashlib.sha256(json.dumps([encoder.max_seq_length, texts], ensure_ascii=False).encode()).hexdigest()[:20]
    p = CACHE/f'upgrade_{name}_{digest}.npy'
    if p.exists(): return np.load(p, allow_pickle=False)
    values = encoder.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    np.save(p, values)
    return values

def score(y, pred):
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0))}

def train_task(train, val, test, prefix, baseline=None, interpretation=''):
    for a,b in [(train,val),(train,test),(val,test)]:
        if set(a.text_key)&set(b.text_key): raise ValueError('Split overlap')
    for frame in [train,val,test]:
        if frame.empty or frame[['review','sentiment','text_key']].isna().any().any(): raise ValueError('Missing data')
    if set(train.sentiment)!=set(LABELS): raise ValueError('Both classes required')
    texts=train.review.astype(str).tolist(); vt=val.review.astype(str).tolist()
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True)
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=50000,sublinear_tf=True)
    w=word.fit_transform(texts); vw=word.transform(vt)
    ch=char.fit_transform(texts); vch=char.transform(vt)
    e=encode(texts,prefix+'_train'); ve=encode(vt,prefix+'_val')
    rows=[]; best=None; best_score=-1
    if baseline is not None:
        metrics=score(val.sentiment,baseline.predict(vt)); best_score=metrics['macro_f1']
        best={'kind':'baseline','pipeline':baseline}
        rows.append({'candidate':'original_baseline',**metrics})
    # Fixed small search. No keyword exceptions, no test-based model selection.
    for ew in [0.0,1.0]:
        x=hstack([w,ch*0.5,csr_matrix(e*ew)],format='csr')
        vx=hstack([vw,vch*0.5,csr_matrix(ve*ew)],format='csr')
        for c in [1.0,10.0]:
            clf=LogisticRegression(C=c,max_iter=2000,random_state=42)
            start=time.perf_counter(); clf.fit(x,train.sentiment)
            metrics=score(val.sentiment,clf.predict(vx))
            rows.append({'candidate':f'word_char_embedding_{ew}_C_{c}',**metrics,'fit_seconds':time.perf_counter()-start})
            print(prefix, rows[-1],flush=True)
            if metrics['macro_f1']>best_score:
                best_score=metrics['macro_f1']; best={'kind':'hybrid','word':word,'char':char,'embedding_weight':ew,'classifier':clf}
    best['validation_macro_f1']=best_score
    best['interpretation']=interpretation
    joblib.dump(best, MODELS/f'{prefix}_v2.joblib')
    pd.DataFrame(rows).to_csv(REPORTS/f'{prefix}_v2_validation.csv',index=False,encoding='utf-8-sig')
    # Arabic old test is reused descriptively after error inspection, not a fresh independent benchmark.
    tt=test.review.astype(str).tolist()
    if best['kind']=='baseline': pred=best['pipeline'].predict(tt)
    else:
        te=encode(tt,prefix+'_test')
        tx=hstack([best['word'].transform(tt),best['char'].transform(tt)*0.5,csr_matrix(te*best['embedding_weight'])],format='csr')
        pred=best['classifier'].predict(tx)
    report={'selection_split':'validation','train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'selected_kind':best['kind'],'validation_macro_f1':best_score,'test_metrics':score(test.sentiment,pred),'interpretation':interpretation}
    if baseline is not None: report['original_baseline_same_test']=score(test.sentiment,baseline.predict(tt))
    (REPORTS/f'{prefix}_v2_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    out=test.copy();out['prediction']=pred
    out.to_csv(REPORTS/f'{prefix}_v2_predictions.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    splits=[pd.read_csv(ROOT/'Data'/'processed'/f'sentiment_{s}.csv') for s in ['train','validation','test']]
    train_task(*splits,prefix='sentiment_ar',baseline=joblib.load(MODELS/'sentiment_classifier.joblib'),interpretation='Arabic v2 selected on existing validation. Old test has already been inspected; reuse is descriptive, not a fresh independent evaluation. Neutral/mixed unsupported.')
    print('DONE: Arabic sentiment v2 selection saved. Original files preserved.')
