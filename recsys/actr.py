"""One fixed ACT-R/PISA-inspired activation baseline, not PISA reproduction."""
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from .data import OFFSET, load_sequences
from .diagnostics import rank_scores, slice_metrics, paired_bootstrap
from .metrics import ranking_metrics
from .repeat_gate import interpretable
from .pilot import atomic_json
ROOT=Path(__file__).resolve().parents[1]
def activation(tokens,times,cutoff,recent=None):
    tokens=np.asarray(tokens); times=np.asarray(times)
    if len(tokens)!=len(times): raise ValueError('alignment')
    if np.any(times>=cutoff): raise ValueError('strict-past violation')
    if recent is not None: tokens,times=tokens[-recent:],times[-recent:]
    weights=np.maximum(cutoff-times,1).astype(np.float64)**(-.5)
    result={}
    for token,weight in zip(tokens,weights):
        if token>=OFFSET:
            item=int(token)-OFFSET; result[item]=result.get(item,0)+float(weight)
    return result

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()  # no options; makes --help safe
    start=time.perf_counter(); out=ROOT/'artifacts/actr-20261021-01';out.mkdir(parents=True,exist_ok=False)
    plan={'alpha':.5,'minimum_age_seconds':1,'variants':['recent128','fullprefix'],'source':'https://arxiv.org/abs/2408.16578',
          'scope':'selection only; fixed activation baseline, not full PISA; no training/grid; same truth/catalog; ties raw item ID; train-popularity fallback'}
    atomic_json(out/'plan.json',plan)
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in ['recsys/actr.py','recsys/data.py','recsys/metrics.py','recsys/diagnostics.py','recsys/repeat_gate.py','research/protocol/case2-v1.json']: z.write(ROOT/f,f)
    p=json.loads((ROOT/'research/protocol/case2-v1.json').read_text()); cut=p['time']['query_cutoffs']['selection_validation']; lo=cut+p['time']['gap_seconds']
    data,_,_,_,_=load_sequences(ROOT/p['data']['sample_path'],p['time']['train_end_exclusive']);truth=data.truth(lo,lo+p['time']['horizon_seconds']);ids,*_=data.queries(cut)
    full,recent={},{}; rankings={'ACTR_recent128':[],'ACTR_fullprefix':[]}; by={u['uid']:u for u in data.users}
    for uid in ids:
        u=by[uid];end=int(np.searchsorted(u['times'],cut));tok=u['tokens'][:end];times=u['times'][:end]
        full[uid]=set(map(int,u['items'][:end]));recent[uid]=set(map(int,u['items'][max(0,end-128):end]))
        for n,limit in [('ACTR_recent128',128),('ACTR_fullprefix',None)]:
            scores=activation(tok,times,cut,limit)
            order=sorted(scores,key=lambda i:(-scores[i],int(data.catalog[i])))[:10]; used=set(order)
            order.extend(i for i in range(10) if i not in used)
            rankings[n].append([int(data.catalog[i]) for i in order[:10]])
    cached=json.loads((ROOT/'artifacts/hybrid-20261021-01/rankings.json').read_text())
    if cached['uids']!=ids: raise ValueError('query mismatch')
    for n in ['PersonalFrequency128_original','PersonalFrequency128_recency_tiebreak','Memory_full_weight_0.5']:rankings[n]=cached['rankings'][n]
    values,rows={},{}
    for n,r in rankings.items():
        overall=ranking_metrics(ids,r,truth)[0];overall.update(interpretable(ids,r,truth));slices,rows[n]=slice_metrics(ids,r,truth,full,recent,data.catalog);values[n]={'overall':overall,**slices}
    comparisons={n+'_minus_'+b:paired_bootstrap(rows[n],rows[b]) for n in ['ACTR_recent128','ACTR_fullprefix'] for b in ['PersonalFrequency128_recency_tiebreak','Memory_full_weight_0.5']}
    report={'plan':plan,'models':values,'comparisons':comparisons,'elapsed_seconds':time.perf_counter()-start,'confirmation_scored':False,'final_test_scored':False}
    atomic_json(out/'report.json',report);atomic_json(out/'rankings.json',{'uids':ids,'rankings':rankings});print(json.dumps(report),flush=True)
if __name__=='__main__':main()
