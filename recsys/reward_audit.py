"""Paired train-only RM fit and full logged selection-event calibration audit."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .data import OFFSET, load_sequences, pack_contexts
from .models import RewardModel, state_hash
from .pilot import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def regression(predictions, labels, uids):
    predictions, labels=np.asarray(predictions),np.asarray(labels)
    errors=predictions-labels
    groups={}
    for uid, error in zip(uids, errors):
        groups.setdefault(int(uid),[]).append(abs(float(error)))
    return {'events':len(labels),'users':len(groups),'MAE_event':float(np.abs(errors).mean()),
            'RMSE_event':float(np.sqrt(np.square(errors).mean())),
            'mean_prediction_bias':float(errors.mean()),
            'MAE_macro_users':float(np.mean([np.mean(x) for x in groups.values()]))}


def calibration(predictions, labels):
    p,y=np.asarray(predictions),np.asarray(labels)
    bins=[]
    for i in range(10):
        take=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        if take.any():
            bins.append({'lo':i/10,'hi':(i+1)/10,'n':int(take.sum()),
                         'mean_prediction':float(p[take].mean()), 'mean_observed_response':float(y[take].mean()),
                         'absolute_calibration_gap':float(abs(p[take].mean()-y[take].mean()))})
    return {'bins':bins,'weighted_absolute_gap':sum(b['n']*b['absolute_calibration_gap'] for b in bins)/len(y),
            'note':'Calibration of conditional expected continuous clipped response, not click/acceptance ECE.'}


def observed_records(data, lo, hi):
    """Every eligible logged event exactly once; entire tie excluded from history."""
    records=[]
    raw_events=0
    for u in data.users:
        times=u['times'][u['starts']]
        groups=np.flatnonzero((times>=lo)&(times<hi))
        if not len(groups): continue
        past=set(map(int,u['items'][:int(u['starts'][groups[0]])]))
        for g in groups:
            first,last=int(u['starts'][g]),int(u['ends'][g])
            recent=set(map(int,u['items'][max(0,first-data.context_size):first]))
            raw_events+=last-first
            for event in range(first,last):
                if first and u['tokens'][event]>=OFFSET:
                    item=int(u['items'][event])
                    kind='recent128' if item in recent else 'older_only' if item in past else 'never_seen'
                    records.append((u['uid'],event,first,int(u['tokens'][event])-OFFSET,float(u['ratios'][event]),kind))
            past.update(map(int,u['items'][first:last]))
    return records,raw_events


def main():
    out=ROOT/'artifacts/reward-audit-20261021-01';out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    proto_path=ROOT/'research/protocol/case2-v1.json';proto=json.loads(proto_path.read_text())
    phash=hashlib.sha256(proto_path.read_bytes()).hexdigest()
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':phash,
          'fit':'Two new RMs, identical initial tensors, shared 5000 train batches of64; history on/off; AdamW lr.001 wd.01, fp32MSE',
          'seed':20261028,'training_steps':5000,'batch':64,'checkpoint_selection':'fixed last step, no validation earlystop',
          'target':'clip played_ratio at100 /100 including zero; logged events only',
          'evaluation':'ALL eligible in-catalog observed events in selection label day once; history strictly before entire event group',
          'controls':['constant train mean','item train mean','paired neural item-only','old calibration RM'],
          'diagnostics':['same fitted history RM with zero history branch','history shuffle permutation among event records'],
          'slices':['recent128/older_only/never_seen','train popularity quartile','train median length <=120,120-300,>300 capped600'],
          'calibration':'ten fixed prediction bins for conditional mean continuous response; no calibration fitting on selection',
          'limits':['No unobserved candidates labeled dislike','No causal acceptance/retention inference without impressions/propensities',
                    'Permutation may sometimes retain the same user and is OOD; no-history diagnostic differs from separately fitted item-only',
                    'One initialization; confirmation/final untouched']}
    atomic_json(out/'plan.json',plan);atomic_json(out/'status.json',{'status':'running','stage':'loading'})
    with zipfile.ZipFile(out/'code-snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['recsys','tests','research/protocol']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(4);torch.manual_seed(plan['seed'])
        data,features,item_means,global_mean,support=load_sequences(ROOT/proto['data']['sample_path'],proto['time']['train_end_exclusive'])
        by_uid={u['uid']:u for u in data.users}
        models={'history':RewardModel(len(data.catalog),features).cuda()}
        models['item_only']=copy.deepcopy(models['history']);models['item_only'].use_history=False
        initial_identical=state_hash(models['history'])==state_hash(models['item_only'])
        if not initial_identical:raise ValueError('RM initial tensors differ')
        opts={n:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01) for n,m in models.items()}
        rng=np.random.default_rng(plan['seed']);trace=[]
        train_start=time.perf_counter()
        for step in range(1,5001):
            batch=data.sample(rng,64);x,m,a,y=[torch.as_tensor(v,device='cuda') for v in batch[:4]]
            losses={}
            for name,model in models.items():
                loss=F.mse_loss(model(x,m,a),y)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite RM loss')
                opts[name].zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opts[name].step();losses[name]=loss.item()
            if step%500==0:
                trace.append({'step':step,'last_batch_MSE':losses})
                atomic_json(out/'status.json',{'status':'running','stage':'training_paired_RMs','step':step,'elapsed_seconds':time.perf_counter()-start})
        torch.cuda.synchronize();train_seconds=time.perf_counter()-train_start
        for n,model in models.items():
            model.eval();model.requires_grad_(False)
            torch.save({'model':model.state_dict(),'width':64,'use_history':model.use_history,
                        'catalog':torch.from_numpy(data.catalog),'protocol_sha256':phash,'steps':5000,
                        'optimizer':opts[n].state_dict(),'sampler_rng':rng.bit_generator.state},out/(n+'-last.pt'))
        del opts,loss,x,m,a,y
        old_path=ROOT/'artifacts/calibration-20261021-01/reward-model.pt'
        old_protocol=json.loads((old_path.parent/'report.json').read_text())['protocol_sha256']
        if old_protocol!=phash:raise ValueError('Old RM protocol mismatch')
        ck=torch.load(old_path,map_location='cpu',weights_only=True)
        if not np.array_equal(ck['catalog'].numpy(),data.catalog):raise ValueError('Old RM catalog mismatch')
        old=RewardModel(len(data.catalog),features,width=ck['width']).cuda()
        old.load_state_dict(ck['model']);old.eval();old.requires_grad_(False);del ck
        cut=proto['time']['query_cutoffs']['selection_validation'];lo=cut+proto['time']['gap_seconds'];hi=lo+proto['time']['horizon_seconds']
        records,raw_events=observed_records(data,lo,hi)
        if not records:raise ValueError('No evaluation events')
        uid=np.array([r[0] for r in records]);action=np.array([r[3] for r in records]);label=np.array([r[4] for r in records])
        permutation=np.random.default_rng(20261029).permutation(len(records))
        pred={n:[] for n in ['history_RM','item_only_RM','old_history_RM','history_RM_zeroed','history_RM_shuffled']}

        def contexts(rows):
            return pack_contexts([data.context(by_uid[r[0]],r[2]) for r in rows],128)

        with torch.inference_mode():
            for j in range(0,len(records),512):
                subset=records[j:j+512];xx,mm=contexts(subset)
                sx,sm=contexts([records[int(i)] for i in permutation[j:j+512]])
                x,m,a=[torch.as_tensor(v,device='cuda') for v in [xx,mm,action[j:j+512]]]
                sx,sm=[torch.as_tensor(v,device='cuda') for v in [sx,sm]]
                pred['history_RM'].extend(models['history'](x,m,a).cpu().tolist())
                pred['item_only_RM'].extend(models['item_only'](x,m,a).cpu().tolist())
                pred['old_history_RM'].extend(old(x,m,a).cpu().tolist())
                pred['history_RM_shuffled'].extend(models['history'](sx,sm,a).cpu().tolist())
                models['history'].use_history=False
                pred['history_RM_zeroed'].extend(models['history'](x,m,a).cpu().tolist())
                models['history'].use_history=True
        pred={n:np.asarray(v) for n,v in pred.items()}
        pred['constant_train_mean']=np.full(len(label),global_mean)
        pred['item_train_mean']=item_means[action]
        kinds=np.array([r[5] for r in records]);lengths=features[action,0]*600
        # features are monotonic log train counts, so train-catalog quartiles are well defined.
        quartiles=np.quantile(features[:,1],[.25,.5,.75]);popbin=np.searchsorted(quartiles,features[action,1],side='right')
        slices={k:kinds==k for k in ['recent128','older_only','never_seen']}
        slices.update({f'popularity_quartile_{k+1}':popbin==k for k in range(4)})
        slices.update({'length_le120':lengths<=120,'length_120_300':(lengths>120)&(lengths<=300),'length_gt300':lengths>300})
        values={}
        for name,p in pred.items():
            values[name]={'overall':regression(p,label,uid),'calibration':calibration(p,label),
                          'slices':{key:regression(p[take],label[take],uid[take]) for key,take in slices.items() if take.any()}}
        # Paired users, keeping all within-user events together; macro MAE difference.
        unique=np.unique(uid);per_user={name:np.array([np.abs(p[uid==u]-label[uid==u]).mean() for u in unique]) for name,p in pred.items()}
        bootstrap_rng=np.random.default_rng(42);indices=bootstrap_rng.integers(0,len(unique),size=(2000,len(unique)))
        comparisons={}
        for other in ['item_only_RM','item_train_mean','old_history_RM','history_RM_zeroed','history_RM_shuffled']:
            delta=per_user['history_RM']-per_user[other]
            comparisons['history_minus_'+other]={'macro_MAE_difference_lower_better':float(delta.mean()),
                'paired_user_bootstrap95':np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
                'caution':'Fixed selection window, no multiplicity correction, no training-seed uncertainty'}
        unique_pairs=len(set((r[0],r[1]) for r in records));assert unique_pairs==len(records)
        report={'plan':plan,'train_support':support,'raw_selection_events':raw_events,'evaluated_unique_events':len(records),
                'evaluated_users':len(unique),'event_catalog_coverage':len(records)/raw_events,
                'zero_response_event_fraction':float((label==0).mean()),'models':values,'comparisons':comparisons,
                'paired_initial_tensors_identical':initial_identical,'training_seconds_paired':train_seconds,'elapsed_seconds':time.perf_counter()-start,
                'confirmation_scored':False,'final_test_scored':False,'old_RM_checkpoint_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest()}
        atomic_json(out/'report.json',report);atomic_json(out/'trace.json',trace)
        atomic_json(out/'status.json',{'status':'completed','elapsed_seconds':report['elapsed_seconds']})
        print(json.dumps({n:v['overall'] for n,v in values.items()},indent=2),flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',{'status':'failed','error':repr(exc)});raise


if __name__=='__main__':main()
