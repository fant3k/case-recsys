"""One locked future-period confirmation; no fitting, selection or final-test scoring."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .data import OFFSET, load_sequences, personal_frequency
from .diagnostics import paired_bootstrap, slice_metrics
from .hybrid import history_distribution, probability_topk
from .metrics import ranking_metrics
from .models import Policy, RewardModel, state_hash
from .pilot import atomic_json
from .repeat_gate import interpretable
from .rl_comparison import bootstrap_mean as _bootstrap_mean, reward_summary

ROOT=Path(__file__).resolve().parents[1]


def bootstrap_mean(values, seed=42):
    result=_bootstrap_mean(values, seed)
    result['caution']='Locked single-candidate confirmation period; paired users, no training-seed/time-generalization uncertainty'
    return result


def confirmation_decision(reward_delta, relevance_margin):
    reward_pass=reward_delta['paired_user_bootstrap95'][0]>0
    relevance_pass=relevance_margin['paired_user_bootstrap95'][0]>=0
    return {'predicted_reward_increase_supported':reward_pass,
            'relevance_within_5percent_supported':relevance_pass,
            'joint_confirmation_pass':reward_pass and relevance_pass,
            'meaning':'Locked offline proxy and relevance hypotheses only; no causal user benefit claim'}


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()  # no options; makes --help safe instead of starting a run
    out=ROOT/'artifacts/confirmation-20261021-01';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    protocol_path=ROOT/'research/protocol/case2-v1.json';proto=json.loads(protocol_path.read_text())
    phash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    checkpoints={
        'reference':'artifacts/day-objective-20261021-01/next_item_plus_day/best.pt',
        'candidate':'artifacts/rl-comparison-20261021-01/RL_capped_seconds_beta0.3/last.pt',
        'continued_SFT':'artifacts/rl-comparison-20261021-01/continued_SFT/last.pt',
        'primary_RM':'artifacts/reward-audit-20261021-01/history-last.pt',
        'alternate_RM':'artifacts/calibration-20261021-01/reward-model.pt'}
    cutoff=proto['time']['query_cutoffs']['confirmation_validation']
    lo=cutoff+proto['time']['gap_seconds'];hi=lo+proto['time']['horizon_seconds']
    plan={'locked_utc':datetime.now(timezone.utc).isoformat(),'kind':'one_time_confirmation','protocol_sha256':phash,
          'query_cutoff':cutoff,'label_window':[lo,hi],
          'primary_candidate':'RL capped_seconds beta0.3 fixed last500 updates; single confirmatory candidate',
          'checkpoints':checkpoints,'checkpoint_sha256':{n:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for n,p in checkpoints.items()},
          'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'recsys/rl_comparison.py',ROOT/'recsys/data.py',ROOT/'recsys/models.py',ROOT/'recsys/metrics.py',ROOT/'recsys/diagnostics.py',protocol_path]},
          'hypothesis':'Predicted normalized capped_seconds improves vs reference AND NDCG remains within5% relative margin',
          'decision':'Both: reward delta lower two-sided95% bootstrap bound >0, relevance mean(NDCGcandidate-.95*NDCGref) lower bound>=0. No candidate selection on this period.',
          'bootstrap':{'draws':2000,'seed':42,'unit':'paired users'},
          'controls':['reference','continued_SFT','PersonalFrequency128','Frequency128_recency_tiebreak','Memory_recent_full_0.5'],
          'reward_scope':'All warm users/full fixed catalog for exact expected rewards, alternateRM sensitivity',
          'ranking_scope':'Warm users with future played>50% day positives, full catalog; OOV stays misses',
          'history':'Strictly before new query; earlier selection events can enter history, no fitted weights/stats change',
          'do_not_change':['checkpoints','hyperparameters','5%margin','candidate','primarymetric','labelwindow'],
          'final_test_scored':False,'limitations':['Same user cohort at a later day, not new users or randomized online A/B',
             'RM agreement does not establish causal utility','Single period and user bootstrap omit all training/time uncertainty']}
    # Lock hashes and criteria before loading evaluation events or computing model scores.
    atomic_json(out/'locked-manifest.json',plan)
    atomic_json(out/'status.json',{'status':'running','stage':'manifest_locked','final_test_scored':False})
    with zipfile.ZipFile(out/'code-snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['recsys','tests','research/protocol']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(4)
        data,features,_,_,support=load_sequences(ROOT/proto['data']['sample_path'],proto['time']['train_end_exclusive'])
        def load(name,rm=False):
            cp=torch.load(ROOT/checkpoints[name],map_location='cpu',weights_only=True)
            if not np.array_equal(cp['catalog'].numpy(),data.catalog):raise ValueError('Catalog mismatch')
            if cp.get('protocol_sha256',phash)!=phash:raise ValueError('Protocol mismatch')
            model=RewardModel(len(data.catalog),features,width=cp['width']) if rm else Policy(**cp['settings'])
            model.load_state_dict(cp['model']);model=model.cuda().eval();model.requires_grad_(False);return model
        if json.loads((ROOT/'artifacts/calibration-20261021-01/report.json').read_text())['protocol_sha256']!=phash:raise ValueError('Old RM protocol mismatch')
        models={n:load(n) for n in ['reference','candidate','continued_SFT']}
        rms={n:load(n,True) for n in ['primary_RM','alternate_RM']}
        hashes={n:state_hash(m) for n,m in {**models,**rms}.items()}
        ids,qx,qm,histories=data.queries(cutoff);truth=data.truth(lo,hi)
        nids,nx,nm,ntruth=data.next_group_queries(lo,hi)
        full,recent={},{};memory=[];recency=[]
        popularity=np.zeros(len(data.catalog),dtype=np.float32)
        by_uid={u['uid']:u for u in data.users}
        for u in data.users:
            train_end=int(np.searchsorted(u['times'],data.train_end));t=u['tokens'][:train_end]
            popularity+=np.bincount(t[t>=OFFSET]-OFFSET,minlength=len(popularity))
        popularity/=popularity.sum()
        for uid in ids:
            u=by_uid[uid];end=int(np.searchsorted(u['times'],cutoff));first=max(0,end-128)
            full[uid]=set(map(int,u['items'][:end]));recent[uid]=set(map(int,u['items'][first:end]))
            p=.5*history_distribution(u['tokens'][first:end],len(data.catalog),popularity)+.5*history_distribution(u['tokens'][:end],len(data.catalog),popularity)
            memory.append(probability_topk(p,data.catalog))
            counts,last=Counter(),{}
            for token,t in zip(u['tokens'][first:end],u['times'][first:end]):
                if token>=OFFSET:a=int(token)-OFFSET;counts[a]+=1;last[a]=int(t)
            order=sorted(counts,key=lambda a:(-counts[a],-last[a],int(data.catalog[a])))[:10];used=set(order)
            order.extend(a for a in range(10) if a not in used);recency.append(data.catalog[order[:10]].tolist())
        lengths=torch.as_tensor(features[:,0],device='cuda')
        primary=torch.empty((len(ids),len(data.catalog)),device='cuda');alternate=torch.empty_like(primary)
        @torch.no_grad()
        def probabilities(model,x,mask):
            values=torch.empty((len(x),len(data.catalog)),device='cuda')
            for j in range(0,len(x),64):
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(torch.as_tensor(x[j:j+64],device='cuda'),torch.as_tensor(mask[j:j+64],device='cuda')).float()
                values[j:j+64]=logits.softmax(-1)
            return values
        reference_probability=probabilities(models['reference'],qx,qm)
        logref=reference_probability.clamp_min(1e-30).log()
        atomic_json(out/'status.json',{'status':'running','stage':'reward_cache','final_test_scored':False})
        with torch.inference_mode():
            for j in range(0,len(ids),16):
                x=torch.as_tensor(qx[j:j+16],device='cuda');m=torch.as_tensor(qm[j:j+16],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):
                    primary[j:j+16]=rms['primary_RM'].all_scores(x,m,512).float()
                    alternate[j:j+16]=rms['alternate_RM'].all_scores(x,m,512).float()
        report={'locked_manifest':plan,'data_support':support,'all_warm_users':len(ids),'active_positive_users':len(truth),
                'models':{},'comparisons':{},'confirmation_scored':True,'final_test_scored':False}
        rows,rankings,reward_rows={},{},{}
        def rank_eval(name,recs):
            metrics,_=ranking_metrics(ids,recs,truth);metrics.update(interpretable(ids,recs,truth))
            slices,user=slice_metrics(ids,recs,truth,full,recent,data.catalog)
            report['models'][name]={'ranking':metrics,**slices};rows[name]=user;rankings[name]=recs
        with torch.inference_mode():
            for name,model in models.items():
                p=reference_probability if name=='reference' else probabilities(model,qx,qm)
                values=p.cpu().numpy();top=np.array([np.lexsort((data.catalog,-v))[:10] for v in values])
                recs=data.catalog[top].tolist();rank_eval(name,recs)
                summary,user=reward_summary(p,logref,primary,alternate,lengths,top)
                report['models'][name]['all_warm_reward']=summary;reward_rows[name]=user
                nextp=probabilities(model,nx,nm).cpu().numpy()
                report['models'][name]['next_group']=ranking_metrics(nids,[probability_topk(v,data.catalog) for v in nextp],ntruth)[0]
                print('EVAL',name,report['models'][name]['ranking'],summary,flush=True)
        rank_eval('PersonalFrequency128',personal_frequency(histories,data.catalog,10))
        rank_eval('Frequency128_recency_tiebreak',recency);rank_eval('Memory_recent_full_0.5',memory)
        # Identical checkpoint compared with itself: test numeric comparator, not online A/A.
        self_difference=paired_bootstrap(rows['reference'],rows['reference'])
        if self_difference['mean_ndcg_difference']!=0 or self_difference['paired_user_bootstrap_95_percentile_interval']!=[0.,0.]:raise ValueError('Self-comparison failed')
        if abs(report['models']['reference']['all_warm_reward']['exact_KL_to_reference'])>1e-5:raise ValueError('Reference self KL failed')
        for name in ['candidate','continued_SFT']:
            candidate=rows[name];ref=rows['reference']
            if [r['uid'] for r in candidate]!=[r['uid'] for r in ref]:raise ValueError('Paired ranking users differ')
            report['comparisons'][name]={'NDCG_delta':paired_bootstrap(candidate,ref),
                '5percent_relevance_margin':bootstrap_mean([a['ndcg']-.95*b['ndcg'] for a,b in zip(candidate,ref)]),
                'reward_deltas':{k:bootstrap_mean(reward_rows[name][k]-reward_rows['reference'][k]) for k in
                                 ['primary_completion','alternate_completion','primary_capped_seconds','alternate_capped_seconds','kl']}}
        c=report['comparisons']['candidate']
        report['decision']=confirmation_decision(c['reward_deltas']['primary_capped_seconds'],c['5percent_relevance_margin'])
        report['candidate_minus_continued_SFT']={'NDCG_delta':paired_bootstrap(rows['candidate'],rows['continued_SFT']),
            'proxy_capped_seconds_delta':bootstrap_mean(reward_rows['candidate']['primary_capped_seconds']-reward_rows['continued_SFT']['primary_capped_seconds'])}
        for name,model in {**models,**rms}.items():
            if state_hash(model)!=hashes[name] or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen component changed')
            if hashlib.sha256((ROOT/checkpoints[name]).read_bytes()).hexdigest()!=plan['checkpoint_sha256'][name]:raise ValueError('Checkpoint changed after lock')
        report.update({'self_comparison_pass':True,'all_frozen_hashes_pass':True,'elapsed_seconds':time.perf_counter()-started})
        atomic_json(out/'report.json',report);atomic_json(out/'rankings.json',{'uids':ids,'rankings':rankings})
        atomic_json(out/'ranking-per-user.json',rows)
        atomic_json(out/'reward-per-user.json',{n:{k:v.tolist() for k,v in values.items()} for n,values in reward_rows.items()})
        atomic_json(out/'status.json',{'status':'completed','decision':report['decision'],'elapsed_seconds':report['elapsed_seconds'],'final_test_scored':False})
        print('CONFIRMATION',json.dumps(report['decision']),flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',{'status':'failed','error':repr(exc),'final_test_scored':False});raise


if __name__=='__main__':main()
