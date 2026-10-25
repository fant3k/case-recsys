"""Fixed-budget H1 RL matrix with matched SFT, exact reward/KL and H5 diagnostics."""
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

from .data import OFFSET, load_sequences
from .day_objective import DaySampler, joint_loss
from .diagnostics import paired_bootstrap, slice_metrics
from .hybrid import probability_topk
from .metrics import ranking_metrics
from .models import Policy, RewardModel, exact_kl, policy_loss, state_hash
from .pilot import atomic_json
from .repeat_gate import interpretable

ROOT=Path(__file__).resolve().parents[1]


def bootstrap_mean(values, seed=42):
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'paired_user_bootstrap95':np.quantile(samples,[.025,.975]).tolist(),
            'caution':'One selection period; no multiple-testing/training-seed correction'}


def reward_summary(probability, log_reference, primary, alternate, lengths, top):
    logp=probability.clamp_min(1e-30).log()
    kl=(probability*(logp-log_reference)).sum(-1)
    idx=torch.as_tensor(top,device=probability.device)
    result={'exact_KL_to_reference':float(kl.mean()),
            'expected_primary_completion':float((probability*primary).sum(-1).mean()),
            'expected_alternate_completion':float((probability*alternate).sum(-1).mean()),
            'expected_primary_capped_seconds':float((probability*primary*lengths).sum(-1).mean()),
            'expected_alternate_capped_seconds':float((probability*alternate*lengths).sum(-1).mean()),
            'expected_primary_equivalent_seconds':float((probability*primary*lengths*600).sum(-1).mean()),
            'expected_alternate_equivalent_seconds':float((probability*alternate*lengths*600).sum(-1).mean()),
            'expected_capped_length_seconds':float((probability*lengths*600).sum(-1).mean()),
            'top10_primary_completion':float(primary.gather(1,idx).mean()),
            'top10_alternate_completion':float(alternate.gather(1,idx).mean()),
            'top10_primary_capped_seconds':float((primary*lengths).gather(1,idx).mean())}
    return result, {'primary_completion':(probability*primary).sum(-1).cpu().numpy(),
                    'alternate_completion':(probability*alternate).sum(-1).cpu().numpy(),
                    'primary_capped_seconds':(probability*primary*lengths).sum(-1).cpu().numpy(),
                    'alternate_capped_seconds':(probability*alternate*lengths).sum(-1).cpu().numpy(),
                    'kl':kl.cpu().numpy()}


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()  # no options; makes --help safe instead of starting a run
    out=ROOT/'artifacts/rl-comparison-20261021-01';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    pp=ROOT/'research/protocol/case2-v1.json';proto=json.loads(pp.read_text());phash=hashlib.sha256(pp.read_bytes()).hexdigest()
    source=ROOT/'artifacts/day-objective-20261021-01/next_item_plus_day/best.pt'
    rm_path=ROOT/'artifacts/reward-audit-20261021-01/history-last.pt'
    other_path=ROOT/'artifacts/calibration-20261021-01/reward-model.pt'
    arms=[{'name':'continued_SFT','kind':'sft','seed':20261031}]
    for reward in ['completion','capped_seconds']:
        for beta in [.03,.3]:
            arms.append({'name':f'RL_{reward}_beta{beta}','kind':'rl','reward':reward,'beta':beta,'seed':20261031})
    arms.append({'name':'RL_completion_beta0.03_seed2','kind':'rl','reward':'completion','beta':.03,'seed':20261032})
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':phash,
          'source_policy':str(source.relative_to(ROOT)),'primary_RM':str(rm_path.relative_to(ROOT)),
          'alternate_RM':str(other_path.relative_to(ROOT)),
          'RM_choice':'new fixed-step RM chosen for lower audited logged-event RMSE and calibration gap; old RM better MAE retained as sensitivity evaluator, not independent ground truth',
          'arms':arms,'steps_per_arm':500,'batch':64,'lr':1e-5,'weight_decay':.01,'grad_clip':1.,
          'optimization':'Fresh AdamW/scaler for all changing policies; same initial weights. FP16 forward FP32 loss. Fixed LAST checkpoint; no selection earlystop.',
          'training_context_seed':20261030,
          'training_contexts':'Same pre-generated user/group train-contained day anchors for ALL arms. Matches joint-SFT loss; narrower support than original unrestricted group sampler; explicit run-level adaptation.',
          'continued_SFT':'Same .5 next CE + .5 unique known day-positive meanCE; fallback nextCE if empty',
          'RL':'On-policy H1 REINFORCE, action-independent EMA baseline (update AFTER step), exact KL(pi||frozen reference). No reward standardization.',
          'reward_units':'completion in[0,1]; capped_seconds=completion*min(trainmedianlength,600)/600 in[0,1]',
          'evaluation':'selection only, all warm users for exact expected rewards/KL, active positive subset for unchanged NDCG. Same 50k catalog full support.',
          'analytic':'H1 q(a|h)*exp(r/beta) normalized for each of4reward/beta, fixed RM and same queries',
          'H5':'64 warm users selected seed20261033 before future-label filtering, sampled5recommendations appended as ITEM TOKENS ONLY. Not user acceptance. No made-up elapsed times.',
          'uncertainty':'Paired user bootstrap2000seed42; relevance preservation requires lower95bound of mean(NDCGpi-.95*NDCGref)>=0, not just point estimate',
          'limits':['Multiple previously explored selection, not independent confirmation','Alternate RM shares source logs, so agreement is not causal validation','Same update/context counts, actual compute differs; report per arm',
                    'H1 analytic control is pointwise reward proxy optimum, not true optimal recommender or H5 optimum'],
          'confirmation_scored':False,'final_test_scored':False}
    atomic_json(out/'plan.json',plan);atomic_json(out/'status.json',{'status':'running','stage':'loading'})
    with zipfile.ZipFile(out/'code-snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['recsys','tests','research/protocol']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(4)
        data,features,_,global_mean,support=load_sequences(ROOT/proto['data']['sample_path'],proto['time']['train_end_exclusive'])
        ck=torch.load(source,map_location='cpu',weights_only=True)
        if ck['protocol_sha256']!=phash or not np.array_equal(ck['catalog'].numpy(),data.catalog):raise ValueError('Source mismatch')
        reference=Policy(**ck['settings']).cuda().eval();reference.load_state_dict(ck['model']);reference.requires_grad_(False)
        del ck
        def load_rm(path):
            c=torch.load(path,map_location='cpu',weights_only=True)
            if not np.array_equal(c['catalog'].numpy(),data.catalog):raise ValueError('RM catalog mismatch')
            if c.get('protocol_sha256',phash)!=phash:raise ValueError('RM protocol mismatch')
            m=RewardModel(len(data.catalog),features,width=c['width']).cuda();m.load_state_dict(c['model'])
            m.eval();m.requires_grad_(False);return m
        if json.loads((other_path.parent/'report.json').read_text())['protocol_sha256']!=phash:raise ValueError('Old RM protocol mismatch')
        primary=load_rm(rm_path);alternate=load_rm(other_path)
        hashes={n:state_hash(m) for n,m in [('reference',reference),('primary_RM',primary),('alternate_RM',alternate)]}
        atomic_json(out/'provenance.json',{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                         'primary_RM_sha256':hashlib.sha256(rm_path.read_bytes()).hexdigest(),
                                         'alternate_RM_sha256':hashlib.sha256(other_path.read_bytes()).hexdigest(),'initial_tensor_hashes':hashes})
        cut=proto['time']['query_cutoffs']['selection_validation'];lo=cut+proto['time']['gap_seconds'];hi=lo+proto['time']['horizon_seconds']
        ids,qx,qm,_=data.queries(cut);truth=data.truth(lo,hi)
        nids,nx,nm,ntruth=data.next_group_queries(lo,hi)
        full,recent={},{}
        for u in data.users:
            end=int(np.searchsorted(u['times'],cut));full[u['uid']]=set(map(int,u['items'][:end]));recent[u['uid']]=set(map(int,u['items'][max(0,end-128):end]))
        sampler=DaySampler(data,proto['time']['gap_seconds'],proto['time']['horizon_seconds'])
        rng=np.random.default_rng(plan['training_context_seed'])
        batches=[sampler.sample(rng,64) for _ in range(plan['steps_per_arm'])]
        lengths=torch.as_tensor(features[:,0],device='cuda')
        atomic_json(out/'status.json',{'status':'running','stage':'caching_all_query_rewards'})
        # Fixed caches permit exact policy expectation on ALL warm queries, not only active users.
        primary_scores=torch.empty((len(ids),len(data.catalog)),device='cuda')
        alternate_scores=torch.empty_like(primary_scores);logref=torch.empty_like(primary_scores)
        with torch.inference_mode():
            for j in range(0,len(ids),16):
                x=torch.as_tensor(qx[j:j+16],device='cuda');m=torch.as_tensor(qm[j:j+16],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):
                    primary_scores[j:j+16]=primary.all_scores(x,m,chunk_size=512).float()
                    alternate_scores[j:j+16]=alternate.all_scores(x,m,chunk_size=512).float()
                    logref[j:j+16]=reference(x,m).float().log_softmax(-1)
        report={'plan':plan,'data_support':support,'all_warm_reward_users':len(ids),'ranking_active_users':len(truth),
                'models':{},'training':{},'comparisons':{},'confirmation_scored':False,'final_test_scored':False}
        rankings,rows,reward_rows={}, {},{}
        uid_to_index={uid:j for j,uid in enumerate(ids)}
        h5indices=np.random.default_rng(20261033).choice(len(ids),64,replace=False)
        h5={}

        @torch.no_grad()
        def policy_prob(model,x,mask):
            p=torch.empty((len(x),len(data.catalog)),device='cuda')
            for j in range(0,len(x),64):
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(torch.as_tensor(x[j:j+64],device='cuda'),torch.as_tensor(mask[j:j+64],device='cuda')).float()
                p[j:j+64]=logits.softmax(-1)
            return p

        @torch.no_grad()
        def evaluate(name,model=None,probability=None,reward=None,beta=None):
            if probability is None:probability=policy_prob(model,qx,qm)
            values=probability.cpu().numpy()
            top=np.array([np.lexsort((data.catalog,-p))[:10] for p in values])
            recs=data.catalog[top].tolist()
            overall,_=ranking_metrics(ids,recs,truth);overall.update(interpretable(ids,recs,truth))
            slices,peruser=slice_metrics(ids,recs,truth,full,recent,data.catalog)
            rewards,individual=reward_summary(probability,logref,primary_scores,alternate_scores,lengths,top)
            result={'ranking':overall,**slices,'all_warm_query_reward':rewards,
                    'top10_mean_capped_length_seconds':float(lengths[torch.as_tensor(top,device='cuda')].mean()*600),
                    'primary_reward_std_over_query_actions':float(primary_scores.std()),
                    'capped_reward_std_over_query_actions':float((primary_scores*lengths).std())}
            if beta is not None:
                chosen='primary_completion' if reward=='completion' else 'primary_capped_seconds'
                result['regularized_primary_objective']=float(np.mean(individual[chosen]-beta*individual['kl']))
            if model is not None:
                pnext=policy_prob(model,nx,nm).cpu().numpy()
                nrecs=[probability_topk(p,data.catalog) for p in pnext]
                result['next_group']=ranking_metrics(nids,nrecs,ntruth)[0]
            rankings[name],rows[name],reward_rows[name]=recs,peruser,individual
            report['models'][name]=result
            atomic_json(out/'partial-report.json',report)
            print('EVAL',name,overall,rewards,flush=True)
            del values

        @torch.no_grad()
        def rollout(name,model):
            x=torch.as_tensor(qx[h5indices],device='cuda');mask=torch.as_tensor(qm[h5indices],device='cuda')
            torch.manual_seed(20261034)
            steps=[];generated=[]
            for t in range(5):
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(x,mask).float();action=torch.distributions.Categorical(logits=logits).sample()
                    r=primary(x,mask,action).float();r_other=alternate(x,mask,action).float()
                steps.append({'step':t+1,'sampled_primary_completion':r.mean().item(),
                              'sampled_alternate_completion':r_other.mean().item(),
                              'sampled_primary_capped_seconds':(r*lengths[action]).mean().item()})
                generated.append(action.cpu().numpy())
                for j,a in enumerate(action):
                    n=int(mask[j].sum())
                    if n<128:x[j,n]=a+OFFSET;mask[j,n]=1
                    else:x[j,:-1]=x[j,1:].clone();x[j,-1]=a+OFFSET
            actions=np.stack(generated,axis=1)
            h5[name]={'users':[int(ids[i]) for i in h5indices], 'generated_item_ids':data.catalog[actions].tolist(),
                      'steps':steps,'within_rollout_repeat_fraction':float(np.mean([5-len(set(a)) for a in actions])/5),
                      'hypothetical_recommendations_not_observed_interactions':True}

        evaluate('reference',model=reference);rollout('reference',reference)
        for arm in arms:
            name=arm['name'];arm_dir=out/name;arm_dir.mkdir()
            torch.manual_seed(arm['seed'])
            policy=copy.deepcopy(reference)
            if state_hash(policy)!=hashes['reference']:raise ValueError('Arm did not start from identical weights')
            policy.requires_grad_(True);policy.train()
            optimizer=torch.optim.AdamW(policy.parameters(),lr=plan['lr'],weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
            baseline=global_mean if arm.get('reward')=='completion' else global_mean*float(lengths.mean())
            trace=[];skipped=0;torch.cuda.reset_peak_memory_stats();train_start=time.perf_counter()
            for step,batch in enumerate(batches,1):
                x,m,target=[torch.as_tensor(a,device='cuda') for a in batch[:3]]
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=policy(x,m).float()
                    if arm['kind']=='sft':
                        owners,day,weights,has=[torch.as_tensor(a,device='cuda') for a in batch[3:7]]
                        loss=joint_loss(logits,target,owners,day,weights,has)
                        observed,kl_value=0.,0.
                    else:
                        with torch.no_grad():
                            ref_logits=reference(x,m).float()
                            actions=torch.distributions.Categorical(logits=logits).sample()
                            rewards=primary(x,m,actions).float()
                            if arm['reward']=='capped_seconds':rewards=rewards*lengths[actions]
                        loss,kl=policy_loss(logits,ref_logits,actions,rewards,baseline,arm['beta'])
                        observed,kl_value=float(rewards.mean()),float(kl)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite optimization loss')
                optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped+=int(scaler.get_scale()<scale)
                if arm['kind']=='rl':baseline=.95*baseline+.05*observed
                if step%50==0:
                    trace.append({'step':step,'last_batch_loss':float(loss),'sampled_predicted_reward':observed,
                                  'batch_exact_KL':kl_value,'EMA_baseline_for_next_batch':baseline})
                    atomic_json(out/'status.json',{'status':'running','arm':name,'step':step,'steps_per_arm':500,
                                                 'elapsed_seconds':time.perf_counter()-started})
            torch.cuda.synchronize();train_seconds=time.perf_counter()-train_start
            info={'updates_attempted':500,'optimizer_updates_skipped':skipped,'contexts':500*64,
                  'seconds_excluding_batch_preparation_and_evaluation':train_seconds,
                  'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            if skipped:raise RuntimeError('Skipped optimizer update invalidates matched-update comparison; retain artifacts')
            policy.eval()
            torch.save({'model':policy.state_dict(),'settings':policy.settings,'optimizer':optimizer.state_dict(),
                        'scaler':scaler.state_dict(),'catalog':torch.from_numpy(data.catalog),'protocol_sha256':phash,
                        'new_steps':500,'baseline':baseline,'arm':arm,'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all()},arm_dir/'last.pt')
            atomic_json(arm_dir/'trace.json',trace);report['training'][name]=info
            evaluate(name,model=policy,reward=arm.get('reward'),beta=arm.get('beta'));rollout(name,policy)
            del policy,optimizer,scaler,logits,loss
            torch.cuda.empty_cache()
        for reward in ['completion','capped_seconds']:
            score=primary_scores if reward=='completion' else primary_scores*lengths
            for beta in [.03,.3]:
                evaluate(f'analytic_{reward}_beta{beta}',probability=(logref+score/beta).softmax(-1),reward=reward,beta=beta)
        for name in report['models']:
            if name=='reference':continue
            refrows=rows['reference'];candidate=rows[name]
            if [v['uid'] for v in refrows]!=[v['uid'] for v in candidate]:raise ValueError('Ranking user mismatch')
            relevance=bootstrap_mean([a['ndcg']-.95*b['ndcg'] for a,b in zip(candidate,refrows)])
            report['comparisons'][name]={'NDCG_delta':paired_bootstrap(candidate,refrows),
                'within_5percent_relative_relevance':relevance,
                'relevance_preserved_supported':relevance['paired_user_bootstrap95'][0]>=0,
                'reward_deltas_all_warm_queries':{k:bootstrap_mean(reward_rows[name][k]-reward_rows['reference'][k])
                      for k in ['primary_completion','alternate_completion','primary_capped_seconds','alternate_capped_seconds','kl']}}
        cached=json.loads((ROOT/'artifacts/hybrid-20261021-01/rankings.json').read_text())
        if cached['uids']!=ids:raise ValueError('Baseline query mismatch')
        report['frequency_controls']={n:ranking_metrics(ids,cached['rankings'][n],truth)[0] for n in
                    ['PersonalFrequency128_original','PersonalFrequency128_recency_tiebreak','Memory_full_weight_0.5']}
        for name,model in [('reference',reference),('primary_RM',primary),('alternate_RM',alternate)]:
            if state_hash(model)!=hashes[name] or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen tensor modified or gradient reached frozen model')
        report['frozen_hashes_and_no_gradients_pass']=True
        report['all_initial_trainable_policies_identical_to_reference']=True
        report['elapsed_seconds']=time.perf_counter()-started
        atomic_json(out/'report.json',report);atomic_json(out/'h5.json',h5)
        atomic_json(out/'rankings.json',{'uids':ids,'rankings':rankings});atomic_json(out/'ranking-per-user.json',rows)
        atomic_json(out/'reward-per-user.json',{n:{k:v.tolist() for k,v in row.items()} for n,row in reward_rows.items()})
        atomic_json(out/'status.json',{'status':'completed','elapsed_seconds':report['elapsed_seconds']})
    except BaseException as exc:
        atomic_json(out/'status.json',{'status':'failed','error':repr(exc)});raise


if __name__=='__main__':main()
