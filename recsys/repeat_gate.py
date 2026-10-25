"""Bounded RepeatNet-inspired disjoint repeat/explore head, frozen GPT-2."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import OFFSET, load_sequences
from .models import Policy, state_hash
from .metrics import ranking_metrics
from .diagnostics import slice_metrics, paired_bootstrap
from .hybrid import probability_topk
from .pilot import atomic_json
ROOT = Path(__file__).resolve().parents[1]

class RepeatHead(nn.Module):
    def __init__(self, width, contextual=False, pointer=False):
        super().__init__()
        self.contextual, self.pointer = contextual, pointer
        self.gate = nn.Linear(width, 1) if contextual else nn.Parameter(torch.zeros(()))
        if contextual:
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        if pointer:
            self.query = nn.Linear(width, 16, bias=False)
            self.key = nn.Linear(width, 16, bias=False)

    def forward(self, hidden, embeddings, x, mask, logits):
        valid = (x >= OFFSET) & mask.bool()
        actions = (x-OFFSET).clamp(min=0)
        scores = (self.key(embeddings)*self.query(hidden)[:, None]).sum(-1)/4 if self.pointer else torch.zeros_like(x, dtype=torch.float32)
        # Zero-score fallback for empty known histories prevents all-masked softmax NaNs.
        safe = valid.clone(); safe[~valid.any(1), 0] = True
        positions = F.softmax(scores.masked_fill(~safe, -torch.inf), -1)*valid
        repeat = torch.zeros_like(logits).scatter_add(1, actions, positions)
        seen = torch.zeros_like(logits).scatter_add(1, actions, valid.float()) > 0
        available = ~seen
        safe_explore = available.clone(); safe_explore[~available.any(1), 0] = True
        explore = F.softmax(logits.masked_fill(~safe_explore, -torch.inf), -1)*available
        gate_logit = self.gate(hidden).squeeze(-1) if self.contextual else self.gate.expand(len(x))
        gate = torch.sigmoid(gate_logit)
        gate = torch.where(valid.any(1), gate, torch.zeros_like(gate))
        gate = torch.where(available.any(1), gate, torch.ones_like(gate))
        return repeat, explore, gate, gate_logit, seen

    def loss(self, output, targets):
        repeat, explore, gate, g, seen = output
        index = targets[:, None]
        is_repeat = seen.gather(1, index).squeeze(1)
        # Supports are disjoint; selecting branch avoids log(0) gradients and is
        # algebraically the same mixture likelihood as logaddexp with -inf.
        rp = repeat.gather(1,index).squeeze(1).clamp_min(1e-30).log()
        ep = explore.gather(1,index).squeeze(1).clamp_min(1e-30).log()
        has_repeat = seen.any(1); has_explore = (~seen).any(1)
        logr = torch.where(has_explore, F.logsigmoid(g), torch.zeros_like(g))
        loge = torch.where(has_repeat, F.logsigmoid(-g), torch.zeros_like(g))
        return -torch.where(is_repeat, logr+rp, loge+ep).mean()

    @staticmethod
    def probabilities(output):
        r,e,g,_,_ = output
        return g[:,None]*r+(1-g[:,None])*e

def interpretable(ids, rankings, truth):
    rows = [sum(i in truth[u] for i in ranking[:10]) for u, ranking in zip(ids, rankings) if u in truth]
    return {'HitRate@10': float(np.mean(np.array(rows)>0)), 'Precision@10': float(np.mean(rows)/10),
            'definition': 'conditional on positive next-day truth; hit means at least one matching observed >50% listened track, not acceptance probability'}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='artifacts/repeat-gate-20261021-01'); parser.add_argument('--steps',type=int,default=1000); a=parser.parse_args()
    out=ROOT/a.out; out.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    seed=20261026; torch.set_num_threads(4); torch.manual_seed(seed)
    protocol_path=ROOT/'research/protocol/case2-v1.json'; protocol=json.loads(protocol_path.read_text()); phash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    recipe={'seed':seed,'steps':a.steps,'batch':64,'eval_steps':[0,500,a.steps],'source':'https://arxiv.org/abs/1812.02646',
            'adaptation':'frozen GPT2 exploration; scalar/context gate; optional embedding pointer16, not original RepeatNet reproduction',
            'objective':'train next-item plain mixture NLL; disjoint known-last128 repeat versus other-catalog explore; no auxiliary mode loss',
            'lr':{'count_scalar':.01,'count_context':.003,'pointer_context':.003},'selection':'best selection next-day NDCG across 0/500/1000 exploratory; confirmation/final untouched',
            'checkpoint':'artifacts/sft-20261021-01/best.pt','protocol_sha256':phash}
    atomic_json(out/'recipe.json',recipe)
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for directory in ['recsys','tests','research/protocol']:
            for p in (ROOT/directory).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts: z.write(p,p.relative_to(ROOT))
    try:
        atomic_json(out/'status.json',{'status':'loading'})
        data,_,_,_,support=load_sequences(ROOT/protocol['data']['sample_path'],protocol['time']['train_end_exclusive'])
        cp=ROOT/recipe['checkpoint']; ck=torch.load(cp,map_location='cpu',weights_only=True)
        if ck['protocol_sha256']!=phash or not np.array_equal(ck['catalog'].numpy(),data.catalog): raise ValueError('checkpoint mismatch')
        model=Policy(**ck['settings']).cuda().eval(); model.load_state_dict(ck['model']); model.requires_grad_(False); del ck
        original=state_hash(model); width=model.settings['width']
        heads=nn.ModuleDict({'count_scalar':RepeatHead(width), 'count_context':RepeatHead(width,True), 'pointer_context':RepeatHead(width,True,True)}).cuda()
        opts={n:torch.optim.Adam(h.parameters(),lr=recipe['lr'][n]) for n,h in heads.items()}
        cutoff=protocol['time']['query_cutoffs']['selection_validation']; lo=cutoff+protocol['time']['gap_seconds']; truth=data.truth(lo,lo+protocol['time']['horizon_seconds'])
        ids,qx,qm,_=data.queries(cutoff)
        full,recent={},{}
        for u in data.users:
            end=int(np.searchsorted(u['times'],cutoff)); full[u['uid']]=set(map(int,u['items'][:end])); recent[u['uid']]=set(map(int,u['items'][max(0,end-128):end]))
        cached=json.loads((ROOT/'artifacts/hybrid-20261021-01/rankings.json').read_text())
        if cached['uids']!=ids: raise ValueError('control query mismatch')
        controls={n:cached['rankings'][n] for n in ['PersonalFrequency128_original','PersonalFrequency128_recency_tiebreak','Memory_full_weight_0.5']}
        def metrics(recs):
            overall=ranking_metrics(ids,recs,truth)[0]; overall.update(interpretable(ids,recs,truth))
            slices,rows=slice_metrics(ids,recs,truth,full,recent,data.catalog)
            return {'overall':overall,**slices},rows
        control_values,control_rows={},{}
        for n,r in controls.items(): control_values[n],control_rows[n]=metrics(r)
        @torch.no_grad()
        def encode(x,mask):
            with torch.autocast('cuda',dtype=torch.float16):
                allh=model.transformer(input_ids=x,attention_mask=mask,use_cache=False).last_hidden_state
                hidden=allh[torch.arange(len(x),device=x.device),mask.sum(1)-1]
                logits=F.linear(hidden,model.transformer.wte.weight[OFFSET:])
            return hidden.float(),model.transformer.wte(x).float(),logits.float()
        evals=[]; selected={}; bestrows={}; bestrank={}
        @torch.no_grad()
        def evaluate(step):
            recs={n:[] for n in ['GPT2_frozen',*heads]}
            for i in range(0,len(ids),64):
                x=torch.as_tensor(qx[i:i+64],device='cuda'); m=torch.as_tensor(qm[i:i+64],device='cuda'); h,e,l=encode(x,m)
                values={'GPT2_frozen':l}
                values.update({n:head.probabilities(head(h,e,x,m,l)) for n,head in heads.items()})
                for n,v in values.items(): recs[n].extend(probability_topk(p,data.catalog) for p in v.cpu().numpy())
            result={'step':step,'models':{}}
            for n,r in recs.items():
                value,rows=metrics(r); result['models'][n]=value
                if n not in selected or value['overall']['NDCG@10']>selected[n]['metrics']['overall']['NDCG@10']:
                    selected[n]={'step':step,'metrics':value}; bestrows[n]=rows; bestrank[n]=r
                    if n in heads: torch.save(heads[n].state_dict(),out/(n+'-best.pt'))
            evals.append(result); atomic_json(out/'evaluations.json',evals)
            print('EVAL',step,{n:v['overall']['NDCG@10'] for n,v in result['models'].items()},flush=True)
        evaluate(0); rng=np.random.default_rng(seed); trace=[]
        for step in range(1,a.steps+1):
            batch=data.sample(rng,64); x,m,target=[torch.as_tensor(b,device='cuda') for b in batch[:3]]; h,e,l=encode(x,m); losses={}
            for n,head in heads.items():
                opt=opts[n]; opt.zero_grad(); loss=head.loss(head(h,e,x,m,l),target)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()): raise FloatingPointError('nonfinite gradient')
                opt.step(); losses[n]=loss.item()
            if step%100==0:
                trace.append({'step':step,'last_batch_NLL':losses}); atomic_json(out/'status.json',{'status':'running','step':step,'elapsed_seconds':time.perf_counter()-start})
            if step in [500,a.steps]: evaluate(step)
        unchanged=state_hash(model)==original and all(p.grad is None for p in model.parameters())
        if not unchanged: raise RuntimeError('frozen policy changed')
        comparisons={}
        for n in selected:
            for base,rows in {**control_rows,'GPT2_frozen':bestrows['GPT2_frozen']}.items(): comparisons[n+'_minus_'+base]=paired_bootstrap(bestrows[n],rows)
        comparisons['pointer_minus_context_count']=paired_bootstrap(bestrows['pointer_context'],bestrows['count_context'])
        comparisons['context_minus_scalar']=paired_bootstrap(bestrows['count_context'],bestrows['count_scalar'])
        report={'recipe':recipe,'data_support':support,'evaluations':evals,'selected':selected,'controls':control_values,'comparisons':comparisons,'trace':trace,
                'frozen_hash_unchanged_and_no_gradients':unchanged,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,
                'confirmation_scored':False,'final_test_scored':False,'limits':['One seed, exploratory selection over checkpoints.','Explore means outside last128, not new to user.','Full-memory control has more history.','Train next-item objective versus next-day evaluation mismatch remains.']}
        atomic_json(out/'report.json',report); atomic_json(out/'rankings.json',{'uids':ids,'rankings':bestrank}); atomic_json(out/'per-user.json',bestrows)
        torch.save({'heads':heads.state_dict(),'optimizers':{n:o.state_dict() for n,o in opts.items()},'step':a.steps,'recipe':recipe},out/'last.pt')
        atomic_json(out/'status.json',{'status':'completed','elapsed_seconds':report['elapsed_seconds']})
    except BaseException as ex:
        atomic_json(out/'status.json',{'status':'failed','error':repr(ex)}); raise

if __name__=='__main__': main()
