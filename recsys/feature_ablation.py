"""Train-only fitting of explicit frequency/elapsed-time residuals to frozen GPT2."""
import argparse
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
from .diagnostics import paired_bootstrap, slice_metrics
from .hybrid import probability_topk
from .metrics import ranking_metrics
from .models import Policy, state_hash
from .pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def explicit_features(contexts, masks, event_times, query_times, vocab_size):
    """Two candidate features: log1p count; exp(-elapsed/1day) at latest listen.

    Times correspond exactly to packed context tokens, including OOV positions.
    No target/current event may enter the context. PAD/UNK/BOS excluded.
    """
    features = np.zeros((len(contexts), vocab_size, 2), dtype=np.float32)
    for j, (tokens, mask, times, query) in enumerate(zip(contexts, masks, event_times, query_times)):
        count = int(np.sum(mask))
        times = np.asarray(times, dtype=np.int64)
        # Empty history is represented by BOS; it carries no event timestamp.
        if len(times) == 0:
            if np.any(np.asarray(tokens)[:count] >= OFFSET):
                raise ValueError('Known tokens without event timestamps')
            continue
        if count != len(times) or np.any(times >= query):
            raise ValueError('Context timestamp alignment or strict-past violation')
        tok = np.asarray(tokens)[:count]
        known = tok >= OFFSET
        actions = tok[known]-OFFSET
        if len(actions):
            counts = np.bincount(actions, minlength=vocab_size)
            features[j, :, 0] = np.log1p(counts)
            np.maximum.at(features[j, :, 1], actions,
                          np.exp2(-(query-times[known]).astype(np.float64)/86400))
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='artifacts/features-20261021-02')
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    protocol_path = ROOT/'research/protocol/case2-v1.json'
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    plan = {'started_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': protocol_hash,
            'steps': args.steps, 'batch': 64, 'seed': 20261024, 'evaluation_steps': [0, 500, args.steps],
            'checkpoint': 'artifacts/sft-20261021-01/best.pt',
            'features': ['log1p candidate count in same last128 events', '2^(-seconds_since_last_candidate_listen/86400), zero if absent'],
            'models': {'GPT2_frequency': 'frozenGPT2 logits + w_count*feature_count',
                       'GPT2_frequency_recency': 'frozenGPT2 logits + w_count*feature_count + w_recency*feature_recency',
                       'Popularity_frequency_recency': 'fixed log train item frequency + same two features'},
            'fit': 'Adam lr.03, initial residual coefficients zero; original train-only user/group/event sampling, CE next-item. One shared sampled batch for all heads. No GPT2/RM/RL weight updates.',
            'selection': 'checkpoint per variant by selection day NDCG; earliest on ties. Show all evaluations.',
            'scope': 'selection only, no confirmation/final evaluation; auxiliary features use same128 context',
            'limits': 'Linear shared candidate bias, not a contextual gate/full memory/production reproduction. No multi-seed claim.'}
    atomic_json(out/'plan.json', plan)
    atomic_json(out/'status.json', {'status': 'running', 'stage': 'loading'})
    with zipfile.ZipFile(out/'code-snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for directory in ['recsys', 'tests', 'research/protocol']:
            for p in (ROOT/directory).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:
                    z.write(p, p.relative_to(ROOT))
    try:
        torch.set_num_threads(4)
        torch.manual_seed(20261024)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        data, _, _, _, support = load_sequences(ROOT/protocol['data']['sample_path'], protocol['time']['train_end_exclusive'])
        checkpoint_path = ROOT/plan['checkpoint']
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if ck['protocol_sha256'] != protocol_hash or not np.array_equal(ck['catalog'].numpy(), data.catalog):
            raise ValueError('Checkpoint protocol/catalog mismatch')
        model = Policy(**ck['settings']).cuda().eval()
        model.load_state_dict(ck['model'])
        model.requires_grad_(False)
        del ck
        original_hash = state_hash(model)
        cutoff = protocol['time']['query_cutoffs']['selection_validation']
        lo = cutoff+protocol['time']['gap_seconds']
        truth = data.truth(lo, lo+protocol['time']['horizon_seconds'])
        ids, qx, qm, _ = data.queries(cutoff)
        by_uid = {u['uid']: u for u in data.users}
        full_seen, recent_seen, query_times = {}, {}, []
        popularity = np.zeros(len(data.catalog), dtype=np.float32)
        for u in data.users:
            end = int(np.searchsorted(u['times'], cutoff))
            tok = u['tokens'][:end]
            popularity += np.bincount(tok[tok >= OFFSET]-OFFSET, minlength=len(popularity))
            full_seen[u['uid']] = set(map(int, u['items'][:end]))
            recent_seen[u['uid']] = set(map(int, u['items'][max(0, end-128):end]))
        for uid in ids:
            u = by_uid[uid]
            end = int(np.searchsorted(u['times'], cutoff))
            query_times.append(u['times'][max(0, end-128):end])
        logpop = torch.as_tensor(np.log(popularity/popularity.sum()), device='cuda')
        heads = {name: torch.nn.Parameter(torch.zeros(1 if name=='GPT2_frequency' else 2, device='cuda')) for name in plan['models']}
        optimizers = {name: torch.optim.Adam([w], lr=.03) for name, w in heads.items()}
        evals, best, best_rows, best_rankings = [], {}, {}, {}
        fixed_rankings = json.loads((ROOT/'artifacts/hybrid-20261021-01/rankings.json').read_text())
        if ids != fixed_rankings['uids']:
            raise ValueError('Cached controls use different query users')
        controls = {name: fixed_rankings['rankings'][name] for name in
                    ['PersonalFrequency128_original', 'PersonalFrequency128_recency_tiebreak', 'Memory_full_weight_0.5']}
        control_rows, control_metrics = {}, {}
        for name, recs in controls.items():
            control_metrics[name] = ranking_metrics(ids, recs, truth)[0]
            control_rows[name] = slice_metrics(ids, recs, truth, full_seen, recent_seen, data.catalog)[1]

        @torch.no_grad()
        def evaluate(step):
            recs = {name: [] for name in ['GPT2_frozen', *heads]}
            for i in range(0, len(ids), 64):
                x = torch.as_tensor(qx[i:i+64], device='cuda')
                mask = torch.as_tensor(qm[i:i+64], device='cuda')
                features = torch.as_tensor(explicit_features(qx[i:i+64], qm[i:i+64], query_times[i:i+64],
                                            [cutoff]*len(x), len(data.catalog)), device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x, mask).float()
                scores = {'GPT2_frozen': logits}
                for name, w in heads.items():
                    base = logpop if name.startswith('Popularity') else logits
                    scores[name] = base+(features[:, :, :len(w)]*w).sum(-1)
                for name, values in scores.items():
                    recs[name].extend(probability_topk(p, data.catalog) for p in values.cpu().numpy())
            result = {'step': step, 'models': {}}
            for name, rankings in recs.items():
                overall, _ = ranking_metrics(ids, rankings, truth)
                slices, per_user = slice_metrics(ids, rankings, truth, full_seen, recent_seen, data.catalog)
                value = {'overall': overall, **slices,
                         'coefficients': heads[name].detach().cpu().tolist() if name in heads else []}
                result['models'][name] = value
                if name not in best or overall['NDCG@10'] > best[name]['metrics']['overall']['NDCG@10']:
                    best[name] = {'step': step, 'metrics': value}
                    best_rows[name], best_rankings[name] = per_user, rankings
            evals.append(result)
            atomic_json(out/'evaluations.json', evals)
            atomic_json(out/'best-heads.json', best)
            print('EVAL', step, {n: v['overall']['NDCG@10'] for n, v in result['models'].items()}, flush=True)

        evaluate(0)
        rng = np.random.default_rng(plan['seed'])
        trace = []
        for step in range(1, args.steps+1):
            batch = data.sample(rng, 64)
            x, mask, target = [torch.as_tensor(a, device='cuda') for a in batch[:3]]
            times, targets_at = [], []
            for uid, event, t in batch[-1]:
                u = by_uid[uid]
                end = int(np.searchsorted(u['times'], t))
                times.append(u['times'][max(0, end-128):end])
                targets_at.append(t)
            features = torch.as_tensor(explicit_features(batch[0], batch[1], times, targets_at, len(data.catalog)), device='cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                logits = model(x, mask).float()
            losses = {}
            for name, w in heads.items():
                base = logpop if name.startswith('Popularity') else logits
                scores = base+(features[:, :, :len(w)]*w).sum(-1)
                loss = F.cross_entropy(scores, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite feature loss')
                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()
                losses[name] = loss.item()
            if step % 100 == 0:
                trace.append({'step': step, 'last_batch_losses': losses,
                              'coefficients': {n: w.detach().cpu().tolist() for n, w in heads.items()}})
                atomic_json(out/'status.json', {'status': 'running', 'step': step, 'total_steps': args.steps,
                                               'elapsed_seconds': time.perf_counter()-start})
            if step in [500, args.steps]:
                evaluate(step)
        frozen_unchanged = state_hash(model) == original_hash
        if not frozen_unchanged or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen GPT2 changed or received gradients')
        comparisons = {}
        for name in best:
            comparisons[name+'_minus_frozenGPT2'] = paired_bootstrap(best_rows[name], best_rows['GPT2_frozen'])
            for control in control_rows:
                comparisons[name+'_minus_'+control] = paired_bootstrap(best_rows[name], control_rows[control])
        comparisons['combined_minus_frequency_only'] = paired_bootstrap(best_rows['GPT2_frequency_recency'], best_rows['GPT2_frequency'])
        report = {'plan': plan, 'data_support': support, 'evaluations': evals, 'selected': best,
                  'controls': control_metrics, 'comparisons': comparisons,
                  'frozen_GPT2_hash_unchanged': frozen_unchanged,
                  'no_GPT2_gradients': True, 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  'elapsed_seconds': time.perf_counter()-start, 'confirmation_scored': False, 'final_test_scored': False,
                  'cautions': ['Exploratory selection only with checkpoint selection, no multiplicity correction.',
                               'Frozen linear residual ablation, not end-to-end retraining or learned user gate.',
                               'Full-memory control has more context; added features use same128 events.',
                               'Train next-item CE differs from selection day ranking, report both honestly.']}
        atomic_json(out/'trace.json', trace)
        atomic_json(out/'report.json', report)
        atomic_json(out/'rankings.json', {'uids': ids, 'selected_rankings': best_rankings})
        atomic_json(out/'per-user.json', best_rows)
        torch.save({'coefficients': {n: w.detach().cpu() for n, w in heads.items()},
                    'optimizers': {n: o.state_dict() for n, o in optimizers.items()},
                    'sampler_rng': rng.bit_generator.state, 'step': args.steps,
                    'protocol_sha256': protocol_hash, 'frozen_policy_sha256': report['checkpoint_sha256']}, out/'last-heads.pt')
        atomic_json(out/'status.json', {'status': 'completed', 'steps': args.steps, 'elapsed_seconds': report['elapsed_seconds']})
    except BaseException as exc:
        atomic_json(out/'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
