"""Bounded selection-only recent/full-history/GPT2 probability mixtures."""
import argparse
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
from .metrics import ranking_metrics
from .models import Policy
from .pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def history_distribution(tokens, size, fallback):
    tokens = np.asarray(tokens)
    known = tokens[tokens >= OFFSET]-OFFSET
    counts = np.bincount(known, minlength=size).astype(np.float32)
    return counts/counts.sum() if counts.sum() else fallback.copy()


def probability_topk(probability, catalog, k=10):
    # Raw ID is the explicit secondary key, including zero-probability ties.
    return catalog[np.lexsort((catalog, -probability))[:k]].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='artifacts/hybrid-20261021-01')
    args = parser.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol_path = ROOT/'research/protocol/case2-v1.json'
    sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol = json.loads(protocol_path.read_text())
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha,
            'scope': 'selection day only; confirmation/final untouched; frozen GPT2',
            'checkpoint': 'artifacts/sft-20261021-01/best.pt',
            'memory_full_weight_grid': [0, .25, .5, .75, 1],
            'gpt2_weight_grid_after_memory_selection': [0, .1, .25, .5, 1],
            'formula_memory': '(1-lambda)*normalized_recent128_counts + lambda*normalized_full_history_counts',
            'formula_hybrid': '(1-alpha)*selected_memory + alpha*GPT2_softmax_temperature1',
            'note': 'Full history includes recent128, not disjoint older-only. Additional context explicitly labeled. No training or learned gate.',
            'ties': 'raw item ID ascending; original frequency fallback preserved separately',
            'empty_known_history_fallback': 'train popularity normalized counts',
            'selection': 'maximize selection NDCG, tie smaller full/GPT2 weight; exploratory in same repeatedly viewed window',
            'bootstrap': '2000 paired user draws seed42; no selection/grid correction; not confirmation'}
    atomic_json(out/'plan.json', plan)
    atomic_json(out/'status.json', {'status': 'running', 'stage': 'loading'})
    with zipfile.ZipFile(out/'code-snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ['recsys', 'tests', 'research/protocol']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:
                    z.write(p, p.relative_to(ROOT))
    try:
        torch.set_num_threads(4)
        data, _, _, _, support = load_sequences(ROOT/protocol['data']['sample_path'], protocol['time']['train_end_exclusive'])
        cutoff = protocol['time']['query_cutoffs']['selection_validation']
        lo = cutoff+protocol['time']['gap_seconds']
        truth = data.truth(lo, lo+protocol['time']['horizon_seconds'])
        ids, x, mask, raw = data.queries(cutoff)
        by_uid = {u['uid']: u for u in data.users}
        full_seen, recent_seen = {}, {}
        popularity = np.zeros(len(data.catalog), dtype=np.float32)
        for u in data.users:
            end = int(np.searchsorted(u['times'], cutoff))
            tok = u['tokens'][:end]
            popularity += np.bincount(tok[tok >= OFFSET]-OFFSET, minlength=len(popularity))
            full_seen[u['uid']] = set(map(int, u['items'][:end]))
            recent_seen[u['uid']] = set(map(int, u['items'][max(0, end-128):end]))
        popularity /= popularity.sum()
        # Evaluate all warm users; activity labels never determine predictions.
        recent, full = [], []
        tie_recs = []
        for uid in ids:
            u = by_uid[uid]
            end = int(np.searchsorted(u['times'], cutoff))
            recent.append(history_distribution(u['tokens'][max(0, end-128):end], len(data.catalog), popularity))
            full.append(history_distribution(u['tokens'][:end], len(data.catalog), popularity))
            counts, last = Counter(), {}
            for token, t in zip(u['tokens'][max(0, end-128):end], u['times'][max(0, end-128):end]):
                if token >= OFFSET:
                    action = int(token)-OFFSET
                    counts[action] += 1
                    last[action] = int(t)
            order = sorted(counts, key=lambda a: (-counts[a], -last[a], int(data.catalog[a])))[:10]
            used = set(order)
            order.extend(a for a in range(10) if a not in used)
            tie_recs.append(data.catalog[order[:10]].tolist())
        recent, full = np.stack(recent), np.stack(full)
        report = {'plan': plan, 'data_support': support, 'methods': {}, 'comparisons': {},
                  'confirmation_scored': False, 'final_test_scored': False}
        rows, rankings = {}, {}

        def evaluate(name, recs):
            overall, _ = ranking_metrics(ids, recs, truth)
            slices, per_user = slice_metrics(ids, recs, truth, full_seen, recent_seen, data.catalog)
            report['methods'][name] = {'overall': overall, **slices}
            rows[name], rankings[name] = per_user, recs
            print(name, overall, flush=True)
            return overall['NDCG@10']

        def ranked(matrix):
            return [probability_topk(p, data.catalog) for p in matrix]

        evaluate('PersonalFrequency128_original', personal_frequency(raw, data.catalog, 10))
        evaluate('PersonalFrequency128_recency_tiebreak', tie_recs)
        best_score, best_lambda = -1, None
        for weight in plan['memory_full_weight_grid']:
            score = evaluate('Memory_full_weight_'+str(weight), ranked((1-weight)*recent+weight*full))
            if score > best_score:
                best_score, best_lambda = score, weight
        memory = (1-best_lambda)*recent+best_lambda*full
        del recent, full
        atomic_json(out/'status.json', {'status': 'running', 'stage': 'gpt2_inference', 'selected_memory_full_weight': best_lambda})
        checkpoint_path = ROOT/plan['checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['protocol_sha256'] != sha or not np.array_equal(checkpoint['catalog'].numpy(), data.catalog):
            raise ValueError('Checkpoint protocol/catalog mismatch')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Policy(**checkpoint['settings']).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        probabilities = np.empty_like(memory)
        with torch.inference_mode():
            for j in range(0, len(ids), 64):
                a = torch.as_tensor(x[j:j+64], device=device)
                b = torch.as_tensor(mask[j:j+64], device=device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device=='cuda'):
                    logits = model(a, b)
                probabilities[j:j+64] = logits.float().softmax(-1).cpu().numpy()
        best_hybrid_score, best_alpha = -1, None
        for alpha in plan['gpt2_weight_grid_after_memory_selection']:
            name = 'Hybrid_GPT2_weight_'+str(alpha)
            score = evaluate(name, ranked((1-alpha)*memory+alpha*probabilities))
            if score > best_hybrid_score:
                best_hybrid_score, best_alpha = score, alpha
        memory_name = 'Memory_full_weight_'+str(best_lambda)
        for name in report['methods']:
            report['comparisons'][name+'_minus_original_frequency'] = paired_bootstrap(rows[name], rows['PersonalFrequency128_original'])
        for alpha in plan['gpt2_weight_grid_after_memory_selection']:
            name = 'Hybrid_GPT2_weight_'+str(alpha)
            report['comparisons'][name+'_minus_selected_memory'] = paired_bootstrap(rows[name], rows[memory_name])
        report.update({'selected_memory_full_weight': best_lambda, 'selected_gpt2_weight': best_alpha,
                       'elapsed_seconds': time.perf_counter()-started,
                       'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                       'finished_utc': datetime.now(timezone.utc).isoformat(),
                       'cautions': ['Selection-only exploratory results, not a confirmed winner.',
                                    'Full-history methods have more context than GPT2 and recent128 baselines.',
                                    'No gate fitted, no SFT/RM/RL update, no next-group tuning.']})
        atomic_json(out/'report.json', report)
        atomic_json(out/'per-user.json', rows)
        atomic_json(out/'rankings.json', {'uids': ids, 'rankings': rankings})
        atomic_json(out/'status.json', {'status': 'completed', 'elapsed_seconds': report['elapsed_seconds'],
                                      'selected_memory_full_weight': best_lambda, 'selected_gpt2_weight': best_alpha})
    except BaseException as exc:
        atomic_json(out/'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
