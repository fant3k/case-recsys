"""Selection-only repeat/novel and fixed recency/transition diagnostics."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .data import OFFSET, load_sequences, personal_frequency
from .metrics import ranking_metrics
from .models import Policy
from .pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def rank_scores(scores, catalog, k=10):
    """Positive candidates first; deterministic train-popularity fallback."""
    order = sorted((int(i) for i, s in scores.items() if s > 0),
                   key=lambda i: (-scores[i], int(catalog[i])))[:k]
    used = set(order)
    order.extend(i for i in range(min(k, len(catalog))) if i not in used)
    return [int(catalog[i]) for i in order[:k]]


def decayed_counts(tokens, times, cutoff, half_life):
    times = np.asarray(times, dtype=np.float64)
    if len(times) and np.any(times >= cutoff):
        raise ValueError('Query history must be strictly earlier')
    # Common factor cancels in ranking; center to avoid stale-history underflow.
    weights = np.exp2((times - times.max())/half_life) if len(times) else []
    result = defaultdict(float)
    for token, weight in zip(tokens, weights):
        if token >= OFFSET:
            result[int(token)-OFFSET] += float(weight)
    return result


def fit_transitions(data, max_gap=1800):
    """One unit of mass per adjacent time-group pair, including OOV mass loss.

    No within-tie ordering; OOV groups do not bridge to later known groups.
    Multiplicity retained via counts, Cartesian pairs normalized by group sizes.
    """
    edges = defaultdict(lambda: defaultdict(float))
    pairs = 0
    for user in data.users:
        for g in range(1, len(user['starts'])):
            a, b = int(user['starts'][g-1]), int(user['starts'][g])
            c = int(user['ends'][g])
            if user['times'][b] >= data.train_end:
                break
            if user['times'][b] - user['times'][a] > max_gap:
                continue
            src, src_n = np.unique(user['tokens'][a:b], return_counts=True)
            dst, dst_n = np.unique(user['tokens'][b:c], return_counts=True)
            for s, sn in zip(src, src_n):
                if s < OFFSET:
                    continue
                row = edges[int(s)-OFFSET]
                for d, dn in zip(dst, dst_n):
                    if d >= OFFSET:
                        row[int(d)-OFFSET] += float(sn*dn)/((b-a)*(c-b))
            pairs += 1
    normalized = {}
    for source, row in edges.items():
        total = sum(row.values())
        normalized[source] = {target: value/total for target, value in row.items()}
    return normalized, {'eligible_adjacent_group_pairs': pairs,
                        'sources': len(edges), 'edges': sum(map(len, edges.values())),
                        'max_gap_seconds': max_gap}


def transition_scores(source_weights, edges, popularity=None):
    scores = defaultdict(float)
    for source, weight in source_weights.items():
        for target, probability in edges.get(source, {}).items():
            scores[target] += weight*probability
    if popularity is not None:
        scores = {target: value/math.sqrt(float(popularity[target]))
                  for target, value in scores.items()}
    return scores


def slice_metrics(uids, recs, truth, full_seen, recent_seen, catalog):
    """DCG contributions use FULL-truth IDCG, so the three pieces add to NDCG."""
    allowed = set(map(int, catalog))
    categories = ('seen_recent128', 'seen_older_only', 'never_seen')
    agg = {c: {'users_with_targets': 0, 'target_pairs': 0, 'catalog_target_pairs': 0,
               'hits': 0, 'macro_recall_when_nonempty': [],
               'ndcg_contribution_all_active_users': 0.0} for c in categories}
    per_user = []
    slots = Counter()
    all_recs = []
    for uid, ranking in zip(uids, recs):
        if uid not in truth:
            continue
        label = lambda i: ('seen_recent128' if i in recent_seen[uid] else
                           'seen_older_only' if i in full_seen[uid] else 'never_seen')
        targets = truth[uid]
        idcg = sum(1/math.log2(j+2) for j in range(min(10, len(targets))))
        row = {'uid': int(uid)}
        for c in categories:
            subset = {i for i in targets if label(i) == c}
            hits = sum(i in subset for i in ranking[:10])
            dcg = sum((i in subset)/math.log2(j+2) for j, i in enumerate(ranking[:10]))
            a = agg[c]
            a['target_pairs'] += len(subset)
            a['catalog_target_pairs'] += len(subset & allowed)
            a['hits'] += hits
            a['ndcg_contribution_all_active_users'] += dcg/idcg
            if subset:
                a['users_with_targets'] += 1
                a['macro_recall_when_nonempty'].append(hits/len(subset))
            row[c+'_ndcg_contribution'] = dcg/idcg
        row['ndcg'] = sum(row[c+'_ndcg_contribution'] for c in categories)
        per_user.append(row)
        slots.update(label(i) for i in ranking[:10])
        all_recs.extend(ranking[:10])
    n = len(per_user)
    for a in agg.values():
        v = a['macro_recall_when_nonempty']
        a['macro_recall_when_nonempty'] = float(np.mean(v)) if v else None
        a['ndcg_contribution_all_active_users'] /= n
        a['pooled_recall'] = a['hits']/a['target_pairs'] if a['target_pairs'] else None
    counts = Counter(all_recs)
    return {'slices': agg, 'recommendation_fraction_on_active_users':
            {c: slots[c]/len(all_recs) for c in categories},
            'unique_recommended_items': len(counts),
            'top_item_slot_fraction': max(counts.values())/len(all_recs)}, per_user


def paired_bootstrap(left, right, draws=2000, seed=42):
    if [r['uid'] for r in left] != [r['uid'] for r in right]:
        raise ValueError('Paired users do not match')
    delta = np.array([a['ndcg']-b['ndcg'] for a, b in zip(left, right)])
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(1)
    return {'mean_ndcg_difference': float(delta.mean()),
            'paired_user_bootstrap_95_percentile_interval': np.quantile(means, [.025, .975]).tolist(),
            'draws': draws, 'seed': seed,
            'caution': 'Exploratory fixed-window interval; no grid/selection multiplicity correction, no training-seed uncertainty.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='artifacts/diagnostics-20261021-01')
    parser.add_argument('--checkpoint', default='artifacts/sft-20261021-01/best.pt')
    args = parser.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    protocol_path = ROOT/'research/protocol/case2-v1.json'
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    plan = {'started_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint': args.checkpoint, 'protocol_sha256': protocol_hash,
            'scope': 'selection_validation only; no confirmation/final scoring',
            'context_events': 128, 'half_lives_seconds': [3600, 86400, 604800],
            'transitions': 'adjacent distinct timestamp groups, train only, max gap1800s; row normalized; recent128 sources weighted with half-life86400s',
            'transition_variants': ['conditional_probability', 'divide_candidate_score_by_sqrt_train_count'],
            'slices': 'seen in recent128 / seen earlier only / never seen in all past; OOV remains in truth',
            'primary': 'unchanged day NDCG@10; next-group recall diagnostic only',
            'selection_warning': 'all new methods exploratory; show full fixed grid, no confirmation tuning'}
    atomic_json(out/'plan.json', plan)
    atomic_json(out/'status.json', {'status': 'running', 'stage': 'loading'})
    try:
        torch.set_num_threads(4)
        data, _, _, _, support = load_sequences(ROOT/protocol['data']['sample_path'],
                                                protocol['time']['train_end_exclusive'])
        checkpoint = torch.load(ROOT/args.checkpoint, map_location='cpu', weights_only=True)
        assert checkpoint['protocol_sha256'] == protocol_hash
        assert np.array_equal(checkpoint['catalog'].numpy(), data.catalog)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Policy(**checkpoint['settings']).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        cutoff = protocol['time']['query_cutoffs']['selection_validation']
        lo = cutoff+protocol['time']['gap_seconds']
        hi = lo+protocol['time']['horizon_seconds']
        ids, x, mask, raw = data.queries(cutoff)
        truth = data.truth(lo, hi)
        next_ids, nx, nm, next_truth = data.next_group_queries(lo, hi)
        full_seen, recent_seen, query_info = {}, {}, {}
        for u in data.users:
            end = int(np.searchsorted(u['times'], cutoff))
            full_seen[u['uid']] = set(map(int, u['items'][:end]))
            recent_seen[u['uid']] = set(map(int, u['items'][max(0, end-128):end]))
            query_info[u['uid']] = (u, end, cutoff)
        next_info = {}
        for uid in next_ids:
            u = query_info[uid][0]
            g = int(np.searchsorted(u['times'][u['starts']], lo))
            end = int(u['starts'][g])
            next_info[uid] = (u, end, int(u['times'][end]))
        popularity = np.zeros(len(data.catalog), dtype=np.float64)
        for u in data.users:
            end = int(np.searchsorted(u['times'], data.train_end))
            tok = u['tokens'][:end]
            popularity += np.bincount(tok[tok >= OFFSET]-OFFSET, minlength=len(popularity))
        atomic_json(out/'status.json', {'status': 'running', 'stage': 'fitting_train_transitions'})
        edges, transition_support = fit_transitions(data)
        print('TRANSITIONS_FIT', transition_support, flush=True)
        atomic_json(out/'status.json', {'status': 'running', 'stage': 'scoring_selection'})

        def predict(xx, mm):
            recs = []
            with torch.inference_mode():
                for i in range(0, len(xx), 64):
                    a = torch.as_tensor(xx[i:i+64], device=device)
                    b = torch.as_tensor(mm[i:i+64], device=device)
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device=='cuda'):
                        logits = model(a, b)
                    recs.extend(data.catalog[logits.topk(10, dim=-1).indices.cpu().numpy()].tolist())
            return recs

        def controls(user_ids, info):
            result = {'MostPop': [data.catalog[:10].tolist() for _ in user_ids]}
            histories = [info[uid][0]['items'][max(0, info[uid][1]-128):info[uid][1]] for uid in user_ids]
            result['PersonalFrequency128'] = personal_frequency(histories, data.catalog, 10)
            for half in plan['half_lives_seconds']:
                recs = []
                for uid in user_ids:
                    u, end, t = info[uid]
                    first = max(0, end-128)
                    scores = decayed_counts(u['tokens'][first:end], u['times'][first:end], t, half)
                    recs.append(rank_scores(scores, data.catalog))
                result['Frequency128_half_life_'+str(half)] = recs
            for corrected in [False, True]:
                recs = []
                for uid in user_ids:
                    u, end, t = info[uid]
                    first = max(0, end-128)
                    weights = decayed_counts(u['tokens'][first:end], u['times'][first:end], t, 86400)
                    scores = transition_scores(weights, edges, popularity if corrected else None)
                    recs.append(rank_scores(scores, data.catalog))
                result['Transitions128'+('_pop_corrected' if corrected else '')] = recs
            return result

        rankings = controls(ids, query_info)
        rankings['GPT2_SFT_best'] = predict(x, mask)
        next_rankings = controls(next_ids, next_info)
        next_rankings['GPT2_SFT_best'] = predict(nx, nm)
        report = {'plan': plan, 'data_support': support, 'transition_support': transition_support,
                  'selection_day': {}, 'next_group': {}, 'comparisons': {},
                  'final_test_scored': False, 'confirmation_scored': False}
        rows = {}
        for name, recs in rankings.items():
            overall, _ = ranking_metrics(ids, recs, truth)
            slices, per_user = slice_metrics(ids, recs, truth, full_seen, recent_seen, data.catalog)
            assert abs(overall['NDCG@10'] - sum(v['ndcg_contribution_all_active_users'] for v in slices['slices'].values())) < 1e-10
            rows[name] = per_user
            report['selection_day'][name] = {'overall': overall, **slices}
            report['next_group'][name] = ranking_metrics(next_ids, next_rankings[name], next_truth)[0]
        for name in rankings:
            if name != 'PersonalFrequency128':
                report['comparisons'][name+'_minus_PersonalFrequency128'] = paired_bootstrap(rows[name], rows['PersonalFrequency128'])
        report['elapsed_seconds'] = time.perf_counter()-start
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out/'report.json', report)
        atomic_json(out/'per-user.json', rows)
        atomic_json(out/'rankings.json', {'uids': ids, 'rankings': rankings,
                                          'next_uids': next_ids, 'next_rankings': next_rankings})
        atomic_json(out/'source-manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), ROOT/'recsys/data.py', ROOT/'recsys/metrics.py', ROOT/'recsys/models.py', protocol_path]})
        atomic_json(out/'status.json', {'status': 'completed', 'elapsed_seconds': report['elapsed_seconds']})
        print(json.dumps({n: v['overall'] for n, v in report['selection_day'].items()}, indent=2), flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
