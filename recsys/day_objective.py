"""Equal-time next-item versus next-item plus train-contained daily positives."""
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
from .models import Policy
from .pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def daily_targets(user, cutoff, train_end, gap=1800, horizon=86400):
    if cutoff+gap+horizon > train_end:
        raise ValueError('Day labels would cross the train boundary')
    lo, hi = np.searchsorted(user['times'], [cutoff+gap, cutoff+gap+horizon])
    positive = user['ratios'][lo:hi] > .5
    raw = np.unique(user['items'][lo:hi][positive])
    tokens = user['tokens'][lo:hi][positive]
    known = np.unique(tokens[tokens >= OFFSET]-OFFSET)
    return known, len(raw)


class DaySampler:
    def __init__(self, data, gap=1800, horizon=86400):
        self.data, self.gap, self.horizon = data, gap, horizon
        self.allowed = {}
        for j, user in enumerate(data.users):
            groups = user['eligible']
            groups = groups[user['times'][user['starts'][groups]]+gap+horizon <= data.train_end]
            if len(groups):
                self.allowed[j] = groups
        self.users = list(self.allowed)
        if not self.users:
            raise ValueError('No train-contained day windows')

    def sample(self, rng, size):
        from .data import pack_contexts
        contexts, actions, owner, day_actions, weights = [], [], [], [], []
        raw_count, known_count, nonempty = 0, 0, []
        for row in range(size):
            j = int(rng.choice(self.users))
            user = self.data.users[j]
            group = int(rng.choice(self.allowed[j]))
            event = int(user['known'][rng.integers(user['first'][group], user['last'][group])])
            end = int(user['starts'][group])
            cutoff = int(user['times'][end])
            assert user['times'][end-1] < cutoff
            contexts.append(self.data.context(user, end))
            actions.append(int(user['tokens'][event])-OFFSET)
            labels, raw = daily_targets(user, cutoff, self.data.train_end, self.gap, self.horizon)
            raw_count += raw
            known_count += len(labels)
            nonempty.append(bool(len(labels)))
            if len(labels):
                owner.extend([row]*len(labels)); day_actions.extend(labels.tolist())
                weights.extend([1/len(labels)]*len(labels))
        x, mask = pack_contexts(contexts, self.data.context_size)
        return x, mask, np.array(actions), np.array(owner, dtype=np.int64), np.array(day_actions, dtype=np.int64), np.array(weights, dtype=np.float32), np.array(nonempty), raw_count, known_count


def joint_loss(logits, next_actions, owners, day_actions, day_weights, has_day):
    next_loss = F.cross_entropy(logits, next_actions, reduction='none')
    day_loss = torch.zeros_like(next_loss)
    if len(owners):
        logp = logits.log_softmax(-1)
        day_loss = day_loss.scatter_add(0, owners, -logp[owners, day_actions]*day_weights)
    return torch.where(has_day, .5*next_loss+.5*day_loss, next_loss).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='artifacts/day-objective-20261021-01')
    parser.add_argument('--seconds-per-arm', type=float, default=90)
    args = parser.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT/'research/protocol/case2-v1.json'
    proto = json.loads(proto_path.read_text())
    proto_hash = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    checkpoint_path = ROOT/'artifacts/sft-20261021-01/best.pt'
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': proto_hash,
            'checkpoint': str(checkpoint_path.relative_to(ROOT)),
            'arms_in_order': ['next_item_control', 'next_item_plus_day'], 'seconds_training_per_arm': args.seconds_per_arm,
            'training_timer': 'sampling/target construction/forward/backward/update/CUDA sync only; evaluation/save excluded; last step may exceed budget',
            'batch': 64, 'day_loss_weight': .5, 'lr': 3e-5, 'weight_decay': .01,
            'optimizer': 'AdamW restored separately from same checkpoint; clip1; AMP fp16, fp32 CE; GradScaler restored',
            'sampling_seed': 20261027,
            'objective': 'next CE control versus .5 next CE + .5 mean CE of unique observed played>50% day items. Same logits. If no known day positives: next CE only; no fabricated dislikes',
            'mask': 'causal prefix ends before entire target group; no invented within-tie next item; past OOV retained',
            'evaluation': 'initial, half and full time budgets; best day NDCG per arm earliest on ties. Auxiliary next-group Recall not selection.',
            'interpretation': 'equal-time warm-start objective ablation; both arms anchor only where query+gap+day<=train_end. Earlier pretrained checkpoint saw all train; no clean inner holdout claimed',
            'scope': 'selection only; no confirmation/final, no RM/RL training, no feature bias'}
    atomic_json(out/'plan.json', plan)
    atomic_json(out/'status.json', {'status': 'running', 'stage': 'loading'})
    with zipfile.ZipFile(out/'code-snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ['recsys', 'tests', 'research/protocol']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:
                    z.write(p, p.relative_to(ROOT))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.set_num_threads(4)
        data, _, _, _, support = load_sequences(ROOT/proto['data']['sample_path'], proto['time']['train_end_exclusive'])
        sampler = DaySampler(data, proto['time']['gap_seconds'], proto['time']['horizon_seconds'])
        support['train_contained_day_users'] = len(sampler.users)
        support['train_contained_day_anchor_groups'] = sum(map(len, sampler.allowed.values()))
        cutoff = proto['time']['query_cutoffs']['selection_validation']
        lo = cutoff+proto['time']['gap_seconds']
        hi = lo+proto['time']['horizon_seconds']
        ids, qx, qm, _ = data.queries(cutoff)
        truth = data.truth(lo, hi)
        nids, nx, nm, ntruth = data.next_group_queries(lo, hi)
        full_seen, recent_seen = {}, {}
        for user in data.users:
            end = int(np.searchsorted(user['times'], cutoff))
            full_seen[user['uid']] = set(map(int, user['items'][:end]))
            recent_seen[user['uid']] = set(map(int, user['items'][max(0, end-128):end]))
        report = {'plan': plan, 'data_support': support, 'arms': {},
                  'confirmation_scored': False, 'final_test_scored': False}
        selected_rows, selected_recs, initial_rows = {}, {}, None
        for arm in plan['arms_in_order']:
            arm_dir = out/arm
            arm_dir.mkdir()
            torch.manual_seed(plan['sampling_seed'])
            ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if ck['protocol_sha256'] != proto_hash or not np.array_equal(ck['catalog'].numpy(), data.catalog):
                raise ValueError('Checkpoint catalog/protocol mismatch')
            initial_step = int(ck.get('step', 0))
            initial_targets = int(ck.get('sampled_target_examples', 0))
            model = Policy(**ck['settings']).cuda()
            model.load_state_dict(ck['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=.01)
            optimizer.load_state_dict(ck['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = plan['lr']
            scaler = torch.amp.GradScaler('cuda')
            scaler.load_state_dict(ck['scaler'])
            del ck
            rng = np.random.default_rng(plan['sampling_seed'])
            history, trace = [], []
            best_score, best_info = -1, None
            elapsed, step, skipped, contexts, targets, prefix_sum = 0., 0, 0, 0, 0, 0
            raw_day, known_day, day_contexts = 0, 0, 0

            @torch.no_grad()
            def predict(x, mask):
                recs = []
                for j in range(0, len(x), 64):
                    a = torch.as_tensor(x[j:j+64], device='cuda')
                    b = torch.as_tensor(mask[j:j+64], device='cuda')
                    with torch.autocast('cuda', dtype=torch.float16):
                        values = model(a, b).float()
                    recs.extend(probability_topk(p, data.catalog) for p in values.cpu().numpy())
                return recs

            def evaluate():
                nonlocal best_score, best_info, initial_rows
                model.eval()
                recs = predict(qx, qm)
                overall, _ = ranking_metrics(ids, recs, truth)
                slices, per_user = slice_metrics(ids, recs, truth, full_seen, recent_seen, data.catalog)
                next_metrics = ranking_metrics(nids, predict(nx, nm), ntruth)[0]
                info = {'step': step, 'training_seconds': elapsed, 'sampled_anchor_contexts': contexts,
                        'target_losses': targets, 'optimizer_updates_skipped': skipped,
                        'ranking': overall, 'next_group': next_metrics, **slices}
                history.append(info)
                if step == 0:
                    if initial_rows is None:
                        initial_rows = per_user
                    elif initial_rows != per_user:
                        raise ValueError('Arms have different initial predictions')
                if overall['NDCG@10'] > best_score:
                    best_score, best_info = overall['NDCG@10'], info
                    selected_rows[arm], selected_recs[arm] = per_user, recs
                    torch.save({'model': model.state_dict(), 'settings': model.settings,
                                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                                'catalog': torch.from_numpy(data.catalog), 'protocol_sha256': proto_hash,
                                'step': initial_step+step, 'new_steps': step, 'arm': arm, 'training_seconds': elapsed,
                                'sampled_target_examples': initial_targets+targets,
                                'sampler_rng': rng.bit_generator.state, 
                                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, arm_dir/'best.pt')
                atomic_json(arm_dir/'evaluations.json', history)
                print('EVAL', arm, step, round(elapsed, 2), overall, flush=True)
                model.train()

            evaluate()
            torch.cuda.reset_peak_memory_stats()
            evaluated_half = False
            while elapsed < args.seconds_per_arm and step < 5000:
                torch.cuda.synchronize()
                t = time.perf_counter()
                batch = sampler.sample(rng, 64)
                x, mask, actions = [torch.as_tensor(a, device='cuda') for a in batch[:3]]
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x, mask).float()
                    if arm == 'next_item_plus_day':
                        owners, day_actions, weights, has_day = [torch.as_tensor(a, device='cuda') for a in batch[3:7]]
                        loss = joint_loss(logits, actions, owners, day_actions, weights, has_day)
                    else:
                        loss = F.cross_entropy(logits, actions)
                raw_day += batch[7]; known_day += batch[8]; day_contexts += int(batch[6].sum())
                current_targets = 64+(batch[8] if arm == 'next_item_plus_day' else 0)
                current_prefix_sum = int(batch[1].sum())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < previous_scale)
                torch.cuda.synchronize()
                elapsed += time.perf_counter()-t
                step += 1
                contexts += 64
                targets += current_targets
                prefix_sum += current_prefix_sum
                if step % 100 == 0:
                    trace.append({'step': step, 'training_seconds': elapsed, 'last_batch_loss': loss.item(),
                                  'anchor_contexts': contexts, 'target_losses': targets})
                    atomic_json(out/'status.json', {'status': 'running', 'arm': arm, 'step': step,
                                                   'training_seconds': elapsed, 'budget_per_arm': args.seconds_per_arm})
                if elapsed >= args.seconds_per_arm/2 and not evaluated_half:
                    evaluate()
                    evaluated_half = True
            if history[-1]['step'] != step:
                evaluate()
            torch.save({'model': model.state_dict(), 'settings': model.settings,
                        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                        'catalog': torch.from_numpy(data.catalog), 'protocol_sha256': proto_hash,
                        'step': initial_step+step, 'new_steps': step, 'arm': arm,
                        'sampled_target_examples': initial_targets+targets, 'sampler_rng': rng.bit_generator.state,
                        
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, arm_dir/'last.pt')
            report['arms'][arm] = {'evaluations': history, 'best': best_info, 'steps': step,
                                   'optimizer_updates_skipped': skipped, 'training_seconds': elapsed,
                                   'sampled_anchor_contexts': contexts, 'target_losses': targets,
                                   'average_context_length': prefix_sum/contexts,
                                   'raw_positive_day_pairs_sampled': raw_day, 'known_positive_day_pairs_sampled': known_day,
                                   'contexts_with_known_day_positives': day_contexts,
                                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                   'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            atomic_json(arm_dir/'trace.json', trace)
            del loss, optimizer, scaler, model, x, mask, actions
            del logits
            torch.cuda.empty_cache()
        report['comparisons'] = {
            'selected_joint_minus_selected_control': paired_bootstrap(selected_rows['next_item_plus_day'], selected_rows['next_item_control']),
            'selected_control_minus_initial': paired_bootstrap(selected_rows['next_item_control'], initial_rows),
            'selected_joint_minus_initial': paired_bootstrap(selected_rows['next_item_plus_day'], initial_rows)}
        cached = json.loads((ROOT/'artifacts/hybrid-20261021-01/rankings.json').read_text())
        if cached['uids'] != ids:
            raise ValueError('Cached controls query mismatch')
        report['controls'] = {}
        for name in ['PersonalFrequency128_original', 'PersonalFrequency128_recency_tiebreak', 'Memory_full_weight_0.5']:
            report['controls'][name] = ranking_metrics(ids, cached['rankings'][name], truth)[0]
        report.update({'elapsed_wall_seconds': time.perf_counter()-started,
                       'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                       'cautions': ['Equal training time, potentially different updates; both arms use train-contained day anchors, changing original anchor support. Shared logits not an added decoder head.',
                                    'Selection checkpoint/one sampling seed, bootstrap uncorrected for selection and repeated exploration.',
                                    'No claim of final winner, causal engagement or retention.']})
        atomic_json(out/'report.json', report)
        atomic_json(out/'rankings.json', {'uids': ids, 'selected_rankings': selected_recs})
        atomic_json(out/'per-user.json', selected_rows)
        atomic_json(out/'status.json', {'status': 'completed', 'elapsed_wall_seconds': report['elapsed_wall_seconds']})
    except BaseException as exc:
        atomic_json(out/'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
