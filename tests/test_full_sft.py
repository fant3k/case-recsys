import unittest
import numpy as np
import torch
from recsys.data import OFFSET, Sequences
from recsys.full_sft import (chunked_cross_entropy, chunked_weighted_cross_entropy, day_targets, group_starts_per_event,
                             lr_factor, prepare_day_positives, sample_targets)
from torch.nn import functional as F
from recsys.models import Policy
from recsys.multitarget import logits_at_positions


def tiny(context_size=6):
    rows = [{'uid': 1, 'timestamp': [1, 1, 2, 3, 3, 3, 4, 5, 5, 6, 9],
             'item_id': [10, 20, 30, 999, 10, 40, 20, 30, 10, 40, 10], 'played_ratio_pct': [90]*11},
            {'uid': 2, 'timestamp': [1, 2, 2, 4, 7], 'item_id': [40, 30, 20, 10, 30], 'played_ratio_pct': [90]*5}]
    data = Sequences(rows, [10, 20, 30, 40], 7, context_size)
    for user in data.users:
        user['group_start'] = group_starts_per_event(user)
    return data


class FullPositionChecks(unittest.TestCase):
    def test_every_target_sees_only_strictly_earlier_groups(self):
        data = tiny()
        audit = []
        for seed in range(30):
            x, mask, owners, positions, actions = sample_targets(data, np.random.default_rng(seed), 4, True, audit=[])
            log = []
            sample_targets(data, np.random.default_rng(seed), 4, True, audit=log)
            self.assertEqual(len(log), len(actions))
            for (user, start, pos, event), owner, action in zip(log, owners, actions):
                times = user['times']
                # every visible input position is strictly before the target's timestamp
                self.assertLess(times[start + pos], times[event])
                self.assertTrue(np.all(times[start:start + pos + 1] < times[event]))
                # the next position after the source already belongs to the target's group or later
                if start + pos + 1 < len(times):
                    self.assertGreaterEqual(times[start + pos + 1], times[event])
                self.assertEqual(action, user['tokens'][event] - OFFSET)
                self.assertGreaterEqual(user['tokens'][event], OFFSET)  # OOV never a target
                self.assertLess(times[event], data.train_end)
                self.assertEqual(x[owner, pos], user['tokens'][start + pos])
                self.assertEqual(mask[owner, pos], 1)
            audit.extend(log)
        self.assertTrue(audit)

    def test_last_arm_has_one_target_per_context_at_final_position(self):
        data = tiny()
        x, mask, owners, positions, actions = sample_targets(data, np.random.default_rng(3), 5, False)
        self.assertEqual(owners.tolist(), list(range(5)))
        self.assertEqual(positions.tolist(), (mask.sum(1) - 1).tolist())

    def test_full_arm_contains_last_arm_anchor_and_more(self):
        data = tiny(context_size=20)
        _, _, o1, p1, a1 = sample_targets(data, np.random.default_rng(5), 8, False)
        _, _, o2, p2, a2 = sample_targets(data, np.random.default_rng(5), 8, True)
        self.assertGreater(len(a2), len(a1))

    def test_source_logits_ignore_target_group_tokens(self):
        torch.manual_seed(0)
        model = Policy(4, context_size=6, width=16, layers=1, heads=2, dropout=0.1).eval()
        x = torch.tensor([[3, 4, 5, 6, 3, 4]])
        mask = torch.ones_like(x)
        with torch.no_grad():
            a = logits_at_positions(model, x, mask, torch.tensor([0]), torch.tensor([2]))
            b = logits_at_positions(model, torch.tensor([[3, 4, 5, 3, 6, 5]]), mask, torch.tensor([0]), torch.tensor([2]))
        torch.testing.assert_close(a, b)

    def test_dropout_setting_roundtrip_and_schedule(self):
        model = Policy(4, context_size=6, width=16, layers=1, heads=2, dropout=0.1)
        self.assertEqual(Policy(**model.settings).settings, model.settings)
        self.assertNotIn('dropout', Policy(4, context_size=6, width=16, layers=1, heads=2).settings)
        self.assertAlmostEqual(lr_factor(0), 0.1)
        self.assertAlmostEqual(lr_factor(0.03), 1.0)
        self.assertAlmostEqual(lr_factor(1.0), 0.1)

    def test_chunked_loss_matches_plain_loss_and_gradients(self):
        torch.manual_seed(1)
        h = torch.randn(37, 8, requires_grad=True)
        w = torch.randn(11, 8, requires_grad=True)
        a = torch.randint(0, 11, (37,))
        plain = F.cross_entropy(F.linear(h, w), a)
        gh, gw = torch.autograd.grad(plain, [h, w])
        chunked = chunked_cross_entropy(h, w, a, chunk=5)
        ch, cw = torch.autograd.grad(chunked, [h, w])
        torch.testing.assert_close(chunked, plain)
        torch.testing.assert_close(ch, gh)
        torch.testing.assert_close(cw, gw)

    def test_day_targets_are_future_window_positives_inside_train(self):
        times = [1, 1, 2, 3, 3, 5, 6, 8, 9, 9, 12, 14, 20]
        items = [10, 20, 30, 10, 40, 20, 999, 30, 10, 20, 40, 10, 30]
        ratios = [90, 90, 30, 90, 90, 60, 90, 90, 10, 90, 90, 90, 90]
        data = Sequences([{'uid': 1, 'timestamp': times, 'item_id': items, 'played_ratio_pct': ratios}], [10, 20, 30, 40], 15, 6)
        user = data.users[0]
        user['group_start'] = group_starts_per_event(user)
        prepare_day_positives(user, 15)
        gap, horizon = 1, 3
        found = 0
        for seed in range(40):
            windows = []
            sample_targets(data, np.random.default_rng(seed), 3, True, windows=windows)
            log = []
            owners, positions, actions, weights = day_targets(windows, np.random.default_rng(seed), 15, gap, horizon, sources=8, positives=2, audit=log)
            for (u, start, pos, c, chosen) in log:
                i = start + pos
                self.assertEqual(c, u['times'][i + 1])
                self.assertLess(u['times'][i], c)
                self.assertLessEqual(c + gap + horizon, 15)
                self.assertEqual(len(set(chosen.tolist())), len(chosen))
                window = (u['times'] >= c + gap) & (u['times'] < c + gap + horizon) & (u['ratios'] > .5) & (u['tokens'] >= OFFSET)
                self.assertTrue(set(chosen.tolist()) <= set(u['tokens'][window].tolist()))
                found += 1
            for j, pos in set(zip(owners.tolist(), positions.tolist())):
                sel = (owners == j) & (positions == pos)
                self.assertAlmostEqual(float(weights[sel].sum()), 1.0, places=5)
        self.assertGreater(found, 0)

    def test_weighted_chunked_loss_matches_plain(self):
        torch.manual_seed(2)
        h = torch.randn(23, 8, requires_grad=True)
        w = torch.randn(9, 8)
        a = torch.randint(0, 9, (23,))
        wt = torch.rand(23)
        plain = (F.cross_entropy(F.linear(h, w), a, reduction='none') * wt).sum() / wt.sum()
        torch.testing.assert_close(chunked_weighted_cross_entropy(h, w, a, wt, chunk=4), plain)


if __name__ == '__main__':
    unittest.main()


class PpsFeedbackChecks(unittest.TestCase):
    def test_pps_bias_counts_prefix_only_and_matches_paper_form(self):
        from recsys.models import pps_bias
        rows = torch.tensor([[3, 4, 3, 1, 5, 0]])
        bias = pps_bias(rows, torch.tensor([2]), 4, 0.01)[0]
        torch.testing.assert_close(bias, torch.log1p(torch.tensor([2., 1., 0., 0.]) / 0.01))
        counts = torch.tensor([2., 1., 0., 0.])
        paper = torch.log((counts + .01) / (counts.max() + .01))
        logits = torch.randn(4)
        torch.testing.assert_close(torch.softmax(logits + bias, 0), torch.softmax(logits + paper, 0))

    def test_feedback_policy_ignores_future_feedback_and_tokens(self):
        from recsys.models import FeedbackPolicy
        torch.manual_seed(0)
        model = FeedbackPolicy(6, context_size=6, width=16, layers=1, heads=2, pps=True, feedback=True).eval()
        x = torch.tensor([[3, 4, 5, 6, 7, 8]]); mask = torch.ones_like(x)
        r = torch.tensor([[1, 2, 3, 4, 5, 1]]); g = torch.tensor([[22, 1, 2, 3, 4, 5]])
        with torch.no_grad():
            h1 = model.states(x, mask, r, g)[0, 2]
            h2 = model.states(torch.tensor([[3, 4, 5, 8, 8, 8]]), mask, torch.tensor([[1, 2, 3, 1, 1, 1]]), torch.tensor([[22, 1, 2, 9, 9, 9]]))[0, 2]
            short = model(x[:, :3], mask[:, :3], r[:, :3], g[:, :3])
        torch.testing.assert_close(h1, h2)
        self.assertEqual(short.shape, (1, 6))

    def test_feedback_buckets_and_positive_targets(self):
        from recsys.full_sft import pack_side, prepare_feedback
        rows = [{'uid': 1, 'timestamp': [1, 1, 3, 10, 10, 50], 'item_id': [10, 20, 30, 10, 40, 20],
                 'played_ratio_pct': [0, 10, 40, 60, 90, 150]}]
        data = Sequences(rows, [10, 20, 30, 40], 100, 8)
        user = data.users[0]
        user['group_start'] = group_starts_per_event(user)
        prepare_feedback(user)
        self.assertEqual(user['ratio_bucket'].tolist(), [1, 2, 3, 4, 5, 5])
        self.assertEqual(user['gap_bucket'][0], 22)
        self.assertEqual(user['gap_bucket'][1], 1)
        self.assertEqual(user['gap_bucket'][2], 1 + int(np.floor(np.log2(3))))
        for seed in range(10):
            log = []
            sample_targets(data, np.random.default_rng(seed), 3, True, audit=log, positive_only=True)
            for u, start, pos, event in log:
                self.assertGreater(u['ratios'][event], 0.5)
        np.testing.assert_array_equal(pack_side([[1, 2], [3]], 3), [[1, 2, 0], [3, 0, 0]])
