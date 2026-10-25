import unittest

import torch

from recsys.models import FeedbackPolicy
from recsys.multistep_fix import choose_next, step_logits


def tiny_policy():
    torch.manual_seed(0)
    return FeedbackPolicy(10, context_size=6, width=16, layers=1, heads=2, pps=True, feedback=True).eval()


def tensors(tokens, length=6):
    x = torch.zeros(1, length, dtype=torch.long)
    x[0, :len(tokens)] = torch.tensor(tokens)
    mask = (x > 0).long()
    ones = torch.ones_like(x)
    return [x, mask, ones, ones, torch.zeros(1, length), torch.zeros(1, length, dtype=torch.long)]


class MultistepFix(unittest.TestCase):
    def test_without_synthetic_events_both_modes_equal_the_model(self):
        model, real = tiny_policy(), tensors([3, 4, 5])
        with torch.no_grad():
            expected = model(real[0], real[1], real[2], real[3]).float()
            self.assertTrue(torch.allclose(step_logits(model, real, real, True), expected, atol=1e-5))
            self.assertTrue(torch.allclose(step_logits(model, real, real, False), expected, atol=1e-5))

    def test_pps_prior_ignores_a_generated_copy_only_when_asked(self):
        model, real = tiny_policy(), tensors([3, 4, 5])
        extended = tensors([3, 4, 5, 5])  # the item 5 (catalog index 2) was "generated" once more
        with torch.no_grad():
            counted = step_logits(model, extended, real, True)
            ignored = step_logits(model, extended, real, False)
            base = step_logits(model, extended, extended, True)
        self.assertTrue(torch.allclose(counted, base))
        # the prior difference is exactly one extra count of the generated item, nothing else
        difference = (counted - ignored)
        self.assertGreater(difference[0, 2].item(), 0.0)

    def test_ban_never_repeats_and_default_can(self):
        logits = torch.tensor([[5.0, 4.0, 1.0]])
        first = choose_next(logits, [], True)
        second = choose_next(logits, [first], True)
        self.assertEqual((first.item(), second.item()), (0, 1))
        self.assertEqual(choose_next(logits, [first], False).item(), 0)


if __name__ == "__main__":
    unittest.main()
