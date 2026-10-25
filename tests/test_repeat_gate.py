import unittest
import torch
from recsys.repeat_gate import RepeatHead
class RepeatGateTests(unittest.TestCase):
    def test_support_normalization_duplicates_empty_allseen(self):
        for pointer in [False,True]:
            head=RepeatHead(4,True,pointer)
            x=torch.tensor([[3,3,4],[2,1,0],[3,4,5]])
            output=head(torch.zeros(3,4),torch.randn(3,3,4),x,torch.ones_like(x),torch.zeros(3,3))
            r,e,g,_,seen=output
            torch.testing.assert_close(head.probabilities(output).sum(1),torch.ones(3))
            self.assertEqual(g[1].item(),0); self.assertEqual(g[2].item(),1)
            self.assertTrue(torch.all((r*e)==0)); self.assertEqual(r[1].sum(),0)
            if not pointer: torch.testing.assert_close(r[0],torch.tensor([2/3,1/3,0]))
            loss=head.loss(output,torch.tensor([0,2,1])); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
    def test_target_outside_context_updates_explore_gate(self):
        head=RepeatHead(4)
        output=head(torch.zeros(1,4),torch.zeros(1,3,4),torch.tensor([[3,3,4]]),torch.ones(1,3),torch.zeros(1,4))
        loss=head.loss(output,torch.tensor([3])); loss.backward()
        self.assertGreater(head.gate.grad.item(),0)
        torch.testing.assert_close(loss,torch.tensor(1.3862944))
if __name__=='__main__': unittest.main()

class ActivationTests(unittest.TestCase):
    def test_strict_past_oov_and_context_cut(self):
        from recsys.actr import activation
        self.assertEqual(activation([3,1,4],[1,2,3],5,recent=2),{1:2**(-.5)})
        with self.assertRaises(ValueError): activation([3],[5],5)
        self.assertAlmostEqual(activation([3,3],[1,4],5)[0],1.5)
