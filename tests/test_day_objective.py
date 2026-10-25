import unittest
import numpy as np
import torch
from recsys.data import Sequences
from recsys.day_objective import daily_targets, DaySampler, joint_loss


class DayObjectiveChecks(unittest.TestCase):
    def test_window_boundaries_duplicates_ratio_and_oov(self):
        row={'uid':1,'timestamp':[1,2,3,4,4,5,6,7], 'item_id':[10,20,10,20,20,999,30,40],
             'played_ratio_pct':[100,100,100,50,100,100,0,100]}
        u=Sequences([row],[10,20,30,40],8,8).users[0]
        labels, raw=daily_targets(u,2,7,gap=1,horizon=4)
        self.assertEqual(labels.tolist(),[0,1])
        self.assertEqual(raw,3)
        with self.assertRaises(ValueError):daily_targets(u,3,7,gap=1,horizon=4)

    def test_sampler_cannot_use_incomplete_train_day(self):
        row={'uid':1,'timestamp':list(range(1,11)), 'item_id':[10]*10,'played_ratio_pct':[100]*10}
        d=Sequences([row],[10],10,8);sampler=DaySampler(d,gap=1,horizon=3)
        batch=sampler.sample(np.random.default_rng(42),100)
        self.assertTrue(np.all(batch[1].sum(1)<=5))
        self.assertEqual(batch[7],100)
        self.assertEqual(batch[8],100)

    def test_joint_loss_unique_day_mean_and_empty_fallback(self):
        logits=torch.tensor([[.3,.5,-.2],[.1,-.1,.2]],requires_grad=True)
        target=torch.tensor([0,1]); owners=torch.tensor([0,0]);day=torch.tensor([1,2])
        actual=joint_loss(logits,target,owners,day,torch.tensor([.5,.5]),torch.tensor([True,False]))
        lp=logits.log_softmax(-1)
        expected=(.5*(-lp[0,0])+.5*(-lp[0,1]-lp[0,2])/2-lp[1,1])/2
        torch.testing.assert_close(actual,expected)
        actual.backward();self.assertTrue(torch.isfinite(logits.grad).all())


if __name__=='__main__':unittest.main()
