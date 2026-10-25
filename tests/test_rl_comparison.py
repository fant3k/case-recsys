import unittest
import numpy as np
import torch
from recsys.rl_comparison import reward_summary, bootstrap_mean


class RLComparisonChecks(unittest.TestCase):
    def test_full_action_expectation_kl_and_delivered_reward_are_distinct(self):
        p=torch.tensor([[.5,.3,.2],[.2,.2,.6]])
        q=torch.tensor([[.4,.4,.2],[.2,.3,.5]])
        r=torch.tensor([[0.,.5,1.],[1.,.5,0.]])
        alt=1-r; lengths=torch.tensor([.2,.5,1.])
        top=np.array([[0,1],[2,1]])
        result,rows=reward_summary(p,q.log(),r,alt,lengths,top)
        self.assertAlmostEqual(result['expected_primary_completion'],float((p*r).sum(1).mean()))
        self.assertAlmostEqual(result['exact_KL_to_reference'],float((p*(p.log()-q.log())).sum(1).mean()))
        self.assertAlmostEqual(result['top10_primary_completion'],.25)
        self.assertAlmostEqual(result['expected_primary_equivalent_seconds'],600*result['expected_primary_capped_seconds'],places=4)
        self.assertEqual(len(rows['kl']),2)
        self.assertNotAlmostEqual(result['top10_primary_completion'],result['expected_primary_completion'])

    def test_paired_interval_preserves_constant_difference(self):
        result=bootstrap_mean(np.ones(10)*.2)
        np.testing.assert_allclose(result['paired_user_bootstrap95'],[.2,.2])


if __name__=='__main__':unittest.main()
