import unittest
import numpy as np
import torch
from recsys.data import Sequences
from recsys.models import RewardModel
from recsys.reward_audit import observed_records, calibration, regression


class RewardAuditChecks(unittest.TestCase):
    def test_all_events_once_zero_retained_and_ties_not_history(self):
        row={'uid':1,'timestamp':[1,2,2,3,4], 'item_id':[10,20,20,999,20], 'played_ratio_pct':[100,0,100,100,100]}
        data=Sequences([row],[10,20],3,128)
        records,raw=observed_records(data,2,5)
        self.assertEqual(raw,4);self.assertEqual(len(records),3)
        self.assertEqual([r[4] for r in records],[0,1,1])
        self.assertEqual([r[5] for r in records],['never_seen','never_seen','recent128'])
        self.assertEqual([r[2] for r in records],[1,1,4])
        self.assertEqual(len({(r[0],r[1]) for r in records}),3)

    def test_conditional_mean_calibration_and_user_weighting(self):
        bins=calibration(np.array([.1,.9]),np.array([0.,1.]))
        self.assertAlmostEqual(bins['weighted_absolute_gap'],.1)
        result=regression([0.,0.,0.],[1.,1.,0.],[1,1,2])
        self.assertAlmostEqual(result['MAE_event'],2/3)
        self.assertAlmostEqual(result['MAE_macro_users'],.5)

    def test_fitted_item_only_is_invariant_to_history(self):
        torch.manual_seed(42)
        model=RewardModel(4,np.zeros((4,2),dtype=np.float32),width=8,use_history=False)
        x=torch.tensor([[3,4],[5,6]]);mask=torch.ones_like(x);a=torch.tensor([1,1])
        with torch.no_grad():p=model(x,mask,a)
        torch.testing.assert_close(p[0],p[1])


if __name__=='__main__':unittest.main()
