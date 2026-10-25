import unittest
from recsys.confirmation import confirmation_decision


class ConfirmationDecisionChecks(unittest.TestCase):
    def test_joint_requires_both_criteria_and_reward_strictly_positive(self):
        val=lambda lo:{'paired_user_bootstrap95':[lo,.1]}
        self.assertTrue(confirmation_decision(val(.001),val(0))['joint_confirmation_pass'])
        self.assertFalse(confirmation_decision(val(0),val(.01))['joint_confirmation_pass'])
        self.assertFalse(confirmation_decision(val(.001),val(-.00001))['joint_confirmation_pass'])


if __name__=='__main__':unittest.main()
