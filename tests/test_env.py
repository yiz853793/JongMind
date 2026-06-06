import unittest
from random import Random

from jongmind.env import RiichiEnv


class RiichiEnvTest(unittest.TestCase):
    def test_step_rejects_action_outside_legal_mask(self) -> None:
        env = RiichiEnv(seed=1)
        obs = env.reset()

        self.assertIn("legal_action_mask", obs)
        self.assertNotIn("hidden_state_for_review_only", obs)
        illegal_action = next(index for index, allowed in enumerate(obs["legal_action_mask"]) if not allowed)

        with self.assertRaises(ValueError):
            env.step(illegal_action)

    def test_random_legal_actions_run_1000_hands_without_crashing(self) -> None:
        random = Random(20260607)

        for hand_index in range(1000):
            env = RiichiEnv(seed=20260607 + hand_index)
            obs = env.reset()
            env.dealer.state.live_wall = []
            done = False
            steps = 0
            while not done:
                legal_ids = [
                    action_id
                    for action_id, allowed in enumerate(env.legal_actions())
                    if allowed
                ]
                self.assertTrue(legal_ids)
                obs, _reward, done, info = env.step(random.choice(legal_ids))
                self.assertNotIn("hidden_state_for_review_only", obs)
                self.assertIn("hidden_state_for_review_only", info)
                steps += 1
                self.assertLess(steps, 1000)

            replay = env.get_replay()
            self.assertTrue(replay["events"])
            self.assertTrue(replay["events"][-1]["done"])


if __name__ == "__main__":
    unittest.main()
