import unittest

from jongmind.game import Seat
from jongmind.rules import MAHJONG_SOUL_4P_RANKED
from jongmind.scoring import MahjongSoulScoring
from jongmind.tiles import sort_tiles
from jongmind_ai.rewards import RewardWeights, analyze_hand, match_terminal_reward, shape_potential_reward, terminal_reward


class RewardTest(unittest.TestCase):
    def test_match_terminal_reward_uses_rank_and_final_score(self) -> None:
        scores = {
            "EAST": 40_000,
            "SOUTH": 30_000,
            "WEST": 20_000,
            "NORTH": 10_000,
        }

        first = match_terminal_reward(scores, Seat.EAST)
        second = match_terminal_reward(scores, Seat.SOUTH)
        third = match_terminal_reward(scores, Seat.WEST)
        fourth = match_terminal_reward(scores, Seat.NORTH)

        self.assertGreater(first.total, second.total)
        self.assertGreater(second.total, 0)
        self.assertLess(third.total, 0)
        self.assertLess(fourth.total, third.total)
        self.assertEqual(first.components["placement"], 3.0)
        self.assertAlmostEqual(first.components["match_score_delta"], 1.5)

    def test_terminal_reward_penalizes_deal_in(self) -> None:
        result = {
            "type": "ron",
            "winners": ["SOUTH"],
            "loser": "EAST",
            "hands": {"SOUTH": {"cost": {"main": 8000}}},
        }

        reward = terminal_reward(result, "EAST", score_delta=-8000)

        self.assertLess(reward.total, 0)
        self.assertIn("deal_in", reward.components)

    def test_terminal_reward_scales_from_tenpai_to_win_to_points(self) -> None:
        tenpai = terminal_reward(
            {"type": "draw", "reason": "exhaustive", "tenpai": ["EAST"]},
            "EAST",
            score_delta=0,
        )
        cheap_win = terminal_reward(
            {
                "type": "ron",
                "winners": ["EAST"],
                "loser": "SOUTH",
                "hands": {"EAST": {"cost": {"main": 1000}}},
            },
            "EAST",
            score_delta=1000,
        )
        mangan_win = terminal_reward(
            {
                "type": "ron",
                "winners": ["EAST"],
                "loser": "SOUTH",
                "hands": {"EAST": {"cost": {"main": 8000}}},
            },
            "EAST",
            score_delta=8000,
        )

        self.assertGreater(cheap_win.total, tenpai.total)
        self.assertGreater(mangan_win.total, cheap_win.total)
        self.assertEqual(RewardWeights().win, 8.0)
        self.assertEqual(RewardWeights().exhaustive_draw_tenpai, 2.0)

    def test_shape_reward_penalizes_open_tenpai_without_yaku(self) -> None:
        scoring = MahjongSoulScoring(MAHJONG_SOUL_4P_RANKED)
        hand = sort_tiles(["2p", "3p", "4p", "2s", "3s", "4s", "5m", "5m", "6m", "7m"])
        state = {
            "dealer": "EAST",
            "round_wind": "EAST",
            "dora_indicators": [],
            "discards": {},
            "melds": {
                "EAST": [
                    {
                        "kind": "chii",
                        "tiles": ["1m", "2m", "3m"],
                        "called_tile": "1m",
                        "open": True,
                    }
                ]
            },
        }

        reward = shape_potential_reward(analyze_hand(hand, state, "EAST", scoring))

        self.assertLess(reward.total, 0)
        self.assertLess(reward.components["no_yaku_tenpai"], 0)

    def test_yakuhai_open_tenpai_is_not_no_yaku(self) -> None:
        scoring = MahjongSoulScoring(MAHJONG_SOUL_4P_RANKED)
        hand = sort_tiles(["2p", "3p", "4p", "2s", "3s", "4s", "5m", "5m", "6m", "7m"])
        state = {
            "dealer": "EAST",
            "round_wind": "EAST",
            "dora_indicators": [],
            "discards": {},
            "melds": {
                "EAST": [
                    {
                        "kind": "pon",
                        "tiles": ["P", "P", "P"],
                        "called_tile": "P",
                        "open": True,
                    }
                ]
            },
        }

        reward = shape_potential_reward(analyze_hand(hand, state, "EAST", scoring))

        self.assertEqual(reward.components["no_yaku_tenpai"], 0.0)
        self.assertGreater(reward.components["expected_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
