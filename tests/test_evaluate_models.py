import unittest

from jongmind.dealer import MahjongDealer, Seat
from jongmind.evaluate_models import _resolve_seed, _target_seat


class EvaluateModelsTest(unittest.TestCase):
    def test_target_seat_rotate_cycles_east_south_west_north(self) -> None:
        seats = [_target_seat(index, "rotate") for index in range(8)]

        self.assertEqual(
            seats,
            [
                Seat.EAST,
                Seat.SOUTH,
                Seat.WEST,
                Seat.NORTH,
                Seat.EAST,
                Seat.SOUTH,
                Seat.WEST,
                Seat.NORTH,
            ],
        )

    def test_fixed_target_seat_does_not_rotate(self) -> None:
        self.assertEqual(_target_seat(99, "NORTH"), Seat.NORTH)

    def test_random_target_seat_is_seeded(self) -> None:
        seats = [_target_seat(index, "random", seed=20260508) for index in range(8)]

        self.assertEqual(seats, [_target_seat(index, "random", seed=20260508) for index in range(8)])
        self.assertTrue(all(seat in Seat for seat in seats))

    def test_resolve_seed_accepts_integer_or_random(self) -> None:
        self.assertEqual(_resolve_seed("12345"), 12345)

        random_seed = _resolve_seed("random")

        self.assertIsInstance(random_seed, int)
        self.assertGreaterEqual(random_seed, 0)

    def test_start_hand_preserves_round_metadata_for_rotated_dealer(self) -> None:
        dealer = MahjongDealer(seed=18)
        dealer.state.hand_number = 5
        dealer.state.scores = {
            Seat.EAST: 27_000,
            Seat.SOUTH: 23_000,
            Seat.WEST: 31_000,
            Seat.NORTH: 19_000,
        }
        dealer.state.honba = 2
        dealer.state.riichi_sticks = 1

        dealer.start_hand()

        self.assertEqual(dealer.state.dealer, Seat.SOUTH)
        self.assertEqual(dealer.state.current_turn, Seat.SOUTH)
        self.assertEqual(dealer.state.hand_number, 5)
        self.assertEqual(dealer.state.round_wind, Seat.SOUTH)
        self.assertEqual(dealer.state.honba, 2)
        self.assertEqual(dealer.state.riichi_sticks, 1)
        self.assertEqual(dealer.state.scores[Seat.WEST], 31_000)

    def test_south_round_uses_south_round_wind_and_rotates_dealer(self) -> None:
        expected_dealers = {
            4: Seat.EAST,
            5: Seat.SOUTH,
            6: Seat.WEST,
            7: Seat.NORTH,
        }

        for hand_number, expected_dealer in expected_dealers.items():
            with self.subTest(hand_number=hand_number):
                dealer = MahjongDealer(seed=20 + hand_number)
                dealer.state.hand_number = hand_number
                views = dealer.start_hand()

                self.assertEqual(dealer.state.round_wind, Seat.SOUTH)
                self.assertEqual(dealer.state.dealer, expected_dealer)
                self.assertEqual(dealer.state.current_turn, expected_dealer)
                self.assertEqual(len(views[expected_dealer]["hand"]), 14)
                for seat in Seat:
                    if seat != expected_dealer:
                        self.assertEqual(len(views[seat]["hand"]), 13)

    def test_run_match_stops_after_negative_score_and_ranks_final_scores(self) -> None:
        module = __import__("jongmind.evaluate_models", fromlist=["PlayerSpec", "run_match"])
        original_play_hand = module._play_hand
        played_hands: list[int] = []

        def fake_play_hand(dealer, agents, max_steps_per_hand):
            played_hands.append(dealer.state.hand_number)
            dealer.state.scores = {
                Seat.EAST: 52_000,
                Seat.SOUTH: -1_000,
                Seat.WEST: 25_000,
                Seat.NORTH: 24_000,
            }
            dealer.state.result = {
                "type": "ron",
                "winners": ["EAST"],
                "loser": "SOUTH",
                "hands": {"EAST": {"cost": {"main": 26_000}, "yaku": ["Riichi"]}},
            }

        module._play_hand = fake_play_hand
        try:
            lineup = {
                Seat.EAST: module.PlayerSpec("east", "random", is_target=True),
                Seat.SOUTH: module.PlayerSpec("south", "random"),
                Seat.WEST: module.PlayerSpec("west", "random"),
                Seat.NORTH: module.PlayerSpec("north", "random"),
            }

            record = module.run_match(
                match_index=0,
                seed=20260511,
                match_type="south",
                target_seat=Seat.EAST,
                lineup=lineup,
                max_steps_per_hand=1,
            )
        finally:
            module._play_hand = original_play_hand

        self.assertEqual(played_hands, [0])
        self.assertEqual(record.hands, 1)
        self.assertEqual(record.players["east"].rank, 1)
        self.assertEqual(record.players["south"].rank, 4)
        self.assertEqual(record.players["south"].final_score, -1_000)


if __name__ == "__main__":
    unittest.main()

class EvaluateModelsPlayerSummaryTest(unittest.TestCase):
    def test_opponent_names_are_stable_for_duplicate_models(self) -> None:
        target, opponents = __import__("jongmind.evaluate_models", fromlist=["_build_player_specs"])._build_player_specs(
            "neural_beginner",
            ("random", "random", "random"),
        )

        self.assertEqual(target.name, "neural_beginner")
        self.assertEqual([opponent.name for opponent in opponents], ["random0", "random1", "random2"])

    def test_player_summary_has_no_composite_score(self) -> None:
        module = __import__("jongmind.evaluate_models", fromlist=["EvalTotals", "_summary"])
        totals = module.EvalTotals(player="random0", model="random")
        totals.matches = 1
        totals.hands = 4
        totals.rank_counts[1] = 1
        totals.final_score_sum = 30000
        totals.rank_sum = 1

        summary = module._summary(totals)

        self.assertNotIn("composite_score", summary)
        self.assertEqual(summary["player"], "random0")
        self.assertEqual(summary["model"], "random")
        self.assertIn("zero_win_match_rate", summary)
        self.assertIn("first_without_win_rate", summary)
        self.assertIn("tenpai_draw_rate", summary)
        self.assertIn("riichi_win_rate", summary)
