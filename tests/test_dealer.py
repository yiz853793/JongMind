import json
import tempfile
import unittest
from queue import Empty, Queue
from pathlib import Path

from jongmind.action_space import DRAW_ACTION_ID, action_id_to_human_readable, human_readable_to_action_id
from jongmind.dealer import DealerCommand, MahjongDealer, Phase, Seat
from jongmind.game import Meld
from jongmind.runtime import broadcast_dealer_result
from jongmind.tiles import sort_tiles
from jongmind.tiles import build_wall, dora_from_indicator


class MahjongDealerTest(unittest.TestCase):
    def test_standard_wall_has_136_tiles(self) -> None:
        wall = build_wall()

        self.assertEqual(len(wall), 136)
        self.assertEqual(len(set(wall)), 34)
        self.assertTrue(all(wall.count(tile) == 4 for tile in set(wall)))

    def test_mahjong_soul_wall_has_three_red_fives(self) -> None:
        wall = build_wall(red_fives=True)

        self.assertEqual(len(wall), 136)
        self.assertEqual(wall.count("0m"), 1)
        self.assertEqual(wall.count("0p"), 1)
        self.assertEqual(wall.count("0s"), 1)
        self.assertEqual(wall.count("5m"), 3)
        self.assertEqual(wall.count("5p"), 3)
        self.assertEqual(wall.count("5s"), 3)

    def test_start_hand_deals_east_14_and_others_13(self) -> None:
        dealer = MahjongDealer(seed=1)
        views = dealer.start_hand()

        self.assertEqual(len(views[Seat.EAST]["hand"]), 14)
        for seat in (Seat.SOUTH, Seat.WEST, Seat.NORTH):
            self.assertEqual(len(views[seat]["hand"]), 13)
        self.assertEqual(dealer.state.phase, Phase.DISCARD)
        self.assertEqual(dealer.state.current_turn, Seat.EAST)
        self.assertEqual(len(dealer.state.live_wall), 69)
        self.assertEqual(len(dealer.state.dead_wall), 14)
        self.assertEqual(len(dealer.state.dora_indicators), 1)
        self.assertEqual(views[Seat.EAST]["wall_count"], 69)
        self.assertEqual(views[Seat.EAST]["dead_wall_count"], 14)
        self.assertEqual(views[Seat.EAST]["dora"], [dora_from_indicator(dealer.state.dora_indicators[0])])

    def test_player_state_includes_public_observation_and_action_mask(self) -> None:
        dealer = MahjongDealer(seed=1)
        views = dealer.start_hand()
        east = views[Seat.EAST]
        east_tile = dealer.state.hands[Seat.EAST][0]
        discard_id = human_readable_to_action_id(f"discard_{east_tile}")

        self.assertIn(discard_id, east["legal_action_ids"])
        self.assertEqual(east["legal_action_mask"][discard_id], 1)
        self.assertEqual(action_id_to_human_readable(discard_id), f"discard_{east_tile}")
        self.assertIn("hand", east["obs_public"])
        self.assertIn("discards", east["obs_public"])
        self.assertNotIn("live_wall", east["obs_public"])
        self.assertNotIn("dead_wall", east["obs_public"])
        self.assertNotIn("ura_dora_indicators", east["obs_public"])

        for seat in (Seat.SOUTH, Seat.WEST, Seat.NORTH):
            dealer.state.hands[seat] = []
        dealer.discard(Seat.EAST, east_tile)
        south = dealer.get_state(Seat.SOUTH)

        self.assertIn(DRAW_ACTION_ID, south["legal_action_ids"])
        self.assertEqual(south["legal_action_mask"][DRAW_ACTION_ID], 1)

    def test_hand_history_log_records_replay_decision_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "hands.jsonl"
            dealer = MahjongDealer(seed=2, hand_log_path=log_path)
            dealer.start_hand()

            tile = dealer.state.hands[Seat.EAST][0]
            dealer.discard(Seat.EAST, tile)
            dealer.close_hand_log(reason="test")

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertFalse(record["completed"])
        self.assertEqual(record["close_reason"], "test")
        command_events = [event for event in record["events"] if event["type"] == "command"]
        self.assertEqual(command_events[0]["action_text"], f"discard_{tile}")
        self.assertIn(command_events[0]["action"], command_events[0]["legal_actions"])
        self.assertEqual(command_events[0]["obs_public"]["phase"], "discard")
        self.assertNotIn("live_wall", command_events[0]["obs_public"])
        self.assertIn("live_wall", command_events[0]["hidden_state_for_review_only"])

    def test_start_hand_deals_current_east_wind_player_14_and_first_discard_is_tedashi(self) -> None:
        dealer = MahjongDealer(seed=19)
        dealer.state.hand_number = 1
        views = dealer.start_hand()
        starting_hand = views[Seat.SOUTH]["hand"]

        self.assertEqual(dealer.state.dealer, Seat.SOUTH)
        self.assertEqual(dealer.state.current_turn, Seat.SOUTH)
        self.assertEqual(len(starting_hand), 14)
        for seat in (Seat.EAST, Seat.WEST, Seat.NORTH):
            self.assertEqual(len(views[seat]["hand"]), 13)

        for tile in set(starting_hand):
            dealer = MahjongDealer(seed=19)
            dealer.state.hand_number = 1
            dealer.start_hand()

            result = dealer.discard(Seat.SOUTH, tile)

            self.assertEqual(result["discard_type"], "tedashi")

    def test_discard_then_next_player_draws(self) -> None:
        dealer = MahjongDealer(seed=2)
        dealer.start_hand()

        east_tile = dealer.state.hands[Seat.EAST][0]
        dealer.discard(Seat.EAST, east_tile)
        self.assertEqual(dealer.state.current_turn, Seat.SOUTH)
        self.assertEqual(dealer.state.phase, Phase.DRAW)

        draw_result = dealer.draw(Seat.SOUTH)
        self.assertEqual(draw_result["event"], "draw")
        self.assertEqual(draw_result["actor"], "SOUTH")
        self.assertEqual(dealer.state.phase, Phase.DISCARD)
        self.assertEqual(len(dealer.state.hands[Seat.SOUTH]), 14)

        drawn_tile = draw_result["tile"]
        discard_result = dealer.discard(Seat.SOUTH, drawn_tile)
        self.assertEqual(discard_result["discard_type"], "tsumogiri")

    def test_discard_marks_tedashi_when_not_last_draw(self) -> None:
        dealer = MahjongDealer(seed=15)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: [],
            Seat.SOUTH: sort_tiles(
                ["1m", "2m", "3m", "4m", "5m", "6m", "1p", "2p", "3p", "1s", "2s", "3s", "P"]
            ),
            Seat.WEST: [],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.SOUTH
        dealer.state.phase = Phase.DRAW
        dealer.state.live_wall.append("9s")
        dealer.draw(Seat.SOUTH)

        result = dealer.discard(Seat.SOUTH, "1m")

        self.assertEqual(result["event"], "discard")
        self.assertEqual(result["actor"], "SOUTH")
        self.assertEqual(result["discard_type"], "tedashi")

    def test_chii_call_opens_meld_and_caller_discards_next(self) -> None:
        dealer = MahjongDealer(seed=4)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: ["5m"],
            Seat.SOUTH: ["3m", "4m"],
            Seat.WEST: [],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD

        discard_result = dealer.discard(Seat.EAST, "5m")
        self.assertEqual(discard_result["event"], "discard_reaction")
        self.assertEqual(dealer.state.phase, Phase.REACTION)
        self.assertEqual(dealer.get_state(Seat.SOUTH)["legal_actions"], ["pass", "chii"])

        call_result = dealer.call(Seat.SOUTH, "chii", ("3m", "4m"))
        self.assertEqual(call_result["event"], "call")
        self.assertEqual(dealer.state.current_turn, Seat.SOUTH)
        self.assertEqual(dealer.state.phase, Phase.DISCARD)
        self.assertEqual(dealer.state.discards[Seat.EAST], [])
        self.assertEqual(dealer.state.hands[Seat.SOUTH], [])
        self.assertEqual(dealer.state.melds[Seat.SOUTH][0].kind, "chii")
        self.assertEqual(dealer.state.melds[Seat.SOUTH][0].tiles, ("3m", "4m", "5m"))

    def test_pon_has_priority_over_chii(self) -> None:
        dealer = MahjongDealer(seed=5)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: ["5m"],
            Seat.SOUTH: ["3m", "4m"],
            Seat.WEST: ["5m", "5m"],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD

        dealer.discard(Seat.EAST, "5m")
        self.assertEqual(dealer.get_state(Seat.SOUTH)["legal_actions"], ["pass", "chii"])
        self.assertEqual(dealer.get_state(Seat.WEST)["legal_actions"], ["pass", "pon"])
        self.assertIsNone(dealer.get_state(Seat.EAST)["pending_reaction"])
        self.assertIsNone(dealer.get_state(Seat.NORTH)["pending_reaction"])
        self.assertEqual(dealer.get_state()["pending_reaction"]["waiting"], ["SOUTH", "WEST"])

        waiting_result = dealer.call(Seat.SOUTH, "chii", ("3m", "4m"))
        self.assertEqual(waiting_result["event"], "reaction_waiting")
        self.assertEqual(dealer.state.phase, Phase.REACTION)
        self.assertEqual(dealer.get_state()["pending_reaction"]["responses"], {"SOUTH": "chii"})
        self.assertEqual(dealer.get_state(Seat.SOUTH)["pending_reaction"]["own_response"], "chii")
        self.assertNotIn("responses", dealer.get_state(Seat.WEST)["pending_reaction"])
        self.assertNotIn("waiting", dealer.get_state(Seat.WEST)["pending_reaction"])

        call_result = dealer.call(Seat.WEST, "pon", ("5m", "5m"))
        self.assertEqual(call_result["event"], "call")
        self.assertEqual(dealer.state.current_turn, Seat.WEST)
        self.assertEqual(dealer.state.phase, Phase.DISCARD)
        self.assertEqual(dealer.state.hands[Seat.SOUTH], ["3m", "4m"])
        self.assertEqual(dealer.state.hands[Seat.WEST], [])
        self.assertEqual(dealer.state.melds[Seat.WEST][0].kind, "pon")

    def test_riichi_player_cannot_call_melds(self) -> None:
        dealer = MahjongDealer(seed=41)
        dealer.start_hand()
        dealer.state.riichi_declared[Seat.SOUTH] = True
        dealer.state.hands[Seat.SOUTH] = sort_tiles(
            ["5m", "5m", "1p", "2p", "3p", "4p", "6p", "7p", "8p", "1s", "2s", "3s", "E"]
        )

        dealer.state.riichi_declared[Seat.SOUTH] = False
        callable_pending = dealer._build_pending_reaction(Seat.EAST, "5m")
        self.assertIsNotNone(callable_pending)
        self.assertIn("pon", callable_pending.options[Seat.SOUTH])  # type: ignore[union-attr]

        dealer.state.riichi_declared[Seat.SOUTH] = True
        pending = dealer._build_pending_reaction(Seat.EAST, "5m")
        if pending is not None and Seat.SOUTH in pending.options:
            self.assertNotIn("pon", pending.options[Seat.SOUTH])
            self.assertNotIn("chii", pending.options[Seat.SOUTH])
            self.assertNotIn("kan", pending.options[Seat.SOUTH])

    def test_broadcast_only_publishes_selected_higher_priority_call(self) -> None:
        dealer = MahjongDealer(seed=16)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: ["5m"],
            Seat.SOUTH: ["3m", "4m"],
            Seat.WEST: ["5m", "5m"],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD
        queues = {seat: Queue() for seat in Seat}

        dealer.discard(Seat.EAST, "5m")
        waiting_result = dealer.call(Seat.SOUTH, "chii", ("3m", "4m"))
        broadcast_dealer_result(
            DealerCommand(kind="call", seat=Seat.SOUTH, action="chii", tiles=("3m", "4m")),
            waiting_result,
            dealer,
            queues,
        )

        for queue in queues.values():
            with self.assertRaises(Empty):
                queue.get_nowait()

        call_result = dealer.call(Seat.WEST, "pon", ("5m", "5m"))
        broadcast_dealer_result(
            DealerCommand(kind="call", seat=Seat.WEST, action="pon", tiles=("5m", "5m")),
            call_result,
            dealer,
            queues,
        )

        for queue in queues.values():
            message = queue.get_nowait()
            self.assertEqual(message["event"], "call")
            self.assertEqual(message["actor"], "WEST")
            self.assertEqual(message["public_event"]["action"], "pon")
            self.assertEqual(message["public_event"]["called_tile"], "5m")
            self.assertEqual(message["public_event"]["meld"]["kind"], "pon")
            self.assertEqual(message["public_event"]["meld"]["owner"], "WEST")
            self.assertNotIn("chii", str(message["public_event"]))
            self.assertNotIn("SOUTH", str(message["public_event"]))

    def test_tsumo_scores_and_finishes_hand(self) -> None:
        dealer = MahjongDealer(seed=6)
        dealer.start_hand()
        dealer.state.hands[Seat.EAST] = sort_tiles(
            ["1m", "2m", "3m", "4m", "5m", "6m", "1p", "2p", "3p", "1s", "2s", "3s", "P", "P"]
        )
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD
        dealer.state.last_draw = "6m"
        dealer.state.turn_count = 5

        result = dealer.win(Seat.EAST)

        self.assertEqual(result["event"], "win")
        self.assertEqual(result["actor"], "EAST")
        self.assertEqual(result["winners"], ["EAST"])
        self.assertEqual(result["win_type"], "tsumo")
        self.assertEqual(dealer.state.phase, Phase.FINISHED)
        self.assertEqual(dealer.state.result["type"], "tsumo")
        self.assertIn("Menzen Tsumo", dealer.state.result["hands"]["EAST"]["yaku"])
        self.assertGreater(dealer.state.scores[Seat.EAST], 25_000)

    def test_ron_response_scores_and_finishes_hand(self) -> None:
        dealer = MahjongDealer(seed=7)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: [],
            Seat.SOUTH: sort_tiles(
                ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p"]
            ),
            Seat.WEST: ["5p"],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.WEST
        dealer.state.phase = Phase.DISCARD

        discard_result = dealer.discard(Seat.WEST, "5p")
        self.assertEqual(discard_result["event"], "discard_reaction")
        self.assertIn("ron", dealer.get_state(Seat.SOUTH)["legal_actions"])

        win_result = dealer.win(Seat.SOUTH)

        self.assertEqual(win_result["event"], "win")
        self.assertEqual(win_result["actor"], "SOUTH")
        self.assertEqual(win_result["winners"], ["SOUTH"])
        self.assertEqual(win_result["loser"], "WEST")
        self.assertEqual(win_result["win_type"], "ron")
        self.assertEqual(dealer.state.phase, Phase.FINISHED)
        self.assertEqual(dealer.state.result["type"], "ron")
        self.assertEqual(dealer.state.result["loser"], "WEST")
        self.assertIn("Tanyao", dealer.state.result["hands"]["SOUTH"]["yaku"])

    def test_riichi_discard_places_stick_after_safe_discard(self) -> None:
        dealer = MahjongDealer(seed=8)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: sort_tiles(
                ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p", "9m"]
            ),
            Seat.SOUTH: [],
            Seat.WEST: [],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD

        result = dealer.discard(Seat.EAST, "9m", riichi=True)

        self.assertEqual(result["event"], "discard")
        self.assertTrue(dealer.state.riichi_declared[Seat.EAST])
        self.assertEqual(dealer.state.riichi_sticks, 1)
        self.assertEqual(dealer.state.scores[Seat.EAST], 24_000)

    def test_riichi_player_must_tsumogiri_after_declaration(self) -> None:
        dealer = MahjongDealer(seed=8)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: sort_tiles(
                ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p", "9m"]
            ),
            Seat.SOUTH: [],
            Seat.WEST: [],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD
        dealer.discard(Seat.EAST, "9m", riichi=True)
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DRAW
        dealer.state.live_wall.append("1p")
        dealer.draw(Seat.EAST)

        state = dealer.get_state(Seat.EAST)

        self.assertEqual(state["legal_discard_tiles"], ["1p"])
        self.assertEqual(state["action_hints"]["discard"]["tiles"], ["1p"])
        with self.assertRaisesRegex(ValueError, "must discard the drawn tile"):
            dealer.discard(Seat.EAST, "2m")
        result = dealer.discard(Seat.EAST, "1p")
        self.assertEqual(result["discard_type"], "tsumogiri")

    def test_riichi_is_prompted_with_discard_candidates(self) -> None:
        dealer = MahjongDealer(seed=10)
        dealer.start_hand()
        dealer.state.hands[Seat.EAST] = sort_tiles(
            ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p", "9m"]
        )
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD

        state = dealer.get_state(Seat.EAST)

        self.assertIn("riichi", state["legal_actions"])
        self.assertIn("9m", state["action_hints"]["riichi"]["discard_tiles"])

    def test_reaction_prompt_includes_call_combinations(self) -> None:
        dealer = MahjongDealer(seed=11)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: ["5m"],
            Seat.SOUTH: ["3m", "4m", "5m", "5m", "5m"],
            Seat.WEST: [],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD

        dealer.discard(Seat.EAST, "5m")
        state = dealer.get_state(Seat.SOUTH)

        self.assertEqual(state["legal_actions"], ["pass", "kan", "pon", "chii"])
        self.assertIn(["3m", "4m"], state["action_hints"]["chii"])
        self.assertIn(["5m", "5m"], state["action_hints"]["pon"])
        self.assertIn(["5m", "5m", "5m"], state["action_hints"]["kan"])

    def test_closed_kan_draws_rinshan_and_reveals_kan_dora(self) -> None:
        dealer = MahjongDealer(seed=9)
        dealer.start_hand()
        dealer.state.hands[Seat.EAST] = sort_tiles(
            ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "5p"]
        )
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD

        result = dealer.declare_closed_kan(Seat.EAST, ("1m", "1m", "1m", "1m"))

        self.assertEqual(result["event"], "closed_kan")
        self.assertEqual(result["actor"], "EAST")
        self.assertEqual(result["action"], "closed_kan")
        self.assertEqual(result["kan_tile"], "1m")
        self.assertEqual(result["call"]["kind"], "closed_kan")
        self.assertEqual(result["rinshan_tile"], result["tile"])
        self.assertEqual(dealer.state.kan_count, 1)
        self.assertEqual(len(dealer.state.dora_indicators), 2)
        self.assertEqual(dealer.state.phase, Phase.DISCARD)
        self.assertTrue(dealer.state.last_draw_was_rinshan)
        self.assertEqual(dealer.state.melds[Seat.EAST][0].kind, "closed_kan")

    def test_added_kan_opens_chankan_reaction_window(self) -> None:
        dealer = MahjongDealer(seed=18)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: ["5p"],
            Seat.SOUTH: sort_tiles(
                ["1m", "1m", "1m", "2m", "3m", "4m", "1s", "2s", "3s", "4p", "6p", "P", "P"]
            ),
            Seat.WEST: [],
            Seat.NORTH: [],
        }
        dealer.state.melds[Seat.EAST] = [
            Meld(
                kind="pon",
                owner=Seat.EAST,
                from_seat=Seat.WEST,
                called_tile="5p",
                tiles=("5p", "5p", "5p"),
            )
        ]
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD
        dealer.state.first_turn = {seat: False for seat in Seat}
        dealer.state.any_call_made = True

        result = dealer.declare_added_kan(Seat.EAST, "5p")

        self.assertEqual(result["event"], "added_kan_reaction")
        self.assertEqual(dealer.state.phase, Phase.REACTION)
        self.assertEqual(dealer.state.kan_count, 0)
        self.assertEqual(dealer.state.melds[Seat.EAST][0].kind, "pon")
        self.assertEqual(dealer.get_state(Seat.SOUTH)["legal_actions"], ["pass", "ron"])
        self.assertEqual(dealer.get_state(Seat.SOUTH)["pending_reaction"]["reaction_type"], "added_kan")

        win_result = dealer.win(Seat.SOUTH)

        self.assertEqual(win_result["event"], "win")
        self.assertEqual(win_result["win_type"], "ron")
        self.assertEqual(win_result["loser"], "EAST")
        self.assertIn("Chankan", dealer.state.result["hands"]["SOUTH"]["yaku"])
        self.assertEqual(dealer.state.melds[Seat.EAST][0].kind, "pon")
        self.assertEqual(dealer.state.kan_count, 0)

    def test_added_kan_completes_after_chankan_pass(self) -> None:
        dealer = MahjongDealer(seed=18)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: ["5p"],
            Seat.SOUTH: sort_tiles(
                ["1m", "1m", "1m", "2m", "3m", "4m", "1s", "2s", "3s", "4p", "6p", "P", "P"]
            ),
            Seat.WEST: [],
            Seat.NORTH: [],
        }
        dealer.state.melds[Seat.EAST] = [
            Meld(
                kind="pon",
                owner=Seat.EAST,
                from_seat=Seat.WEST,
                called_tile="5p",
                tiles=("5p", "5p", "5p"),
            )
        ]
        dealer.state.current_turn = Seat.EAST
        dealer.state.phase = Phase.DISCARD
        dealer.state.first_turn = {seat: False for seat in Seat}
        dealer.state.any_call_made = True

        dealer.declare_added_kan(Seat.EAST, "5p")
        result = dealer.pass_reaction(Seat.SOUTH)

        self.assertEqual(result["event"], "added_kan")
        self.assertEqual(result["actor"], "EAST")
        self.assertEqual(result["action"], "added_kan")
        self.assertEqual(result["kan_tile"], "5p")
        self.assertEqual(result["call"]["kind"], "added_kan")
        self.assertEqual(result["rinshan_tile"], result["tile"])
        self.assertEqual(dealer.state.phase, Phase.DISCARD)
        self.assertEqual(dealer.state.current_turn, Seat.EAST)
        self.assertEqual(dealer.state.kan_count, 1)
        self.assertEqual(len(dealer.state.dora_indicators), 2)
        self.assertEqual(dealer.state.hands[Seat.EAST], [result["tile"]])
        self.assertEqual(dealer.state.melds[Seat.EAST][0].kind, "added_kan")

    def test_dealer_ron_score_payment_is_applied(self) -> None:
        dealer = MahjongDealer(seed=12)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: sort_tiles(
                ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p"]
            ),
            Seat.SOUTH: ["5p"],
            Seat.WEST: [],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.SOUTH
        dealer.state.phase = Phase.DISCARD
        dealer.state.turn_count = 5
        dealer.state.first_turn[Seat.EAST] = False
        dealer.state.dora_indicators = ["4p"]

        dealer.discard(Seat.SOUTH, "5p")
        dealer.win(Seat.EAST)

        self.assertEqual(dealer.state.result["hands"]["EAST"]["cost"]["main"], 12_000)
        self.assertEqual(dealer.state.scores[Seat.EAST], 37_000)
        self.assertEqual(dealer.state.scores[Seat.SOUTH], 13_000)
        self.assertEqual(sum(dealer.state.scores.values()), 100_000)

    def test_current_dealer_ron_payment_is_applied_to_rotated_dealer(self) -> None:
        dealer = MahjongDealer(seed=17)
        dealer.start_hand()
        dealer.state.dealer = Seat.SOUTH
        dealer.state.hand_number = 1
        dealer.state.hands = {
            Seat.EAST: [],
            Seat.SOUTH: sort_tiles(
                ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p"]
            ),
            Seat.WEST: ["5p"],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.WEST
        dealer.state.phase = Phase.DISCARD
        dealer.state.turn_count = 5
        dealer.state.first_turn[Seat.SOUTH] = False
        dealer.state.dora_indicators = ["4p"]

        dealer.discard(Seat.WEST, "5p")
        dealer.win(Seat.SOUTH)

        self.assertEqual(dealer.state.result["hands"]["SOUTH"]["cost"]["main"], 12_000)
        self.assertEqual(dealer.state.scores[Seat.SOUTH], 37_000)
        self.assertEqual(dealer.state.scores[Seat.WEST], 13_000)
        self.assertEqual(sum(dealer.state.scores.values()), 100_000)

    def test_child_tsumo_score_payment_is_applied(self) -> None:
        dealer = MahjongDealer(seed=13)
        dealer.start_hand()
        dealer.state.hands[Seat.SOUTH] = sort_tiles(
            ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p", "5p"]
        )
        dealer.state.current_turn = Seat.SOUTH
        dealer.state.phase = Phase.DISCARD
        dealer.state.last_draw = "5p"
        dealer.state.turn_count = 5
        dealer.state.first_turn[Seat.SOUTH] = False
        dealer.state.dora_indicators = ["N"]

        dealer.win(Seat.SOUTH)

        self.assertEqual(dealer.state.result["hands"]["SOUTH"]["cost"]["main"], 3900)
        self.assertEqual(dealer.state.result["hands"]["SOUTH"]["cost"]["additional"], 2000)
        self.assertEqual(dealer.state.scores[Seat.SOUTH], 32_900)
        self.assertEqual(dealer.state.scores[Seat.EAST], 21_100)
        self.assertEqual(dealer.state.scores[Seat.WEST], 23_000)
        self.assertEqual(dealer.state.scores[Seat.NORTH], 23_000)
        self.assertEqual(sum(dealer.state.scores.values()), 100_000)

    def test_honba_bonus_is_applied_to_tsumo_payments(self) -> None:
        dealer = MahjongDealer(seed=13)
        dealer.start_hand()
        dealer.state.honba = 2
        dealer.state.hands[Seat.SOUTH] = sort_tiles(
            ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p", "5p"]
        )
        dealer.state.current_turn = Seat.SOUTH
        dealer.state.phase = Phase.DISCARD
        dealer.state.last_draw = "5p"
        dealer.state.turn_count = 5
        dealer.state.first_turn[Seat.SOUTH] = False
        dealer.state.dora_indicators = ["N"]

        dealer.win(Seat.SOUTH)

        self.assertEqual(dealer.state.result["hands"]["SOUTH"]["cost"]["main_bonus"], 200)
        self.assertEqual(dealer.state.result["hands"]["SOUTH"]["cost"]["additional_bonus"], 200)
        self.assertEqual(dealer.state.scores[Seat.SOUTH], 33_500)
        self.assertEqual(dealer.state.scores[Seat.EAST], 20_900)
        self.assertEqual(dealer.state.scores[Seat.WEST], 22_800)
        self.assertEqual(dealer.state.scores[Seat.NORTH], 22_800)
        self.assertEqual(sum(dealer.state.scores.values()), 100_000)

    def test_honba_bonus_is_applied_to_ron_payment(self) -> None:
        dealer = MahjongDealer(seed=7)
        dealer.start_hand()
        dealer.state.honba = 1
        dealer.state.hands = {
            Seat.EAST: [],
            Seat.SOUTH: sort_tiles(
                ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p"]
            ),
            Seat.WEST: ["5p"],
            Seat.NORTH: [],
        }
        dealer.state.current_turn = Seat.WEST
        dealer.state.phase = Phase.DISCARD
        dealer.state.turn_count = 5
        dealer.state.first_turn[Seat.SOUTH] = False
        dealer.state.dora_indicators = ["N"]

        dealer.discard(Seat.WEST, "5p")
        dealer.win(Seat.SOUTH)

        self.assertEqual(dealer.state.result["hands"]["SOUTH"]["cost"]["main_bonus"], 300)
        self.assertEqual(dealer.state.scores[Seat.SOUTH], 30_500)
        self.assertEqual(dealer.state.scores[Seat.WEST], 19_500)
        self.assertEqual(sum(dealer.state.scores.values()), 100_000)

    def test_riichi_stick_is_awarded_to_tsumo_winner(self) -> None:
        dealer = MahjongDealer(seed=13)
        dealer.start_hand()
        dealer.state.riichi_sticks = 1
        dealer.state.scores[Seat.EAST] -= 1000
        dealer.state.hands[Seat.SOUTH] = sort_tiles(
            ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p", "5p"]
        )
        dealer.state.current_turn = Seat.SOUTH
        dealer.state.phase = Phase.DISCARD
        dealer.state.last_draw = "5p"
        dealer.state.turn_count = 5
        dealer.state.first_turn[Seat.SOUTH] = False
        dealer.state.dora_indicators = ["N"]

        dealer.win(Seat.SOUTH)

        self.assertEqual(dealer.state.result["hands"]["SOUTH"]["cost"]["kyoutaku_bonus"], 1000)
        self.assertEqual(dealer.state.scores[Seat.SOUTH], 33_900)
        self.assertEqual(dealer.state.riichi_sticks, 0)
        self.assertEqual(sum(dealer.state.scores.values()), 100_000)

    def test_exhaustive_draw_tenpai_payment_is_applied(self) -> None:
        dealer = MahjongDealer(seed=14)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: sort_tiles(
                ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p"]
            ),
            Seat.SOUTH: ["1m", "1m", "2m", "2m", "4m", "4m", "7p", "8p", "1s", "3s", "5s", "E", "C"],
            Seat.WEST: ["1m", "2m", "4m", "5m", "8m", "9m", "2p", "4p", "6p", "8p", "1s", "5s", "C"],
            Seat.NORTH: ["1m", "1m", "2m", "2m", "4m", "4m", "7p", "8p", "1s", "3s", "5s", "E", "C"],
        }
        dealer.state.live_wall = []

        dealer._finish_exhaustive_draw()

        self.assertEqual(dealer.state.result["tenpai"], ["EAST"])
        self.assertEqual(dealer.state.scores[Seat.EAST], 28_000)
        self.assertEqual(dealer.state.scores[Seat.SOUTH], 24_000)
        self.assertEqual(dealer.state.scores[Seat.WEST], 24_000)
        self.assertEqual(dealer.state.scores[Seat.NORTH], 24_000)
        self.assertEqual(sum(dealer.state.scores.values()), 100_000)

    def test_exhaustive_draw_tenpai_payment_counts_open_hands(self) -> None:
        dealer = MahjongDealer(seed=14)
        dealer.start_hand()
        dealer.state.hands = {
            Seat.EAST: sort_tiles(["1m", "1m", "2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s"]),
            Seat.SOUTH: ["1m", "1m", "2m", "2m", "4m", "4m", "7p", "8p", "1s", "3s", "5s", "E", "C"],
            Seat.WEST: ["1m", "2m", "4m", "5m", "8m", "9m", "2p", "4p", "6p", "8p", "1s", "5s", "C"],
            Seat.NORTH: ["1m", "1m", "2m", "2m", "4m", "4m", "7p", "8p", "1s", "3s", "5s", "E", "C"],
        }
        dealer.state.melds[Seat.EAST] = [
            Meld(
                kind="pon",
                owner=Seat.EAST,
                from_seat=Seat.SOUTH,
                called_tile="P",
                tiles=("P", "P", "P"),
            )
        ]
        dealer.state.live_wall = []

        dealer._finish_exhaustive_draw()

        self.assertEqual(dealer.state.result["tenpai"], ["EAST"])
        self.assertEqual(dealer.state.scores[Seat.EAST], 28_000)
        self.assertEqual(dealer.state.scores[Seat.SOUTH], 24_000)
        self.assertEqual(dealer.state.scores[Seat.WEST], 24_000)
        self.assertEqual(dealer.state.scores[Seat.NORTH], 24_000)
        self.assertEqual(sum(dealer.state.scores.values()), 100_000)

    def test_cannot_discard_tile_not_in_hand(self) -> None:
        dealer = MahjongDealer(seed=3)
        dealer.start_hand()

        with self.assertRaises(ValueError):
            dealer.discard(Seat.EAST, "missing")


if __name__ == "__main__":
    unittest.main()
