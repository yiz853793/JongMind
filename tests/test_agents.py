import unittest

import torch

from jongmind_ai import (
    AkochanAgent,
    MahjongAIHeuristicAgent,
    MjaiManueAgent,
    OpenCallAgent,
    ShantenAgent,
    TileEfficiencyAgent,
    create_model,
    model_names,
)
from jongmind_ai.base import forced_tsumogiri_tile, legal_discard_tiles
from jongmind_ai.features import (
    FEATURE_SIZE,
    REACTION_ACTIONS,
    REACTION_FEATURE_SIZE,
    TILE_TYPES,
    legal_discard_mask,
    legal_reaction_mask,
)
from jongmind_ai.neural_beginner import CallNet, DiscardNet, NeuralBeginnerAgent
from jongmind.tiles import sort_tiles


class TileEfficiencyAgentTest(unittest.TestCase):
    def test_discards_isolated_tile_when_it_preserves_tenpai(self) -> None:
        agent = TileEfficiencyAgent()
        hand = sort_tiles(
            ["2m", "3m", "4m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5p", "2s", "3s", "4s"]
        )

        decision = agent.choose_discard(hand, {"discards": {}, "melds": {}, "dora": []})

        self.assertEqual(decision.tile, "9m")
        self.assertEqual(decision.shanten, 0)
        self.assertGreater(decision.ukeire, 0)

    def test_tile_efficiency_scores_all_legal_discards(self) -> None:
        agent = TileEfficiencyAgent()
        hand = sort_tiles(
            ["2m", "3m", "4m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5p", "2s", "3s", "4s"]
        )

        scores = agent.score_discards(hand, {"discards": {}, "melds": {}, "dora": []})
        decision = agent.choose_discard(hand, {"discards": {}, "melds": {}, "dora": []})
        best_tile = max(scores.values(), key=lambda item: item.score).tile

        self.assertEqual(set(scores), set(hand))
        self.assertEqual(best_tile, decision.tile)

    def test_shanten_agent_discards_to_preserve_tenpai(self) -> None:
        agent = ShantenAgent()
        hand = sort_tiles(
            ["2m", "3m", "4m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5p", "2s", "3s", "4s"]
        )

        decision = agent.choose_discard(hand, {"discards": {}, "melds": {}, "dora": []})

        self.assertEqual(decision.shanten, 0)

    def test_open_call_agent_uses_first_available_call(self) -> None:
        agent = OpenCallAgent()

        decision = agent.choose_reaction(
            {
                "action_hints": {
                    "discarder": "EAST",
                    "tile": "5m",
                    "pon": [["5m", "5m"]],
                    "chii": [["3m", "4m"]],
                }
            }
        )

        self.assertEqual(decision.action, "pon")
        self.assertEqual(decision.tiles, ("5m", "5m"))

    def test_tile_efficiency_call_agent_calls_tanyao_chii(self) -> None:
        agent = TileEfficiencyAgent(allow_calls=True, call_min_score_delta=-100.0)

        decision = agent.choose_reaction(
            {
                "seat": "SOUTH",
                "current_turn": "SOUTH",
                "dealer": "EAST",
                "round_wind": "EAST",
                "legal_actions": ["pass", "chii"],
                "hand": ["3m", "4m", "2p", "3p", "4p", "5p", "6p", "7p", "2s", "3s", "4s", "6s", "8s"],
                "action_hints": {
                    "discarder": "EAST",
                    "tile": "5m",
                    "chii": [["3m", "4m"]],
                },
                "discards": {"EAST": ["5m"]},
                "melds": {},
                "dora": [],
            }
        )

        self.assertEqual(decision.action, "chii")
        self.assertEqual(decision.tiles, ("3m", "4m"))

    def test_tile_efficiency_call_agent_passes_chii_without_open_yaku_path(self) -> None:
        agent = TileEfficiencyAgent(allow_calls=True, call_min_score_delta=-100.0)

        decision = agent.choose_reaction(
            {
                "seat": "SOUTH",
                "current_turn": "SOUTH",
                "dealer": "EAST",
                "round_wind": "EAST",
                "legal_actions": ["pass", "chii"],
                "hand": ["3m", "4m", "1p", "2p", "3p", "5p", "6p", "7p", "2s", "3s", "4s", "E", "E"],
                "action_hints": {
                    "discarder": "EAST",
                    "tile": "5m",
                    "chii": [["3m", "4m"]],
                },
                "discards": {"EAST": ["5m"]},
                "melds": {},
                "dora": [],
            }
        )

        self.assertIsNone(decision.action)

    def test_tile_efficiency_call_agent_calls_value_honor_pon(self) -> None:
        agent = TileEfficiencyAgent(allow_calls=True, call_min_score_delta=-100.0)

        decision = agent.choose_reaction(
            {
                "seat": "WEST",
                "current_turn": "WEST",
                "dealer": "EAST",
                "round_wind": "EAST",
                "legal_actions": ["pass", "pon"],
                "hand": ["P", "P", "2m", "3m", "4m", "4p", "5p", "6p", "3s", "4s", "5s", "7s", "8s"],
                "action_hints": {
                    "discarder": "SOUTH",
                    "tile": "P",
                    "pon": [["P", "P"]],
                },
                "discards": {"SOUTH": ["P"]},
                "melds": {},
                "dora": [],
            }
        )

        self.assertEqual(decision.action, "pon")
        self.assertEqual(decision.tiles, ("P", "P"))

    def test_mjai_manue_teacher_protects_early_closed_chii(self) -> None:
        agent = MjaiManueAgent()

        decision = agent.choose_reaction(
            {
                "seat": "SOUTH",
                "current_turn": "SOUTH",
                "dealer": "EAST",
                "round_wind": "EAST",
                "turn_count": 12,
                "wall_count": 56,
                "legal_actions": ["pass", "chii"],
                "hand": ["3m", "4m", "2p", "3p", "4p", "5p", "6p", "7p", "2s", "3s", "4s", "6s", "8s"],
                "action_hints": {
                    "discarder": "EAST",
                    "tile": "5m",
                    "chii": [["3m", "4m"]],
                },
                "discards": {"EAST": ["5m"]},
                "melds": {},
                "dora": [],
                "scores": {"EAST": 25000, "SOUTH": 25000, "WEST": 25000, "NORTH": 25000},
                "riichi_declared": {},
            }
        )

        self.assertIsNone(decision.action)

    def test_expected_value_teachers_call_value_honor_pon(self) -> None:
        state = {
            "seat": "WEST",
            "current_turn": "WEST",
            "dealer": "WEST",
            "round_wind": "EAST",
            "turn_count": 16,
            "wall_count": 52,
            "legal_actions": ["pass", "pon"],
            "hand": ["P", "P", "2m", "3m", "4m", "4p", "5p", "6p", "3s", "4s", "5s", "7s", "8s"],
            "action_hints": {
                "discarder": "SOUTH",
                "tile": "P",
                "pon": [["P", "P"]],
            },
            "discards": {"SOUTH": ["P"]},
            "melds": {},
            "dora": [],
            "scores": {"EAST": 25000, "SOUTH": 25000, "WEST": 25000, "NORTH": 25000},
            "riichi_declared": {},
        }

        for model_name in ("mjai_manue", "akochan", "mahjong_ai"):
            with self.subTest(model_name=model_name):
                decision = create_model(model_name).choose_reaction(state)
                self.assertEqual(decision.action, "pon")
                self.assertEqual(decision.tiles, ("P", "P"))

    def test_akochan_teacher_prefers_genbutsu_against_riichi(self) -> None:
        agent = AkochanAgent()
        hand = ["1m", "5m", "1p", "1p", "9p", "9p", "1s", "1s", "9s", "9s", "E", "E", "P", "P"]
        state = {
            "seat": "SOUTH",
            "current_turn": "SOUTH",
            "dealer": "EAST",
            "round_wind": "EAST",
            "turn_count": 42,
            "wall_count": 32,
            "discards": {"EAST": ["1m"], "SOUTH": [], "WEST": [], "NORTH": []},
            "melds": {},
            "dora": [],
            "scores": {"EAST": 25000, "SOUTH": 25000, "WEST": 25000, "NORTH": 25000},
            "riichi_declared": {"EAST": True},
        }

        scores = agent.score_discards(hand, state)

        self.assertGreater(scores["1m"].score, scores["5m"].score)

    def test_dealer_has_lower_mjai_manue_call_margin(self) -> None:
        agent = MjaiManueAgent()
        state = {
            "seat": "SOUTH",
            "current_turn": "SOUTH",
            "dealer": "EAST",
            "round_wind": "EAST",
            "turn_count": 34,
            "wall_count": 44,
            "legal_actions": ["pass", "chii"],
            "hand": ["3m", "4m", "2p", "3p", "4p", "5p", "6p", "7p", "2s", "3s", "4s", "6s", "8s"],
            "action_hints": {
                "discarder": "EAST",
                "tile": "5m",
                "chii": [["3m", "4m"]],
            },
            "discards": {"EAST": ["5m"]},
            "melds": {},
            "dora": [],
            "scores": {"EAST": 25000, "SOUTH": 25000, "WEST": 25000, "NORTH": 25000},
            "riichi_declared": {},
        }
        analysis = agent._score_call_candidate(state["hand"], state, "chii", ("3m", "4m"))
        self.assertIsNotNone(analysis)
        assert analysis is not None
        strategy = agent._call_strategy(state["hand"], state, "chii", ("3m", "4m"), analysis, agent._pass_reaction_score(state["hand"], state))
        self.assertIsNotNone(strategy)
        assert strategy is not None

        child_margin = agent._call_margin(state["hand"], state, "chii", analysis, strategy)
        dealer_state = dict(state)
        dealer_state["dealer"] = "SOUTH"
        dealer_margin = agent._call_margin(dealer_state["hand"], dealer_state, "chii", analysis, strategy)

        self.assertLess(dealer_margin, child_margin)

    def test_neural_beginner_reaction_uses_call_head_when_available(self) -> None:
        agent = NeuralBeginnerAgent(checkpoint_path="missing-neural-beginner.pt")
        agent.call_model = CallNet()
        with torch.no_grad():
            for parameter in agent.call_model.parameters():
                parameter.zero_()
            agent.call_model.network[-1].bias[REACTION_ACTIONS.index("pon")] = 5.0

        decision = agent.choose_reaction(
            {
                "seat": "EAST",
                "current_turn": "SOUTH",
                "dealer": "EAST",
                "round_wind": "EAST",
                "legal_actions": ["pass", "pon"],
                "hand": ["P", "P", "1m", "2m", "3m"],
                "action_hints": {
                    "discarder": "SOUTH",
                    "tile": "P",
                    "pon": [["P", "P"]],
                },
                "discards": {"SOUTH": ["P"]},
                "melds": {},
                "scores": {"EAST": 25_000, "SOUTH": 25_000, "WEST": 25_000, "NORTH": 25_000},
                "riichi_declared": {},
            }
        )

        self.assertEqual(decision.action, "pon")
        self.assertEqual(decision.tiles, ("P", "P"))

    def test_resmlp_actor_critic_heads_return_policy_and_value(self) -> None:
        discard_model = DiscardNet()
        call_model = CallNet()

        discard_logits, discard_value = discard_model.actor_critic(torch.zeros(2, FEATURE_SIZE))
        call_logits, call_value = call_model.actor_critic(torch.zeros(2, REACTION_FEATURE_SIZE))

        self.assertEqual(tuple(discard_logits.shape), (2, len(TILE_TYPES)))
        self.assertEqual(tuple(discard_value.shape), (2,))
        self.assertEqual(tuple(call_logits.shape), (2, len(REACTION_ACTIONS)))
        self.assertEqual(tuple(call_value.shape), (2,))

    def test_reaction_mask_only_allows_calls_with_candidates(self) -> None:
        mask = legal_reaction_mask(
            {
                "legal_actions": ["pass", "pon", "chii"],
                "action_hints": {"pon": [["P", "P"]], "chii": []},
            }
        )

        self.assertTrue(mask[REACTION_ACTIONS.index("pass")])
        self.assertTrue(mask[REACTION_ACTIONS.index("pon")])
        self.assertFalse(mask[REACTION_ACTIONS.index("chii")])

    def test_forced_tsumogiri_limits_discard_choices(self) -> None:
        hand = ["1m", "2m", "3m", "9p"]
        state = {
            "legal_discard_tiles": ["9p"],
            "action_hints": {"discard": {"tiles": ["9p"], "forced_tsumogiri": True}},
        }

        mask = legal_discard_mask(hand, state)

        self.assertEqual(legal_discard_tiles(hand, state), ["9p"])
        self.assertEqual(forced_tsumogiri_tile(hand, state), "9p")
        self.assertTrue(mask[TILE_TYPES.index("9p")])
        self.assertFalse(mask[TILE_TYPES.index("1m")])

    def test_model_registry_creates_builtin_model(self) -> None:
        self.assertIn("tile_efficiency", model_names())
        self.assertIn("tile_efficiency_call", model_names())
        self.assertIn("mjai_manue", model_names())
        self.assertIn("akochan", model_names())
        self.assertIn("mahjong_ai", model_names())

        agent = create_model("tile_efficiency")
        mjai_manue = create_model("mjai_manue")
        akochan = create_model("akochan")
        mahjong_ai = create_model("mahjong_ai")

        self.assertIsInstance(agent, TileEfficiencyAgent)
        self.assertIsInstance(mjai_manue, MjaiManueAgent)
        self.assertIsInstance(akochan, AkochanAgent)
        self.assertIsInstance(mahjong_ai, MahjongAIHeuristicAgent)


if __name__ == "__main__":
    unittest.main()
