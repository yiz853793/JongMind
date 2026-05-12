import io
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

import jongmind.train_policy_random as train_policy_random
from jongmind.dealer import MahjongDealer
from jongmind.game import PendingReaction, Phase, Seat
from jongmind.train_policy_random import (
    PolicyStep,
    _add_match_reward,
    _add_terminal_reward,
    _discard_defense_reward,
    _discard_shape_reward,
    _early_stop_reason,
    _hand_terminal_reward,
    _load_checkpoint,
    _opponent_agents,
    _play_hand,
    _parse_opponents,
    _parse_validation_opponents,
    _parse_placement_rewards,
    _print_validation,
    _rank,
    _resolve_planner_seed,
    _resolve_validation_seed,
    _resolve_seed,
    _reward_weights_from_args,
    _rolling_metrics,
    _run_validation,
    _teacher_discard_prior_logits,
    _teacher_prior_weight,
    _update_policy,
    _validation_pass_reason,
    _validation_precheck_reason,
    _state_after_discard_for_reward,
    _target_seat,
    MatchTrainStats,
    NEWBIE_ATTACK_WEIGHTS,
    NEWBIE_ATTACK_MATCH_WEIGHTS,
    ValidationRecord,
)
from jongmind.rules import MAHJONG_SOUL_4P_RANKED
from jongmind.scoring import MahjongSoulScoring
from jongmind.tiles import sort_tiles
from jongmind_ai.features import FEATURE_SIZE, TILE_TYPES, legal_discard_mask
from jongmind_ai.neural_beginner import CallNet, DiscardNet
from jongmind_ai.rewards import RewardBreakdown


class PolicyTrainingTest(unittest.TestCase):
    def test_target_seat_rotate_cycles_all_seats(self) -> None:
        seats = [_target_seat(index, "rotate") for index in range(4)]

        self.assertEqual(seats, [Seat.EAST, Seat.SOUTH, Seat.WEST, Seat.NORTH])

    def test_random_target_seat_is_seeded(self) -> None:
        seats = [_target_seat(index, "random", seed=20260508) for index in range(8)]

        self.assertEqual(seats, [_target_seat(index, "random", seed=20260508) for index in range(8)])
        self.assertTrue(all(seat in Seat for seat in seats))

    def test_resolve_seed_accepts_integer_or_random(self) -> None:
        self.assertEqual(_resolve_seed("12345"), 12345)

        random_seed = _resolve_seed("random")

        self.assertIsInstance(random_seed, int)
        self.assertGreaterEqual(random_seed, 0)

    def test_resolve_planner_seed_defaults_to_game_seed(self) -> None:
        self.assertEqual(_resolve_planner_seed("", game_seed=12345), 12345)
        self.assertEqual(_resolve_planner_seed("same", game_seed=12345), 12345)
        self.assertEqual(_resolve_planner_seed("67890", game_seed=12345), 67890)

    def test_rank_uses_score_then_seat_order_tiebreak(self) -> None:
        scores = {
            Seat.EAST: 25_000,
            Seat.SOUTH: 30_000,
            Seat.WEST: 25_000,
            Seat.NORTH: 20_000,
        }

        self.assertEqual(_rank(scores, Seat.SOUTH), 1)
        self.assertEqual(_rank(scores, Seat.EAST), 2)
        self.assertEqual(_rank(scores, Seat.WEST), 3)

    def test_policy_step_combines_three_reward_layers(self) -> None:
        step = PolicyStep(
            log_prob=torch.tensor(0.0),
            entropy=torch.tensor(0.0),
            shape_reward=0.25,
        )

        _add_terminal_reward([step], RewardBreakdown(total=1.0, components={}))
        _add_match_reward([step], RewardBreakdown(total=3.0, components={}))

        self.assertAlmostEqual(step.total_reward, 4.25)

    def test_terminal_and_match_rewards_are_distributed_across_decisions(self) -> None:
        steps = [
            PolicyStep(log_prob=torch.tensor(0.0), entropy=torch.tensor(0.0), shape_reward=0.0),
            PolicyStep(log_prob=torch.tensor(0.0), entropy=torch.tensor(0.0), shape_reward=0.0),
        ]

        _add_terminal_reward(steps, RewardBreakdown(total=10.0, components={}))
        _add_match_reward(steps, RewardBreakdown(total=4.0, components={}))

        self.assertEqual([step.terminal_reward for step in steps], [5.0, 5.0])
        self.assertEqual([step.match_reward for step in steps], [2.0, 2.0])
        self.assertAlmostEqual(sum(step.terminal_reward for step in steps), 10.0)
        self.assertAlmostEqual(sum(step.match_reward for step in steps), 4.0)

    def test_teacher_prior_weight_linearly_decays(self) -> None:
        self.assertEqual(_teacher_prior_weight(0, start=1.5, end=0.0, decay_matches=100), 1.5)
        self.assertEqual(_teacher_prior_weight(50, start=1.5, end=0.0, decay_matches=100), 0.75)
        self.assertEqual(_teacher_prior_weight(150, start=1.5, end=0.0, decay_matches=100), 0.0)

    def test_teacher_discard_prior_prefers_higher_teacher_scores(self) -> None:
        teacher = SimpleNamespace(
            score_discards=lambda hand, state: {
                "1m": SimpleNamespace(score=0.0),
                "2m": SimpleNamespace(score=2.0),
                "3m": SimpleNamespace(score=1.0),
            }
        )
        hand = ["1m", "2m", "3m"]

        logits = _teacher_discard_prior_logits(
            teacher,
            hand,
            {},
            legal_discard_mask(hand, {}),
            temperature=1.0,
        )

        self.assertEqual(float(logits[TILE_TYPES.index("2m")]), 0.0)
        self.assertLess(float(logits[TILE_TYPES.index("1m")]), float(logits[TILE_TYPES.index("3m")]))
        self.assertLess(float(logits[TILE_TYPES.index("3m")]), 0.0)

    def test_policy_update_clips_large_value_targets(self) -> None:
        model = DiscardNet()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        logits, value = model.actor_critic(torch.zeros(1, FEATURE_SIZE))
        distribution = torch.distributions.Categorical(logits=logits.squeeze(0))
        step = PolicyStep(
            log_prob=distribution.log_prob(torch.tensor(0)),
            entropy=distribution.entropy(),
            shape_reward=1000.0,
            value=value.squeeze(0),
        )

        loss = _update_policy(
            model=model,
            optimizer=optimizer,
            traces=[step],
            entropy_coef=0.0,
            shape_ce_coef=0.0,
            value_loss_coef=1.0,
            value_target_clip=2.0,
            normalize_advantages=False,
            grad_clip=1.0,
        )

        self.assertIsNotNone(loss)
        self.assertLess(loss or 0.0, 20.0)

    def test_validation_ignores_teacher_prior_args(self) -> None:
        original_create_model = train_policy_random.create_model
        requested_models: list[str] = []

        def create_model_without_teacher(model_name: str, seed: int) -> object:
            requested_models.append(model_name)
            if model_name == "tile_efficiency_call":
                raise AssertionError("validation must not instantiate the teacher prior")
            return original_create_model(model_name, seed)

        train_policy_random.create_model = create_model_without_teacher
        try:
            _run_validation(
                model=DiscardNet(),
                call_model=CallNet(),
                args=Namespace(
                    validation_matches=1,
                    validation_progress_every=0,
                    validation_seed=20260512,
                    validation_opponents=("random", "random", "random"),
                    validation_match_type="east",
                    validation_target_seat="rotate",
                    disable_calls=False,
                    max_steps_per_hand=1,
                    teacher_prior="tile_efficiency_call",
                    early_stop_max_avg_rank=4.0,
                    early_stop_min_first_rate=0.0,
                    early_stop_max_fourth_rate=1.0,
                    early_stop_min_avg_score=-100000.0,
                    early_stop_min_win_rate=0.0,
                    early_stop_max_open_rate=1.0,
                ),
                after_match=50,
                validation_index=0,
            )
        finally:
            train_policy_random.create_model = original_create_model

        self.assertEqual(requested_models, ["random", "random", "random"])

    def test_validation_prints_threshold_result(self) -> None:
        record = ValidationRecord(
            validation_index=0,
            after_match=50,
            timestamp="2026-05-12T00:00:00+08:00",
            seed=20260512,
            opponents=("random", "random", "shanten"),
            match_type="east",
            target_seat="rotate",
            matches=50,
            hands=200,
            average_rank=2.5,
            first_rate=0.2,
            fourth_rate=0.3,
            average_score=23000.0,
            win_rate=0.08,
            open_rate=0.12,
            errors=0,
            passed=False,
            reason="matches=50 avg_rank=2.50/1.60",
        )
        output = io.StringIO()

        with redirect_stdout(output):
            _print_validation(record)

        self.assertIn("status=FAIL", output.getvalue())
        self.assertIn("validation[1] result matches=50 avg_rank=2.50/1.60", output.getvalue())

    def test_opponent_win_penalty_pushes_against_passive_draw_play(self) -> None:
        reward = _hand_terminal_reward(
            result={"type": "tsumo", "winners": ["SOUTH"], "loser": None},
            target_seat=Seat.EAST,
            score_delta=-1000,
            opponent_win_penalty=-0.25,
        )

        self.assertIn("opponent_win", reward.components)
        self.assertLess(reward.total, 0)

    def test_own_win_is_largest_newbie_terminal_signal(self) -> None:
        win_reward = _hand_terminal_reward(
            result={
                "type": "tsumo",
                "winners": ["EAST"],
                "loser": None,
                "hands": {"EAST": {"cost": {"main": 2000, "additional": 1000}}},
            },
            target_seat=Seat.EAST,
            score_delta=4000,
            opponent_win_penalty=-0.5,
        )
        tenpai_reward = _hand_terminal_reward(
            result={"type": "draw", "reason": "exhaustive", "tenpai": ["EAST"]},
            target_seat=Seat.EAST,
            score_delta=1500,
            opponent_win_penalty=-0.5,
        )

        self.assertGreater(win_reward.total, tenpai_reward.total)
        self.assertEqual(win_reward.components["win"], 12.0)
        self.assertEqual(tenpai_reward.components["exhaustive_draw_tenpai"], 2.0)

    def test_match_reward_penalizes_third_and_fourth_more_than_it_rewards_top_two(self) -> None:
        self.assertEqual(NEWBIE_ATTACK_MATCH_WEIGHTS.placement, (1.0, 0.0, -2.0, -6.0))

    def test_parse_opponents_accepts_three_registered_models(self) -> None:
        self.assertEqual(
            _parse_opponents("tile_efficiency,shanten,open_call"),
            ("tile_efficiency", "shanten", "open_call"),
        )

    def test_parse_validation_opponents_falls_back_to_training_opponents(self) -> None:
        fallback = ("random", "shanten", "tile_efficiency")

        self.assertEqual(_parse_validation_opponents("", fallback), fallback)
        self.assertEqual(
            _parse_validation_opponents("shanten,shanten,shanten", fallback),
            ("shanten", "shanten", "shanten"),
        )

    def test_resolve_validation_seed_can_reuse_game_seed(self) -> None:
        self.assertEqual(_resolve_validation_seed("same", game_seed=123), 123)
        self.assertEqual(_resolve_validation_seed("456", game_seed=123), 456)

    def test_validation_pass_reason_reuses_early_stop_thresholds_by_default(self) -> None:
        metrics = _rolling_metrics(
            [
                _stat(rank=1, final_score=30_000, wins=1, hands=4),
                _stat(rank=2, final_score=27_000, wins=1, hands=4),
            ]
        )
        args = Namespace(
            early_stop_max_avg_rank=1.6,
            early_stop_min_first_rate=0.5,
            early_stop_max_fourth_rate=0.0,
            early_stop_min_avg_score=28_000,
            early_stop_min_win_rate=0.2,
            early_stop_max_open_rate=1.0,
            validation_max_avg_rank=None,
            validation_min_first_rate=None,
            validation_max_fourth_rate=None,
            validation_min_avg_score=None,
            validation_min_win_rate=None,
            validation_max_open_rate=None,
        )

        passed, reason = _validation_pass_reason(metrics, errors=0, args=args)

        self.assertTrue(passed)
        self.assertIn("avg_rank=1.50/1.60", reason)

    def test_validation_pass_reason_can_use_different_validation_thresholds(self) -> None:
        metrics = _rolling_metrics([_stat(rank=1, final_score=30_000, wins=1, hands=4)])
        args = Namespace(
            early_stop_max_avg_rank=1.6,
            early_stop_min_first_rate=0.5,
            early_stop_max_fourth_rate=0.0,
            early_stop_min_avg_score=28_000,
            early_stop_min_win_rate=0.2,
            early_stop_max_open_rate=1.0,
            validation_max_avg_rank=1.0,
            validation_min_first_rate=1.0,
            validation_max_fourth_rate=0.0,
            validation_min_avg_score=35_000,
            validation_min_win_rate=0.0,
            validation_max_open_rate=1.0,
        )

        passed, reason = _validation_pass_reason(metrics, errors=0, args=args)

        self.assertFalse(passed)
        self.assertIn("avg_score=30000/35000", reason)

    def test_validation_precheck_can_skip_validation_before_window_is_ready(self) -> None:
        args = Namespace(validation_precheck_window=3)

        passed, reason = _validation_precheck_reason([_stat(rank=1, final_score=30_000, wins=1)], args)

        self.assertFalse(passed)
        self.assertIn("available=1", reason)

    def test_validation_precheck_uses_recent_training_thresholds(self) -> None:
        stats = [
            _stat(rank=4, final_score=10_000, wins=0, hands=4),
            _stat(rank=1, final_score=31_000, wins=1, hands=4),
            _stat(rank=2, final_score=28_000, wins=1, hands=4),
        ]
        args = Namespace(
            validation_precheck_window=2,
            validation_precheck_max_avg_rank=1.6,
            validation_precheck_min_first_rate=0.5,
            validation_precheck_max_fourth_rate=0.0,
            validation_precheck_min_avg_score=28_000,
            validation_precheck_min_win_rate=0.2,
            validation_precheck_max_open_rate=1.0,
            early_stop_max_avg_rank=9.0,
            early_stop_min_first_rate=0.0,
            early_stop_max_fourth_rate=1.0,
            early_stop_min_avg_score=0.0,
            early_stop_min_win_rate=0.0,
            early_stop_max_open_rate=1.0,
            validation_max_avg_rank=None,
            validation_min_first_rate=None,
            validation_max_fourth_rate=None,
            validation_min_avg_score=None,
            validation_min_win_rate=None,
            validation_max_open_rate=None,
        )

        passed, reason = _validation_precheck_reason(stats, args)

        self.assertTrue(passed)
        self.assertIn("window=2", reason)

    def test_parse_placement_rewards_accepts_four_numbers(self) -> None:
        self.assertEqual(_parse_placement_rewards("1,0,-2,-6"), (1.0, 0.0, -2.0, -6.0))

    def test_reward_weights_can_raise_expected_value_weight(self) -> None:
        weights = _reward_weights_from_args(
            Namespace(
                deal_in_penalty=-6.0,
                score_delta_weight=0.0008,
                expected_value_weight=0.0025,
            )
        )

        self.assertEqual(weights.deal_in, -6.0)
        self.assertEqual(weights.score_delta, 0.0008)
        self.assertEqual(weights.expected_value, 0.0025)

    def test_opponent_agents_fill_non_target_seats(self) -> None:
        agents = _opponent_agents(("random", "shanten", "open_call"), Seat.SOUTH, seed=123)

        self.assertEqual(set(agents), {Seat.EAST, Seat.WEST, Seat.NORTH})
        self.assertNotIn(Seat.SOUTH, agents)

    def test_disable_calls_passes_target_reaction_without_opponent_lookup(self) -> None:
        dealer = MahjongDealer(seed=1)
        dealer.start_hand()
        dealer.state.current_turn = Seat.NORTH
        dealer.state.phase = Phase.REACTION
        dealer.state.discards[Seat.NORTH].append("3m")
        dealer.state.pending_reaction = PendingReaction(
            discarder=Seat.NORTH,
            tile="3m",
            options={Seat.EAST: ["chii"]},
        )
        traces: list[PolicyStep] = []

        with self.assertRaisesRegex(RuntimeError, "step limit"):
            _play_hand(
                dealer=dealer,
                model=DiscardNet(),
                call_model=CallNet(),
                target_seat=Seat.EAST,
                opponents={},
                traces=traces,
                temperature=0.25,
                shape_scale=1.0,
                defense_scale=1.0,
                call_shape_scale=0.0,
                reward_weights=NEWBIE_ATTACK_WEIGHTS,
                allow_calls=False,
                max_steps_per_hand=1,
            )

        self.assertIsNone(dealer.state.pending_reaction)
        self.assertEqual(dealer.state.current_turn, Seat.EAST)
        self.assertEqual(traces, [])

    def test_rolling_metrics_use_recent_match_and_hand_counts(self) -> None:
        metrics = _rolling_metrics(
            [
                _stat(rank=1, final_score=30_000, wins=1, hands=4, open_hands=2),
                _stat(rank=4, final_score=20_000, wins=0, hands=6, open_hands=3),
            ]
        )

        self.assertEqual(metrics.matches, 2)
        self.assertEqual(metrics.hands, 10)
        self.assertEqual(metrics.first_rate, 0.5)
        self.assertEqual(metrics.fourth_rate, 0.5)
        self.assertEqual(metrics.win_rate, 0.1)
        self.assertEqual(metrics.open_rate, 0.5)

    def test_early_stop_requires_all_thresholds_in_latest_window(self) -> None:
        stats = [
            _stat(rank=4, final_score=10_000, wins=0, hands=4),
            _stat(rank=1, final_score=31_000, wins=1, hands=4),
            _stat(rank=1, final_score=29_000, wins=1, hands=4),
        ]
        args = Namespace(
            early_stop_window=2,
            early_stop_max_avg_rank=1.1,
            early_stop_min_first_rate=1.0,
            early_stop_max_fourth_rate=0.0,
            early_stop_min_avg_score=28_000,
            early_stop_min_win_rate=0.1,
            early_stop_max_open_rate=1.0,
        )

        reason = _early_stop_reason(stats, args)

        self.assertIsNotNone(reason)
        self.assertIn("window=2", reason or "")

    def test_early_stop_stays_off_when_window_is_disabled(self) -> None:
        args = Namespace(early_stop_window=0)

        self.assertIsNone(_early_stop_reason([_stat(rank=1, final_score=40_000, wins=2)], args))

    def test_discard_shape_reward_is_relative_to_available_discards(self) -> None:
        scoring = MahjongSoulScoring(MAHJONG_SOUL_4P_RANKED)
        hand = sort_tiles(
            ["2m", "3m", "4m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5p", "2s", "3s", "4s"]
        )
        state = {"discards": {}, "melds": {}, "dora_indicators": [], "dealer": "EAST", "round_wind": "EAST"}

        good = _discard_shape_reward(hand, "9m", state, "EAST", NEWBIE_ATTACK_WEIGHTS, scoring)
        bad = _discard_shape_reward(hand, "3m", state, "EAST", NEWBIE_ATTACK_WEIGHTS, scoring)

        self.assertGreater(good.total, 0)
        self.assertLess(bad.total, 0)

    def test_discard_shape_reward_state_counts_candidate_discard_as_visible(self) -> None:
        state = {"discards": {"EAST": ["1m"], "SOUTH": []}}

        reward_state = _state_after_discard_for_reward(state, "EAST", "9m")

        self.assertEqual(reward_state["discards"]["EAST"], ["1m", "9m"])
        self.assertEqual(state["discards"]["EAST"], ["1m"])

    def test_defense_reward_penalizes_danger_against_riichi(self) -> None:
        state = {
            "dealer": "SOUTH",
            "wall_count": 24,
            "dora": ["5m"],
            "dora_indicators": [],
            "discards": {"EAST": [], "SOUTH": ["1m"], "WEST": [], "NORTH": []},
            "melds": {},
            "riichi_declared": {"SOUTH": True},
        }

        reward = _discard_defense_reward("0m", state, "EAST")

        self.assertLess(reward.total, 0)
        self.assertIn("danger", reward.components)

    def test_defense_reward_prefers_genbutsu_against_riichi(self) -> None:
        state = {
            "dealer": "SOUTH",
            "wall_count": 24,
            "dora": [],
            "dora_indicators": [],
            "discards": {"EAST": [], "SOUTH": ["5m"], "WEST": [], "NORTH": []},
            "melds": {},
            "riichi_declared": {"SOUTH": True},
        }

        safe = _discard_defense_reward("5m", state, "EAST")
        risky = _discard_defense_reward("6m", state, "EAST")

        self.assertGreater(safe.total, risky.total)

    def test_loading_checkpoint_keeps_requested_learning_rate(self) -> None:
        model = DiscardNet()
        source_optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": source_optimizer.state_dict(),
                },
                checkpoint_path,
            )

            loaded_model = DiscardNet()
            loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=0.003)
            _load_checkpoint(loaded_model, loaded_optimizer, checkpoint_path, learning_rate=0.003)

        self.assertEqual(loaded_optimizer.param_groups[0]["lr"], 0.003)

    def test_loading_checkpoint_can_skip_optimizer_state(self) -> None:
        model = DiscardNet()
        source_optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": source_optimizer.state_dict(),
                },
                checkpoint_path,
            )

            loaded_model = DiscardNet()
            loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=0.003)
            _load_checkpoint(
                loaded_model,
                loaded_optimizer,
                checkpoint_path,
                learning_rate=0.003,
                load_optimizer=False,
            )

        self.assertEqual(loaded_optimizer.state, {})
        self.assertEqual(loaded_optimizer.param_groups[0]["lr"], 0.003)


def _stat(
    rank: int,
    final_score: int,
    wins: int,
    hands: int = 4,
    open_hands: int = 0,
    error: str | None = None,
) -> MatchTrainStats:
    return MatchTrainStats(
        match_index=0,
        seed=0,
        timestamp="2026-05-09T00:00:00+08:00",
        target_seat="EAST",
        hands=hands,
        decisions=0,
        rank=rank,
        final_score=final_score,
        wins=wins,
        tsumo_wins=0,
        open_hands=open_hands,
        calls=0,
        tenpai_draws=0,
        noten_draws=0,
        opponent_wins=0,
        deal_ins=0,
        riichi_hands=0,
        mangan_plus_wins=0,
        yakuman_wins=0,
        shape_reward=0.0,
        defense_reward=0.0,
        terminal_reward=0.0,
        match_reward=0.0,
        total_reward=0.0,
        loss=0.0,
        error=error,
    )


if __name__ == "__main__":
    unittest.main()
