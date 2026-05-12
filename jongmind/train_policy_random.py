"""Train the neural discard/call policy from game rewards."""

from __future__ import annotations

import argparse
import json
import random
import secrets
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from jongmind.dealer import MahjongDealer
from jongmind.game import DealerCommand, Phase, Seat
from jongmind.tiles import is_suited, is_terminal_or_honor, normalize_tile, sort_tiles, tile_rank
from jongmind.evaluate_models import _is_mangan_plus, _is_yakuman, _play_hand as _eval_play_hand
from jongmind_ai import MODEL_NAMES, create_model
from jongmind_ai.base import forced_tsumogiri_tile, legal_discard_tiles
from jongmind_ai.features import (
    REACTION_ACTIONS,
    TILE_TYPES,
    encode_reaction_state,
    encode_state,
    legal_discard_mask,
    legal_reaction_mask,
)
from jongmind_ai.neural_beginner import DEFAULT_CHECKPOINT, CallNet, DiscardNet, load_state_dict_flexible
from jongmind_ai.rewards import (
    MatchRewardWeights,
    RewardBreakdown,
    RewardWeights,
    analyze_hand,
    match_terminal_reward,
    shape_potential_reward,
    terminal_reward,
)


SEAT_NAMES = tuple(seat.name for seat in Seat)
NEWBIE_ATTACK_WEIGHTS = RewardWeights(
    shanten=0.12,
    ukeire=0.003,
    effective_tile_types=0.01,
    tenpai=1.0,
    winning_tiles=0.01,
    winning_tile_types=0.03,
    expected_value=0.0010,
    no_yaku_tenpai=-2.5,
    win=12.0,
    deal_in=-8.0,
    exhaustive_draw_tenpai=2.0,
    exhaustive_draw_noten=-2.0,
    score_delta=0.00055,
)
NEWBIE_ATTACK_MATCH_WEIGHTS = MatchRewardWeights(
    placement=(1.0, 0.0, -2.0, -6.0),
    score_delta=0.00003,
)


@dataclass
class PolicyStep:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    shape_reward: float
    value: torch.Tensor | None = None
    shape_logits: torch.Tensor | None = None
    shape_target: int | None = None
    auxiliary_logits: torch.Tensor | None = None
    auxiliary_target: int | None = None
    auxiliary_ce_coef: float = 0.0
    defense_reward: float = 0.0
    terminal_reward: float = 0.0
    match_reward: float = 0.0

    @property
    def total_reward(self) -> float:
        return self.shape_reward + self.defense_reward + self.terminal_reward + self.match_reward


@dataclass(frozen=True)
class MatchTrainStats:
    match_index: int
    seed: int
    timestamp: str
    target_seat: str
    hands: int
    decisions: int
    rank: int
    final_score: int
    wins: int
    tsumo_wins: int
    open_hands: int
    calls: int
    tenpai_draws: int
    noten_draws: int
    opponent_wins: int
    deal_ins: int
    riichi_hands: int
    mangan_plus_wins: int
    yakuman_wins: int
    shape_reward: float
    defense_reward: float
    terminal_reward: float
    match_reward: float
    total_reward: float
    loss: float | None
    error: str | None = None


@dataclass(frozen=True)
class RollingMetrics:
    matches: int
    hands: int
    average_rank: float
    first_rate: float
    fourth_rate: float
    average_score: float
    win_rate: float
    open_rate: float


@dataclass(frozen=True)
class ValidationRecord:
    validation_index: int
    after_match: int
    timestamp: str
    seed: int
    opponents: tuple[str, str, str]
    match_type: str
    target_seat: str
    matches: int
    hands: int
    average_rank: float
    first_rate: float
    fourth_rate: float
    average_score: float
    win_rate: float
    open_rate: float
    errors: int
    passed: bool
    reason: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--match-type", choices=("east", "south"), default="east")
    parser.add_argument(
        "--seed",
        default="20260508",
        help="Game base seed integer, or random to generate one at startup.",
    )
    parser.add_argument(
        "--planner-seed",
        default="",
        help=(
            "Policy/model seed integer, random to generate one, "
            "or empty/same to reuse --seed."
        ),
    )
    parser.add_argument(
        "--target-seat",
        choices=(*SEAT_NAMES, "rotate", "random"),
        default="rotate",
        help="Seat trained by the policy, or rotate to reduce seat bias.",
    )
    parser.add_argument(
        "--opponents",
        default="random,random,random",
        help=f"Comma-separated opponent models for the non-target seats. Choices: {', '.join(MODEL_NAMES)}.",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional checkpoint to initialize from. Empty means random weights.",
    )
    parser.add_argument(
        "--reset-call-model",
        action="store_true",
        help="When loading --init-checkpoint, keep discard weights but reinitialize the call head.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When loading --init-checkpoint, skip optimizer state and start AdamW fresh.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--shape-ce-coef",
        type=float,
        default=0.1,
        help="Auxiliary loss weight for the reward-best local discard.",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="Critic MSE loss weight for actor-critic policy updates.",
    )
    parser.add_argument(
        "--value-target-clip",
        type=float,
        default=20.0,
        help="Clamp per-decision critic targets to +/- this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=20.0,
        help="Clamp distributed terminal/match rewards per decision. Use 0 to disable.",
    )
    parser.add_argument(
        "--teacher-prior",
        default="",
        help=(
            "Optional model used as a behavior prior during RL sampling. "
            f"Choices: {', '.join(MODEL_NAMES)}. Empty disables it."
        ),
    )
    parser.add_argument(
        "--teacher-prior-weight-start",
        type=float,
        default=0.0,
        help="Initial additive logit weight for --teacher-prior.",
    )
    parser.add_argument(
        "--teacher-prior-weight-end",
        type=float,
        default=0.0,
        help="Final additive logit weight after --teacher-prior-decay-matches.",
    )
    parser.add_argument(
        "--teacher-prior-decay-matches",
        type=int,
        default=0,
        help="Number of matches over which the teacher-prior weight linearly decays.",
    )
    parser.add_argument(
        "--teacher-prior-temperature",
        type=float,
        default=0.45,
        help="Softmax temperature for converting teacher discard scores into logits.",
    )
    parser.add_argument(
        "--teacher-prior-action-bias",
        type=float,
        default=2.0,
        help="Reaction logit bias added to the action selected by the teacher prior.",
    )
    parser.add_argument(
        "--teacher-reaction-ce-coef",
        type=float,
        default=0.0,
        help=(
            "Extra supervised cross-entropy loss that trains the raw reaction head "
            "toward --teacher-prior. This keeps the bare validation policy from "
            "forgetting pass/call boundaries after the sampling prior fades."
        ),
    )
    parser.add_argument(
        "--call-action-penalty",
        type=float,
        default=0.0,
        help="Immediate reward subtracted whenever the target policy makes a non-ron call.",
    )
    parser.add_argument(
        "--call-shape-scale",
        type=float,
        default=0.05,
        help="Scale for local call-vs-pass shanten/ukeire shaping rewards.",
    )
    parser.add_argument(
        "--disable-calls",
        action="store_true",
        help="Force the target model to pass non-ron reaction prompts.",
    )
    parser.add_argument(
        "--no-advantage-normalize",
        action="store_true",
        help="Disable per-match advantage centering/scaling.",
    )
    parser.add_argument(
        "--shape-scale",
        type=float,
        default=1.0,
        help="Scale for per-discard shanten/ukeire/value shaping rewards.",
    )
    parser.add_argument(
        "--defense-scale",
        type=float,
        default=0.3,
        help="Scale for discard danger penalties when opponents are threatening.",
    )
    parser.add_argument(
        "--terminal-scale",
        type=float,
        default=1.0,
        help="Scale for hand-end win/tenpai/deal-in/score-delta rewards.",
    )
    parser.add_argument(
        "--match-scale",
        type=float,
        default=0.03,
        help="Scale for match-end placement/final-score rewards.",
    )
    parser.add_argument(
        "--opponent-win-penalty",
        type=float,
        default=-0.5,
        help="Extra hand-end penalty when another player wins and the target did not.",
    )
    parser.add_argument(
        "--deal-in-penalty",
        type=float,
        default=NEWBIE_ATTACK_WEIGHTS.deal_in,
        help="Terminal reward component when the target deals in.",
    )
    parser.add_argument(
        "--bust-penalty",
        type=float,
        default=-20.0,
        help="Extra terminal penalty when the target score falls below zero.",
    )
    parser.add_argument(
        "--score-delta-weight",
        type=float,
        default=NEWBIE_ATTACK_WEIGHTS.score_delta,
        help="Hand-end score delta reward multiplier.",
    )
    parser.add_argument(
        "--expected-value-weight",
        type=float,
        default=NEWBIE_ATTACK_WEIGHTS.expected_value,
        help=(
            "Reward multiplier for estimated hand value in shape rewards and "
            "actual win value at hand end."
        ),
    )
    parser.add_argument(
        "--placement-rewards",
        default=",".join(str(value) for value in NEWBIE_ATTACK_MATCH_WEIGHTS.placement),
        help="Comma-separated match placement rewards for 1st,2nd,3rd,4th.",
    )
    parser.add_argument(
        "--match-score-delta-weight",
        type=float,
        default=NEWBIE_ATTACK_MATCH_WEIGHTS.score_delta,
        help="Match-end final score delta reward multiplier.",
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-steps-per-hand", type=int, default=2000)
    parser.add_argument(
        "--early-stop-window",
        type=int,
        default=0,
        help="Stop when the latest N matches satisfy all early-stop thresholds. 0 disables early stop.",
    )
    parser.add_argument("--early-stop-max-avg-rank", type=float, default=1.6)
    parser.add_argument("--early-stop-min-first-rate", type=float, default=0.65)
    parser.add_argument("--early-stop-max-fourth-rate", type=float, default=0.10)
    parser.add_argument("--early-stop-min-avg-score", type=float, default=28_000.0)
    parser.add_argument("--early-stop-min-win-rate", type=float, default=0.08)
    parser.add_argument(
        "--early-stop-max-open-rate",
        type=float,
        default=1.0,
        help="Optional open-hand cap for early stop. 1.0 effectively disables this constraint.",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=0,
        help=(
            "Run a fixed validation set every N training matches and stop when "
            "validation thresholds pass. 0 disables validation early stop."
        ),
    )
    parser.add_argument(
        "--validation-matches",
        type=int,
        default=100,
        help="Number of fixed validation matches per validation run.",
    )
    parser.add_argument(
        "--validation-opponents",
        default="",
        help="Comma-separated validation opponents. Empty reuses --opponents.",
    )
    parser.add_argument(
        "--validation-match-type",
        choices=("same", "east", "south"),
        default="same",
        help="Validation match type. 'same' reuses --match-type.",
    )
    parser.add_argument(
        "--validation-target-seat",
        choices=(*SEAT_NAMES, "rotate", "random", "same"),
        default="rotate",
        help="Validation target seat policy. 'same' reuses --target-seat.",
    )
    parser.add_argument(
        "--validation-seed",
        default="20260510",
        help="Fixed validation base seed integer, random, or same to use --seed.",
    )
    parser.add_argument(
        "--validation-log",
        default="",
        help="JSONL validation log path. Empty creates a sibling file next to --output-log.",
    )
    parser.add_argument(
        "--validation-progress-every",
        type=int,
        default=10,
        help=(
            "Print validation progress every N validation matches. "
            "Use 0 to print only validation start/end."
        ),
    )
    parser.add_argument(
        "--validation-precheck-window",
        type=int,
        default=0,
        help=(
            "Only run validation when the latest N training matches pass a cheap "
            "precheck. 0 disables precheck and validates every --validation-every matches."
        ),
    )
    parser.add_argument("--validation-precheck-max-avg-rank", type=float, default=None)
    parser.add_argument("--validation-precheck-min-first-rate", type=float, default=None)
    parser.add_argument("--validation-precheck-max-fourth-rate", type=float, default=None)
    parser.add_argument("--validation-precheck-min-avg-score", type=float, default=None)
    parser.add_argument("--validation-precheck-min-win-rate", type=float, default=None)
    parser.add_argument("--validation-precheck-max-open-rate", type=float, default=None)
    parser.add_argument("--validation-max-avg-rank", type=float, default=None)
    parser.add_argument("--validation-min-first-rate", type=float, default=None)
    parser.add_argument("--validation-max-fourth-rate", type=float, default=None)
    parser.add_argument("--validation-min-avg-score", type=float, default=None)
    parser.add_argument("--validation-min-win-rate", type=float, default=None)
    parser.add_argument("--validation-max-open-rate", type=float, default=None)
    parser.add_argument(
        "--output-log",
        default="outputs/policy_random_training.jsonl",
        help="JSONL training log path.",
    )
    parser.add_argument(
        "--review-log",
        default="",
        help=(
            "JSONL decision-review log path. Empty creates a sibling file "
            "next to --output-log; use 'off' to disable."
        ),
    )
    parser.add_argument(
        "--review-include-wall",
        action="store_true",
        help=(
            "Include private live/dead wall snapshots in the review log only. "
            "These hidden tiles are never passed to the policy model."
        ),
    )
    args = parser.parse_args()
    args.seed = _resolve_seed(args.seed)
    args.planner_seed = _resolve_planner_seed(args.planner_seed, args.seed)
    args.opponents = _parse_opponents(args.opponents)
    args.teacher_prior = _parse_optional_model(args.teacher_prior, "--teacher-prior")
    args.validation_opponents = _parse_validation_opponents(args.validation_opponents, args.opponents)
    args.validation_seed = _resolve_validation_seed(args.validation_seed, args.seed)
    args.validation_match_type = args.match_type if args.validation_match_type == "same" else args.validation_match_type
    args.validation_target_seat = args.target_seat if args.validation_target_seat == "same" else args.validation_target_seat
    args.placement_rewards = _parse_placement_rewards(args.placement_rewards)
    reward_weights = _reward_weights_from_args(args)
    match_reward_weights = _match_reward_weights_from_args(args)

    random.seed(args.seed)
    torch.manual_seed(args.planner_seed)
    print(f"game_seed={args.seed} planner_seed={args.planner_seed} opponents={','.join(args.opponents)}")
    if args.teacher_prior:
        print(
            "teacher_prior="
            f"model={args.teacher_prior} "
            f"weight={args.teacher_prior_weight_start:g}->{args.teacher_prior_weight_end:g} "
            f"decay_matches={args.teacher_prior_decay_matches}"
        )
    if args.validation_every > 0:
        print(
            "validation="
            f"every={args.validation_every} matches={args.validation_matches} "
            f"precheck_window={args.validation_precheck_window} "
            f"seed={args.validation_seed} opponents={','.join(args.validation_opponents)} "
            f"match_type={args.validation_match_type} target_seat={args.validation_target_seat}"
        )

    model = DiscardNet()
    call_model = CallNet()
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *call_model.parameters()],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    if args.init_checkpoint:
        _load_checkpoint(
            model,
            optimizer,
            Path(args.init_checkpoint),
            learning_rate=args.learning_rate,
            call_model=call_model,
            load_call_model=not args.reset_call_model,
            load_optimizer=not (args.reset_optimizer or args.reset_call_model),
        )

    checkpoint_path = Path(args.checkpoint)
    log_path = Path(args.output_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    review_path = _review_log_path(args.review_log, log_path)
    review_handle = None
    if review_path is not None:
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_handle = open(review_path, "w", encoding="utf-8")

    validation_path = _validation_log_path(args.validation_log, log_path) if args.validation_every > 0 else None
    validation_handle = None
    if validation_path is not None:
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_handle = open(validation_path, "w", encoding="utf-8")

    stats: list[MatchTrainStats] = []
    validation_records: list[ValidationRecord] = []
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            for match_index in range(args.matches):
                seed = args.seed + match_index
                target_seat = _target_seat(match_index, args.target_seat, args.seed)
                review_records: list[dict[str, Any]] = []
                teacher_prior_weight = _teacher_prior_weight(
                    match_index=match_index,
                    start=args.teacher_prior_weight_start,
                    end=args.teacher_prior_weight_end,
                    decay_matches=args.teacher_prior_decay_matches,
                )
                result = train_match(
                    model=model,
                    call_model=call_model,
                    optimizer=optimizer,
                    match_index=match_index,
                    seed=seed,
                    target_seat=target_seat,
                    opponents=args.opponents,
                    match_type=args.match_type,
                    entropy_coef=args.entropy_coef,
                    shape_ce_coef=args.shape_ce_coef,
                    value_loss_coef=args.value_loss_coef,
                    value_target_clip=args.value_target_clip,
                    normalize_advantages=not args.no_advantage_normalize,
                    temperature=args.temperature,
                    grad_clip=args.grad_clip,
                    shape_scale=args.shape_scale,
                    defense_scale=args.defense_scale,
                    call_shape_scale=args.call_shape_scale,
                    terminal_scale=args.terminal_scale,
                    match_scale=args.match_scale,
                    reward_clip=args.reward_clip,
                    opponent_win_penalty=args.opponent_win_penalty,
                    bust_penalty=args.bust_penalty,
                    reward_weights=reward_weights,
                    match_reward_weights=match_reward_weights,
                    allow_calls=not args.disable_calls,
                    teacher_prior_name=args.teacher_prior,
                    teacher_prior_weight=teacher_prior_weight,
                    teacher_prior_temperature=args.teacher_prior_temperature,
                    teacher_prior_action_bias=args.teacher_prior_action_bias,
                    teacher_reaction_ce_coef=args.teacher_reaction_ce_coef,
                    call_action_penalty=args.call_action_penalty,
                    max_steps_per_hand=args.max_steps_per_hand,
                    review_records=review_records if review_handle is not None else None,
                    review_include_wall=args.review_include_wall,
                )
                stats.append(result)
                log.write(_json_line(result) + "\n")
                log.flush()
                if review_handle is not None:
                    for record in review_records:
                        review_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    review_handle.flush()
                _print_match(result, args.matches)

                if (match_index + 1) % args.save_every == 0:
                    _save_checkpoint(checkpoint_path, model, call_model, optimizer, args, stats[-1])

                validation_record = _maybe_run_validation(
                    model=model,
                    call_model=call_model,
                    args=args,
                    after_match=match_index + 1,
                    validation_index=len(validation_records),
                    training_stats=stats,
                )
                if validation_record is not None:
                    validation_records.append(validation_record)
                    _print_validation(validation_record)
                    if validation_handle is not None:
                        validation_handle.write(json.dumps(asdict(validation_record), ensure_ascii=False, sort_keys=True) + "\n")
                        validation_handle.flush()
                    if validation_record.passed:
                        print(f"validation_early_stop: {validation_record.reason}")
                        _save_checkpoint(checkpoint_path, model, call_model, optimizer, args, stats[-1])
                        break

                early_stop_reason = _early_stop_reason(stats, args)
                if early_stop_reason is not None:
                    print(f"early_stop: {early_stop_reason}")
                    _save_checkpoint(checkpoint_path, model, call_model, optimizer, args, stats[-1])
                    break
    finally:
        if review_handle is not None:
            review_handle.close()
        if validation_handle is not None:
            validation_handle.close()

    _save_checkpoint(checkpoint_path, model, call_model, optimizer, args, stats[-1] if stats else None)
    _print_summary(stats, checkpoint_path, log_path)
    if validation_records:
        _print_validation_summary(validation_records, validation_path)

    if any(stat.error for stat in stats):
        raise SystemExit(1)


def train_match(
    model: DiscardNet,
    call_model: CallNet,
    optimizer: torch.optim.Optimizer,
    match_index: int,
    seed: int,
    target_seat: Seat,
    opponents: tuple[str, str, str],
    match_type: str,
    entropy_coef: float,
    shape_ce_coef: float,
    value_loss_coef: float,
    value_target_clip: float,
    normalize_advantages: bool,
    temperature: float,
    grad_clip: float,
    shape_scale: float,
    defense_scale: float,
    call_shape_scale: float,
    terminal_scale: float,
    match_scale: float,
    reward_clip: float,
    opponent_win_penalty: float,
    bust_penalty: float,
    reward_weights: RewardWeights,
    match_reward_weights: MatchRewardWeights,
    allow_calls: bool,
    max_steps_per_hand: int,
    teacher_prior_name: str = "",
    teacher_prior_weight: float = 0.0,
    teacher_prior_temperature: float = 0.45,
    teacher_prior_action_bias: float = 2.0,
    teacher_reaction_ce_coef: float = 0.0,
    call_action_penalty: float = 0.0,
    review_records: list[dict[str, Any]] | None = None,
    review_include_wall: bool = False,
) -> MatchTrainStats:
    model.train()
    call_model.train()
    dealer = MahjongDealer(seed=seed)
    scores = {seat: dealer.rules.starting_points for seat in Seat}
    opponent_agents = _opponent_agents(opponents, target_seat, seed)
    teacher_prior = (
        create_model(teacher_prior_name, seed=seed * 100 + 97)
        if teacher_prior_name and teacher_prior_weight > 0
        else None
    )
    traces: list[PolicyStep] = []
    max_hand_number = 4 if match_type == "east" else 8
    hand_number = 0
    honba = 0
    riichi_sticks = 0
    hands_played = 0
    wins = 0
    tsumo_wins = 0
    open_hands = 0
    calls = 0
    tenpai_draws = 0
    noten_draws = 0
    opponent_wins = 0
    deal_ins = 0
    riichi_hands = 0
    mangan_plus_wins = 0
    yakuman_wins = 0

    try:
        while hand_number < max_hand_number:
            dealer.state.scores = scores.copy()
            dealer.state.hand_number = hand_number
            dealer.state.max_hand_number = max_hand_number
            dealer.state.round_wind = Seat.EAST if hand_number < 4 else Seat.SOUTH
            dealer.state.honba = honba
            dealer.state.riichi_sticks = riichi_sticks
            dealer.start_hand()
            dealer_seat = dealer.state.dealer
            hand_start_score = scores[target_seat]
            hand_trace_start = len(traces)
            hand_review_start = len(review_records) if review_records is not None else 0

            _play_hand(
                dealer=dealer,
                model=model,
                call_model=call_model,
                target_seat=target_seat,
                opponents=opponent_agents,
                traces=traces,
                temperature=temperature,
                shape_scale=shape_scale,
                defense_scale=defense_scale,
                call_shape_scale=call_shape_scale,
                reward_weights=reward_weights,
                allow_calls=allow_calls,
                teacher_prior=teacher_prior,
                teacher_prior_weight=teacher_prior_weight,
                teacher_prior_temperature=teacher_prior_temperature,
                teacher_prior_action_bias=teacher_prior_action_bias,
                teacher_reaction_ce_coef=teacher_reaction_ce_coef,
                call_action_penalty=call_action_penalty,
                max_steps_per_hand=max_steps_per_hand,
                review_records=review_records,
                match_index=match_index,
                hand_number=hand_number,
                max_hand_number=max_hand_number,
                review_include_wall=review_include_wall,
            )
            calls += _target_call_count(dealer, target_seat)
            hands_played += 1
            if _target_hand_is_open(dealer, target_seat):
                open_hands += 1
            if dealer.state.riichi_declared[target_seat]:
                riichi_hands += 1

            result = dealer.state.result or {}
            scores = dealer.state.scores.copy()
            riichi_sticks = dealer.state.riichi_sticks
            hand_score_delta = scores[target_seat] - hand_start_score
            if review_records is not None:
                _annotate_hand_reviews(
                    review_records[hand_review_start:],
                    result=result,
                    target_seat=target_seat,
                    score_delta=hand_score_delta,
                    scores=scores,
                )

            hand_reward = _hand_terminal_reward(
                result=result,
                target_seat=target_seat,
                score_delta=hand_score_delta,
                opponent_win_penalty=opponent_win_penalty,
                bust_penalty=bust_penalty,
                final_score=scores[target_seat],
                weights=reward_weights,
            )
            _add_terminal_reward(
                traces[hand_trace_start:],
                hand_reward,
                scale=terminal_scale,
                clip=reward_clip,
            )

            if result.get("type") in {"ron", "tsumo"}:
                winners = [Seat[name] for name in result.get("winners", [])]
                if target_seat in winners:
                    wins += 1
                    if result.get("type") == "tsumo":
                        tsumo_wins += 1
                    target_hand_value = result.get("hands", {}).get(target_seat.name, {})
                    if _is_mangan_plus(target_hand_value):
                        mangan_plus_wins += 1
                    if _is_yakuman(target_hand_value):
                        yakuman_wins += 1
                else:
                    opponent_wins += 1
                if result.get("loser") == target_seat.name:
                    deal_ins += 1

                if dealer_seat in winners:
                    honba += 1
                else:
                    honba = 0
                    hand_number += 1
            else:
                tenpai = {Seat[name] for name in result.get("tenpai", [])}
                if target_seat in tenpai:
                    tenpai_draws += 1
                else:
                    noten_draws += 1
                honba += 1
                if dealer_seat not in tenpai:
                    hand_number += 1

            if any(score < 0 for score in scores.values()):
                break

        match_reward = match_terminal_reward(scores, target_seat, weights=match_reward_weights)
        _add_match_reward(traces, match_reward, scale=match_scale, clip=reward_clip)
        loss_value = _update_policy(
            model=model,
            optimizer=optimizer,
            traces=traces,
            entropy_coef=entropy_coef,
            shape_ce_coef=shape_ce_coef,
            value_loss_coef=value_loss_coef,
            value_target_clip=value_target_clip,
            normalize_advantages=normalize_advantages,
            grad_clip=grad_clip,
        )
        return _match_stats(
            match_index=match_index,
            seed=seed,
            target_seat=target_seat,
            scores=scores,
            hands_played=hands_played,
            wins=wins,
            tsumo_wins=tsumo_wins,
            open_hands=open_hands,
            calls=calls,
            tenpai_draws=tenpai_draws,
            noten_draws=noten_draws,
            opponent_wins=opponent_wins,
            deal_ins=deal_ins,
            riichi_hands=riichi_hands,
            mangan_plus_wins=mangan_plus_wins,
            yakuman_wins=yakuman_wins,
            traces=traces,
            loss_value=loss_value,
        )
    except Exception as exc:
        return MatchTrainStats(
            match_index=match_index,
            seed=seed,
            timestamp=_timestamp(),
            target_seat=target_seat.name,
            hands=hands_played,
            decisions=len(traces),
            rank=4,
            final_score=scores[target_seat],
            wins=wins,
            tsumo_wins=tsumo_wins,
            open_hands=open_hands,
            calls=calls,
            tenpai_draws=tenpai_draws,
            noten_draws=noten_draws,
            opponent_wins=opponent_wins,
            deal_ins=deal_ins,
            riichi_hands=riichi_hands,
            mangan_plus_wins=mangan_plus_wins,
            yakuman_wins=yakuman_wins,
            shape_reward=sum(step.shape_reward for step in traces),
            defense_reward=sum(step.defense_reward for step in traces),
            terminal_reward=sum(step.terminal_reward for step in traces),
            match_reward=sum(step.match_reward for step in traces),
            total_reward=sum(step.total_reward for step in traces),
            loss=None,
            error=str(exc),
        )


def _play_hand(
    dealer: MahjongDealer,
    model: DiscardNet,
    call_model: CallNet,
    target_seat: Seat,
    opponents: dict[Seat, Any],
    traces: list[PolicyStep],
    temperature: float,
    shape_scale: float,
    defense_scale: float,
    call_shape_scale: float,
    reward_weights: RewardWeights,
    allow_calls: bool,
    max_steps_per_hand: int,
    teacher_prior: Any | None = None,
    teacher_prior_weight: float = 0.0,
    teacher_prior_temperature: float = 0.45,
    teacher_prior_action_bias: float = 2.0,
    teacher_reaction_ce_coef: float = 0.0,
    call_action_penalty: float = 0.0,
    review_records: list[dict[str, Any]] | None = None,
    match_index: int = 0,
    hand_number: int = 0,
    max_hand_number: int = 4,
    review_include_wall: bool = False,
) -> None:
    responded_reactions: set[tuple[int, str, str, Seat]] = set()
    for _ in range(max_steps_per_hand):
        state = dealer.get_state()
        phase = state["phase"]
        if phase == Phase.FINISHED.value:
            return

        if phase == Phase.DRAW.value:
            seat = Seat[state["current_turn"]]
            dealer.handle(DealerCommand(kind="draw", seat=seat))
            continue

        if phase == Phase.DISCARD.value:
            seat = Seat[state["current_turn"]]
            player_state = dealer.get_state(seat)
            legal_actions = player_state["legal_actions"]
            if "tsumo" in legal_actions:
                dealer.handle(DealerCommand(kind="win", seat=seat, action="tsumo"))
                continue
            if "abortive_draw" in legal_actions:
                dealer.handle(DealerCommand(kind="abortive_draw", seat=seat))
                continue

            if seat == target_seat:
                forced_tile = forced_tsumogiri_tile(player_state["hand"], player_state)
                if forced_tile is not None:
                    tile = forced_tile
                else:
                    tile, log_prob, entropy, value, shape_logits = _sample_discard(
                        model,
                        player_state["hand"],
                        player_state,
                        temperature,
                        teacher_prior=teacher_prior,
                        teacher_prior_weight=teacher_prior_weight,
                        teacher_prior_temperature=teacher_prior_temperature,
                    )
                    potentials = _discard_shape_potentials(
                        hand=player_state["hand"],
                        state=player_state,
                        seat_name=seat.name,
                        weights=reward_weights,
                        scoring=dealer.scoring,
                    )
                    shape_reward = _shape_reward_from_potentials(tile, potentials)
                    defense_reward = _discard_defense_reward(tile, player_state, seat.name)
                    shape_target = TILE_TYPES.index(_best_shape_tile(potentials))
                    traces.append(
                        PolicyStep(
                            log_prob=log_prob,
                            entropy=entropy,
                            shape_logits=shape_logits,
                            shape_target=shape_target,
                            shape_reward=shape_reward.total * shape_scale,
                            value=value,
                            defense_reward=defense_reward.total * defense_scale,
                        )
                    )
                    if review_records is not None:
                        review_records.append(
                            _discard_review_record(
                                match_index=match_index,
                                hand_number=hand_number,
                                max_hand_number=max_hand_number,
                                state=player_state,
                                seat=seat,
                                chosen_tile=tile,
                                potentials=potentials,
                                shape_reward=shape_reward,
                                defense_reward=defense_reward,
                                scoring=dealer.scoring,
                                private_wall=_private_wall_review_context(dealer) if review_include_wall else None,
                            )
                        )
            else:
                decision = opponents[seat].choose_discard(player_state["hand"], player_state)
                tile = decision.tile

            riichi = (
                "riichi" in legal_actions
                and tile in player_state.get("action_hints", {}).get("riichi", {}).get("discard_tiles", [])
            )
            dealer.handle(DealerCommand(kind="discard", seat=seat, tile=tile, riichi=riichi))
            continue

        if phase == Phase.REACTION.value:
            pending = dealer.state.pending_reaction
            if pending is None:
                raise RuntimeError("reaction phase without pending reaction")
            seat = pending.waiting_seats[0]
            player_state = dealer.get_state(seat)
            legal_actions = player_state["legal_actions"]
            reaction_key = (player_state["turn_count"], pending.discarder.name, pending.tile, seat)
            if reaction_key in responded_reactions:
                dealer.handle(DealerCommand(kind="pass", seat=seat))
                continue
            responded_reactions.add(reaction_key)
            if "ron" in legal_actions:
                dealer.handle(DealerCommand(kind="win", seat=seat, action="ron"))
                continue
            if seat == target_seat:
                if not allow_calls:
                    dealer.handle(DealerCommand(kind="pass", seat=seat))
                    continue
                else:
                    action, tiles, log_prob, entropy, value, raw_call_logits = _sample_reaction(
                        call_model,
                        player_state,
                        temperature,
                        teacher_prior=teacher_prior,
                        teacher_prior_weight=teacher_prior_weight,
                        teacher_prior_action_bias=teacher_prior_action_bias,
                    )
                    shape_reward = _reaction_shape_reward(
                        action=action,
                        tiles=tiles,
                        dealer=dealer,
                        state=player_state,
                        seat=seat,
                        weights=reward_weights,
                    )
                    teacher_reaction_target = _teacher_reaction_target_index(teacher_prior, player_state)
                    call_penalty = float(call_action_penalty) if action is not None else 0.0
                    traces.append(
                        PolicyStep(
                            log_prob=log_prob,
                            entropy=entropy,
                            shape_reward=shape_reward.total * call_shape_scale - call_penalty,
                            value=value,
                            auxiliary_logits=raw_call_logits if teacher_reaction_target is not None else None,
                            auxiliary_target=teacher_reaction_target,
                            auxiliary_ce_coef=teacher_reaction_ce_coef,
                        )
                    )
                    if review_records is not None:
                        review_records.append(
                            _reaction_review_record(
                                match_index=match_index,
                                hand_number=hand_number,
                                max_hand_number=max_hand_number,
                                state=player_state,
                                seat=seat,
                                action=action,
                                tiles=tiles,
                                shape_reward=shape_reward,
                                scoring=dealer.scoring,
                                private_wall=_private_wall_review_context(dealer) if review_include_wall else None,
                            )
                        )
                    if action is not None:
                        dealer.handle(
                            DealerCommand(
                                kind="call",
                                seat=seat,
                                action=action,
                                tiles=tiles,
                            )
                        )
                    else:
                        dealer.handle(DealerCommand(kind="pass", seat=seat))
            else:
                reaction = _choose_reaction(opponents[seat], player_state, legal_actions)
                if reaction is not None:
                    dealer.handle(
                        DealerCommand(
                            kind="call",
                            seat=seat,
                            action=reaction[0],
                            tiles=reaction[1],
                        )
                    )
                else:
                    dealer.handle(DealerCommand(kind="pass", seat=seat))
            continue

        raise RuntimeError(f"unknown phase: {phase}")

    raise RuntimeError(f"step limit reached: {max_steps_per_hand}")


def _sample_discard(
    model: DiscardNet,
    hand: list[str],
    state: dict[str, Any],
    temperature: float,
    teacher_prior: Any | None = None,
    teacher_prior_weight: float = 0.0,
    teacher_prior_temperature: float = 0.45,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = encode_state(hand, state).unsqueeze(0)
    logits, value = model.actor_critic(features)
    logits = logits.squeeze(0)
    mask = legal_discard_mask(hand, state)
    if teacher_prior is not None and teacher_prior_weight > 0:
        logits = logits + teacher_prior_weight * _teacher_discard_prior_logits(
            teacher_prior,
            hand,
            state,
            mask,
            teacher_prior_temperature,
            dtype=logits.dtype,
            device=logits.device,
        )
    masked_logits = logits.masked_fill(~mask, -1_000_000_000.0)
    distribution = torch.distributions.Categorical(logits=masked_logits / max(temperature, 1e-6))
    action = distribution.sample()
    tile = TILE_TYPES[int(action.item())]
    return tile, distribution.log_prob(action), distribution.entropy(), value.squeeze(0), masked_logits


def _sample_reaction(
    model: CallNet,
    state: dict[str, Any],
    temperature: float,
    teacher_prior: Any | None = None,
    teacher_prior_weight: float = 0.0,
    teacher_prior_action_bias: float = 2.0,
) -> tuple[str | None, tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hand = state.get("hand") or []
    features = encode_reaction_state(hand, state).unsqueeze(0)
    logits, value = model.actor_critic(features)
    logits = logits.squeeze(0)
    mask = legal_reaction_mask(state)
    raw_masked_logits = logits.masked_fill(~mask, -1_000_000_000.0)
    if teacher_prior is not None and teacher_prior_weight > 0:
        logits = logits + teacher_prior_weight * _teacher_reaction_prior_logits(
            teacher_prior,
            state,
            mask,
            teacher_prior_action_bias,
            dtype=logits.dtype,
            device=logits.device,
        )
    masked_logits = logits.masked_fill(~mask, -1_000_000_000.0)
    distribution = torch.distributions.Categorical(logits=masked_logits / max(temperature, 1e-6))
    action_index = distribution.sample()
    action = REACTION_ACTIONS[int(action_index.item())]
    if action == "pass":
        return None, (), distribution.log_prob(action_index), distribution.entropy(), value.squeeze(0), raw_masked_logits

    candidates = (state.get("action_hints") or {}).get(action) or []
    if not candidates:
        return None, (), distribution.log_prob(action_index), distribution.entropy(), value.squeeze(0), raw_masked_logits
    return action, tuple(candidates[0]), distribution.log_prob(action_index), distribution.entropy(), value.squeeze(0), raw_masked_logits


def _teacher_discard_prior_logits(
    teacher: Any,
    hand: list[str],
    state: dict[str, Any],
    mask: torch.Tensor,
    temperature: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    logits = torch.zeros(len(TILE_TYPES), dtype=dtype, device=device)
    scorer = getattr(teacher, "score_discards", None)
    if scorer is None:
        chooser = getattr(teacher, "choose_discard", None)
        if chooser is None:
            return logits
        decision = chooser(hand, state)
        tile = getattr(decision, "tile", None)
        if tile in TILE_TYPES and bool(mask[TILE_TYPES.index(tile)].item()):
            logits[TILE_TYPES.index(tile)] = 1.0
        return logits

    raw_scores = scorer(hand, state)
    legal_scores: list[tuple[int, float]] = []
    for tile, score in raw_scores.items():
        if tile not in TILE_TYPES:
            continue
        tile_index = TILE_TYPES.index(tile)
        if not bool(mask[tile_index].item()):
            continue
        value = float(getattr(score, "score", score))
        legal_scores.append((tile_index, value))

    if not legal_scores:
        return logits

    best_score = max(value for _, value in legal_scores)
    scale = max(float(temperature), 1e-6)
    for tile_index, value in legal_scores:
        logits[tile_index] = max(-8.0, min(0.0, (value - best_score) / scale))
    return logits


def _teacher_reaction_prior_logits(
    teacher: Any,
    state: dict[str, Any],
    mask: torch.Tensor,
    action_bias: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    logits = torch.zeros(len(REACTION_ACTIONS), dtype=dtype, device=device)
    legal_actions = list(state.get("legal_actions") or [])
    reaction = _choose_reaction(teacher, state, legal_actions)
    action = "pass" if reaction is None else reaction[0]
    if action not in REACTION_ACTIONS:
        return logits
    action_index = REACTION_ACTIONS.index(action)
    if bool(mask[action_index].item()):
        logits[action_index] = float(action_bias)
    return logits


def _teacher_reaction_target_index(teacher: Any | None, state: dict[str, Any]) -> int | None:
    if teacher is None:
        return None
    mask = legal_reaction_mask(state)
    reaction = _choose_reaction(teacher, state, list(state.get("legal_actions") or []))
    action = "pass" if reaction is None else reaction[0]
    if action not in REACTION_ACTIONS:
        return None
    action_index = REACTION_ACTIONS.index(action)
    if not bool(mask[action_index].item()):
        return None
    return action_index


def _reaction_shape_reward(
    action: str | None,
    tiles: tuple[str, ...],
    dealer: MahjongDealer,
    state: dict[str, Any],
    seat: Seat,
    weights: RewardWeights,
) -> RewardBreakdown:
    potentials = _reaction_shape_potentials(dealer, state, seat, weights)
    chosen_key = (action or "pass", tiles if action is not None else ())
    chosen = potentials.get(chosen_key, potentials[("pass", ())])
    average = sum(potentials.values()) / max(1, len(potentials))
    best = max(potentials.values())
    components = {
        "relative_call_shape": chosen - average,
        "chosen_call_shape": chosen,
        "average_call_shape": average,
        "best_call_gap": chosen - best,
    }
    return RewardBreakdown(total=components["relative_call_shape"], components=components)


def _reaction_shape_potentials(
    dealer: MahjongDealer,
    state: dict[str, Any],
    seat: Seat,
    weights: RewardWeights,
) -> dict[tuple[str, tuple[str, ...]], float]:
    hand = state.get("hand") or []
    potentials: dict[tuple[str, tuple[str, ...]], float] = {
        ("pass", ()): shape_potential_reward(
            analyze_hand(hand, state, seat.name, dealer.scoring),
            weights,
        ).total
    }
    hints = state.get("action_hints") or {}
    for action in ("chii", "pon", "kan"):
        for raw_tiles in hints.get(action) or []:
            tiles = tuple(raw_tiles)
            try:
                after_call_hand = _hand_after_removing_tiles(hand, tiles)
            except ValueError:
                continue
            call_state = _state_after_call_for_reward(state, seat, action, tiles)
            if action == "kan":
                potentials[(action, tiles)] = shape_potential_reward(
                    analyze_hand(after_call_hand, call_state, seat.name, dealer.scoring),
                    weights,
                ).total
                continue

            discard_potentials = _discard_shape_potentials(
                hand=after_call_hand,
                state=call_state,
                seat_name=seat.name,
                weights=weights,
                scoring=dealer.scoring,
            )
            if discard_potentials:
                potentials[(action, tiles)] = max(discard_potentials.values())
    return potentials


def _hand_after_removing_tiles(hand: list[str], tiles: tuple[str, ...]) -> list[str]:
    remaining = hand[:]
    for tile in tiles:
        remaining.remove(tile)
    return remaining


def _state_after_call_for_reward(
    state: dict[str, Any],
    seat: Seat,
    action: str,
    tiles: tuple[str, ...],
) -> dict[str, Any]:
    hints = state.get("action_hints") or {}
    called_tile = hints.get("tile")
    discarder = hints.get("discarder")
    reward_state = dict(state)
    discards = {
        discard_seat: list(discard_tiles)
        for discard_seat, discard_tiles in state.get("discards", {}).items()
    }
    if called_tile is not None and discarder in discards and discards[discarder]:
        if discards[discarder][-1] == called_tile:
            discards[discarder].pop()
    melds = {
        meld_seat: [dict(meld) for meld in seat_melds]
        for meld_seat, seat_melds in state.get("melds", {}).items()
    }
    seat_melds = melds.setdefault(seat.name, [])
    seat_melds.append(
        {
            "kind": action,
            "owner": seat.name,
            "from_seat": discarder,
            "called_tile": called_tile,
            "tiles": sort_tiles([*tiles, called_tile]) if called_tile is not None else list(tiles),
            "open": True,
        }
    )
    reward_state["discards"] = discards
    reward_state["melds"] = melds
    return reward_state


def _discard_shape_reward(
    hand: list[str],
    chosen_tile: str,
    state: dict[str, Any],
    seat_name: str,
    weights: RewardWeights,
    scoring: Any,
) -> RewardBreakdown:
    return _shape_reward_from_potentials(
        chosen_tile,
        _discard_shape_potentials(hand, state, seat_name, weights, scoring),
    )


def _discard_shape_potentials(
    hand: list[str],
    state: dict[str, Any],
    seat_name: str,
    weights: RewardWeights,
    scoring: Any,
) -> dict[str, float]:
    potentials: dict[str, float] = {}
    for tile in dict.fromkeys(legal_discard_tiles(hand, state)):
        after_hand = hand[:]
        after_hand.remove(tile)
        reward_state = _state_after_discard_for_reward(state, seat_name, tile)
        metrics = analyze_hand(after_hand, reward_state, seat_name, scoring)
        potentials[tile] = shape_potential_reward(metrics, weights).total
    return potentials


def _shape_reward_from_potentials(
    chosen_tile: str,
    potentials: dict[str, float],
) -> RewardBreakdown:
    chosen = potentials[chosen_tile]
    average = sum(potentials.values()) / max(1, len(potentials))
    best = max(potentials.values())
    components = {
        "relative_shape": chosen - average,
        "chosen_shape": chosen,
        "average_shape": average,
        "best_gap": chosen - best,
    }
    return RewardBreakdown(total=components["relative_shape"], components=components)


def _best_shape_tile(potentials: dict[str, float]) -> str:
    return max(potentials, key=lambda tile: (potentials[tile], -TILE_TYPES.index(tile)))


def _state_after_discard_for_reward(
    state: dict[str, Any],
    seat_name: str,
    tile: str,
) -> dict[str, Any]:
    reward_state = dict(state)
    discards = {
        discard_seat: list(tiles)
        for discard_seat, tiles in state.get("discards", {}).items()
    }
    seat_discards = discards.setdefault(seat_name, [])
    seat_discards.append(tile)
    reward_state["discards"] = discards
    return reward_state


def _discard_defense_reward(tile: str, state: dict[str, Any], seat_name: str) -> RewardBreakdown:
    threats = _threatening_opponents(state, seat_name)
    if not threats:
        return RewardBreakdown(total=0.0, components={})

    danger = 0.0
    genbutsu_safe = 0.0
    for threat in threats:
        if _is_genbutsu(tile, state, threat["seat"]):
            genbutsu_safe += 0.12 * threat["weight"]
            continue
        danger += _tile_danger(tile, state) * threat["weight"]

    components = {
        "danger": -danger,
        "genbutsu_safe": genbutsu_safe,
    }
    return RewardBreakdown(total=sum(components.values()), components=components)


def _threatening_opponents(state: dict[str, Any], seat_name: str) -> list[dict[str, Any]]:
    riichi_declared = state.get("riichi_declared") or {}
    melds = state.get("melds") or {}
    dealer = state.get("dealer")
    threats: list[dict[str, Any]] = []
    for opponent in SEAT_NAMES:
        if opponent == seat_name:
            continue
        open_melds = sum(
            1
            for meld in melds.get(opponent, [])
            if meld.get("open", True) and meld.get("kind") != "closed_kan"
        )
        is_riichi = bool(riichi_declared.get(opponent, False))
        if not is_riichi and open_melds < 2:
            continue
        weight = 1.0 if is_riichi else 0.35 + 0.15 * min(open_melds, 4)
        if opponent == dealer:
            weight += 0.2
        wall_count = int(state.get("wall_count", 70))
        if wall_count <= 30:
            weight += 0.15
        if wall_count <= 18:
            weight += 0.20
        threats.append({"seat": opponent, "weight": weight})
    return threats


def _is_genbutsu(tile: str, state: dict[str, Any], opponent: str) -> bool:
    normalized = normalize_tile(tile)
    return normalized in {
        normalize_tile(discard)
        for discard in (state.get("discards") or {}).get(opponent, [])
    }


def _tile_danger(tile: str, state: dict[str, Any]) -> float:
    normalized = normalize_tile(tile)
    visible = _visible_tile_count(normalized, state)
    if visible >= 4:
        return 0.0

    danger = 0.45
    dora = {normalize_tile(dora_tile) for dora_tile in state.get("dora", [])}
    if normalized in dora:
        danger += 0.55
    if tile.startswith("0"):
        danger += 0.30
    if is_suited(tile):
        rank = tile_rank(tile)
        if 3 <= rank <= 7:
            danger += 0.15
        elif rank in (1, 9):
            danger -= 0.08
    elif visible <= 1:
        danger += 0.20
    if is_terminal_or_honor(tile):
        danger -= 0.03
    if visible == 2:
        danger *= 0.65
    elif visible == 3:
        danger *= 0.35
    return max(0.0, danger)


def _visible_tile_count(normalized_tile: str, state: dict[str, Any]) -> int:
    count = 0
    for discards in (state.get("discards") or {}).values():
        count += sum(1 for tile in discards if normalize_tile(tile) == normalized_tile)
    for melds in (state.get("melds") or {}).values():
        for meld in melds:
            count += sum(1 for tile in meld.get("tiles", []) if normalize_tile(tile) == normalized_tile)
    count += sum(1 for tile in state.get("dora_indicators", []) if normalize_tile(tile) == normalized_tile)
    return count


def _choose_reaction(
    agent: Any,
    state: dict[str, Any],
    legal_actions: list[str],
) -> tuple[str, tuple[str, ...]] | None:
    chooser = getattr(agent, "choose_reaction", None)
    if chooser is None:
        return None
    decision = chooser(state)
    if decision.action is None or decision.action not in legal_actions:
        return None
    return decision.action, decision.tiles




class _CurrentPolicyAgent:
    """Greedy wrapper around the currently trained policy for validation only."""

    def __init__(self, model: DiscardNet, call_model: CallNet, allow_calls: bool = True) -> None:
        self.model = model
        self.call_model = call_model
        self.allow_calls = allow_calls

    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> Any:
        with torch.no_grad():
            features = encode_state(hand, state).unsqueeze(0)
            logits = self.model(features).squeeze(0)
            mask = legal_discard_mask(hand, state)
            logits = logits.masked_fill(~mask, -1_000_000_000.0)
            tile = TILE_TYPES[int(torch.argmax(logits).item())]
        return type("DiscardDecision", (), {"tile": tile})()

    def choose_reaction(self, state: dict[str, Any]) -> Any:
        if not self.allow_calls:
            return type("CallDecision", (), {"action": None, "tiles": ()})()
        hand = state.get("hand") or []
        with torch.no_grad():
            features = encode_reaction_state(hand, state).unsqueeze(0)
            logits = self.call_model(features).squeeze(0)
            mask = legal_reaction_mask(state)
            logits = logits.masked_fill(~mask, -1_000_000_000.0)
            action = REACTION_ACTIONS[int(torch.argmax(logits).item())]
        if action == "pass":
            return type("CallDecision", (), {"action": None, "tiles": ()})()
        candidates = (state.get("action_hints") or {}).get(action) or []
        if not candidates:
            return type("CallDecision", (), {"action": None, "tiles": ()})()
        return type("CallDecision", (), {"action": action, "tiles": tuple(candidates[0])})()


def _maybe_run_validation(
    model: DiscardNet,
    call_model: CallNet,
    args: argparse.Namespace,
    after_match: int,
    validation_index: int,
    training_stats: list[MatchTrainStats],
) -> ValidationRecord | None:
    every = int(getattr(args, "validation_every", 0) or 0)
    if every <= 0 or after_match % every != 0:
        return None

    precheck_passed, precheck_reason = _validation_precheck_reason(training_stats, args)
    if not precheck_passed:
        print(
            f"validation[{validation_index + 1}] skipped after_match={after_match} "
            f"precheck=FAIL {precheck_reason}",
            flush=True,
        )
        return None

    if precheck_reason:
        print(
            f"validation[{validation_index + 1}] precheck=PASS after_match={after_match} "
            f"{precheck_reason}",
            flush=True,
        )

    return _run_validation(
        model=model,
        call_model=call_model,
        args=args,
        after_match=after_match,
        validation_index=validation_index,
    )


def _run_validation(
    model: DiscardNet,
    call_model: CallNet,
    args: argparse.Namespace,
    after_match: int,
    validation_index: int,
) -> ValidationRecord:
    was_training = model.training
    call_was_training = call_model.training
    model.eval()
    call_model.eval()
    validation_matches = max(1, int(args.validation_matches))
    progress_every = max(0, int(getattr(args, "validation_progress_every", 10) or 0))
    stats: list[MatchTrainStats] = []
    _print_validation_start(
        validation_index=validation_index,
        after_match=after_match,
        validation_matches=validation_matches,
        seed=int(args.validation_seed),
        opponents=args.validation_opponents,
        match_type=args.validation_match_type,
        target_seat=args.validation_target_seat,
    )
    try:
        for offset in range(validation_matches):
            seed = int(args.validation_seed) + offset
            target_seat = _target_seat(offset, args.validation_target_seat, int(args.validation_seed))
            stats.append(
                _validation_match(
                    model=model,
                    call_model=call_model,
                    match_index=offset,
                    seed=seed,
                    target_seat=target_seat,
                    opponents=args.validation_opponents,
                    match_type=args.validation_match_type,
                    allow_calls=not args.disable_calls,
                    max_steps_per_hand=args.max_steps_per_hand,
                )
            )
            completed = offset + 1
            if progress_every > 0 and (completed % progress_every == 0 or completed == validation_matches):
                _print_validation_progress(
                    validation_index=validation_index,
                    completed=completed,
                    total=validation_matches,
                    stats=stats,
                )
    finally:
        if was_training:
            model.train()
        if call_was_training:
            call_model.train()

    metrics = _rolling_metrics(stats)
    errors = sum(1 for stat in stats if stat.error)
    passed, reason = _validation_pass_reason(metrics, errors, args)
    return ValidationRecord(
        validation_index=validation_index,
        after_match=after_match,
        timestamp=_timestamp(),
        seed=int(args.validation_seed),
        opponents=args.validation_opponents,
        match_type=args.validation_match_type,
        target_seat=args.validation_target_seat,
        matches=metrics.matches,
        hands=metrics.hands,
        average_rank=metrics.average_rank,
        first_rate=metrics.first_rate,
        fourth_rate=metrics.fourth_rate,
        average_score=metrics.average_score,
        win_rate=metrics.win_rate,
        open_rate=metrics.open_rate,
        errors=errors,
        passed=passed,
        reason=reason,
    )


def _validation_match(
    model: DiscardNet,
    call_model: CallNet,
    match_index: int,
    seed: int,
    target_seat: Seat,
    opponents: tuple[str, str, str],
    match_type: str,
    allow_calls: bool,
    max_steps_per_hand: int,
) -> MatchTrainStats:
    dealer = MahjongDealer(seed=seed)
    scores = {seat: dealer.rules.starting_points for seat in Seat}
    agents: dict[Seat, Any] = {target_seat: _CurrentPolicyAgent(model, call_model, allow_calls=allow_calls)}
    opponent_index = 0
    for seat in Seat:
        if seat == target_seat:
            continue
        agents[seat] = create_model(opponents[opponent_index], seed=seed * 100 + int(seat))
        opponent_index += 1

    max_hand_number = 4 if match_type == "east" else 8
    hand_number = 0
    honba = 0
    riichi_sticks = 0
    hands_played = 0
    wins = 0
    tsumo_wins = 0
    open_hands = 0
    calls = 0
    tenpai_draws = 0
    noten_draws = 0
    opponent_wins = 0
    deal_ins = 0
    riichi_hands = 0
    mangan_plus_wins = 0
    yakuman_wins = 0

    try:
        while hand_number < max_hand_number:
            dealer.state.scores = scores.copy()
            dealer.state.hand_number = hand_number
            dealer.state.max_hand_number = max_hand_number
            dealer.state.round_wind = Seat.EAST if hand_number < 4 else Seat.SOUTH
            dealer.state.honba = honba
            dealer.state.riichi_sticks = riichi_sticks
            dealer.start_hand()
            dealer_seat = dealer.state.dealer

            _eval_play_hand(dealer, agents, max_steps_per_hand=max_steps_per_hand)
            calls += _target_call_count(dealer, target_seat)
            hands_played += 1
            if _target_hand_is_open(dealer, target_seat):
                open_hands += 1
            if dealer.state.riichi_declared[target_seat]:
                riichi_hands += 1

            result = dealer.state.result or {}
            scores = dealer.state.scores.copy()
            riichi_sticks = dealer.state.riichi_sticks

            if result.get("type") in {"ron", "tsumo"}:
                winners = [Seat[name] for name in result.get("winners", [])]
                if target_seat in winners:
                    wins += 1
                    if result.get("type") == "tsumo":
                        tsumo_wins += 1
                    target_hand_value = result.get("hands", {}).get(target_seat.name, {})
                    if _is_mangan_plus(target_hand_value):
                        mangan_plus_wins += 1
                    if _is_yakuman(target_hand_value):
                        yakuman_wins += 1
                else:
                    opponent_wins += 1
                if result.get("loser") == target_seat.name:
                    deal_ins += 1

                if dealer_seat in winners:
                    honba += 1
                else:
                    honba = 0
                    hand_number += 1
            else:
                tenpai = {Seat[name] for name in result.get("tenpai", [])}
                if target_seat in tenpai:
                    tenpai_draws += 1
                else:
                    noten_draws += 1
                honba += 1
                if dealer_seat not in tenpai:
                    hand_number += 1

            if any(score < 0 for score in scores.values()):
                break

        return _match_stats(
            match_index=match_index,
            seed=seed,
            target_seat=target_seat,
            scores=scores,
            hands_played=hands_played,
            wins=wins,
            tsumo_wins=tsumo_wins,
            open_hands=open_hands,
            calls=calls,
            tenpai_draws=tenpai_draws,
            noten_draws=noten_draws,
            opponent_wins=opponent_wins,
            deal_ins=deal_ins,
            riichi_hands=riichi_hands,
            mangan_plus_wins=mangan_plus_wins,
            yakuman_wins=yakuman_wins,
            traces=[],
            loss_value=None,
        )
    except Exception as exc:
        return MatchTrainStats(
            match_index=match_index,
            seed=seed,
            timestamp=_timestamp(),
            target_seat=target_seat.name,
            hands=hands_played,
            decisions=0,
            rank=4,
            final_score=scores[target_seat],
            wins=wins,
            tsumo_wins=tsumo_wins,
            open_hands=open_hands,
            calls=calls,
            tenpai_draws=tenpai_draws,
            noten_draws=noten_draws,
            opponent_wins=opponent_wins,
            deal_ins=deal_ins,
            riichi_hands=riichi_hands,
            mangan_plus_wins=mangan_plus_wins,
            yakuman_wins=yakuman_wins,
            shape_reward=0.0,
            defense_reward=0.0,
            terminal_reward=0.0,
            match_reward=0.0,
            total_reward=0.0,
            loss=None,
            error=str(exc),
        )


def _validation_precheck_reason(
    stats: list[MatchTrainStats],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    window = int(getattr(args, "validation_precheck_window", 0) or 0)
    if window <= 0:
        return True, ""
    if len(stats) < window:
        return False, f"window={window} available={len(stats)}"

    recent = stats[-window:]
    errors = sum(1 for stat in recent if stat.error)
    if errors:
        return False, f"window={window} errors={errors}"

    metrics = _rolling_metrics(recent)
    max_avg_rank = _precheck_threshold(args, "max_avg_rank", _validation_threshold(args, "max_avg_rank", args.early_stop_max_avg_rank))
    min_first_rate = _precheck_threshold(args, "min_first_rate", _validation_threshold(args, "min_first_rate", args.early_stop_min_first_rate))
    max_fourth_rate = _precheck_threshold(args, "max_fourth_rate", _validation_threshold(args, "max_fourth_rate", args.early_stop_max_fourth_rate))
    min_avg_score = _precheck_threshold(args, "min_avg_score", _validation_threshold(args, "min_avg_score", args.early_stop_min_avg_score))
    min_win_rate = _precheck_threshold(args, "min_win_rate", _validation_threshold(args, "min_win_rate", args.early_stop_min_win_rate))
    max_open_rate = _precheck_threshold(args, "max_open_rate", _validation_threshold(args, "max_open_rate", args.early_stop_max_open_rate))
    checks = [
        metrics.average_rank <= max_avg_rank,
        metrics.first_rate >= min_first_rate,
        metrics.fourth_rate <= max_fourth_rate,
        metrics.average_score >= min_avg_score,
        metrics.win_rate >= min_win_rate,
        metrics.open_rate <= max_open_rate,
    ]
    reason = (
        f"window={window} avg_rank={metrics.average_rank:.2f}/{max_avg_rank:.2f} "
        f"first_rate={metrics.first_rate * 100:.1f}%/{min_first_rate * 100:.1f}% "
        f"fourth_rate={metrics.fourth_rate * 100:.1f}%/{max_fourth_rate * 100:.1f}% "
        f"avg_score={metrics.average_score:.0f}/{min_avg_score:.0f} "
        f"win_rate={metrics.win_rate * 100:.2f}%/{min_win_rate * 100:.2f}% "
        f"open_rate={metrics.open_rate * 100:.2f}%/{max_open_rate * 100:.2f}%"
    )
    return all(checks), reason


def _precheck_threshold(args: argparse.Namespace, suffix: str, fallback: float) -> float:
    value = getattr(args, f"validation_precheck_{suffix}", None)
    return fallback if value is None else float(value)


def _validation_pass_reason(
    metrics: RollingMetrics,
    errors: int,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    max_avg_rank = _validation_threshold(args, "max_avg_rank", args.early_stop_max_avg_rank)
    min_first_rate = _validation_threshold(args, "min_first_rate", args.early_stop_min_first_rate)
    max_fourth_rate = _validation_threshold(args, "max_fourth_rate", args.early_stop_max_fourth_rate)
    min_avg_score = _validation_threshold(args, "min_avg_score", args.early_stop_min_avg_score)
    min_win_rate = _validation_threshold(args, "min_win_rate", args.early_stop_min_win_rate)
    max_open_rate = _validation_threshold(args, "max_open_rate", args.early_stop_max_open_rate)
    checks = [
        errors == 0,
        metrics.average_rank <= max_avg_rank,
        metrics.first_rate >= min_first_rate,
        metrics.fourth_rate <= max_fourth_rate,
        metrics.average_score >= min_avg_score,
        metrics.win_rate >= min_win_rate,
        metrics.open_rate <= max_open_rate,
    ]
    reason = (
        f"matches={metrics.matches} avg_rank={metrics.average_rank:.2f}/{max_avg_rank:.2f} "
        f"first_rate={metrics.first_rate * 100:.1f}%/{min_first_rate * 100:.1f}% "
        f"fourth_rate={metrics.fourth_rate * 100:.1f}%/{max_fourth_rate * 100:.1f}% "
        f"avg_score={metrics.average_score:.0f}/{min_avg_score:.0f} "
        f"win_rate={metrics.win_rate * 100:.2f}%/{min_win_rate * 100:.2f}% "
        f"open_rate={metrics.open_rate * 100:.2f}%/{max_open_rate * 100:.2f}% "
        f"errors={errors}"
    )
    return all(checks), reason


def _validation_threshold(args: argparse.Namespace, suffix: str, fallback: float) -> float:
    value = getattr(args, f"validation_{suffix}", None)
    return fallback if value is None else float(value)


def _validation_log_path(raw: str, log_path: Path) -> Path:
    if raw:
        return Path(raw)
    return log_path.with_name(f"{log_path.stem}_validation.jsonl")



def _review_log_path(raw: str, log_path: Path) -> Path | None:
    if raw.lower() in {"off", "none", "false", "0"}:
        return None
    if raw:
        return Path(raw)
    return log_path.with_name(f"{log_path.stem}_review.jsonl")


def _discard_review_record(
    match_index: int,
    hand_number: int,
    max_hand_number: int,
    state: dict[str, Any],
    seat: Seat,
    chosen_tile: str,
    potentials: dict[str, float],
    shape_reward: RewardBreakdown,
    defense_reward: RewardBreakdown,
    scoring: Any,
    private_wall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hand = list(state.get("hand") or [])
    after_hand = hand[:]
    if chosen_tile in after_hand:
        after_hand.remove(chosen_tile)
    metrics = analyze_hand(after_hand, state, seat.name, scoring)
    threats = _threatening_opponents(state, seat.name)
    danger_summary = _discard_danger_summary(hand, chosen_tile, state, threats, potentials)
    record = {
        **_base_review_context(match_index, hand_number, max_hand_number, state, seat),
        "kind": "discard",
        "chosen_tile": chosen_tile,
        "shanten_after": metrics.shanten,
        "ukeire_after": metrics.ukeire,
        "effective_tile_types_after": metrics.effective_tile_types,
        "is_tenpai_after": metrics.shanten == 0,
        "winning_tiles_after": metrics.winning_tiles,
        "winning_tile_types_after": metrics.winning_tile_types,
        "expected_value_after": round(float(metrics.expected_value), 3),
        "has_yaku_after": metrics.expected_value > 0.0,
        "chosen_shape": round(float(potentials.get(chosen_tile, 0.0)), 6),
        "best_shape_tile": _best_shape_tile(potentials) if potentials else None,
        "best_shape_gap": round(float(shape_reward.components.get("best_gap", 0.0)), 6),
        "relative_shape_reward": round(float(shape_reward.total), 6),
        "defense_reward": round(float(defense_reward.total), 6),
        **danger_summary,
    }
    if private_wall is not None:
        record["private_wall"] = private_wall
    return record


def _reaction_review_record(
    match_index: int,
    hand_number: int,
    max_hand_number: int,
    state: dict[str, Any],
    seat: Seat,
    action: str | None,
    tiles: tuple[str, ...],
    shape_reward: RewardBreakdown,
    scoring: Any,
    private_wall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hand = list(state.get("hand") or [])
    metrics = analyze_hand(hand, state, seat.name, scoring)
    hints = state.get("action_hints") or {}
    called_tile = hints.get("tile")
    record = {
        **_base_review_context(match_index, hand_number, max_hand_number, state, seat),
        "kind": "reaction",
        "reaction_action": action or "pass",
        "called_tile": called_tile,
        "used_tiles": list(tiles),
        "discarder": hints.get("discarder"),
        "shanten_before_reaction": metrics.shanten,
        "ukeire_before_reaction": metrics.ukeire,
        "expected_value_before_reaction": round(float(metrics.expected_value), 3),
        "relative_call_shape_reward": round(float(shape_reward.total), 6),
        "call_best_gap": round(float(shape_reward.components.get("best_call_gap", 0.0)), 6),
    }
    if private_wall is not None:
        record["private_wall"] = private_wall
    return record


def _base_review_context(
    match_index: int,
    hand_number: int,
    max_hand_number: int,
    state: dict[str, Any],
    seat: Seat,
) -> dict[str, Any]:
    scores = _state_scores(state)
    ordered = sorted(Seat, key=lambda score_seat: (-scores[score_seat], int(score_seat)))
    rank = ordered.index(seat) + 1
    own_score = scores[seat]
    first_score = scores[ordered[0]]
    last_score = scores[ordered[-1]]
    prev_score = scores[ordered[rank - 2]] if rank > 1 else own_score
    next_score = scores[ordered[rank]] if rank < 4 else own_score
    melds = state.get("melds") or {}
    riichi_declared = state.get("riichi_declared") or {}
    dealer_name = state.get("dealer") or "EAST"
    round_wind = state.get("round_wind") or ("SOUTH" if hand_number >= 4 else "EAST")
    return {
        "match_index": match_index,
        "hand_number": hand_number,
        "max_hand_number": max_hand_number,
        "round_wind": round_wind,
        "kyoku": hand_number % 4 + 1,
        "round_label": f"{round_wind}_{hand_number % 4 + 1}",
        "is_last_hand": hand_number >= max_hand_number - 1,
        "seat": seat.name,
        "dealer": dealer_name,
        "is_dealer": seat.name == dealer_name,
        "turn_count": int(state.get("turn_count", 0) or 0),
        "approx_junme": round(float(state.get("turn_count", 0) or 0) / 4.0, 2),
        "wall_count": int(state.get("wall_count", 0) or 0),
        "own_score": own_score,
        "rank": rank,
        "gap_to_first": own_score - first_score,
        "gap_to_prev": own_score - prev_score,
        "gap_to_next": own_score - next_score,
        "gap_to_last": own_score - last_score,
        "scores": {score_seat.name: scores[score_seat] for score_seat in Seat},
        "riichi_opponents": [
            score_seat.name
            for score_seat in Seat
            if score_seat != seat and riichi_declared.get(score_seat.name, False)
        ],
        "open_meld_counts": {
            score_seat.name: _open_meld_count_for_review(melds.get(score_seat.name, []))
            for score_seat in Seat
        },
    }


def _state_scores(state: dict[str, Any]) -> dict[Seat, int]:
    raw_scores = state.get("scores") or {}
    return {seat: int(raw_scores.get(seat.name, 25_000)) for seat in Seat}


def _discard_danger_summary(
    hand: list[str],
    chosen_tile: str,
    state: dict[str, Any],
    threats: list[dict[str, Any]],
    potentials: dict[str, float],
) -> dict[str, Any]:
    unique_tiles = list(dict.fromkeys(hand))
    chosen_danger = _candidate_danger(chosen_tile, state, threats)
    if unique_tiles:
        safest_tile = min(unique_tiles, key=lambda tile: (_candidate_danger(tile, state, threats), TILE_TYPES.index(tile)))
        safest_danger = _candidate_danger(safest_tile, state, threats)
    else:
        safest_tile = None
        safest_danger = 0.0
    safe_tile_count = sum(1 for tile in unique_tiles if _candidate_danger(tile, state, threats) <= 0.05)
    best_shape = max(potentials.values()) if potentials else 0.0
    safest_shape = potentials.get(safest_tile, 0.0) if safest_tile is not None else 0.0
    return {
        "heuristic_chosen_danger": round(float(chosen_danger), 6),
        "chosen_visible_safe_to": _genbutsu_to_seats(chosen_tile, state),
        "heuristic_safe_tile_count": safe_tile_count,
        "heuristic_safest_tile": safest_tile,
        "heuristic_safest_danger": round(float(safest_danger), 6),
        "shape_loss_if_heuristic_safest": round(float(best_shape - safest_shape), 6),
    }


def _genbutsu_to_seats(tile: str, state: dict[str, Any]) -> list[str]:
    return [seat for seat in SEAT_NAMES if _is_genbutsu(tile, state, seat)]


def _candidate_danger(tile: str, state: dict[str, Any], threats: list[dict[str, Any]]) -> float:
    danger = 0.0
    for threat in threats:
        if _is_genbutsu(tile, state, threat["seat"]):
            continue
        danger += _tile_danger(tile, state) * float(threat["weight"])
    return danger


def _private_wall_review_context(dealer: MahjongDealer) -> dict[str, Any]:
    """Hidden wall snapshot for offline review only.

    The policy never receives these fields.  They are useful for post-hand
    analysis such as whether a pushed tile could realistically have changed
    future draws, or whether an early safe discard later mattered.
    """
    live_wall = list(dealer.state.live_wall)
    dead_wall = list(dealer.state.dead_wall)
    return {
        "live_wall_remaining": live_wall,
        "live_wall_draw_order": list(reversed(live_wall)),
        "next_live_draw": live_wall[-1] if live_wall else None,
        "dead_wall": dead_wall,
        "dora_indicators": list(dealer.state.dora_indicators),
        "ura_dora_indicators": list(dealer.state.ura_dora_indicators),
        "rinshan_draws_remaining": dead_wall[dealer.state.rinshan_draws_used :],
    }


def _open_meld_count_for_review(melds: list[dict[str, Any]]) -> int:
    return sum(
        1
        for meld in melds
        if bool(meld.get("open", True)) and meld.get("kind") != "closed_kan"
    )


def _annotate_hand_reviews(
    records: list[dict[str, Any]],
    result: dict[str, Any],
    target_seat: Seat,
    score_delta: int,
    scores: dict[Seat, int],
) -> None:
    winners = set(result.get("winners") or [])
    target_won = target_seat.name in winners
    target_dealt_in = result.get("loser") == target_seat.name
    ordered = sorted(Seat, key=lambda seat: (-scores[seat], int(seat)))
    final_rank = ordered.index(target_seat) + 1
    for record in records:
        record.update(
            {
                "outcome_type": result.get("type"),
                "outcome_reason": result.get("reason"),
                "target_won_hand": target_won,
                "target_dealt_in": target_dealt_in,
                "opponent_won_hand": result.get("type") in {"ron", "tsumo"} and not target_won,
                "hand_score_delta": score_delta,
                "final_score_after_hand": scores[target_seat],
                "final_rank_after_hand": final_rank,
            }
        )

def _update_policy(
    model: DiscardNet,
    optimizer: torch.optim.Optimizer,
    traces: list[PolicyStep],
    entropy_coef: float,
    shape_ce_coef: float,
    value_loss_coef: float,
    value_target_clip: float,
    normalize_advantages: bool,
    grad_clip: float,
) -> float | None:
    if not traces:
        return None

    log_probs = torch.stack([step.log_prob for step in traces])
    entropies = torch.stack([step.entropy for step in traces])
    returns = torch.tensor([step.total_reward for step in traces], dtype=torch.float32)
    returns = _clamp_tensor(returns, value_target_clip)
    values = torch.stack([step.value for step in traces]) if all(step.value is not None for step in traces) else None
    advantages = returns - values.detach() if values is not None else returns
    if normalize_advantages and len(traces) > 1:
        advantages = advantages - advantages.mean()
        std = advantages.std(unbiased=False)
        if float(std.item()) > 1e-6:
            advantages = advantages / (std + 1e-6)

    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns) if values is not None else torch.tensor(0.0)
    entropy_loss = -entropy_coef * entropies.mean()
    shape_losses = [
        F.cross_entropy(step.shape_logits.unsqueeze(0), torch.tensor([step.shape_target]))
        for step in traces
        if step.shape_logits is not None and step.shape_target is not None
    ]
    shape_loss = torch.stack(shape_losses).mean() if shape_losses else torch.tensor(0.0)
    auxiliary_losses = [
        float(step.auxiliary_ce_coef)
        * F.cross_entropy(step.auxiliary_logits.unsqueeze(0), torch.tensor([step.auxiliary_target]))
        for step in traces
        if (
            step.auxiliary_logits is not None
            and step.auxiliary_target is not None
            and step.auxiliary_ce_coef > 0
        )
    ]
    auxiliary_loss = torch.stack(auxiliary_losses).mean() if auxiliary_losses else torch.tensor(0.0)
    loss = policy_loss + value_loss_coef * value_loss + shape_ce_coef * shape_loss + auxiliary_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    if grad_clip > 0:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
    optimizer.step()
    return float(loss.item())


def _clamp_tensor(values: torch.Tensor, limit: float) -> torch.Tensor:
    if limit <= 0:
        return values
    return values.clamp(min=-float(limit), max=float(limit))


def _hand_terminal_reward(
    result: dict[str, Any],
    target_seat: Seat,
    score_delta: int,
    opponent_win_penalty: float,
    bust_penalty: float = -20.0,
    final_score: int | None = None,
    weights: RewardWeights = NEWBIE_ATTACK_WEIGHTS,
) -> RewardBreakdown:
    reward = terminal_reward(
        result,
        target_seat.name,
        score_delta=score_delta,
        weights=weights,
    )
    components = dict(reward.components)
    if result.get("type") in {"ron", "tsumo"} and target_seat.name not in set(result.get("winners") or []):
        components["opponent_win"] = opponent_win_penalty
    if final_score is not None and final_score < 0:
        components["bust"] = bust_penalty
    return RewardBreakdown(total=sum(components.values()), components=components)


def _add_terminal_reward(
    traces: list[PolicyStep],
    reward: RewardBreakdown,
    scale: float = 1.0,
    clip: float = 0.0,
) -> None:
    if not traces:
        return
    value = _clip_float(reward.total * scale / len(traces), clip)
    for step in traces:
        step.terminal_reward = value


def _add_match_reward(
    traces: list[PolicyStep],
    reward: RewardBreakdown,
    scale: float = 1.0,
    clip: float = 0.0,
) -> None:
    if not traces:
        return
    value = _clip_float(reward.total * scale / len(traces), clip)
    for step in traces:
        step.match_reward = value


def _clip_float(value: float, limit: float) -> float:
    if limit <= 0:
        return value
    return max(-float(limit), min(float(limit), value))


def _target_hand_is_open(dealer: MahjongDealer, target_seat: Seat) -> bool:
    return any(meld.open and meld.kind != "closed_kan" for meld in dealer.state.melds[target_seat])


def _target_call_count(dealer: MahjongDealer, target_seat: Seat) -> int:
    return sum(
        1
        for meld in dealer.state.melds[target_seat]
        if meld.open and meld.kind != "closed_kan"
    )


def _rolling_metrics(stats: list[MatchTrainStats]) -> RollingMetrics:
    matches = len(stats)
    hands = sum(stat.hands for stat in stats)
    return RollingMetrics(
        matches=matches,
        hands=hands,
        average_rank=sum(stat.rank for stat in stats) / max(1, matches),
        first_rate=sum(1 for stat in stats if stat.rank == 1) / max(1, matches),
        fourth_rate=sum(1 for stat in stats if stat.rank == 4) / max(1, matches),
        average_score=sum(stat.final_score for stat in stats) / max(1, matches),
        win_rate=sum(stat.wins for stat in stats) / max(1, hands),
        open_rate=sum(stat.open_hands for stat in stats) / max(1, hands),
    )


def _early_stop_reason(stats: list[MatchTrainStats], args: argparse.Namespace) -> str | None:
    window = int(getattr(args, "early_stop_window", 0))
    if window <= 0 or len(stats) < window:
        return None

    recent = stats[-window:]
    if any(stat.error for stat in recent):
        return None

    metrics = _rolling_metrics(recent)
    checks = [
        metrics.average_rank <= args.early_stop_max_avg_rank,
        metrics.first_rate >= args.early_stop_min_first_rate,
        metrics.fourth_rate <= args.early_stop_max_fourth_rate,
        metrics.average_score >= args.early_stop_min_avg_score,
        metrics.win_rate >= args.early_stop_min_win_rate,
        metrics.open_rate <= args.early_stop_max_open_rate,
    ]
    if not all(checks):
        return None

    return (
        f"window={window} avg_rank={metrics.average_rank:.2f} "
        f"first_rate={metrics.first_rate * 100:.1f}% "
        f"fourth_rate={metrics.fourth_rate * 100:.1f}% "
        f"avg_score={metrics.average_score:.0f} "
        f"win_rate={metrics.win_rate * 100:.2f}% "
        f"open_rate={metrics.open_rate * 100:.2f}%"
    )


def _match_stats(
    match_index: int,
    seed: int,
    target_seat: Seat,
    scores: dict[Seat, int],
    hands_played: int,
    wins: int,
    tsumo_wins: int,
    open_hands: int,
    calls: int,
    tenpai_draws: int,
    noten_draws: int,
    opponent_wins: int,
    deal_ins: int,
    riichi_hands: int,
    mangan_plus_wins: int,
    yakuman_wins: int,
    traces: list[PolicyStep],
    loss_value: float | None,
) -> MatchTrainStats:
    return MatchTrainStats(
        match_index=match_index,
        seed=seed,
        timestamp=_timestamp(),
        target_seat=target_seat.name,
        hands=hands_played,
        decisions=len(traces),
        rank=_rank(scores, target_seat),
        final_score=scores[target_seat],
        wins=wins,
        tsumo_wins=tsumo_wins,
        open_hands=open_hands,
        calls=calls,
        tenpai_draws=tenpai_draws,
        noten_draws=noten_draws,
        opponent_wins=opponent_wins,
        deal_ins=deal_ins,
        riichi_hands=riichi_hands,
        mangan_plus_wins=mangan_plus_wins,
        yakuman_wins=yakuman_wins,
        shape_reward=sum(step.shape_reward for step in traces),
        defense_reward=sum(step.defense_reward for step in traces),
        terminal_reward=sum(step.terminal_reward for step in traces),
        match_reward=sum(step.match_reward for step in traces),
        total_reward=sum(step.total_reward for step in traces),
        loss=loss_value,
    )


def _target_seat(match_index: int, target_seat: str, seed: int = 0) -> Seat:
    if target_seat == "rotate":
        return Seat(match_index % 4)
    if target_seat == "random":
        return Seat(random.Random(seed + match_index).randrange(len(Seat)))
    return Seat[target_seat]


def _parse_opponents(raw: str | tuple[str, str, str]) -> tuple[str, str, str]:
    if isinstance(raw, tuple):
        return raw
    opponents = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(opponents) != 3:
        raise ValueError("--opponents must contain exactly three model names")
    unknown = [model_name for model_name in opponents if model_name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown opponent models: {unknown}; choose from {MODEL_NAMES}")
    return opponents


def _parse_optional_model(raw: str, option_name: str) -> str:
    model_name = raw.strip()
    if model_name and model_name not in MODEL_NAMES:
        raise ValueError(f"unknown {option_name}: {model_name}; choose from {MODEL_NAMES}")
    return model_name


def _parse_validation_opponents(raw: str, fallback: tuple[str, str, str]) -> tuple[str, str, str]:
    return fallback if not raw.strip() else _parse_opponents(raw)


def _teacher_prior_weight(
    match_index: int,
    start: float,
    end: float,
    decay_matches: int,
) -> float:
    if decay_matches <= 0:
        return max(0.0, float(start))
    progress = min(1.0, max(0.0, float(match_index) / float(decay_matches)))
    weight = float(start) + (float(end) - float(start)) * progress
    return max(0.0, weight)


def _parse_placement_rewards(raw: str | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if isinstance(raw, tuple):
        return raw
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 4:
        raise ValueError("--placement-rewards must contain exactly four comma-separated numbers")
    return values


def _reward_weights_from_args(args: argparse.Namespace) -> RewardWeights:
    return replace(
        NEWBIE_ATTACK_WEIGHTS,
        deal_in=args.deal_in_penalty,
        score_delta=args.score_delta_weight,
        expected_value=args.expected_value_weight,
    )


def _match_reward_weights_from_args(args: argparse.Namespace) -> MatchRewardWeights:
    return replace(
        NEWBIE_ATTACK_MATCH_WEIGHTS,
        placement=args.placement_rewards,
        score_delta=args.match_score_delta_weight,
    )


def _opponent_agents(
    opponents: tuple[str, str, str],
    target_seat: Seat,
    seed: int,
) -> dict[Seat, Any]:
    agents: dict[Seat, Any] = {}
    opponent_index = 0
    for seat in Seat:
        if seat == target_seat:
            continue
        model_name = opponents[opponent_index]
        agents[seat] = create_model(model_name, seed=seed * 100 + int(seat))
        opponent_index += 1
    return agents


def _resolve_seed(raw: str | int) -> int:
    if isinstance(raw, int):
        return raw
    if raw.lower() == "random":
        return secrets.randbelow(1_000_000_000)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("--seed must be an integer or random") from exc


def _resolve_planner_seed(raw: str | int | None, game_seed: int) -> int:
    if raw is None:
        return game_seed
    if isinstance(raw, str) and raw.lower() in {"", "same"}:
        return game_seed
    try:
        return _resolve_seed(raw)
    except ValueError as exc:
        raise ValueError("--planner-seed must be an integer, random, same, or empty") from exc


def _resolve_validation_seed(raw: str | int | None, game_seed: int) -> int:
    if raw is None:
        return game_seed
    if isinstance(raw, str) and raw.lower() in {"", "same"}:
        return game_seed
    try:
        return _resolve_seed(raw)
    except ValueError as exc:
        raise ValueError("--validation-seed must be an integer, random, same, or empty") from exc


def _rank(scores: dict[Seat, int], target_seat: Seat) -> int:
    ordered = sorted(Seat, key=lambda seat: (-scores[seat], int(seat)))
    return ordered.index(target_seat) + 1


def _load_checkpoint(
    model: DiscardNet,
    optimizer: torch.optim.Optimizer,
    path: Path,
    learning_rate: float | None = None,
    call_model: CallNet | None = None,
    load_call_model: bool = True,
    load_optimizer: bool = True,
) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    fully_loaded = load_state_dict_flexible(model, state_dict, name="discard_model")
    call_state_dict = checkpoint.get("call_model_state")
    if call_model is not None and call_state_dict is not None and load_call_model:
        fully_loaded = load_state_dict_flexible(call_model, call_state_dict, name="call_model") and fully_loaded
    elif call_model is not None and call_state_dict is not None and not load_call_model:
        print("reset call model: skipped call_model_state")
        fully_loaded = False
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None and load_optimizer and fully_loaded:
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError as exc:
            print(f"skipped optimizer state from {path}: {exc}")
    elif optimizer_state is not None:
        print("reset optimizer: skipped optimizer_state")
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    print(f"loaded: {path}")


def _save_checkpoint(
    path: Path,
    model: DiscardNet,
    call_model: CallNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    latest: MatchTrainStats | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "call_model_state": call_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tile_types": list(TILE_TYPES),
            "trainer": "policy_random",
            "args": vars(args),
            "latest": asdict(latest) if latest is not None else None,
        },
        path,
    )
    print(f"saved: {path}")


def _print_match(stat: MatchTrainStats, total_matches: int) -> None:
    loss_text = "N/A" if stat.loss is None else f"{stat.loss:.3f}"
    print(
        f"time={stat.timestamp} match={stat.match_index + 1}/{total_matches} seed={stat.seed} "
        f"seat={stat.target_seat} rank={stat.rank} score={stat.final_score} "
        f"wins={stat.wins} calls={stat.calls} open_hands={stat.open_hands} "
        f"deal_ins={stat.deal_ins} decisions={stat.decisions} "
        f"reward={stat.total_reward:.3f} loss={loss_text} error={stat.error}"
    )


def _print_summary(
    stats: list[MatchTrainStats],
    checkpoint_path: Path,
    log_path: Path,
) -> None:
    if not stats:
        return
    matches = len(stats)
    first_rate = sum(1 for stat in stats if stat.rank == 1) / matches
    fourth_rate = sum(1 for stat in stats if stat.rank == 4) / matches
    average_rank = sum(stat.rank for stat in stats) / matches
    average_score = sum(stat.final_score for stat in stats) / matches
    average_calls = sum(stat.calls for stat in stats) / matches
    open_rate = sum(stat.open_hands for stat in stats) / max(1, sum(stat.hands for stat in stats))
    average_defense = sum(stat.defense_reward for stat in stats) / matches
    print("\n=== policy training summary ===")
    print(f"matches       {matches}")
    print(f"first_rate    {first_rate * 100:.2f}%")
    print(f"fourth_rate   {fourth_rate * 100:.2f}%")
    print(f"average_rank  {average_rank:.2f}")
    print(f"average_score {average_score:.0f}")
    print(f"average_calls {average_calls:.2f}")
    print(f"open_rate     {open_rate * 100:.2f}%")
    print(f"avg_defense   {average_defense:.2f}")
    print(f"checkpoint    {checkpoint_path}")
    print(f"log           {log_path}")


def _print_validation_start(
    validation_index: int,
    after_match: int,
    validation_matches: int,
    seed: int,
    opponents: tuple[str, str, str],
    match_type: str,
    target_seat: str,
) -> None:
    print(
        f"validation[{validation_index + 1}] start after_match={after_match} "
        f"matches={validation_matches} seed={seed} "
        f"opponents={','.join(opponents)} "
        f"match_type={match_type} target_seat={target_seat}",
        flush=True,
    )


def _print_validation_progress(
    validation_index: int,
    completed: int,
    total: int,
    stats: list[MatchTrainStats],
) -> None:
    metrics = _rolling_metrics(stats)
    errors = sum(1 for stat in stats if stat.error)
    print(
        f"validation[{validation_index + 1}] progress={completed}/{total} "
        f"avg_rank={metrics.average_rank:.2f} "
        f"first_rate={metrics.first_rate * 100:.1f}% "
        f"fourth_rate={metrics.fourth_rate * 100:.1f}% "
        f"avg_score={metrics.average_score:.0f} "
        f"win_rate={metrics.win_rate * 100:.2f}% "
        f"open_rate={metrics.open_rate * 100:.2f}% "
        f"errors={errors}",
        flush=True,
    )


def _print_validation(record: ValidationRecord) -> None:
    status = "PASS" if record.passed else "FAIL"
    print(
        f"validation[{record.validation_index + 1}] after_match={record.after_match} "
        f"status={status} avg_rank={record.average_rank:.2f} "
        f"first_rate={record.first_rate * 100:.1f}% "
        f"fourth_rate={record.fourth_rate * 100:.1f}% "
        f"avg_score={record.average_score:.0f} "
        f"win_rate={record.win_rate * 100:.2f}% "
        f"open_rate={record.open_rate * 100:.2f}% errors={record.errors}"
    )
    if record.reason:
        print(f"validation[{record.validation_index + 1}] result {record.reason}")


def _print_validation_summary(records: list[ValidationRecord], path: Path | None) -> None:
    latest = records[-1]
    print("\n=== validation summary ===")
    print(f"runs          {len(records)}")
    print(f"latest_status {'PASS' if latest.passed else 'FAIL'}")
    print(f"average_rank  {latest.average_rank:.2f}")
    print(f"first_rate    {latest.first_rate * 100:.2f}%")
    print(f"fourth_rate   {latest.fourth_rate * 100:.2f}%")
    print(f"average_score {latest.average_score:.0f}")
    print(f"win_rate      {latest.win_rate * 100:.2f}%")
    print(f"open_rate     {latest.open_rate * 100:.2f}%")
    if latest.reason:
        print(f"latest_result {latest.reason}")
    if path is not None:
        print(f"validation_log {path}")


def _json_line(stat: MatchTrainStats) -> str:
    import json

    return json.dumps(asdict(stat), ensure_ascii=False, sort_keys=True)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
