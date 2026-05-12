"""Supervised pretraining for the neural Mahjong policy.

Use an existing baseline agent (``tile_efficiency`` or ``shanten``) as a
teacher.  This learns basic discard/shape decisions before policy-gradient RL,
which avoids the sparse-reward problem where a random policy rarely wins and
therefore gets almost no useful signal.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from jongmind.dealer import MahjongDealer
from jongmind.game import DealerCommand, Phase, Seat
from jongmind_ai import MODEL_NAMES, create_model
from jongmind_ai.base import forced_tsumogiri_tile
from jongmind_ai.features import (
    REACTION_ACTIONS,
    TILE_TYPES,
    encode_reaction_state,
    encode_state,
    legal_discard_mask,
    legal_reaction_mask,
)
from jongmind_ai.neural_beginner import DEFAULT_CHECKPOINT, CallNet, DiscardNet
from jongmind.train_policy_random import (
    _choose_reaction,
    _load_checkpoint,
    _opponent_agents,
    _parse_opponents,
    _rank,
    _resolve_planner_seed,
    _resolve_seed,
    _target_seat,
)

SEAT_NAMES = tuple(seat.name for seat in Seat)


@dataclass(frozen=True)
class ImitationMatchStats:
    match_index: int
    seed: int
    timestamp: str
    target_seat: str
    hands: int
    discard_samples: int
    discard_correct: int
    call_pass_samples: int
    call_pass_correct: int
    updates: int
    loss: float
    rank: int
    final_score: int
    wins: int
    deal_ins: int
    open_hands: int
    calls: int
    error: str | None = None

    @property
    def discard_accuracy(self) -> float:
        return self.discard_correct / max(1, self.discard_samples)

    @property
    def call_pass_accuracy(self) -> float:
        return self.call_pass_correct / max(1, self.call_pass_samples)


class OnlineBatcher:
    def __init__(self, optimizer: torch.optim.Optimizer, batch_size: int, grad_clip: float) -> None:
        self.optimizer = optimizer
        self.batch_size = max(1, batch_size)
        self.grad_clip = grad_clip
        self.losses: list[torch.Tensor] = []
        self.total_loss = 0.0
        self.samples = 0
        self.updates = 0

    def add(self, loss: torch.Tensor, weight: float = 1.0) -> None:
        if weight <= 0:
            return
        weighted = loss * float(weight)
        self.losses.append(weighted)
        self.total_loss += float(weighted.detach().item())
        self.samples += 1
        if len(self.losses) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.losses:
            return
        loss = torch.stack(self.losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            params: list[torch.nn.Parameter] = []
            for group in self.optimizer.param_groups:
                params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        self.optimizer.step()
        self.losses.clear()
        self.updates += 1

    @property
    def average_loss(self) -> float:
        return self.total_loss / max(1, self.samples)


@dataclass
class HandImitationCounters:
    discard_samples: int = 0
    discard_correct: int = 0
    call_pass_samples: int = 0
    call_pass_correct: int = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=MODEL_NAMES, default="tile_efficiency")
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--match-type", choices=("east", "south"), default="east")
    parser.add_argument("--target-seat", choices=(*SEAT_NAMES, "rotate", "random"), default="rotate")
    parser.add_argument("--opponents", default="random,random,shanten")
    parser.add_argument("--seed", default="20260510")
    parser.add_argument("--planner-seed", default="")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--discard-target",
        choices=("hard", "soft"),
        default="hard",
        help="Use hard one-tile imitation or soft teacher scores over every legal discard.",
    )
    parser.add_argument(
        "--discard-temperature",
        type=float,
        default=0.35,
        help="Soft target temperature. Lower values stay closer to the teacher's top discard.",
    )
    parser.add_argument("--max-steps-per-hand", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output-log", default="outputs/imitation_training.jsonl")
    parser.add_argument(
        "--train-call-pass",
        action="store_true",
        help="Also train the reaction head to pass non-ron prompts, giving later RL a conservative call-head start.",
    )
    parser.add_argument(
        "--train-call-teacher",
        action="store_true",
        help="Train and execute reaction decisions from --call-teacher instead of forcing pass.",
    )
    parser.add_argument(
        "--call-teacher",
        default="",
        help="Optional model name for reaction imitation, for example open_call.",
    )
    parser.add_argument("--call-pass-weight", type=float, default=0.25)
    parser.add_argument(
        "--call-pass-bias",
        type=float,
        default=3.0,
        help="Initial output bias for the pass action in CallNet.",
    )
    args = parser.parse_args()

    args.seed = _resolve_seed(args.seed)
    args.planner_seed = _resolve_planner_seed(args.planner_seed, args.seed)
    args.opponents = _parse_opponents(args.opponents)
    if args.call_teacher and args.call_teacher not in MODEL_NAMES:
        raise ValueError(f"unknown --call-teacher: {args.call_teacher}; choose from {MODEL_NAMES}")

    random.seed(args.seed)
    torch.manual_seed(args.planner_seed)

    model = DiscardNet()
    call_model = CallNet()
    _set_call_pass_bias(call_model, args.call_pass_bias)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *call_model.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.init_checkpoint:
        _load_checkpoint(
            model,
            optimizer,
            Path(args.init_checkpoint),
            learning_rate=args.learning_rate,
            call_model=call_model,
            load_call_model=True,
            load_optimizer=not args.reset_optimizer,
        )
        _set_call_pass_bias(call_model, args.call_pass_bias)

    print(
        f"imitation teacher={args.teacher} matches={args.matches} "
        f"seed={args.seed} planner_seed={args.planner_seed} "
        f"opponents={','.join(args.opponents)} discard_target={args.discard_target} "
        f"train_call_pass={args.train_call_pass} train_call_teacher={args.train_call_teacher} "
        f"call_teacher={args.call_teacher or args.teacher}"
    )

    checkpoint_path = Path(args.checkpoint)
    log_path = Path(args.output_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats: list[ImitationMatchStats] = []
    with open(log_path, "w", encoding="utf-8") as log:
        for match_index in range(args.matches):
            seed = args.seed + match_index
            target_seat = _target_seat(match_index, args.target_seat, args.seed)
            result = train_imitation_match(
                model=model,
                call_model=call_model,
                optimizer=optimizer,
                match_index=match_index,
                seed=seed,
                target_seat=target_seat,
                teacher_name=args.teacher,
                opponents=args.opponents,
                match_type=args.match_type,
                batch_size=args.batch_size,
                grad_clip=args.grad_clip,
                train_call_pass=args.train_call_pass,
                train_call_teacher=args.train_call_teacher,
                call_teacher_name=args.call_teacher,
                call_pass_weight=args.call_pass_weight,
                discard_target=args.discard_target,
                discard_temperature=args.discard_temperature,
                max_steps_per_hand=args.max_steps_per_hand,
            )
            stats.append(result)
            log.write(json.dumps(_stats_dict(result), ensure_ascii=False, sort_keys=True) + "\n")
            log.flush()
            if args.progress_every > 0 and ((match_index + 1) % args.progress_every == 0 or result.error):
                _print_match(result, args.matches)
            if (match_index + 1) % args.save_every == 0:
                _save_imitation_checkpoint(checkpoint_path, model, call_model, optimizer, args, result)

    _save_imitation_checkpoint(checkpoint_path, model, call_model, optimizer, args, stats[-1] if stats else None)
    _print_summary(stats, checkpoint_path, log_path)
    if any(stat.error for stat in stats):
        raise SystemExit(1)


def train_imitation_match(
    model: DiscardNet,
    call_model: CallNet,
    optimizer: torch.optim.Optimizer,
    match_index: int,
    seed: int,
    target_seat: Seat,
    teacher_name: str,
    opponents: tuple[str, str, str],
    match_type: str,
    batch_size: int,
    grad_clip: float,
    train_call_pass: bool,
    train_call_teacher: bool,
    call_teacher_name: str,
    call_pass_weight: float,
    discard_target: str,
    discard_temperature: float,
    max_steps_per_hand: int,
) -> ImitationMatchStats:
    model.train()
    call_model.train()
    dealer = MahjongDealer(seed=seed)
    scores = {seat: dealer.rules.starting_points for seat in Seat}
    teacher = create_model(teacher_name, seed=seed * 100 + 999)
    call_teacher = create_model(call_teacher_name, seed=seed * 100 + 1999) if call_teacher_name else teacher
    opponent_agents = _opponent_agents(opponents, target_seat, seed)
    batcher = OnlineBatcher(optimizer, batch_size=batch_size, grad_clip=grad_clip)

    max_hand_number = 4 if match_type == "east" else 8
    hand_number = 0
    honba = 0
    riichi_sticks = 0
    hands_played = 0
    wins = 0
    deal_ins = 0
    open_hands = 0
    calls = 0
    counters = HandImitationCounters()

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
            before_calls = _target_call_count(dealer, target_seat)

            _play_imitation_hand(
                dealer=dealer,
                model=model,
                call_model=call_model,
                teacher=teacher,
                call_teacher=call_teacher,
                target_seat=target_seat,
                opponents=opponent_agents,
                batcher=batcher,
                counters=counters,
                train_call_pass=train_call_pass,
                train_call_teacher=train_call_teacher,
                call_pass_weight=call_pass_weight,
                discard_target=discard_target,
                discard_temperature=discard_temperature,
                max_steps_per_hand=max_steps_per_hand,
            )

            batcher.flush()
            hands_played += 1
            calls += _target_call_count(dealer, target_seat) - before_calls
            if _target_hand_is_open(dealer, target_seat):
                open_hands += 1

            result = dealer.state.result or {}
            scores = dealer.state.scores.copy()
            riichi_sticks = dealer.state.riichi_sticks
            if result.get("type") in {"ron", "tsumo"}:
                winners = [Seat[name] for name in result.get("winners", [])]
                if target_seat in winners:
                    wins += 1
                if result.get("loser") == target_seat.name:
                    deal_ins += 1
                if dealer_seat in winners:
                    honba += 1
                else:
                    honba = 0
                    hand_number += 1
            else:
                tenpai = {Seat[name] for name in result.get("tenpai", [])}
                honba += 1
                if dealer_seat not in tenpai:
                    hand_number += 1

            if any(score < 0 for score in scores.values()):
                break

        batcher.flush()
        return ImitationMatchStats(
            match_index=match_index,
            seed=seed,
            timestamp=_timestamp(),
            target_seat=target_seat.name,
            hands=hands_played,
            discard_samples=counters.discard_samples,
            discard_correct=counters.discard_correct,
            call_pass_samples=counters.call_pass_samples,
            call_pass_correct=counters.call_pass_correct,
            updates=batcher.updates,
            loss=batcher.average_loss,
            rank=_rank(scores, target_seat),
            final_score=scores[target_seat],
            wins=wins,
            deal_ins=deal_ins,
            open_hands=open_hands,
            calls=calls,
        )
    except Exception as exc:
        batcher.flush()
        return ImitationMatchStats(
            match_index=match_index,
            seed=seed,
            timestamp=_timestamp(),
            target_seat=target_seat.name,
            hands=hands_played,
            discard_samples=counters.discard_samples,
            discard_correct=counters.discard_correct,
            call_pass_samples=counters.call_pass_samples,
            call_pass_correct=counters.call_pass_correct,
            updates=batcher.updates,
            loss=batcher.average_loss,
            rank=_rank(scores, target_seat),
            final_score=scores[target_seat],
            wins=wins,
            deal_ins=deal_ins,
            open_hands=open_hands,
            calls=calls,
            error=str(exc),
        )


def _play_imitation_hand(
    dealer: MahjongDealer,
    model: DiscardNet,
    call_model: CallNet,
    teacher: Any,
    call_teacher: Any,
    target_seat: Seat,
    opponents: dict[Seat, Any],
    batcher: OnlineBatcher,
    counters: HandImitationCounters,
    train_call_pass: bool,
    train_call_teacher: bool,
    call_pass_weight: float,
    discard_target: str,
    discard_temperature: float,
    max_steps_per_hand: int,
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
                    decision = teacher.choose_discard(player_state["hand"], player_state)
                    tile = decision.tile
                    _add_discard_supervision(
                        model=model,
                        teacher=teacher,
                        hand=player_state["hand"],
                        state=player_state,
                        teacher_tile=tile,
                        batcher=batcher,
                        counters=counters,
                        target_mode=discard_target,
                        temperature=discard_temperature,
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
                teacher_reaction = None
                if train_call_teacher:
                    teacher_reaction = _choose_reaction(call_teacher, player_state, legal_actions)
                    target_action = teacher_reaction[0] if teacher_reaction is not None else "pass"
                    _add_call_action_supervision(
                        call_model=call_model,
                        state=player_state,
                        target_action=target_action,
                        batcher=batcher,
                        counters=counters,
                        weight=call_pass_weight,
                    )
                elif train_call_pass:
                    _add_call_action_supervision(
                        call_model=call_model,
                        state=player_state,
                        target_action="pass",
                        batcher=batcher,
                        counters=counters,
                        weight=call_pass_weight,
                    )
                if teacher_reaction is not None:
                    dealer.handle(
                        DealerCommand(
                            kind="call",
                            seat=seat,
                            action=teacher_reaction[0],
                            tiles=teacher_reaction[1],
                        )
                    )
                else:
                    dealer.handle(DealerCommand(kind="pass", seat=seat))
            else:
                reaction = _choose_reaction(opponents[seat], player_state, legal_actions)
                if reaction is None:
                    dealer.handle(DealerCommand(kind="pass", seat=seat))
                else:
                    dealer.handle(DealerCommand(kind="call", seat=seat, action=reaction[0], tiles=reaction[1]))
            continue

        raise RuntimeError(f"unknown phase: {phase}")

    raise RuntimeError(f"step limit reached: {max_steps_per_hand}")


def _add_discard_supervision(
    model: DiscardNet,
    teacher: Any,
    hand: list[str],
    state: dict[str, Any],
    teacher_tile: str,
    batcher: OnlineBatcher,
    counters: HandImitationCounters,
    target_mode: str,
    temperature: float,
) -> None:
    if teacher_tile not in TILE_TYPES:
        return
    features = encode_state(hand, state).unsqueeze(0)
    logits = model(features).squeeze(0)
    mask = legal_discard_mask(hand, state)
    masked_logits = logits.masked_fill(~mask, -1_000_000_000.0)
    if target_mode == "soft":
        target_distribution = _soft_discard_target(
            teacher=teacher,
            hand=hand,
            state=state,
            fallback_tile=teacher_tile,
            mask=mask,
            temperature=temperature,
        )
        log_probs = F.log_softmax(masked_logits, dim=0)
        loss = -(target_distribution * log_probs).sum()
    else:
        target = torch.tensor([TILE_TYPES.index(teacher_tile)], dtype=torch.long)
        loss = F.cross_entropy(masked_logits.unsqueeze(0), target)
    batcher.add(loss)
    counters.discard_samples += 1
    predicted = TILE_TYPES[int(torch.argmax(masked_logits).item())]
    if predicted == teacher_tile:
        counters.discard_correct += 1


def _soft_discard_target(
    teacher: Any,
    hand: list[str],
    state: dict[str, Any],
    fallback_tile: str,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scores = _teacher_discard_scores(teacher, hand, state)
    if not scores:
        return _one_hot_discard_target(fallback_tile)

    target_logits = torch.full((len(TILE_TYPES),), -1_000_000_000.0, dtype=torch.float32)
    scale = max(1e-6, float(temperature))
    for tile, score in scores.items():
        if tile not in TILE_TYPES:
            continue
        index = TILE_TYPES.index(tile)
        if bool(mask[index]):
            target_logits[index] = float(score) / scale

    if float(target_logits.max().item()) <= -999_999_999.0:
        return _one_hot_discard_target(fallback_tile)
    return torch.softmax(target_logits, dim=0)


def _teacher_discard_scores(teacher: Any, hand: list[str], state: dict[str, Any]) -> dict[str, float]:
    scorer = getattr(teacher, "score_discards", None)
    if scorer is None:
        return {}
    raw_scores = scorer(hand, state)
    scores: dict[str, float] = {}
    for tile, value in raw_scores.items():
        score = getattr(value, "score", value)
        scores[tile] = float(score)
    return scores


def _one_hot_discard_target(tile: str) -> torch.Tensor:
    target = torch.zeros(len(TILE_TYPES), dtype=torch.float32)
    if tile in TILE_TYPES:
        target[TILE_TYPES.index(tile)] = 1.0
    return target


def _add_call_action_supervision(
    call_model: CallNet,
    state: dict[str, Any],
    target_action: str,
    batcher: OnlineBatcher,
    counters: HandImitationCounters,
    weight: float,
) -> None:
    legal_actions = state.get("legal_actions") or []
    if target_action not in REACTION_ACTIONS or target_action not in legal_actions:
        return
    hand = state.get("hand") or []
    features = encode_reaction_state(hand, state).unsqueeze(0)
    logits = call_model(features).squeeze(0)
    mask = legal_reaction_mask(state)
    masked_logits = logits.masked_fill(~mask, -1_000_000_000.0)
    target_index = REACTION_ACTIONS.index(target_action)
    if not bool(mask[target_index]):
        return
    target = torch.tensor([target_index], dtype=torch.long)
    loss = F.cross_entropy(masked_logits.unsqueeze(0), target)
    batcher.add(loss, weight=weight)
    counters.call_pass_samples += 1
    predicted = REACTION_ACTIONS[int(torch.argmax(masked_logits).item())]
    if predicted == target_action:
        counters.call_pass_correct += 1


def _target_call_count(dealer: MahjongDealer, target_seat: Seat) -> int:
    return sum(1 for meld in dealer.state.melds[target_seat] if meld.open)


def _target_hand_is_open(dealer: MahjongDealer, target_seat: Seat) -> bool:
    return any(meld.open for meld in dealer.state.melds[target_seat])


def _set_call_pass_bias(call_model: CallNet, value: float) -> None:
    with torch.no_grad():
        final_layer = call_model.network[-1]
        final_layer.bias.zero_()
        final_layer.bias[REACTION_ACTIONS.index("pass")] = float(value)


def _save_imitation_checkpoint(
    path: Path,
    model: DiscardNet,
    call_model: CallNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    latest: ImitationMatchStats | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "call_model_state": call_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tile_types": list(TILE_TYPES),
            "trainer": "imitation",
            "args": vars(args),
            "latest": asdict(latest) if latest is not None else None,
        },
        path,
    )
    print(f"saved: {path}")


def _stats_dict(stats: ImitationMatchStats) -> dict[str, Any]:
    data = asdict(stats)
    data["discard_accuracy"] = stats.discard_accuracy
    data["call_pass_accuracy"] = stats.call_pass_accuracy
    data["call_accuracy"] = stats.call_pass_accuracy
    return data


def _print_match(stats: ImitationMatchStats, total_matches: int) -> None:
    print(
        f"time={stats.timestamp} match={stats.match_index + 1}/{total_matches} "
        f"seed={stats.seed} seat={stats.target_seat} rank={stats.rank} score={stats.final_score} "
        f"wins={stats.wins} deal_ins={stats.deal_ins} calls={stats.calls} open_hands={stats.open_hands} "
        f"discard_samples={stats.discard_samples} discard_acc={stats.discard_accuracy:.2%} "
        f"call_samples={stats.call_pass_samples} call_acc={stats.call_pass_accuracy:.2%} "
        f"loss={stats.loss:.4f} updates={stats.updates} error={stats.error}"
    )


def _print_summary(stats: list[ImitationMatchStats], checkpoint_path: Path, log_path: Path) -> None:
    if not stats:
        print("no imitation matches were run")
        return
    total_discard = sum(item.discard_samples for item in stats)
    total_discard_correct = sum(item.discard_correct for item in stats)
    total_call = sum(item.call_pass_samples for item in stats)
    total_call_correct = sum(item.call_pass_correct for item in stats)
    total_hands = sum(item.hands for item in stats)
    average_rank = sum(item.rank for item in stats) / len(stats)
    average_score = sum(item.final_score for item in stats) / len(stats)
    win_rate = sum(item.wins for item in stats) / max(1, total_hands)
    open_rate = sum(item.open_hands for item in stats) / max(1, total_hands)
    deal_in_rate = sum(item.deal_ins for item in stats) / max(1, total_hands)
    print("\n=== imitation summary ===")
    print(f"总对局数     {len(stats)}")
    print(f"总手数       {total_hands}")
    print(f"弃牌样本     {total_discard}")
    print(f"弃牌准确率   {total_discard_correct / max(1, total_discard):.2%}")
    print(f"call samples  {total_call}")
    print(f"call accuracy {total_call_correct / max(1, total_call):.2%}")
    print(f"平均顺位     {average_rank:.2f}")
    print(f"平均点数     {average_score:.0f}")
    print(f"和牌率       {win_rate:.2%}")
    print(f"放铳率       {deal_in_rate:.2%}")
    print(f"副露率       {open_rate:.2%}")
    print(f"checkpoint   {checkpoint_path}")
    print(f"log          {log_path}")


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
