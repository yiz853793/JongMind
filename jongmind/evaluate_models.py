"""Evaluate Mahjong models over east/south matches.

The evaluator keeps the old command-line interface, but records and summarizes
all four players instead of only the target model.  Opponents are assigned stable
names such as random0, random1, random2 so duplicate models can be compared
fairly.
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

from jongmind.dealer import MahjongDealer
from jongmind.game import DealerCommand, Phase, Seat
from jongmind_ai import MODEL_NAMES, DiscardAgent, create_model


SEAT_NAMES = tuple(seat.name for seat in Seat)


@dataclass(frozen=True)
class PlayerSpec:
    name: str
    model: str
    is_target: bool = False


@dataclass
class PlayerMatchRecord:
    player: str
    model: str
    seat: str
    rank: int
    final_score: int
    wins: int
    tsumo_wins: int
    deal_ins: int
    open_hands: int
    riichi_hands: int
    riichi_wins: int
    draw_hands: int
    exhaustive_draws: int
    tenpai_draws: int
    mangan_plus_wins: int
    yakuman_wins: int
    perfect: bool
    max_renchan: int
    average_win_turn: float | None


@dataclass
class MatchRecord:
    match_index: int
    seed: int
    target_seat: str
    hands: int
    final_scores: dict[str, int]
    players: dict[str, PlayerMatchRecord]
    error: str | None = None


@dataclass
class PlayerHandStats:
    wins: int = 0
    tsumo_wins: int = 0
    deal_ins: int = 0
    open_hands: int = 0
    riichi_hands: int = 0
    riichi_wins: int = 0
    draw_hands: int = 0
    exhaustive_draws: int = 0
    tenpai_draws: int = 0
    mangan_plus_wins: int = 0
    yakuman_wins: int = 0
    max_renchan: int = 0
    current_renchan: int = 0
    win_turn_sum: float = 0.0
    win_turn_count: int = 0


@dataclass
class EvalTotals:
    player: str
    model: str
    matches: int = 0
    hands: int = 0
    rank_counts: dict[int, int] = field(default_factory=lambda: {rank: 0 for rank in range(1, 5)})
    final_score_sum: int = 0
    rank_sum: int = 0
    wins: int = 0
    tsumo_wins: int = 0
    deal_ins: int = 0
    open_hands: int = 0
    riichi_hands: int = 0
    riichi_wins: int = 0
    draw_hands: int = 0
    exhaustive_draws: int = 0
    tenpai_draws: int = 0
    mangan_plus_wins: int = 0
    yakuman_wins: int = 0
    perfect_matches: int = 0
    bust_matches: int = 0
    no_win_matches: int = 0
    first_without_win_matches: int = 0
    zero_win_matches: int = 0
    zero_win_first_matches: int = 0
    max_renchan: int = 0
    win_turn_sum: float = 0.0
    win_turn_count: int = 0
    errors: int = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="neural_beginner", choices=MODEL_NAMES)
    parser.add_argument(
        "--opponents",
        default="random,random,random",
        help="Comma-separated opponent models for the non-target seats.",
    )
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--match-type", choices=("east", "south"), default="east")
    parser.add_argument(
        "--seed",
        default="20260508",
        help="Base seed integer, or random to generate one at startup.",
    )
    parser.add_argument(
        "--target-seat",
        choices=(*SEAT_NAMES, "rotate", "random"),
        default="rotate",
        help="Target seat for every match, rotate deterministically, or randomize at match start.",
    )
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument(
        "--hand-log",
        default="",
        help="Optional JSONL hand history path. Use 'auto' to write hands.jsonl in the eval run dir.",
    )
    parser.add_argument("--max-steps-per-hand", type=int, default=2000)
    args = parser.parse_args()
    args.seed = _resolve_seed(args.seed)
    print(f"base_seed={args.seed}")

    opponents = _parse_opponents(args.opponents)
    target_player, opponent_players = _build_player_specs(args.target, opponents)
    run_dir = Path(args.output_dir) / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    hand_log_path = _hand_log_path(args.hand_log, run_dir)

    records: list[MatchRecord] = []
    totals: dict[str, EvalTotals] = {
        player.name: EvalTotals(player=player.name, model=player.model)
        for player in (target_player, *opponent_players)
    }
    for match_index in range(args.matches):
        seed = args.seed + match_index
        target_seat = _target_seat(match_index, args.target_seat, args.seed)
        lineup = _lineup(target_player, opponent_players, target_seat)
        record = run_match(
            match_index=match_index,
            seed=seed,
            match_type=args.match_type,
            target_seat=target_seat,
            lineup=lineup,
            max_steps_per_hand=args.max_steps_per_hand,
            hand_log_path=hand_log_path,
        )
        records.append(record)
        _accumulate(totals, record)
        _print_match_record(record, total_matches=args.matches)

    player_summaries = {player: _summary(player_totals) for player, player_totals in totals.items()}
    summary: dict[str, Any] = {
        "target": args.target,
        "opponents": list(opponents),
        "players": player_summaries,
        "match_type": args.match_type,
        "seed": args.seed,
        "target_seat": args.target_seat,
        "output_dir": str(run_dir),
        "hand_log": str(hand_log_path) if hand_log_path is not None else None,
        "errors": sum(player_summary["errors"] for player_summary in player_summaries.values()),
        "zero_win_matches": sum(1 for record in records if _record_total_wins(record) == 0),
        "zero_win_match_rate": (
            sum(1 for record in records if _record_total_wins(record) == 0) / max(1, len(records))
        ),
    }
    _write_outputs(run_dir, records, summary)
    _print_summary(summary)

    if summary["errors"]:
        raise SystemExit(1)


def run_match(
    match_index: int,
    seed: int,
    match_type: str,
    target_seat: Seat,
    lineup: dict[Seat, PlayerSpec],
    max_steps_per_hand: int = 2000,
    hand_log_path: Path | None = None,
) -> MatchRecord:
    dealer = MahjongDealer(
        seed=seed,
        hand_log_path=hand_log_path,
        record_context={
            "match_index": match_index,
            "match_type": match_type,
            "target_seat": target_seat.name,
            "lineup": {
                seat.name: {
                    "player": player.name,
                    "model": player.model,
                    "is_target": player.is_target,
                }
                for seat, player in lineup.items()
            },
        }
        if hand_log_path is not None
        else None,
    )
    scores = {seat: dealer.rules.starting_points for seat in Seat}
    agents: dict[Seat, DiscardAgent] = {
        seat: create_model(player.model, seed=seed * 100 + int(seat))
        for seat, player in lineup.items()
    }
    stats = {seat: PlayerHandStats() for seat in Seat}
    max_hand_number = 4 if match_type == "east" else 8
    hand_number = 0
    honba = 0
    riichi_sticks = 0
    hands_played = 0

    try:
        while hand_number < max_hand_number:
            round_wind = Seat.EAST if hand_number < 4 else Seat.SOUTH
            dealer.state.scores = scores.copy()
            dealer.state.hand_number = hand_number
            dealer.state.max_hand_number = max_hand_number
            dealer.state.round_wind = round_wind
            dealer.state.honba = honba
            dealer.state.riichi_sticks = riichi_sticks
            dealer.start_hand()
            dealer_seat = dealer.state.dealer

            _play_hand(dealer, agents, max_steps_per_hand=max_steps_per_hand)
            hands_played += 1

            result = dealer.state.result or {}
            scores = dealer.state.scores.copy()
            riichi_sticks = dealer.state.riichi_sticks
            for seat in Seat:
                if any(meld.open and meld.kind != "closed_kan" for meld in dealer.state.melds[seat]):
                    stats[seat].open_hands += 1
                if dealer.state.riichi_declared[seat]:
                    stats[seat].riichi_hands += 1

            if result.get("type") in {"ron", "tsumo"}:
                winners = [Seat[name] for name in result.get("winners", [])]
                loser_name = result.get("loser")
                loser = Seat[loser_name] if loser_name is not None else None
                for winner in winners:
                    stats[winner].wins += 1
                    if dealer.state.riichi_declared[winner]:
                        stats[winner].riichi_wins += 1
                    if result.get("type") == "tsumo":
                        stats[winner].tsumo_wins += 1
                    stats[winner].win_turn_sum += max(1.0, dealer.state.turn_count / 4.0)
                    stats[winner].win_turn_count += 1
                    hand_value = result.get("hands", {}).get(winner.name, {})
                    if _is_mangan_plus(hand_value):
                        stats[winner].mangan_plus_wins += 1
                    if _is_yakuman(hand_value):
                        stats[winner].yakuman_wins += 1
                if loser is not None:
                    stats[loser].deal_ins += 1

                dealer_keeps = dealer_seat in winners
                if dealer_keeps:
                    honba += 1
                else:
                    honba = 0
                    hand_number += 1
            else:
                tenpai = {Seat[name] for name in result.get("tenpai", [])}
                for seat in Seat:
                    stats[seat].draw_hands += 1
                    if result.get("reason") == "exhaustive":
                        stats[seat].exhaustive_draws += 1
                        if seat in tenpai:
                            stats[seat].tenpai_draws += 1
                dealer_keeps = dealer_seat in tenpai
                honba += 1
                if not dealer_keeps:
                    hand_number += 1

            if dealer_keeps:
                stats[dealer_seat].current_renchan += 1
                stats[dealer_seat].max_renchan = max(
                    stats[dealer_seat].max_renchan,
                    stats[dealer_seat].current_renchan,
                )
            else:
                stats[dealer_seat].current_renchan = 0

            if any(score < 0 for score in scores.values()):
                break

        return _make_match_record(
            match_index=match_index,
            seed=seed,
            target_seat=target_seat,
            hands_played=hands_played,
            lineup=lineup,
            scores=scores,
            stats=stats,
        )
    except Exception as exc:
        return _make_match_record(
            match_index=match_index,
            seed=seed,
            target_seat=target_seat,
            hands_played=hands_played,
            lineup=lineup,
            scores=scores,
            stats=stats,
            error=str(exc),
        )


def _make_match_record(
    match_index: int,
    seed: int,
    target_seat: Seat,
    hands_played: int,
    lineup: dict[Seat, PlayerSpec],
    scores: dict[Seat, int],
    stats: dict[Seat, PlayerHandStats],
    error: str | None = None,
) -> MatchRecord:
    ranks = _ranks(scores)
    players: dict[str, PlayerMatchRecord] = {}
    for seat in Seat:
        player = lineup[seat]
        seat_stats = stats[seat]
        players[player.name] = PlayerMatchRecord(
            player=player.name,
            model=player.model,
            seat=seat.name,
            rank=ranks[seat],
            final_score=scores[seat],
            wins=seat_stats.wins,
            tsumo_wins=seat_stats.tsumo_wins,
            deal_ins=seat_stats.deal_ins,
            open_hands=seat_stats.open_hands,
            riichi_hands=seat_stats.riichi_hands,
            riichi_wins=seat_stats.riichi_wins,
            draw_hands=seat_stats.draw_hands,
            exhaustive_draws=seat_stats.exhaustive_draws,
            tenpai_draws=seat_stats.tenpai_draws,
            mangan_plus_wins=seat_stats.mangan_plus_wins,
            yakuman_wins=seat_stats.yakuman_wins,
            perfect=seat_stats.deal_ins == 0 and seat_stats.mangan_plus_wins >= 4,
            max_renchan=seat_stats.max_renchan,
            average_win_turn=(
                seat_stats.win_turn_sum / seat_stats.win_turn_count
                if seat_stats.win_turn_count
                else None
            ),
        )
    return MatchRecord(
        match_index=match_index,
        seed=seed,
        target_seat=target_seat.name,
        hands=hands_played,
        final_scores={lineup[seat].name: scores[seat] for seat in Seat},
        players=players,
        error=error,
    )


def _play_hand(
    dealer: MahjongDealer,
    agents: dict[Seat, DiscardAgent],
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
            decision = agents[seat].choose_discard(player_state["hand"], player_state)
            riichi = (
                "riichi" in legal_actions
                and decision.tile in player_state.get("action_hints", {}).get("riichi", {}).get("discard_tiles", [])
            )
            dealer.handle(DealerCommand(kind="discard", seat=seat, tile=decision.tile, riichi=riichi))
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
            reaction = _choose_reaction(agents[seat], player_state, legal_actions)
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


def _choose_reaction(agent: Any, state: dict[str, Any], legal_actions: list[str]) -> tuple[str, tuple[str, ...]] | None:
    chooser = getattr(agent, "choose_reaction", None)
    if chooser is None:
        return None
    decision = chooser(state)
    if decision.action is None or decision.action not in legal_actions:
        return None
    return decision.action, decision.tiles


def _rank(scores: dict[Seat, int], target_seat: Seat) -> int:
    return _ranks(scores)[target_seat]


def _ranks(scores: dict[Seat, int]) -> dict[Seat, int]:
    ordered = sorted(Seat, key=lambda seat: (-scores[seat], int(seat)))
    return {seat: rank + 1 for rank, seat in enumerate(ordered)}


def _is_mangan_plus(hand_value: dict[str, Any]) -> bool:
    cost = hand_value.get("cost") or {}
    return int(cost.get("total", 0)) >= 8000 or int(cost.get("main", 0)) >= 8000


def _is_yakuman(hand_value: dict[str, Any]) -> bool:
    cost = hand_value.get("cost") or {}
    yaku_level = str(cost.get("yaku_level", "")).lower()
    return "yakuman" in yaku_level or int(cost.get("total", 0)) >= 32_000 or int(cost.get("main", 0)) >= 32_000


def _target_seat(match_index: int, target_seat: str, seed: int = 0) -> Seat:
    if target_seat == "rotate":
        return Seat(match_index % 4)
    if target_seat == "random":
        return Seat(Random(seed + match_index).randrange(len(Seat)))
    return Seat[target_seat]


def _resolve_seed(raw: str | int) -> int:
    if isinstance(raw, int):
        return raw
    if raw.lower() == "random":
        return secrets.randbelow(1_000_000_000)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("--seed must be an integer or random") from exc


def _parse_opponents(raw: str) -> tuple[str, str, str]:
    opponents = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(opponents) != 3:
        raise ValueError("--opponents must contain exactly three model names")
    unknown = [model_name for model_name in opponents if model_name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown opponent models: {unknown}; choose from {MODEL_NAMES}")
    return opponents


def _build_player_specs(target: str, opponents: tuple[str, str, str]) -> tuple[PlayerSpec, tuple[PlayerSpec, PlayerSpec, PlayerSpec]]:
    target_player = PlayerSpec(name=target, model=target, is_target=True)
    counts: dict[str, int] = {}
    opponent_players: list[PlayerSpec] = []
    for model in opponents:
        index = counts.get(model, 0)
        counts[model] = index + 1
        opponent_players.append(PlayerSpec(name=f"{model}{index}", model=model))
    return target_player, (opponent_players[0], opponent_players[1], opponent_players[2])


def _lineup(
    target_player: PlayerSpec | str,
    opponents: tuple[PlayerSpec, PlayerSpec, PlayerSpec] | tuple[str, str, str],
    target_seat: Seat,
) -> dict[Seat, PlayerSpec]:
    # Keep compatibility with the old helper shape used by earlier tests or scripts:
    # _lineup("neural_beginner", ("random", "random", "random"), Seat.EAST)
    if isinstance(target_player, str):
        target_player, opponents = _build_player_specs(target_player, opponents)  # type: ignore[arg-type]

    lineup: dict[Seat, PlayerSpec] = {}
    opponent_index = 0
    for seat in Seat:
        if seat == target_seat:
            lineup[seat] = target_player
        else:
            lineup[seat] = opponents[opponent_index]  # type: ignore[index]
            opponent_index += 1
    return lineup


def _hand_log_path(raw: str, run_dir: Path) -> Path | None:
    value = raw.strip()
    if not value:
        return None
    if value.lower() == "auto":
        return run_dir / "hands.jsonl"
    return Path(value)


def _accumulate(totals: dict[str, EvalTotals], record: MatchRecord) -> None:
    total_wins = _record_total_wins(record)
    for player_name, player_record in record.players.items():
        player_totals = totals[player_name]
        player_totals.matches += 1
        player_totals.hands += record.hands
        player_totals.rank_counts[player_record.rank] += 1
        player_totals.final_score_sum += player_record.final_score
        player_totals.rank_sum += player_record.rank
        player_totals.wins += player_record.wins
        player_totals.tsumo_wins += player_record.tsumo_wins
        player_totals.deal_ins += player_record.deal_ins
        player_totals.open_hands += player_record.open_hands
        player_totals.riichi_hands += player_record.riichi_hands
        player_totals.riichi_wins += player_record.riichi_wins
        player_totals.draw_hands += player_record.draw_hands
        player_totals.exhaustive_draws += player_record.exhaustive_draws
        player_totals.tenpai_draws += player_record.tenpai_draws
        player_totals.mangan_plus_wins += player_record.mangan_plus_wins
        player_totals.yakuman_wins += player_record.yakuman_wins
        player_totals.perfect_matches += 1 if player_record.perfect else 0
        player_totals.bust_matches += 1 if player_record.final_score < 0 else 0
        player_totals.no_win_matches += 1 if player_record.wins == 0 else 0
        player_totals.first_without_win_matches += (
            1 if player_record.rank == 1 and player_record.wins == 0 else 0
        )
        player_totals.zero_win_matches += 1 if total_wins == 0 else 0
        player_totals.zero_win_first_matches += (
            1 if total_wins == 0 and player_record.rank == 1 else 0
        )
        player_totals.max_renchan = max(player_totals.max_renchan, player_record.max_renchan)
        if player_record.average_win_turn is not None and player_record.wins:
            player_totals.win_turn_sum += player_record.average_win_turn * player_record.wins
            player_totals.win_turn_count += player_record.wins
        player_totals.errors += 1 if record.error else 0


def _record_total_wins(record: MatchRecord) -> int:
    return sum(player.wins for player in record.players.values())


def _summary(totals: EvalTotals) -> dict[str, Any]:
    matches = max(1, totals.matches)
    hands = max(1, totals.hands)
    riichi_hands = max(1, totals.riichi_hands)
    exhaustive_draws = max(1, totals.exhaustive_draws)
    return {
        "player": totals.player,
        "model": totals.model,
        "matches": totals.matches,
        "hands": totals.hands,
        "first_rate": totals.rank_counts[1] / matches,
        "second_rate": totals.rank_counts[2] / matches,
        "third_rate": totals.rank_counts[3] / matches,
        "fourth_rate": totals.rank_counts[4] / matches,
        "bust_rate": totals.bust_matches / matches,
        "average_score": totals.final_score_sum / matches,
        "average_rank": totals.rank_sum / matches,
        "max_renchan": totals.max_renchan,
        "average_win_turn": (
            totals.win_turn_sum / totals.win_turn_count if totals.win_turn_count else None
        ),
        "win_rate": totals.wins / hands,
        "tsumo_rate": totals.tsumo_wins / hands,
        "deal_in_rate": totals.deal_ins / hands,
        "open_rate": totals.open_hands / hands,
        "riichi_rate": totals.riichi_hands / hands,
        "riichi_win_rate": totals.riichi_wins / riichi_hands,
        "draw_rate": totals.draw_hands / hands,
        "exhaustive_draw_rate": totals.exhaustive_draws / hands,
        "tenpai_draw_rate": totals.tenpai_draws / exhaustive_draws,
        "no_win_match_rate": totals.no_win_matches / matches,
        "first_without_win_rate": totals.first_without_win_matches / matches,
        "zero_win_match_rate": totals.zero_win_matches / matches,
        "zero_win_first_rate": totals.zero_win_first_matches / matches,
        "riichi_wins": totals.riichi_wins,
        "draw_hands": totals.draw_hands,
        "exhaustive_draws": totals.exhaustive_draws,
        "tenpai_draws": totals.tenpai_draws,
        "mangan_plus_wins": totals.mangan_plus_wins,
        "yakuman_wins": totals.yakuman_wins,
        "perfect_matches": totals.perfect_matches,
        "errors": totals.errors,
    }


def _write_outputs(run_dir: Path, records: list[MatchRecord], summary: dict[str, Any]) -> None:
    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
    with open(run_dir / "matches.jsonl", "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
    with open(run_dir / "matches.csv", "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "match_index",
            "seed",
            "target_seat",
            "hands",
            "error",
            "player",
            "model",
            "seat",
            "rank",
            "final_score",
            "wins",
            "tsumo_wins",
            "deal_ins",
            "open_hands",
            "riichi_hands",
            "riichi_wins",
            "draw_hands",
            "exhaustive_draws",
            "tenpai_draws",
            "mangan_plus_wins",
            "yakuman_wins",
            "perfect",
            "max_renchan",
            "average_win_turn",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for player_record in record.players.values():
                row = {
                    "match_index": record.match_index,
                    "seed": record.seed,
                    "target_seat": record.target_seat,
                    "hands": record.hands,
                    "error": record.error,
                    **asdict(player_record),
                }
                writer.writerow(row)


def _print_match_record(record: MatchRecord, total_matches: int) -> None:
    print(f"\n=== match {record.match_index + 1}/{total_matches} seed={record.seed} hands={record.hands} ===")
    if record.error:
        print(f"error={record.error}")
    headers = ["座位", "玩家", "模型", "排位", "点数", "和牌", "自摸", "放铳", "副露", "立直"]
    rows: list[list[str]] = []
    ordered = sorted(record.players.values(), key=lambda player: Seat[player.seat])
    for player in ordered:
        rows.append(
            [
                player.seat,
                player.player,
                player.model,
                str(player.rank),
                str(player.final_score),
                str(player.wins),
                str(player.tsumo_wins),
                str(player.deal_ins),
                str(player.open_hands),
                str(player.riichi_hands),
            ]
        )
    print(_format_table(headers, rows, right_aligned={3, 4, 5, 6, 7, 8, 9}))


def _format_table(headers: list[str], rows: list[list[str]], right_aligned: set[int] | None = None) -> str:
    right_aligned = right_aligned or set()
    table_rows = [headers, *rows]
    widths = [
        max(_display_width(row[index]) for row in table_rows)
        for index in range(len(headers))
    ]
    lines: list[str] = []
    for row_index, row in enumerate(table_rows):
        cells = [
            _pad_display(
                row[index],
                widths[index],
                align_right=index in right_aligned and row_index > 0,
            )
            for index in range(len(headers))
        ]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def _pad_display(value: str, width: int, align_right: bool = False) -> str:
    padding = max(0, width - _display_width(value))
    if align_right:
        return " " * padding + value
    return value + " " * padding


def _display_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        for char in value
    )


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n=== evaluation summary ===")
    print(f"输出目录     {summary['output_dir']}")
    if summary.get("hand_log"):
        print(f"hand_log      {summary['hand_log']}")
    for player_name, player_summary in summary["players"].items():
        print(f"\n--- {player_name} ({player_summary['model']}) ---")
        print(f"总对局数     {player_summary['matches']}")
        print(f"总手数       {player_summary['hands']}")
        print(f"一位率       {_percent(player_summary['first_rate'])}")
        print(f"二位率       {_percent(player_summary['second_rate'])}")
        print(f"三位率       {_percent(player_summary['third_rate'])}")
        print(f"四位率       {_percent(player_summary['fourth_rate'])}")
        print(f"被飞率       {_percent(player_summary['bust_rate'])}")
        print(f"平均打点     {player_summary['average_score']:.0f}")
        print(f"平均顺位     {player_summary['average_rank']:.2f}")
        print(f"最大连庄     {player_summary['max_renchan']}")
        win_turn = player_summary["average_win_turn"]
        print(f"和了巡数     {win_turn:.2f}" if win_turn is not None else "和了巡数     N/A")
        print(f"和牌率       {_percent(player_summary['win_rate'])}")
        print(f"自摸率       {_percent(player_summary['tsumo_rate'])}")
        print(f"放铳率       {_percent(player_summary['deal_in_rate'])}")
        print(f"副露率       {_percent(player_summary['open_rate'])}")
        print(f"立直率       {_percent(player_summary['riichi_rate'])}")
        print(f"立直和牌率   {_percent(player_summary['riichi_win_rate'])}")
        print(f"流局率       {_percent(player_summary['draw_rate'])}")
        print(f"流局听牌率   {_percent(player_summary['tenpai_draw_rate'])}")
        print(f"零和牌场率   {_percent(player_summary['zero_win_match_rate'])}")
        print(f"无和一位率   {_percent(player_summary['first_without_win_rate'])}")
        print(f"役满次数     {player_summary['yakuman_wins']}")
        print(f"完美对局     {player_summary['perfect_matches']}")


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    main()
