"""Reward shaping helpers for reinforcement-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jongmind.game import Seat
from jongmind.rules import MAHJONG_SOUL_4P_RANKED
from jongmind.scoring import MahjongSoulScoring, WinContext
from jongmind.tiles import build_wall, normalize_tile, sort_tiles


@dataclass(frozen=True)
class RewardWeights:
    shanten: float = 0.15
    ukeire: float = 0.003
    effective_tile_types: float = 0.01
    tenpai: float = 1.0
    winning_tiles: float = 0.01
    winning_tile_types: float = 0.03
    expected_value: float = 0.0006
    no_yaku_tenpai: float = -1.5
    win: float = 8.0
    deal_in: float = -5.0
    exhaustive_draw_tenpai: float = 2.0
    exhaustive_draw_noten: float = -1.5
    score_delta: float = 0.00025


@dataclass(frozen=True)
class MatchRewardWeights:
    placement: tuple[float, float, float, float] = (3.0, 1.0, -1.0, -3.0)
    score_delta: float = 0.0001
    starting_points: int = 25_000


@dataclass(frozen=True)
class EffectiveDraw:
    tile: str
    count: int


@dataclass(frozen=True)
class HandMetrics:
    shanten: int
    ukeire: int
    effective_tile_types: int
    winning_tiles: int
    winning_tile_types: int
    expected_value: float


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]


@dataclass(frozen=True)
class _RewardMeld:
    kind: str
    tiles: tuple[str, ...]
    called_tile: str
    open: bool = True


_FULL_WALL = build_wall(red_fives=True)
_TILE_KINDS = tuple(dict.fromkeys(sort_tiles(_FULL_WALL)))
_WALL_COUNTS = {tile: _FULL_WALL.count(tile) for tile in _TILE_KINDS}


def analyze_hand(
    hand: list[str],
    state: dict[str, Any],
    seat_name: str = "EAST",
    scoring: MahjongSoulScoring | None = None,
) -> HandMetrics:
    scoring = scoring or MahjongSoulScoring(MAHJONG_SOUL_4P_RANKED)
    sorted_hand = sort_tiles(hand)
    shanten = scoring.shanten_count(sorted_hand)
    effective_draws = _effective_draws(sorted_hand, state, shanten, scoring)
    ukeire = sum(draw.count for draw in effective_draws)
    expected_value = _expected_win_value(sorted_hand, state, seat_name, shanten, effective_draws, scoring)
    winning_tiles = ukeire if shanten == 0 else 0
    winning_tile_types = len(effective_draws) if shanten == 0 else 0
    return HandMetrics(
        shanten=shanten,
        ukeire=ukeire,
        effective_tile_types=len(effective_draws),
        winning_tiles=winning_tiles,
        winning_tile_types=winning_tile_types,
        expected_value=expected_value,
    )


def shape_potential_reward(
    metrics: HandMetrics,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    components = {
        "shanten": -metrics.shanten * weights.shanten,
        "ukeire": metrics.ukeire * weights.ukeire,
        "effective_tile_types": metrics.effective_tile_types * weights.effective_tile_types,
        "tenpai": weights.tenpai if metrics.shanten == 0 else 0.0,
        "winning_tiles": metrics.winning_tiles * weights.winning_tiles,
        "winning_tile_types": metrics.winning_tile_types * weights.winning_tile_types,
        "expected_value": metrics.expected_value * weights.expected_value,
        "no_yaku_tenpai": (
            weights.no_yaku_tenpai
            if metrics.shanten == 0 and metrics.winning_tiles > 0 and metrics.expected_value <= 0.0
            else 0.0
        ),
    }
    return RewardBreakdown(total=sum(components.values()), components=components)


def step_shape_reward(
    before_hand: list[str],
    after_hand: list[str],
    state: dict[str, Any],
    seat_name: str = "EAST",
    weights: RewardWeights = RewardWeights(),
    scoring: MahjongSoulScoring | None = None,
) -> RewardBreakdown:
    scoring = scoring or MahjongSoulScoring(MAHJONG_SOUL_4P_RANKED)
    before = shape_potential_reward(analyze_hand(before_hand, state, seat_name, scoring), weights)
    after = shape_potential_reward(analyze_hand(after_hand, state, seat_name, scoring), weights)
    components = {
        name: after.components.get(name, 0.0) - before.components.get(name, 0.0)
        for name in after.components
    }
    return RewardBreakdown(total=sum(components.values()), components=components)


def terminal_reward(
    result: dict[str, Any],
    seat_name: str,
    score_delta: int | None = None,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    components: dict[str, float] = {}
    result_type = result.get("type")
    winners = set(result.get("winners") or [])

    if score_delta is not None:
        components["score_delta"] = score_delta * weights.score_delta

    if result_type in {"ron", "tsumo"}:
        if seat_name in winners:
            components["win"] = weights.win
            components["win_value"] = _winner_cost_value(result, seat_name) * weights.expected_value
        elif result.get("loser") == seat_name:
            components["deal_in"] = weights.deal_in
    elif result_type == "draw" and result.get("reason") == "exhaustive":
        tenpai = set(result.get("tenpai") or [])
        if seat_name in tenpai:
            components["exhaustive_draw_tenpai"] = weights.exhaustive_draw_tenpai
        else:
            components["exhaustive_draw_noten"] = weights.exhaustive_draw_noten

    return RewardBreakdown(total=sum(components.values()), components=components)


def match_terminal_reward(
    scores: dict[str | Seat, int],
    seat: str | Seat,
    weights: MatchRewardWeights = MatchRewardWeights(),
) -> RewardBreakdown:
    seat_name = _seat_name(seat)
    normalized_scores = {_seat_name(score_seat): score for score_seat, score in scores.items()}
    ranks = _score_ranks(normalized_scores)
    rank = ranks[seat_name]
    score_delta = normalized_scores[seat_name] - weights.starting_points
    components = {
        "placement": weights.placement[rank - 1],
        "match_score_delta": score_delta * weights.score_delta,
    }
    return RewardBreakdown(total=sum(components.values()), components=components)


def _effective_draws(
    hand: list[str],
    state: dict[str, Any],
    current_shanten: int,
    scoring: MahjongSoulScoring,
) -> tuple[EffectiveDraw, ...]:
    if len(hand) % 3 != 1:
        return ()

    visible_counts = _visible_counts(hand, state)
    draws: list[EffectiveDraw] = []
    for tile in _TILE_KINDS:
        visible = visible_counts.get(tile, 0)
        remaining = _WALL_COUNTS[tile] - visible
        if remaining <= 0:
            continue
        next_hand = sort_tiles([*hand, tile])
        if scoring.shanten_count(next_hand) < current_shanten:
            draws.append(EffectiveDraw(tile=tile, count=remaining))
    return tuple(draws)


def _visible_counts(hand: list[str], state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tile in hand:
        counts[tile] = counts.get(tile, 0) + 1
    for tile in state.get("dora_indicators", []):
        counts[tile] = counts.get(tile, 0) + 1
    for discards in state.get("discards", {}).values():
        for tile in discards:
            counts[tile] = counts.get(tile, 0) + 1
    for melds in state.get("melds", {}).values():
        for meld in melds:
            for tile in meld.get("tiles", []):
                counts[tile] = counts.get(tile, 0) + 1
    return counts


def _expected_win_value(
    hand: list[str],
    state: dict[str, Any],
    seat_name: str,
    shanten: int,
    effective_draws: tuple[EffectiveDraw, ...],
    scoring: MahjongSoulScoring,
) -> float:
    if shanten != 0 or not effective_draws:
        return 0.0

    seat = _seat_from_name(seat_name)
    dealer = _seat_from_name(state.get("dealer"), Seat.EAST)
    round_wind = _seat_from_name(state.get("round_wind"), Seat.EAST)
    melds = _state_melds(state, seat.name)
    is_closed = scoring.is_closed_hand(melds)
    total_value = 0.0
    total_weight = 0
    for draw in effective_draws:
        win_hand = sort_tiles([*hand, draw.tile])
        value = scoring.estimate_hand_value(
            concealed_tiles=win_hand,
            win_tile=draw.tile,
            melds=melds,
            seat_index=int(seat),
            dealer_index=int(dealer),
            round_wind_index=int(round_wind),
            dora_indicators=state.get("dora_indicators", []),
            context=WinContext(is_tsumo=True, is_riichi=is_closed),
        )
        if value.ok and value.cost is not None:
            total_value += _cost_value(value.cost) * draw.count
            total_weight += draw.count
    if total_weight == 0:
        return 0.0
    return total_value / total_weight


def _state_melds(state: dict[str, Any], seat_name: str) -> list[_RewardMeld]:
    melds: list[_RewardMeld] = []
    for meld in state.get("melds", {}).get(seat_name, []):
        tiles = tuple(meld.get("tiles") or ())
        called_tile = meld.get("called_tile") or (tiles[-1] if tiles else "")
        melds.append(
            _RewardMeld(
                kind=meld.get("kind", "pon"),
                tiles=tiles,
                called_tile=called_tile,
                open=bool(meld.get("open", True)),
            )
        )
    return melds


def _seat_from_name(name: str | None, default: Seat = Seat.EAST) -> Seat:
    if name is None:
        return default
    return Seat[name]


def _seat_name(seat: str | Seat) -> str:
    if isinstance(seat, Seat):
        return seat.name
    return seat


def _score_ranks(scores: dict[str, int]) -> dict[str, int]:
    seat_order = {seat.name: int(seat) for seat in Seat}
    ordered = sorted(
        scores,
        key=lambda seat_name: (-scores[seat_name], seat_order.get(seat_name, 99), seat_name),
    )
    return {seat_name: rank for rank, seat_name in enumerate(ordered, start=1)}


def _winner_cost_value(result: dict[str, Any], seat_name: str) -> int:
    hand = (result.get("hands") or {}).get(seat_name) or {}
    return _cost_value(hand.get("cost") or {})


def _cost_value(cost: dict[str, Any]) -> int:
    if "total" in cost:
        return int(cost["total"])
    main = int(cost.get("main", 0))
    additional = int(cost.get("additional", 0))
    if additional:
        return main + additional * 2
    return main
