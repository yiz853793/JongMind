"""Feature encoding for neural agents.

The neural policy intentionally stays lightweight (MLP-friendly), so the
encoder exposes several high-level but objective Mahjong features instead of
requiring the network to rediscover everything from raw tile counts.
"""

from __future__ import annotations

from typing import Any

import torch

from jongmind.tiles import build_wall, sort_tiles
from jongmind_ai.base import legal_discard_tiles
from jongmind_ai.rewards import HandMetrics, analyze_hand


SEAT_NAMES = ("EAST", "SOUTH", "WEST", "NORTH")
TILE_TYPES = tuple(dict.fromkeys(sort_tiles(build_wall(red_fives=True))))
TILE_TO_INDEX = {tile: index for index, tile in enumerate(TILE_TYPES)}

# 10 tile-count groups: own hand, dora, 4 discard rivers, 4 meld collections.
BASE_FEATURE_SIZE = len(TILE_TYPES) * 10 + 4 + 4 + 4 + 4 + 3
OFFENSE_FEATURE_SIZE = 8
SITUATION_FEATURE_SIZE = 4 + 4 + 16
FEATURE_SIZE = BASE_FEATURE_SIZE + OFFENSE_FEATURE_SIZE + SITUATION_FEATURE_SIZE

REACTION_ACTIONS = ("pass", "chii", "pon", "kan")
REACTION_FEATURE_SIZE = FEATURE_SIZE + len(TILE_TYPES) + 4 + len(REACTION_ACTIONS)


def tile_index(tile: str) -> int:
    return TILE_TO_INDEX[tile]


def encode_state(hand: list[str], state: dict[str, Any]) -> torch.Tensor:
    features: list[float] = []
    features.extend(_tile_counts(hand))
    features.extend(_tile_counts(state.get("dora", [])))

    discards = state.get("discards", {})
    for seat in SEAT_NAMES:
        features.extend(_tile_counts(discards.get(seat, [])))

    melds = state.get("melds", {})
    for seat in SEAT_NAMES:
        meld_tiles: list[str] = []
        for meld in melds.get(seat, []):
            meld_tiles.extend(meld.get("tiles", []))
        features.extend(_tile_counts(meld_tiles))

    scores = state.get("scores", {})
    features.extend(float(scores.get(seat, 25_000)) / 50_000.0 for seat in SEAT_NAMES)

    riichi_declared = state.get("riichi_declared", {})
    features.extend(1.0 if riichi_declared.get(seat, False) else 0.0 for seat in SEAT_NAMES)
    features.extend(_one_hot_seat(state.get("dealer")))
    features.extend(_one_hot_seat(state.get("current_turn")))

    features.append(float(state.get("wall_count", 70)) / 70.0)
    features.append(float(state.get("honba", 0)) / 8.0)
    features.append(float(state.get("riichi_sticks", 0)) / 4.0)

    features.extend(_offense_features(hand, state))
    features.extend(_situation_features(state))
    return torch.tensor(features, dtype=torch.float32)


def encode_reaction_state(hand: list[str], state: dict[str, Any]) -> torch.Tensor:
    hints = state.get("action_hints") or {}
    pending = state.get("pending_reaction") or {}
    pending_tile = hints.get("tile") or pending.get("tile")
    discarder = hints.get("discarder") or pending.get("discarder")
    legal_actions = set(state.get("legal_actions") or ())

    features = encode_state(hand, state).tolist()
    features.extend(_tile_one_hot(pending_tile))
    features.extend(_one_hot_seat(discarder))
    features.extend(1.0 if action in legal_actions else 0.0 for action in REACTION_ACTIONS)
    return torch.tensor(features, dtype=torch.float32)


def legal_discard_mask(hand: list[str], state: dict[str, Any] | None = None) -> torch.Tensor:
    mask = torch.zeros(len(TILE_TYPES), dtype=torch.bool)
    choices = legal_discard_tiles(hand, state or {})
    for tile in choices:
        mask[TILE_TO_INDEX[tile]] = True
    return mask


def legal_reaction_mask(state: dict[str, Any]) -> torch.Tensor:
    hints = state.get("action_hints") or {}
    legal_actions = set(state.get("legal_actions") or ())
    mask = torch.zeros(len(REACTION_ACTIONS), dtype=torch.bool)
    for index, action in enumerate(REACTION_ACTIONS):
        if action == "pass":
            mask[index] = "pass" in legal_actions
        else:
            mask[index] = action in legal_actions and bool(hints.get(action))
    if not mask.any():
        mask[REACTION_ACTIONS.index("pass")] = True
    return mask


def empty_state() -> dict[str, Any]:
    return {
        "seat": "EAST",
        "current_turn": "EAST",
        "dealer": "EAST",
        "round_wind": "EAST",
        "hand_number": 0,
        "max_hand_number": 4,
        "turn_count": 0,
        "wall_count": 70,
        "dora": [],
        "dora_indicators": [],
        "discards": {seat: [] for seat in SEAT_NAMES},
        "melds": {seat: [] for seat in SEAT_NAMES},
        "scores": {seat: 25_000 for seat in SEAT_NAMES},
        "riichi_declared": {seat: False for seat in SEAT_NAMES},
        "honba": 0,
        "riichi_sticks": 0,
    }


def _tile_counts(tiles: list[str]) -> list[float]:
    counts = [0.0] * len(TILE_TYPES)
    for tile in tiles:
        if tile in TILE_TO_INDEX:
            counts[TILE_TO_INDEX[tile]] += 1.0 / 4.0
    return counts


def _tile_one_hot(tile: str | None) -> list[float]:
    values = [0.0] * len(TILE_TYPES)
    if tile in TILE_TO_INDEX:
        values[TILE_TO_INDEX[tile]] = 1.0
    return values


def _one_hot_seat(seat_name: str | None) -> list[float]:
    return [1.0 if seat_name == seat else 0.0 for seat in SEAT_NAMES]


def _offense_features(hand: list[str], state: dict[str, Any]) -> list[float]:
    metrics = _best_current_metrics(hand, state)
    shanten = max(-1, min(metrics.shanten, 6))
    expected_value = max(0.0, min(metrics.expected_value, 24_000.0))
    return [
        float(shanten) / 6.0,
        min(metrics.ukeire, 80) / 80.0,
        min(metrics.effective_tile_types, 34) / 34.0,
        1.0 if metrics.shanten == 0 else 0.0,
        min(metrics.winning_tiles, 80) / 80.0,
        min(metrics.winning_tile_types, 34) / 34.0,
        expected_value / 24_000.0,
        1.0 if metrics.expected_value > 0.0 else 0.0,
    ]


def _best_current_metrics(hand: list[str], state: dict[str, Any]) -> HandMetrics:
    seat_name = state.get("seat") or state.get("current_turn") or "EAST"
    try:
        # A discard decision usually sees 14 tiles.  Ukeire is more meaningful
        # after discarding one tile, so expose the best post-discard offensive
        # metrics instead of a zero-ukeire 14-tile snapshot.
        if len(hand) % 3 == 2 and hand:
            best: HandMetrics | None = None
            for tile in dict.fromkeys(hand):
                after = hand[:]
                after.remove(tile)
                metrics = analyze_hand(after, state, seat_name)
                if best is None or _metric_key(metrics) > _metric_key(best):
                    best = metrics
            if best is not None:
                return best
        return analyze_hand(hand, state, seat_name)
    except Exception:
        return HandMetrics(
            shanten=6,
            ukeire=0,
            effective_tile_types=0,
            winning_tiles=0,
            winning_tile_types=0,
            expected_value=0.0,
        )


def _metric_key(metrics: HandMetrics) -> tuple[float, float, float, float]:
    return (
        -float(metrics.shanten),
        float(metrics.expected_value),
        float(metrics.ukeire),
        float(metrics.effective_tile_types),
    )


def _situation_features(state: dict[str, Any]) -> list[float]:
    seat_name = state.get("seat") or state.get("current_turn") or "EAST"
    dealer = state.get("dealer") or "EAST"
    hand_number = int(state.get("hand_number", 0) or 0)
    max_hand_number = int(state.get("max_hand_number", 0) or 0)
    if max_hand_number <= 0:
        max_hand_number = 8 if hand_number >= 4 or state.get("round_wind") == "SOUTH" else 4
    max_hand_number = max(1, max_hand_number)
    remaining_hands = max(0, max_hand_number - hand_number - 1)

    scores = _scores_by_name(state)
    score_context = _score_context(scores, seat_name)
    riichi_declared = state.get("riichi_declared") or {}
    melds = state.get("melds") or {}
    open_counts = {
        seat: _open_meld_count(melds.get(seat, []))
        for seat in SEAT_NAMES
    }
    opponent_open_counts = [count for seat, count in open_counts.items() if seat != seat_name]
    riichi_opponents = sum(
        1 for seat in SEAT_NAMES if seat != seat_name and riichi_declared.get(seat, False)
    )
    return [
        *_one_hot_seat(state.get("round_wind")),
        *_one_hot_seat(seat_name),
        hand_number / max(1.0, float(max_hand_number - 1)),
        remaining_hands / float(max_hand_number),
        1.0 if hand_number >= max_hand_number - 1 else 0.0,
        float(state.get("turn_count", 0) or 0) / 70.0,
        1.0 if seat_name == dealer else 0.0,
        scores.get(seat_name, 25_000) / 50_000.0,
        score_context["rank"] / 4.0,
        score_context["gap_to_first"] / 50_000.0,
        score_context["gap_to_prev"] / 50_000.0,
        score_context["gap_to_next"] / 50_000.0,
        score_context["gap_to_last"] / 50_000.0,
        riichi_opponents / 3.0,
        (max(opponent_open_counts) if opponent_open_counts else 0) / 4.0,
        open_counts.get(dealer, 0) / 4.0,
        open_counts.get(seat_name, 0) / 4.0,
        1.0 if riichi_declared.get(dealer, False) else 0.0,
    ]


def _scores_by_name(state: dict[str, Any]) -> dict[str, int]:
    raw_scores = state.get("scores") or {}
    scores: dict[str, int] = {}
    for seat in SEAT_NAMES:
        value = raw_scores.get(seat, 25_000)
        scores[seat] = int(value)
    return scores


def _score_context(scores: dict[str, int], seat_name: str) -> dict[str, float]:
    ordered = sorted(SEAT_NAMES, key=lambda seat: (-scores.get(seat, 25_000), SEAT_NAMES.index(seat)))
    rank_index = ordered.index(seat_name) if seat_name in ordered else 0
    own_score = scores.get(seat_name, 25_000)
    first_score = scores.get(ordered[0], own_score)
    last_score = scores.get(ordered[-1], own_score)
    prev_score = scores.get(ordered[rank_index - 1], own_score) if rank_index > 0 else own_score
    next_score = scores.get(ordered[rank_index + 1], own_score) if rank_index + 1 < len(ordered) else own_score
    return {
        "rank": float(rank_index + 1),
        "gap_to_first": float(own_score - first_score),
        "gap_to_prev": float(own_score - prev_score),
        "gap_to_next": float(own_score - next_score),
        "gap_to_last": float(own_score - last_score),
    }


def _open_meld_count(melds: list[dict[str, Any]]) -> int:
    return sum(
        1
        for meld in melds
        if bool(meld.get("open", True)) and meld.get("kind") != "closed_kan"
    )
