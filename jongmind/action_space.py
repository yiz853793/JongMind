"""Unified action ids and public observations for AI-facing dealer state."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any

from jongmind.tiles import build_wall, is_suited, normalize_tile, sort_tiles, tile_rank, tile_suit


TILE_TYPES = tuple(dict.fromkeys(sort_tiles(build_wall(red_fives=True))))
TILE_TO_ACTION_ID = {tile: index for index, tile in enumerate(TILE_TYPES)}

DRAW_ACTION_ID = len(TILE_TYPES)
TSUMO_ACTION_ID = DRAW_ACTION_ID + 1
RON_ACTION_ID = TSUMO_ACTION_ID + 1
RIICHI_ACTION_ID = RON_ACTION_ID + 1
PASS_ACTION_ID = RIICHI_ACTION_ID + 1
ABORTIVE_DRAW_ACTION_ID = PASS_ACTION_ID + 1
FIRST_CALL_ACTION_ID = ABORTIVE_DRAW_ACTION_ID + 1

FIXED_ACTION_TEXT = {
    DRAW_ACTION_ID: "draw",
    TSUMO_ACTION_ID: "tsumo",
    RON_ACTION_ID: "ron",
    RIICHI_ACTION_ID: "riichi",
    PASS_ACTION_ID: "pass",
    ABORTIVE_DRAW_ACTION_ID: "abortive_draw",
}
TEXT_TO_FIXED_ACTION_ID = {text: action_id for action_id, text in FIXED_ACTION_TEXT.items()}


@dataclass(frozen=True)
class ActionSpec:
    id: int
    text: str
    kind: str
    tile: str | None = None
    tiles: tuple[str, ...] = ()
    riichi_discard_tiles: tuple[str, ...] = ()

    def public_view(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
        }
        if self.tile is not None:
            view["tile"] = self.tile
        if self.tiles:
            view["tiles"] = list(self.tiles)
        if self.riichi_discard_tiles:
            view["riichi_discard_tiles"] = list(self.riichi_discard_tiles)
        return view


def legal_action_specs(state: dict[str, Any]) -> list[ActionSpec]:
    legal = set(state.get("legal_actions") or [])
    hints = state.get("action_hints") or {}
    specs: list[ActionSpec] = []

    if "discard" in legal:
        for tile in _legal_discard_tiles(state, hints):
            specs.append(ActionSpec(TILE_TO_ACTION_ID[tile], f"discard_{tile}", "discard", tile=tile))
    if "draw" in legal:
        specs.append(ActionSpec(DRAW_ACTION_ID, "draw", "draw"))
    if "tsumo" in legal:
        specs.append(ActionSpec(TSUMO_ACTION_ID, "tsumo", "tsumo"))
    if "ron" in legal:
        tile = _pending_tile(state, hints)
        specs.append(ActionSpec(RON_ACTION_ID, "ron", "ron", tile=tile))
    if "riichi" in legal:
        discard_tiles = tuple(hints.get("riichi", {}).get("discard_tiles", ()))
        specs.append(
            ActionSpec(
                RIICHI_ACTION_ID,
                "riichi",
                "riichi",
                riichi_discard_tiles=discard_tiles,
            )
        )
    if "pass" in legal:
        specs.append(ActionSpec(PASS_ACTION_ID, "pass", "pass"))
    if "abortive_draw" in legal:
        specs.append(ActionSpec(ABORTIVE_DRAW_ACTION_ID, "abortive_draw", "abortive_draw"))

    for action in ("chii", "pon", "kan"):
        if action not in legal:
            continue
        for tiles in hints.get(action, ()):
            tiles_tuple = tuple(tiles)
            text = _call_text(action, tiles_tuple)
            specs.append(ActionSpec(human_readable_to_action_id(text), text, action, tiles=tiles_tuple))

    if "closed_kan" in legal:
        for tiles in hints.get("closed_kan", ()):
            tiles_tuple = tuple(tiles)
            text = _call_text("closed_kan", tiles_tuple)
            specs.append(ActionSpec(human_readable_to_action_id(text), text, "closed_kan", tiles=tiles_tuple))

    if "added_kan" in legal:
        for candidate in hints.get("added_kan", ()):
            tile = candidate.get("tile") if isinstance(candidate, dict) else candidate
            if tile is None:
                continue
            text = _call_text("added_kan", (tile,))
            specs.append(ActionSpec(human_readable_to_action_id(text), text, "added_kan", tile=tile))

    return _dedupe_specs(specs)


def legal_action_ids(state: dict[str, Any]) -> list[int]:
    return [spec.id for spec in legal_action_specs(state)]


def legal_action_mask(state: dict[str, Any]) -> list[int]:
    legal_ids = set(legal_action_ids(state))
    return [1 if action_id in legal_ids else 0 for action_id in range(ACTION_SPACE_SIZE)]


def public_observation(state: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "seat",
        "hand",
        "current_turn",
        "dealer",
        "phase",
        "wall_count",
        "dead_wall_count",
        "dora_indicators",
        "dora",
        "discards",
        "melds",
        "hand_counts",
        "scores",
        "riichi_declared",
        "turn_count",
        "round_wind",
        "hand_number",
        "max_hand_number",
        "honba",
        "riichi_sticks",
        "kan_count",
        "legal_actions",
        "action_hints",
        "legal_action_ids",
        "legal_action_mask",
        "pending_reaction",
        "result",
    )
    return {field: state[field] for field in fields if field in state}


def action_id_to_human_readable(action_id: int) -> str:
    if 0 <= action_id < len(TILE_TYPES):
        return f"discard_{TILE_TYPES[action_id]}"
    if action_id in FIXED_ACTION_TEXT:
        return FIXED_ACTION_TEXT[action_id]
    if action_id in CALL_ID_TO_TEXT:
        return CALL_ID_TO_TEXT[action_id]
    raise ValueError(f"unknown action id: {action_id}")


def human_readable_to_action_id(action: str | dict[str, Any]) -> int:
    text = _action_text(action)
    if text.startswith("discard_"):
        tile = text.removeprefix("discard_")
        if tile not in TILE_TO_ACTION_ID:
            raise ValueError(f"unknown discard tile action: {text}")
        return TILE_TO_ACTION_ID[tile]
    if text in TEXT_TO_FIXED_ACTION_ID:
        return TEXT_TO_FIXED_ACTION_ID[text]
    if text in CALL_TEXT_TO_ID:
        return CALL_TEXT_TO_ID[text]
    raise ValueError(f"unknown action text: {text}")


def _legal_discard_tiles(state: dict[str, Any], hints: dict[str, Any]) -> list[str]:
    discard_hint = hints.get("discard") or {}
    legal_tiles = state.get("legal_discard_tiles") or discard_hint.get("tiles")
    tiles = legal_tiles if legal_tiles else state.get("hand", [])
    unique_tiles = []
    seen: set[str] = set()
    for tile in tiles:
        if tile in TILE_TO_ACTION_ID and tile not in seen:
            seen.add(tile)
            unique_tiles.append(tile)
    return unique_tiles


def _pending_tile(state: dict[str, Any], hints: dict[str, Any]) -> str | None:
    pending = state.get("pending_reaction") or {}
    return hints.get("tile") or pending.get("tile")


def _dedupe_specs(specs: list[ActionSpec]) -> list[ActionSpec]:
    seen: set[int] = set()
    unique: list[ActionSpec] = []
    for spec in specs:
        if spec.id in seen:
            continue
        seen.add(spec.id)
        unique.append(spec)
    return unique


def _action_text(action: str | dict[str, Any]) -> str:
    if isinstance(action, str):
        return action
    kind = str(action.get("kind") or action.get("action") or "")
    if kind == "discard":
        return f"discard_{action['tile']}"
    if kind in {"chii", "pon", "kan", "closed_kan"}:
        return _call_text(kind, tuple(action.get("tiles") or ()))
    if kind == "added_kan":
        tile = action.get("tile")
        tiles = (tile,) if tile is not None else tuple(action.get("tiles") or ())
        return _call_text(kind, tiles)
    return kind


def _call_text(action: str, tiles: tuple[str, ...]) -> str:
    return "_".join((action, *sort_tiles(tiles)))


def _build_call_action_tables() -> tuple[dict[str, int], dict[int, str]]:
    texts: list[str] = []
    for action, tiles in _call_action_candidates():
        texts.append(_call_text(action, tiles))
    text_to_id = {
        text: FIRST_CALL_ACTION_ID + index
        for index, text in enumerate(sorted(set(texts)))
    }
    return text_to_id, {action_id: text for text, action_id in text_to_id.items()}


def _call_action_candidates() -> list[tuple[str, tuple[str, ...]]]:
    candidates: list[tuple[str, tuple[str, ...]]] = []
    for pair in combinations_with_replacement(TILE_TYPES, 2):
        if _can_match_same_base(pair):
            candidates.append(("pon", tuple(sort_tiles(pair))))
        if _can_complete_sequence(pair):
            candidates.append(("chii", tuple(sort_tiles(pair))))
    for triple in combinations_with_replacement(TILE_TYPES, 3):
        if _can_match_same_base(triple):
            candidates.append(("kan", tuple(sort_tiles(triple))))
    for quad in combinations_with_replacement(TILE_TYPES, 4):
        if _can_match_same_base(quad):
            candidates.append(("closed_kan", tuple(sort_tiles(quad))))
    for tile in TILE_TYPES:
        candidates.append(("added_kan", (tile,)))
    return candidates


def _can_match_same_base(tiles: tuple[str, ...]) -> bool:
    return len({normalize_tile(tile) for tile in tiles}) == 1


def _can_complete_sequence(tiles: tuple[str, str]) -> bool:
    if any(not is_suited(tile) for tile in tiles):
        return False
    if len({tile_suit(tile) for tile in tiles}) != 1:
        return False
    ranks = sorted(tile_rank(tile) for tile in tiles)
    for called_rank in range(1, 10):
        all_ranks = sorted((*ranks, called_rank))
        if all_ranks[1] == all_ranks[0] + 1 and all_ranks[2] == all_ranks[1] + 1:
            return True
    return False


CALL_TEXT_TO_ID, CALL_ID_TO_TEXT = _build_call_action_tables()
ACTION_SPACE_SIZE = FIRST_CALL_ACTION_ID + len(CALL_TEXT_TO_ID)
