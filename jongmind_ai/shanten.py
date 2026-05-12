"""Shanten-only baseline model."""

from __future__ import annotations

from typing import Any

from jongmind_ai.base import DiscardDecision, legal_discard_tiles
from jongmind_ai.tile_efficiency import DiscardScore
from jongmind.rules import MAHJONG_SOUL_4P_RANKED
from jongmind.scoring import MahjongSoulScoring
from jongmind.tiles import build_wall, is_terminal_or_honor, normalize_tile, sort_tiles


class ShantenAgent:
    """Minimize shanten after discard, with small deterministic tie-breakers."""

    def __init__(self) -> None:
        self.scoring = MahjongSoulScoring(MAHJONG_SOUL_4P_RANKED)
        self.tile_kinds = tuple(dict.fromkeys(sort_tiles(build_wall(red_fives=True))))

    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> DiscardDecision:
        if not hand:
            raise ValueError("cannot discard from an empty hand")

        candidates: list[tuple[tuple[int, int, int], DiscardScore]] = []
        for score in self.score_discards(hand, state).values():
            key = (score.shanten, score.discard_value, self._tile_order(score.tile))
            candidates.append((key, score))

        best = min(candidates, key=lambda item: item[0])[1]
        return DiscardDecision(
            tile=best.tile,
            shanten=best.shanten,
            ukeire=0,
            reason=f"min_shanten={best.shanten}",
        )

    def score_discards(self, hand: list[str], state: dict[str, Any]) -> dict[str, DiscardScore]:
        scores: dict[str, DiscardScore] = {}
        for tile in dict.fromkeys(legal_discard_tiles(hand, state)):
            remaining = hand[:]
            remaining.remove(tile)
            shanten = self.scoring.shanten_count(remaining)
            discard_value = self._discard_value(tile, state)
            scores[tile] = DiscardScore(
                tile=tile,
                shanten=shanten,
                ukeire=0,
                discard_value=discard_value,
                score=-4.0 * float(shanten) - 0.05 * float(discard_value),
            )
        return scores

    def _discard_value(self, tile: str, state: dict[str, Any]) -> int:
        value = 0
        normalized = normalize_tile(tile)
        if not is_terminal_or_honor(tile):
            value += 1
        if tile.startswith("0"):
            value += 3
        if normalized in state.get("dora", []):
            value += 3
        return value

    def _tile_order(self, tile: str) -> int:
        return self.tile_kinds.index(tile)
