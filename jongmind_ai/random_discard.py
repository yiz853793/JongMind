"""Random discard baseline model."""

from __future__ import annotations

from random import Random
from typing import Any

from jongmind_ai.base import DiscardDecision, legal_discard_tiles


class RandomDiscardAgent:
    def __init__(self, seed: int) -> None:
        self.random = Random(seed)

    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> DiscardDecision:
        tile = self.random.choice(legal_discard_tiles(hand, state))
        return DiscardDecision(tile=tile, shanten=99, ukeire=0, reason="random")
