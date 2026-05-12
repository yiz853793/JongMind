"""Beginner baseline AI."""

from __future__ import annotations

from typing import Any

from jongmind_ai.base import DiscardDecision
from jongmind_ai.tile_efficiency import TileEfficiencyAgent


class BeginnerAgent:
    """Closed tile-efficiency beginner AI.

    The goal of this model is not to be clever. It is a first AI baseline that
    should consistently beat random by making every discard minimize shanten
    and maximize immediate ukeire, while avoiding open-call complexity for now.
    """

    def __init__(self) -> None:
        self.tile_efficiency = TileEfficiencyAgent()

    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> DiscardDecision:
        return self.tile_efficiency.choose_discard(hand, state)
