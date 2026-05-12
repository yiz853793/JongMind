"""Open-call baseline model."""

from __future__ import annotations

from typing import Any

from jongmind_ai.base import CallDecision, DiscardDecision
from jongmind_ai.tile_efficiency import TileEfficiencyAgent


class OpenCallAgent:
    """Tile-efficiency discard model that calls the first legal meld.

    This is intentionally simple: it makes public chii/pon/kan events happen
    while still using tile-efficiency for discard decisions after opening.
    """

    def __init__(self) -> None:
        self.discard_agent = TileEfficiencyAgent()

    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> DiscardDecision:
        return self.discard_agent.choose_discard(hand, state)

    def choose_reaction(self, state: dict[str, Any]) -> CallDecision:
        hints = state.get("action_hints") or {}
        for action in ("kan", "pon", "chii"):
            candidates = hints.get(action) or []
            if candidates:
                return CallDecision(
                    action=action,
                    tiles=tuple(candidates[0]),
                    reason=f"open_call_first_{action}",
                )
        return CallDecision(action=None)
