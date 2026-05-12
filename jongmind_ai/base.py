"""Shared model types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class DiscardDecision:
    tile: str
    shanten: int
    ukeire: int
    reason: str


@dataclass(frozen=True)
class CallDecision:
    action: str | None
    tiles: tuple[str, ...] = ()
    reason: str = "pass"


@runtime_checkable
class DiscardAgent(Protocol):
    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> DiscardDecision:
        ...


@runtime_checkable
class ReactionAgent(Protocol):
    def choose_reaction(self, state: dict[str, Any]) -> CallDecision:
        ...


def legal_discard_tiles(hand: list[str], state: dict[str, Any]) -> list[str]:
    """Return the discard choices exposed by the dealer, preserving hand copies."""
    hints = state.get("action_hints") or {}
    discard_hint = hints.get("discard") or {}
    legal_tiles = state.get("legal_discard_tiles") or discard_hint.get("tiles")
    if not legal_tiles:
        return hand[:]

    legal_set = set(legal_tiles)
    choices = [tile for tile in hand if tile in legal_set]
    return choices or hand[:]


def forced_tsumogiri_tile(hand: list[str], state: dict[str, Any]) -> str | None:
    hints = state.get("action_hints") or {}
    discard_hint = hints.get("discard") or {}
    if not discard_hint.get("forced_tsumogiri"):
        return None
    choices = legal_discard_tiles(hand, state)
    return choices[0] if len(choices) == 1 else None
