"""Shared game types and public state views."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jongmind.tiles import dora_from_indicator


class Seat(int, Enum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3

    @property
    def next(self) -> "Seat":
        return Seat((int(self) + 1) % 4)


class Phase(str, Enum):
    DRAW = "draw"
    DISCARD = "discard"
    REACTION = "reaction"
    FINISHED = "finished"


@dataclass(frozen=True)
class DealerCommand:
    kind: str
    seat: Seat | None = None
    tile: str | None = None
    action: str | None = None
    tiles: tuple[str, ...] = ()
    riichi: bool = False


@dataclass(frozen=True)
class Meld:
    kind: str
    owner: Seat
    from_seat: Seat
    called_tile: str
    tiles: tuple[str, ...]
    open: bool = True

    def public_view(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "owner": self.owner.name,
            "from_seat": self.from_seat.name,
            "called_tile": self.called_tile,
            "tiles": list(self.tiles),
            "open": self.open,
        }


@dataclass
class PendingReaction:
    discarder: Seat
    tile: str
    options: dict[Seat, list[str]]
    from_draw: bool = False
    responses: dict[Seat, DealerCommand] = field(default_factory=dict)
    reaction_type: str = "discard"

    @property
    def waiting_seats(self) -> list[Seat]:
        return [seat for seat in self.options if seat not in self.responses]

    def public_view(self, seat: Seat | None = None) -> dict[str, Any] | None:
        if seat is None:
            return {
                "discarder": self.discarder.name,
                "tile": self.tile,
                "reaction_type": self.reaction_type,
                "waiting": [waiting_seat.name for waiting_seat in self.waiting_seats],
                "discard_type": "tsumogiri" if self.from_draw else "tedashi",
                "responses": {
                    response_seat.name: (response.action or response.kind)
                    for response_seat, response in self.responses.items()
                },
            }
        if seat not in self.options:
            return None

        view: dict[str, Any] = {
            "discarder": self.discarder.name,
            "tile": self.tile,
            "reaction_type": self.reaction_type,
            "discard_type": "tsumogiri" if self.from_draw else "tedashi",
        }
        response = self.responses.get(seat)
        if response is None:
            view["legal_actions"] = ["pass", *self.options[seat]]
        else:
            view["own_response"] = response.action or response.kind
        return view


@dataclass
class DealerState:
    hands: dict[Seat, list[str]] = field(default_factory=lambda: {seat: [] for seat in Seat})
    discards: dict[Seat, list[str]] = field(default_factory=lambda: {seat: [] for seat in Seat})
    melds: dict[Seat, list[Meld]] = field(default_factory=lambda: {seat: [] for seat in Seat})
    live_wall: list[str] = field(default_factory=list)
    dead_wall: list[str] = field(default_factory=list)
    dora_indicators: list[str] = field(default_factory=list)
    ura_dora_indicators: list[str] = field(default_factory=list)
    scores: dict[Seat, int] = field(default_factory=lambda: {seat: 25_000 for seat in Seat})
    riichi_declared: dict[Seat, bool] = field(default_factory=lambda: {seat: False for seat in Seat})
    ippatsu_active: dict[Seat, bool] = field(default_factory=lambda: {seat: False for seat in Seat})
    first_turn: dict[Seat, bool] = field(default_factory=lambda: {seat: True for seat in Seat})
    current_turn: Seat = Seat.EAST
    dealer: Seat = Seat.EAST
    phase: Phase = Phase.FINISHED
    last_draw: str | None = None
    last_draw_by_seat: dict[Seat, str | None] = field(default_factory=lambda: {seat: None for seat in Seat})
    last_draw_was_rinshan: bool = False
    turn_count: int = 0
    round_wind: Seat = Seat.EAST
    hand_number: int = 0
    max_hand_number: int = 4
    honba: int = 0
    riichi_sticks: int = 0
    kan_count: int = 0
    rinshan_draws_used: int = 0
    any_call_made: bool = False
    pending_reaction: PendingReaction | None = None
    pending_riichi_seat: Seat | None = None
    result: dict[str, Any] | None = None

    def public_view(self, seat: Seat | None = None) -> dict[str, Any]:
        legal_actions: list[str] = []
        if seat is not None:
            if self.phase == Phase.DRAW and seat == self.current_turn:
                legal_actions = ["draw"]
            elif self.phase == Phase.DISCARD and seat == self.current_turn:
                legal_actions = ["discard"]
            elif (
                self.phase == Phase.REACTION
                and self.pending_reaction is not None
                and seat in self.pending_reaction.options
                and seat not in self.pending_reaction.responses
            ):
                legal_actions = ["pass", *self.pending_reaction.options[seat]]

        view: dict[str, Any] = {
            "current_turn": self.current_turn.name,
            "dealer": self.dealer.name,
            "phase": self.phase.value,
            "wall_count": len(self.live_wall),
            "dead_wall_count": len(self.dead_wall),
            "dora_indicators": self.dora_indicators[:],
            "dora": [dora_from_indicator(tile) for tile in self.dora_indicators],
            "discards": {seat.name: tiles[:] for seat, tiles in self.discards.items()},
            "melds": {
                seat.name: [meld.public_view() for meld in melds]
                for seat, melds in self.melds.items()
            },
            "hand_counts": {seat.name: len(hand) for seat, hand in self.hands.items()},
            "scores": {seat.name: score for seat, score in self.scores.items()},
            "riichi_declared": {
                seat.name: declared for seat, declared in self.riichi_declared.items()
            },
            "turn_count": self.turn_count,
            "round_wind": self.round_wind.name,
            "hand_number": self.hand_number,
            "max_hand_number": self.max_hand_number,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "kan_count": self.kan_count,
            "legal_actions": legal_actions,
            "pending_reaction": (
                self.pending_reaction.public_view(seat)
                if self.pending_reaction is not None
                else None
            ),
            "result": self.result,
        }
        if seat is not None:
            view["seat"] = seat.name
            view["hand"] = self.hands[seat][:]
        return view
