"""Rule presets for supported mahjong variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSet:
    name: str
    players: int
    starting_points: int
    target_points: int
    uma: tuple[int, int, int, int]
    oka: int
    red_fives: bool
    open_tanyao: bool
    atozuke: bool
    kuikae: bool
    abortive_draws: bool
    multiple_ron: bool
    allow_negative_end: bool


MAHJONG_SOUL_4P_RANKED = RuleSet(
    name="Mahjong Soul 4-player ranked",
    players=4,
    starting_points=25_000,
    target_points=30_000,
    uma=(15, 5, -5, -15),
    oka=0,
    red_fives=True,
    open_tanyao=True,
    atozuke=True,
    kuikae=False,
    abortive_draws=True,
    multiple_ron=True,
    allow_negative_end=True,
)
