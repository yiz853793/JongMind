"""Tile utilities.

The internal notation follows common riichi tooling:
- 1m..9m: characters
- 1p..9p: dots
- 1s..9s: bamboo
- 0m/0p/0s: red fives
- E/S/W/N: winds
- C/F/P: red, green, white dragons
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from mahjong.tile import TilesConverter


SUITS = ("m", "p", "s")
HONORS = ("E", "S", "W", "N", "C", "F", "P")
RED_FIVES = ("0m", "0p", "0s")
HONOR_TO_MPSZ = {
    "E": "1",
    "S": "2",
    "W": "3",
    "N": "4",
    "P": "5",
    "F": "6",
    "C": "7",
}


def build_wall(red_fives: bool = False) -> list[str]:
    suited = [f"{rank}{suit}" for suit in SUITS for rank in range(1, 10)]
    wall = [tile for tile in suited + list(HONORS) for _ in range(4)]
    if red_fives:
        for suit in SUITS:
            wall.remove(f"5{suit}")
            wall.append(f"0{suit}")
    return wall


def normalize_tile(tile: str) -> str:
    if tile in RED_FIVES:
        return f"5{tile[1]}"
    return tile


def is_suited(tile: str) -> bool:
    normalized = normalize_tile(tile)
    return len(normalized) == 2 and normalized[1] in SUITS


def tile_rank(tile: str) -> int:
    if not is_suited(tile):
        raise ValueError(f"{tile} is not a suited tile")
    return int(normalize_tile(tile)[0])


def tile_suit(tile: str) -> str:
    if not is_suited(tile):
        raise ValueError(f"{tile} is not a suited tile")
    return normalize_tile(tile)[1]


def is_terminal_or_honor(tile: str) -> bool:
    normalized = normalize_tile(tile)
    if normalized in HONORS:
        return True
    return is_suited(normalized) and tile_rank(normalized) in (1, 9)


def sort_tiles(tiles: Iterable[str]) -> list[str]:
    order = {tile: index for index, tile in enumerate(_sort_order())}
    return sorted(tiles, key=lambda tile: order[tile])


def dora_from_indicator(indicator: str) -> str:
    normalized = normalize_tile(indicator)
    if normalized[1:] in SUITS:
        rank = int(normalized[0])
        return f"{rank % 9 + 1}{normalized[1]}"
    if normalized in ("E", "S", "W", "N"):
        winds = ("E", "S", "W", "N")
        return winds[(winds.index(normalized) + 1) % len(winds)]
    dragons = ("C", "F", "P")
    return dragons[(dragons.index(normalized) + 1) % len(dragons)]


def count_tiles(tiles: Iterable[str]) -> Counter[str]:
    return Counter(normalize_tile(tile) for tile in tiles)


def tiles_to_mpsz(tiles: Iterable[str]) -> str:
    groups = {"m": [], "p": [], "s": [], "z": []}
    for tile in tiles:
        if tile in RED_FIVES:
            groups[tile[1]].append("0")
        elif is_suited(tile):
            normalized = normalize_tile(tile)
            groups[normalized[1]].append(normalized[0])
        elif tile in HONOR_TO_MPSZ:
            groups["z"].append(HONOR_TO_MPSZ[tile])
        else:
            raise ValueError(f"unknown tile: {tile}")

    result = ""
    for suit in ("m", "p", "s", "z"):
        if groups[suit]:
            result += "".join(groups[suit]) + suit
    return result


def tiles_to_136(tiles: Iterable[str]) -> list[int]:
    return TilesConverter.one_line_string_to_136_array(
        tiles_to_mpsz(tiles),
        has_aka_dora=True,
    )


def tiles_to_34(tiles: Iterable[str]) -> list[int]:
    return TilesConverter.to_34_array(tiles_to_136(tiles))


def tile_to_136(tile: str) -> int:
    return tiles_to_136([tile])[0]


def _sort_order() -> list[str]:
    tiles: list[str] = []
    for suit in SUITS:
        tiles.extend(f"{rank}{suit}" for rank in range(1, 5))
        tiles.extend((f"0{suit}", f"5{suit}"))
        tiles.extend(f"{rank}{suit}" for rank in range(6, 10))
    tiles.extend(HONORS)
    return tiles
