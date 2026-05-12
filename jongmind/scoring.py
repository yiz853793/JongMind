"""Mahjong Soul-like Japanese mahjong hand evaluation.

The project owns the table flow, while this module delegates yaku/fu/point
calculation to the maintained ``mahjong`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from mahjong.agari import Agari
from mahjong.constants import EAST, NORTH, SOUTH, WEST
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld as MahjongMeld
from mahjong.shanten import Shanten

from jongmind.rules import RuleSet
from jongmind.tiles import tile_to_136, tiles_to_136, tiles_to_34


WIND_TO_34 = {
    0: EAST,
    1: SOUTH,
    2: WEST,
    3: NORTH,
}


@dataclass(frozen=True)
class WinContext:
    is_tsumo: bool = False
    is_riichi: bool = False
    is_ippatsu: bool = False
    is_rinshan: bool = False
    is_chankan: bool = False
    is_haitei: bool = False
    is_houtei: bool = False
    is_daburu_riichi: bool = False
    is_tenhou: bool = False
    is_chiihou: bool = False
    is_renhou: bool = False
    kyoutaku_number: int = 0
    tsumi_number: int = 0


@dataclass(frozen=True)
class HandValue:
    ok: bool
    error: str | None
    han: int | None = None
    fu: int | None = None
    cost: dict[str, int | str] | None = None
    yaku: tuple[str, ...] = ()
    fu_details: tuple[dict[str, Any], ...] = ()


class MahjongSoulScoring:
    def __init__(self, rules: RuleSet) -> None:
        self.rules = rules
        self.calculator = HandCalculator()
        self.shanten = Shanten()
        self.agari = Agari()

    def is_closed_hand(self, melds: Iterable[Any]) -> bool:
        return not any(getattr(meld, "open", True) for meld in melds)

    def shanten_count(self, concealed_tiles: list[str]) -> int:
        return self.shanten.calculate_shanten(tiles_to_34(concealed_tiles))

    def is_tenpai(self, concealed_tiles: list[str]) -> bool:
        return self.shanten_count(concealed_tiles) == 0

    def estimate_hand_value(
        self,
        concealed_tiles: list[str],
        win_tile: str,
        melds: list[Any],
        seat_index: int,
        dealer_index: int,
        round_wind_index: int,
        dora_indicators: list[str],
        context: WinContext,
    ) -> HandValue:
        meld_tiles = [
            tile
            for meld in melds
            for tile in getattr(meld, "tiles")
        ]
        tiles = tiles_to_136([*concealed_tiles, *meld_tiles])
        win_tile_136 = tile_to_136(win_tile)
        mahjong_melds = [self._convert_meld(meld) for meld in melds]
        dora_136 = [tile_to_136(tile) for tile in dora_indicators]

        result = self.calculator.estimate_hand_value(
            tiles,
            win_tile_136,
            melds=mahjong_melds,
            dora_indicators=dora_136,
            config=self._hand_config(
                seat_index=seat_index,
                dealer_index=dealer_index,
                round_wind_index=round_wind_index,
                context=context,
            ),
        )

        if result.error:
            return HandValue(ok=False, error=result.error)

        return HandValue(
            ok=True,
            error=None,
            han=result.han,
            fu=result.fu,
            cost=result.cost,
            yaku=tuple(str(yaku) for yaku in result.yaku),
            fu_details=tuple(result.fu_details or ()),
        )

    def _hand_config(
        self,
        seat_index: int,
        dealer_index: int,
        round_wind_index: int,
        context: WinContext,
    ) -> HandConfig:
        player_wind_index = (seat_index - dealer_index) % 4
        return HandConfig(
            is_tsumo=context.is_tsumo,
            is_riichi=context.is_riichi,
            is_ippatsu=context.is_ippatsu,
            is_rinshan=context.is_rinshan,
            is_chankan=context.is_chankan,
            is_haitei=context.is_haitei,
            is_houtei=context.is_houtei,
            is_daburu_riichi=context.is_daburu_riichi,
            is_tenhou=context.is_tenhou,
            is_chiihou=context.is_chiihou,
            is_renhou=context.is_renhou,
            player_wind=WIND_TO_34[player_wind_index],
            round_wind=WIND_TO_34[round_wind_index],
            kyoutaku_number=context.kyoutaku_number,
            tsumi_number=context.tsumi_number,
            options=OptionalRules(
                has_open_tanyao=self.rules.open_tanyao,
                has_aka_dora=self.rules.red_fives,
                has_double_yakuman=True,
                kiriage=False,
                fu_for_open_pinfu=True,
                fu_for_pinfu_tsumo=False,
                renhou_as_yakuman=False,
                limit_to_sextuple_yakuman=True,
            ),
        )

    def _convert_meld(self, meld: Any) -> MahjongMeld:
        meld_type = {
            "chii": MahjongMeld.CHI,
            "pon": MahjongMeld.PON,
            "kan": MahjongMeld.KAN,
            "closed_kan": MahjongMeld.KAN,
            "added_kan": MahjongMeld.SHOUMINKAN,
        }[getattr(meld, "kind")]
        tiles = tiles_to_136(getattr(meld, "tiles"))
        called_tile = tile_to_136(getattr(meld, "called_tile"))
        return MahjongMeld(
            meld_type=meld_type,
            tiles=tiles,
            opened=getattr(meld, "open", True),
            called_tile=called_tile,
        )
