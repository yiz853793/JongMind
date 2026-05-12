"""Tile-efficiency baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jongmind_ai.base import CallDecision, DiscardDecision, legal_discard_tiles
from jongmind.rules import MAHJONG_SOUL_4P_RANKED
from jongmind.scoring import MahjongSoulScoring
from jongmind.tiles import (
    HONORS,
    build_wall,
    is_suited,
    is_terminal_or_honor,
    normalize_tile,
    sort_tiles,
    tile_suit,
)


SEAT_NAMES = ("EAST", "SOUTH", "WEST", "NORTH")
WIND_TILES = ("E", "S", "W", "N")
DRAGON_TILES = ("C", "F", "P")


@dataclass(frozen=True)
class DiscardScore:
    tile: str
    shanten: int
    ukeire: int
    discard_value: int
    score: float


@dataclass(frozen=True)
class CallAnalysis:
    score: DiscardScore
    after_call_hand: tuple[str, ...]
    call_state: dict[str, Any]


@dataclass(frozen=True)
class TeacherProfile:
    name: str
    speed_weight: float
    value_weight: float
    defense_weight: float
    placement_weight: float
    base_call_margin: float
    closed_call_margin: float
    dealer_attack_bonus: float
    dealer_call_bonus: float
    threat_call_margin: float
    route_call_bonus: float
    cheap_call_penalty: float
    open_value_floor: int
    closed_riichi_bonus: int
    late_tenpai_bonus: float


MJAI_MANUE_PROFILE = TeacherProfile(
    name="mjai_manue",
    speed_weight=0.20,
    value_weight=1.00,
    defense_weight=5.80,
    placement_weight=0.25,
    base_call_margin=0.34,
    closed_call_margin=0.18,
    dealer_attack_bonus=0.12,
    dealer_call_bonus=0.12,
    threat_call_margin=0.42,
    route_call_bonus=0.10,
    cheap_call_penalty=0.38,
    open_value_floor=1800,
    closed_riichi_bonus=1600,
    late_tenpai_bonus=0.18,
)


AKOCHAN_PROFILE = TeacherProfile(
    name="akochan",
    speed_weight=0.16,
    value_weight=1.15,
    defense_weight=7.20,
    placement_weight=0.55,
    base_call_margin=0.42,
    closed_call_margin=0.28,
    dealer_attack_bonus=0.20,
    dealer_call_bonus=0.16,
    threat_call_margin=0.62,
    route_call_bonus=0.08,
    cheap_call_penalty=0.62,
    open_value_floor=2200,
    closed_riichi_bonus=1800,
    late_tenpai_bonus=0.12,
)


MAHJONG_AI_PROFILE = TeacherProfile(
    name="mahjong_ai",
    speed_weight=0.24,
    value_weight=0.88,
    defense_weight=5.20,
    placement_weight=0.20,
    base_call_margin=0.30,
    closed_call_margin=0.45,
    dealer_attack_bonus=0.14,
    dealer_call_bonus=0.10,
    threat_call_margin=0.45,
    route_call_bonus=0.28,
    cheap_call_penalty=0.48,
    open_value_floor=2000,
    closed_riichi_bonus=1500,
    late_tenpai_bonus=0.22,
)


class TileEfficiencyAgent:
    """Shanten/ukeire baseline.

    The model minimizes shanten after discard. For tied discards, it maximizes
    ukeire: the number of visible-unaccounted tiles that improve shanten on the
    next draw. It is deliberately attack-only.  By default it stays closed so
    old baselines remain stable; pass ``allow_calls=True`` for a call-capable
    variant that uses the same tile-efficiency scoring.
    """

    def __init__(self, allow_calls: bool = False, call_min_score_delta: float = 0.25) -> None:
        self.allow_calls = allow_calls
        self.call_min_score_delta = call_min_score_delta
        self.scoring = MahjongSoulScoring(MAHJONG_SOUL_4P_RANKED)
        self.full_wall = build_wall(red_fives=True)
        self.tile_kinds = tuple(dict.fromkeys(sort_tiles(self.full_wall)))
        self.wall_counts = {tile: self.full_wall.count(tile) for tile in self.tile_kinds}

    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> DiscardDecision:
        if not hand:
            raise ValueError("cannot discard from an empty hand")

        candidates: list[tuple[tuple[int, int, int, int], DiscardScore]] = []
        for score in self.score_discards(hand, state).values():
            stable_order = sort_tiles([score.tile])[0]
            key = (score.shanten, -score.ukeire, score.discard_value, self._tile_order(stable_order))
            candidates.append((key, score))

        best = min(candidates, key=lambda item: item[0])[1]
        return DiscardDecision(
            tile=best.tile,
            shanten=best.shanten,
            ukeire=best.ukeire,
            reason=f"min_shanten={best.shanten}, max_ukeire={best.ukeire}",
        )

    def choose_reaction(self, state: dict[str, Any]) -> CallDecision:
        if not self.allow_calls:
            return CallDecision(action=None, reason="tile_efficiency_closed_pass")

        hand = list(state.get("hand") or [])
        hints = state.get("action_hints") or {}
        legal_actions = set(state.get("legal_actions") or ())
        if not hand:
            return CallDecision(action=None, reason="tile_efficiency_no_hand_pass")

        pass_score = self._pass_reaction_score(hand, state)
        candidates: list[tuple[tuple[float, int, int], str, tuple[str, ...], CallAnalysis, str]] = []
        for action in ("pon", "chii", "kan"):
            if action not in legal_actions:
                continue
            for raw_tiles in hints.get(action) or []:
                tiles = tuple(raw_tiles)
                analysis = self._score_call_candidate(hand, state, action, tiles)
                if analysis is None:
                    continue
                strategy = self._call_strategy(hand, state, action, tiles, analysis, pass_score)
                if strategy is None:
                    continue
                score_value = analysis.score.score
                if score_value < pass_score + self.call_min_score_delta:
                    continue
                priority = {"pon": 3, "chii": 2, "kan": 1}[action]
                candidates.append(((score_value, priority, -self._call_tile_order(tiles)), action, tiles, analysis, strategy))

        if not candidates:
            return CallDecision(
                action=None,
                reason=f"tile_efficiency_call_pass score={pass_score:.3f}",
            )

        _, action, tiles, analysis, strategy = max(candidates, key=lambda item: item[0])
        call_score = analysis.score
        return CallDecision(
            action=action,
            tiles=tiles,
            reason=(
                f"tile_efficiency_call_{strategy}_{action} "
                f"post_discard={call_score.tile} shanten={call_score.shanten} ukeire={call_score.ukeire}"
            ),
        )

    def score_discards(self, hand: list[str], state: dict[str, Any]) -> dict[str, DiscardScore]:
        """Score every legal discard for soft imitation targets.

        The hard policy still uses the exact lexicographic rule above.  The
        scalar score is intentionally smooth so training can learn that several
        near-equivalent discards are acceptable instead of treating all non-best
        tiles as equally wrong.
        """
        scores: dict[str, DiscardScore] = {}
        for tile in dict.fromkeys(legal_discard_tiles(hand, state)):
            remaining = hand[:]
            remaining.remove(tile)
            shanten = self.scoring.shanten_count(remaining)
            ukeire = self._ukeire(remaining, state, shanten)
            discard_value = self._discard_value(tile, state)
            scores[tile] = DiscardScore(
                tile=tile,
                shanten=shanten,
                ukeire=ukeire,
                discard_value=discard_value,
                score=self._soft_score(shanten, ukeire, discard_value),
            )
        return scores

    def _ukeire(self, tiles_after_discard: list[str], state: dict[str, Any], current_shanten: int) -> int:
        visible_counts = self._visible_counts(tiles_after_discard, state)
        ukeire = 0
        for tile in self.tile_kinds:
            if visible_counts.get(tile, 0) >= self.wall_counts[tile]:
                continue
            next_tiles = sort_tiles([*tiles_after_discard, tile])
            if self.scoring.shanten_count(next_tiles) < current_shanten:
                ukeire += self.wall_counts[tile] - visible_counts.get(tile, 0)
        return ukeire

    def _visible_counts(self, own_tiles: list[str], state: dict[str, Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tile in own_tiles:
            counts[tile] = counts.get(tile, 0) + 1
        for discards in state.get("discards", {}).values():
            for tile in discards:
                counts[tile] = counts.get(tile, 0) + 1
        for melds in state.get("melds", {}).values():
            for meld in melds:
                for tile in meld.get("tiles", []):
                    counts[tile] = counts.get(tile, 0) + 1
        return counts

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

    def _soft_score(self, shanten: int, ukeire: int, discard_value: int) -> float:
        return -4.0 * float(shanten) + min(float(ukeire), 80.0) / 20.0 - 0.05 * float(discard_value)

    def _tile_order(self, tile: str) -> int:
        return self.tile_kinds.index(tile)

    def _pass_reaction_score(self, hand: list[str], state: dict[str, Any]) -> float:
        sorted_hand = sort_tiles(hand)
        shanten = self.scoring.shanten_count(sorted_hand)
        ukeire = self._ukeire(sorted_hand, state, shanten) if len(sorted_hand) % 3 == 1 else 0
        return self._soft_score(shanten, ukeire, 0)

    def _score_call_candidate(
        self,
        hand: list[str],
        state: dict[str, Any],
        action: str,
        tiles: tuple[str, ...],
    ) -> CallAnalysis | None:
        try:
            after_call_hand = self._hand_after_removing_tiles(hand, tiles)
        except ValueError:
            return None

        call_state = self._state_after_call(state, action, tiles)
        if action == "kan":
            shanten = self.scoring.shanten_count(after_call_hand)
            return CallAnalysis(
                score=DiscardScore(
                    tile=tiles[0] if tiles else "",
                    shanten=shanten,
                    ukeire=0,
                    discard_value=0,
                    score=self._soft_score(shanten, 0, 0) - 1.0,
                ),
                after_call_hand=tuple(after_call_hand),
                call_state=call_state,
            )

        scores = self.score_discards(after_call_hand, call_state)
        if not scores:
            return None
        return CallAnalysis(
            score=max(scores.values(), key=lambda item: item.score),
            after_call_hand=tuple(after_call_hand),
            call_state=call_state,
        )

    def _call_strategy(
        self,
        hand: list[str],
        state: dict[str, Any],
        action: str,
        tiles: tuple[str, ...],
        analysis: CallAnalysis,
        pass_score: float,
    ) -> str | None:
        if action == "kan":
            return self._kan_strategy(state, tiles, analysis, pass_score)

        called_tile = self._called_tile(state)
        meld_tiles = self._meld_tiles(state, tiles)
        concealed_after_discard = self._concealed_after_post_call_discard(analysis)
        seat_name = self._seat_name(state)
        combined_tiles = [
            *concealed_after_discard,
            *meld_tiles,
            *self._own_open_meld_tiles(state, seat_name),
        ]

        if action == "pon" and called_tile is not None and self._is_value_honor(called_tile, state):
            if self._passes_threat_gate(state, analysis):
                return "yakuhai"

        if self._is_open_tanyao_path(combined_tiles, meld_tiles):
            if self._passes_threat_gate(state, analysis):
                return "tanyao"

        if self._is_flush_path(combined_tiles, meld_tiles):
            if self._passes_threat_gate(state, analysis):
                return "flush"

        if self._has_existing_open_yaku(state, seat_name, combined_tiles):
            if analysis.score.score >= pass_score + 0.75 and self._passes_threat_gate(state, analysis):
                return "open_yaku_push"

        if self._is_late_tenpai_push(state, analysis, pass_score):
            return "late_tenpai"

        return None

    def _kan_strategy(
        self,
        state: dict[str, Any],
        tiles: tuple[str, ...],
        analysis: CallAnalysis,
        pass_score: float,
    ) -> str | None:
        called_tile = self._called_tile(state)
        if called_tile is None or not self._is_value_honor(called_tile, state):
            return None
        if self._has_active_threat(state) or int(state.get("turn_count", 0) or 0) >= 48:
            return None
        if analysis.score.shanten > 1 or analysis.score.score < pass_score + 1.0:
            return None
        if not tiles:
            return None
        return "yakuhai_kan"

    def _concealed_after_post_call_discard(self, analysis: CallAnalysis) -> list[str]:
        concealed = list(analysis.after_call_hand)
        discard = analysis.score.tile
        if discard in concealed:
            concealed.remove(discard)
        return sort_tiles(concealed)

    def _called_tile(self, state: dict[str, Any]) -> str | None:
        hints = state.get("action_hints") or {}
        tile = hints.get("tile")
        return str(tile) if tile is not None else None

    def _meld_tiles(self, state: dict[str, Any], tiles: tuple[str, ...]) -> list[str]:
        called_tile = self._called_tile(state)
        if called_tile is None:
            return sort_tiles(tiles)
        return sort_tiles([*tiles, called_tile])

    def _own_open_meld_tiles(self, state: dict[str, Any], seat_name: str) -> list[str]:
        tiles: list[str] = []
        for meld in (state.get("melds") or {}).get(seat_name, []):
            if not meld.get("open", True):
                continue
            tiles.extend(meld.get("tiles") or [])
        return tiles

    def _seat_name(self, state: dict[str, Any]) -> str:
        return str(state.get("seat") or state.get("current_turn") or "EAST")

    def _is_value_honor(self, tile: str, state: dict[str, Any]) -> bool:
        normalized = normalize_tile(tile)
        if normalized in DRAGON_TILES:
            return True
        return normalized in {self._seat_wind_tile(state), self._round_wind_tile(state)}

    def _seat_wind_tile(self, state: dict[str, Any]) -> str:
        seat_name = self._seat_name(state)
        dealer_name = str(state.get("dealer") or "EAST")
        try:
            seat_index = SEAT_NAMES.index(seat_name)
            dealer_index = SEAT_NAMES.index(dealer_name)
        except ValueError:
            return "E"
        return WIND_TILES[(seat_index - dealer_index) % 4]

    def _round_wind_tile(self, state: dict[str, Any]) -> str:
        round_wind = str(state.get("round_wind") or "EAST")
        try:
            return WIND_TILES[SEAT_NAMES.index(round_wind)]
        except ValueError:
            return "E"

    def _is_open_tanyao_path(self, combined_tiles: list[str], meld_tiles: list[str]) -> bool:
        if not combined_tiles or not meld_tiles:
            return False
        if any(is_terminal_or_honor(tile) for tile in meld_tiles):
            return False
        return not any(is_terminal_or_honor(tile) for tile in combined_tiles)

    def _is_flush_path(self, combined_tiles: list[str], meld_tiles: list[str]) -> bool:
        if not combined_tiles or not meld_tiles:
            return False

        suited_tiles = [tile for tile in combined_tiles if is_suited(tile)]
        if len(suited_tiles) < 8:
            return False
        suits = {tile_suit(tile) for tile in suited_tiles}
        if len(suits) != 1:
            return False
        return all((tile in HONORS) or tile_suit(tile) in suits for tile in combined_tiles if is_suited(tile) or tile in HONORS)

    def _has_existing_open_yaku(self, state: dict[str, Any], seat_name: str, combined_tiles: list[str]) -> bool:
        open_melds = [
            meld
            for meld in (state.get("melds") or {}).get(seat_name, [])
            if meld.get("open", True)
        ]
        if not open_melds:
            return False

        for meld in open_melds:
            called_tile = meld.get("called_tile")
            if called_tile is not None and self._is_value_honor(str(called_tile), state):
                return True

        return self._is_open_tanyao_path(combined_tiles, combined_tiles) or self._is_flush_path(combined_tiles, combined_tiles)

    def _is_late_tenpai_push(self, state: dict[str, Any], analysis: CallAnalysis, pass_score: float) -> bool:
        turn_count = int(state.get("turn_count", 0) or 0)
        if turn_count < 52 or self._has_active_threat(state):
            return False
        return analysis.score.shanten <= 0 and analysis.score.score >= pass_score + 1.25

    def _passes_threat_gate(self, state: dict[str, Any], analysis: CallAnalysis) -> bool:
        if not self._has_active_threat(state):
            return True
        return analysis.score.shanten <= 1

    def _has_active_threat(self, state: dict[str, Any]) -> bool:
        seat_name = self._seat_name(state)
        riichi_declared = state.get("riichi_declared") or {}
        if any(seat != seat_name and declared for seat, declared in riichi_declared.items()):
            return True

        turn_count = int(state.get("turn_count", 0) or 0)
        if turn_count < 36:
            return False
        for seat, melds in (state.get("melds") or {}).items():
            if seat == seat_name:
                continue
            open_count = sum(1 for meld in melds if meld.get("open", True))
            if open_count >= 2:
                return True
        return False

    def _hand_after_removing_tiles(self, hand: list[str], tiles: tuple[str, ...]) -> list[str]:
        remaining = hand[:]
        for tile in tiles:
            remaining.remove(tile)
        return sort_tiles(remaining)

    def _state_after_call(self, state: dict[str, Any], action: str, tiles: tuple[str, ...]) -> dict[str, Any]:
        hints = state.get("action_hints") or {}
        called_tile = hints.get("tile")
        discarder = hints.get("discarder")
        seat_name = state.get("seat") or state.get("current_turn") or "EAST"
        call_state = dict(state)
        discards = {
            seat: list(seat_discards)
            for seat, seat_discards in (state.get("discards") or {}).items()
        }
        if called_tile is not None and discarder in discards and discards[discarder]:
            if discards[discarder][-1] == called_tile:
                discards[discarder].pop()
        melds = {
            seat: [dict(meld) for meld in seat_melds]
            for seat, seat_melds in (state.get("melds") or {}).items()
        }
        melds.setdefault(seat_name, []).append(
            {
                "kind": action,
                "owner": seat_name,
                "from_seat": discarder,
                "called_tile": called_tile,
                "tiles": sort_tiles([*tiles, called_tile]) if called_tile is not None else list(tiles),
                "open": True,
            }
        )
        call_state["discards"] = discards
        call_state["melds"] = melds
        call_state["legal_discard_tiles"] = None
        return call_state

    def _call_tile_order(self, tiles: tuple[str, ...]) -> int:
        orders = [self._tile_order(tile) for tile in tiles if tile in self.tile_kinds]
        return min(orders) if orders else 0


class ExpectedValueTeacherAgent(TileEfficiencyAgent):
    """Expected-value teacher profile for discard and call supervision.

    These teachers are local heuristic implementations inspired by open-source
    mahjong projects.  They do not vendor or translate external engine code:
    the goal is to expose comparable teaching styles through JongMind's normal
    model interface.
    """

    def __init__(self, profile: TeacherProfile, call_min_ev_delta: float | None = None) -> None:
        super().__init__(
            allow_calls=True,
            call_min_score_delta=profile.base_call_margin if call_min_ev_delta is None else call_min_ev_delta,
        )
        self.profile = profile

    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> DiscardDecision:
        if not hand:
            raise ValueError("cannot discard from an empty hand")

        best = max(
            self.score_discards(hand, state).values(),
            key=lambda score: (
                score.score,
                -score.shanten,
                score.ukeire,
                -self._tile_order(sort_tiles([score.tile])[0]),
            ),
        )
        return DiscardDecision(
            tile=best.tile,
            shanten=best.shanten,
            ukeire=best.ukeire,
            reason=(
                f"{self.profile.name}_ev score={best.score:.3f} "
                f"shanten={best.shanten} ukeire={best.ukeire}"
            ),
        )

    def choose_reaction(self, state: dict[str, Any]) -> CallDecision:
        hand = list(state.get("hand") or [])
        hints = state.get("action_hints") or {}
        legal_actions = set(state.get("legal_actions") or ())
        if not hand:
            return CallDecision(action=None, reason=f"{self.profile.name}_no_hand_pass")

        pass_score = self._pass_reaction_score(hand, state)
        candidates: list[tuple[tuple[float, int, int], str, tuple[str, ...], CallAnalysis, str, float]] = []
        for action in ("pon", "chii", "kan"):
            if action not in legal_actions:
                continue
            for raw_tiles in hints.get(action) or []:
                tiles = tuple(raw_tiles)
                analysis = self._score_call_candidate(hand, state, action, tiles)
                if analysis is None:
                    continue
                strategy = self._call_strategy(hand, state, action, tiles, analysis, pass_score)
                if strategy is None:
                    continue
                if self._protect_closed_hand(hand, state, action, analysis, strategy):
                    continue
                call_score = self._call_ev_score(analysis, state, action, strategy)
                margin = self._call_margin(hand, state, action, analysis, strategy)
                if call_score < pass_score + margin:
                    continue
                priority = {"pon": 3, "chii": 2, "kan": 1}[action]
                candidates.append(
                    (
                        (call_score, priority, -self._call_tile_order(tiles)),
                        action,
                        tiles,
                        analysis,
                        strategy,
                        margin,
                    )
                )

        if not candidates:
            return CallDecision(
                action=None,
                reason=f"{self.profile.name}_call_pass ev={pass_score:.3f}",
            )

        key, action, tiles, analysis, strategy, margin = max(candidates, key=lambda item: item[0])
        call_score = analysis.score
        return CallDecision(
            action=action,
            tiles=tiles,
            reason=(
                f"{self.profile.name}_call_{strategy}_{action} "
                f"ev={key[0]:.3f} margin={margin:.3f} "
                f"post_discard={call_score.tile} shanten={call_score.shanten} ukeire={call_score.ukeire}"
            ),
        )

    def score_discards(self, hand: list[str], state: dict[str, Any]) -> dict[str, DiscardScore]:
        scores: dict[str, DiscardScore] = {}
        for tile in dict.fromkeys(legal_discard_tiles(hand, state)):
            remaining = hand[:]
            remaining.remove(tile)
            shanten = self.scoring.shanten_count(remaining)
            ukeire = self._ukeire(remaining, state, shanten)
            discard_value = self._discard_value(tile, state)
            scores[tile] = DiscardScore(
                tile=tile,
                shanten=shanten,
                ukeire=ukeire,
                discard_value=discard_value,
                score=self._expected_point_score(
                    remaining,
                    state,
                    discard_tile=tile,
                    shanten=shanten,
                    ukeire=ukeire,
                    open_hand=self._is_hand_open(state),
                ),
            )
        return scores

    def _pass_reaction_score(self, hand: list[str], state: dict[str, Any]) -> float:
        sorted_hand = sort_tiles(hand)
        shanten = self.scoring.shanten_count(sorted_hand)
        ukeire = self._ukeire(sorted_hand, state, shanten) if len(sorted_hand) % 3 == 1 else 0
        return self._expected_point_score(
            sorted_hand,
            state,
            discard_tile=None,
            shanten=shanten,
            ukeire=ukeire,
            open_hand=self._is_hand_open(state),
        )

    def _call_ev_score(self, analysis: CallAnalysis, state: dict[str, Any], action: str, strategy: str) -> float:
        score = analysis.score.score + self.profile.route_call_bonus
        if strategy == "yakuhai":
            score += 0.55
        elif strategy == "flush":
            score += 0.22
        elif strategy == "late_tenpai":
            score += self.profile.late_tenpai_bonus
        elif strategy == "tanyao":
            score += 0.08

        concealed = self._concealed_after_post_call_discard(analysis)
        value_points = self._estimated_value_points(concealed, analysis.call_state, open_hand=True)
        if value_points < self.profile.open_value_floor and analysis.score.shanten > 0:
            score -= self.profile.cheap_call_penalty
        if action == "kan":
            score -= 0.55
        return score

    def _call_margin(
        self,
        hand: list[str],
        state: dict[str, Any],
        action: str,
        analysis: CallAnalysis,
        strategy: str,
    ) -> float:
        margin = self.profile.base_call_margin
        if not self._is_hand_open(state):
            margin += self.profile.closed_call_margin
        if action == "chii":
            margin += 0.08
        if strategy == "yakuhai":
            margin -= 0.25
        elif strategy in {"flush", "late_tenpai"}:
            margin -= 0.16
        if self._is_dealer(state):
            margin -= self.profile.dealer_call_bonus

        rank, leader_gap = self._placement(state)
        if rank == 1 and leader_gap <= 0:
            margin += 0.18 * self.profile.placement_weight
        elif rank >= 3:
            margin -= 0.16 * self.profile.placement_weight

        if self._has_active_threat(state):
            margin += self.profile.threat_call_margin
            if analysis.score.shanten <= 0:
                margin -= 0.20

        turn_count = int(state.get("turn_count", 0) or 0)
        if turn_count >= 52 and analysis.score.shanten <= 0:
            margin -= self.profile.late_tenpai_bonus

        return max(0.05, margin)

    def _protect_closed_hand(
        self,
        hand: list[str],
        state: dict[str, Any],
        action: str,
        analysis: CallAnalysis,
        strategy: str,
    ) -> bool:
        if self._is_hand_open(state):
            return False
        if strategy == "yakuhai" and action == "pon":
            return False

        turn_count = int(state.get("turn_count", 0) or 0)
        pass_shanten = self.scoring.shanten_count(sort_tiles(hand))
        closed_value = self._estimated_value_points(sort_tiles(hand), state, open_hand=False)
        open_value = self._estimated_value_points(
            self._concealed_after_post_call_discard(analysis),
            analysis.call_state,
            open_hand=True,
        )

        if action == "chii" and turn_count <= 28 and pass_shanten <= 1 and not self._is_dealer(state):
            return True
        if self.profile.name == "mahjong_ai" and action == "chii" and turn_count <= 36 and pass_shanten <= 1:
            return True
        if turn_count <= 36 and pass_shanten <= analysis.score.shanten and closed_value >= open_value + 1200:
            return True
        if strategy == "tanyao" and turn_count <= 24 and open_value < self.profile.open_value_floor:
            return True
        return False

    def _expected_point_score(
        self,
        hand_after_discard: list[str],
        state: dict[str, Any],
        *,
        discard_tile: str | None,
        shanten: int,
        ukeire: int,
        open_hand: bool,
    ) -> float:
        hora_prob = self._hora_probability(shanten, ukeire, state)
        avg_hora_points = self._estimated_value_points(hand_after_discard, state, open_hand=open_hand)
        unsafe_prob = self._discard_danger(discard_tile, state) if discard_tile is not None else 0.0
        avg_hoju_points = self._expected_deal_in_points(discard_tile, state)
        expected_points = (
            (1.0 - unsafe_prob) * hora_prob * (avg_hora_points / 1000.0)
            - unsafe_prob * (avg_hoju_points / 1000.0)
        )
        speed = -float(max(shanten, 0)) + min(float(ukeire), 80.0) / 32.0
        placement = self._placement_bias(state)
        dealer_bonus = self.profile.dealer_attack_bonus if self._is_dealer(state) else 0.0
        return (
            self.profile.value_weight * expected_points
            + self.profile.speed_weight * speed
            + self.profile.placement_weight * placement
            + dealer_bonus
        )

    def _hora_probability(self, shanten: int, ukeire: int, state: dict[str, Any]) -> float:
        if shanten < 0:
            return 0.95
        base_by_shanten = {
            0: 0.34,
            1: 0.15,
            2: 0.055,
            3: 0.020,
            4: 0.008,
        }
        base = base_by_shanten.get(shanten, 0.003)
        ukeire_factor = 0.30 + 0.70 * min(float(ukeire), 40.0) / 40.0
        wall_count = int(state.get("wall_count", 52) or 52)
        turn_factor = max(0.20, min(1.25, (wall_count / 4.0) / 12.0))
        if self._is_dealer(state):
            turn_factor *= 1.08
        return min(0.92, base * ukeire_factor * turn_factor)

    def _estimated_value_points(self, hand_tiles: list[str], state: dict[str, Any], *, open_hand: bool) -> int:
        seat_name = self._seat_name(state)
        combined_tiles = sort_tiles([*hand_tiles, *self._own_open_meld_tiles(state, seat_name)])
        shanten = self.scoring.shanten_count(hand_tiles) if hand_tiles else 6
        value = 1000

        dora_tiles = {normalize_tile(tile) for tile in state.get("dora", [])}
        dora_count = 0
        for tile in combined_tiles:
            normalized = normalize_tile(tile)
            if tile.startswith("0") or normalized in dora_tiles:
                dora_count += 1
        value += 1000 * dora_count

        if self._all_simples(combined_tiles):
            value += 800 if open_hand else 600
        value += self._value_honor_bonus(combined_tiles, state)
        value += self._flush_value_bonus(combined_tiles)
        value += self._toitoi_value_bonus(combined_tiles)

        if not open_hand and shanten <= 1:
            value += self.profile.closed_riichi_bonus
        if open_hand and value < self.profile.open_value_floor:
            value -= 250
        if self._is_dealer(state):
            value = int(value * 1.15)
        return max(1000, value)

    def _expected_deal_in_points(self, discard_tile: str | None, state: dict[str, Any]) -> int:
        if discard_tile is None or not self._has_active_threat(state):
            return 0
        value = 3900
        if any(declared for seat, declared in (state.get("riichi_declared") or {}).items() if seat != self._seat_name(state)):
            value = 5200
        dora_tiles = {normalize_tile(tile) for tile in state.get("dora", [])}
        if discard_tile.startswith("0") or normalize_tile(discard_tile) in dora_tiles:
            value += 2000
        dealer = str(state.get("dealer") or "EAST")
        if any(seat == dealer for seat, _strength in self._active_threats(state)):
            value = int(value * 1.35)
        return value

    def _discard_danger(self, tile: str | None, state: dict[str, Any]) -> float:
        if tile is None:
            return 0.0
        threats = self._active_threats(state)
        if not threats:
            return 0.0

        normalized = normalize_tile(tile)
        dora_tiles = {normalize_tile(dora) for dora in state.get("dora", [])}
        visible_count = self._public_visible_kind_count(normalized, state)
        danger = 0.0
        for seat, strength in threats:
            discards = {normalize_tile(discard) for discard in (state.get("discards") or {}).get(seat, [])}
            if normalized in discards:
                continue

            risk = 0.08 + 0.14 * strength
            if tile.startswith("0") or normalized in dora_tiles:
                risk += 0.08
            if is_suited(normalized):
                rank = int(normalized[0])
                if rank in {4, 5, 6}:
                    risk += 0.06
                elif rank in {2, 3, 7, 8}:
                    risk += 0.03
                else:
                    risk -= 0.04
                if self._is_suji_like_safe(normalized, seat, state):
                    risk -= 0.05
            else:
                risk += 0.04

            if visible_count >= 3:
                risk -= 0.12
            elif visible_count == 2:
                risk -= 0.05
            if seat == str(state.get("dealer") or "EAST"):
                risk += 0.03
            danger += max(0.01, risk)
        return max(0.0, min(0.85, danger))

    def _active_threats(self, state: dict[str, Any]) -> list[tuple[str, float]]:
        seat_name = self._seat_name(state)
        threats: list[tuple[str, float]] = []
        riichi_declared = state.get("riichi_declared") or {}
        for seat in SEAT_NAMES:
            if seat == seat_name:
                continue
            if riichi_declared.get(seat):
                threats.append((seat, 1.0))
                continue
            turn_count = int(state.get("turn_count", 0) or 0)
            open_count = sum(
                1
                for meld in (state.get("melds") or {}).get(seat, [])
                if meld.get("open", True)
            )
            if turn_count >= 36 and open_count >= 2:
                threats.append((seat, min(0.85, 0.45 + 0.12 * (open_count - 2))))
        return threats

    def _placement(self, state: dict[str, Any]) -> tuple[int, int]:
        seat_name = self._seat_name(state)
        scores = state.get("scores") or {}
        if not scores:
            return (2, 0)
        seat_scores = {seat: int(scores.get(seat, 25000) or 25000) for seat in SEAT_NAMES}
        my_score = seat_scores.get(seat_name, 25000)
        ordered = sorted(SEAT_NAMES, key=lambda seat: (-seat_scores[seat], SEAT_NAMES.index(seat)))
        rank = ordered.index(seat_name) + 1 if seat_name in ordered else 2
        leader_gap = max(seat_scores.values()) - my_score
        return rank, leader_gap

    def _placement_bias(self, state: dict[str, Any]) -> float:
        rank, leader_gap = self._placement(state)
        if rank == 1 and leader_gap <= 0:
            return 0.10
        if rank == 2:
            return 0.02
        if rank == 3:
            return 0.12
        return 0.18

    def _is_dealer(self, state: dict[str, Any]) -> bool:
        return self._seat_name(state) == str(state.get("dealer") or "EAST")

    def _is_hand_open(self, state: dict[str, Any]) -> bool:
        seat_name = self._seat_name(state)
        return bool(self._own_open_meld_tiles(state, seat_name))

    def _all_simples(self, tiles: list[str]) -> bool:
        return bool(tiles) and all(not is_terminal_or_honor(tile) for tile in tiles)

    def _value_honor_bonus(self, tiles: list[str], state: dict[str, Any]) -> int:
        value_honors = {self._seat_wind_tile(state), self._round_wind_tile(state), *DRAGON_TILES}
        counts: dict[str, int] = {}
        for tile in tiles:
            normalized = normalize_tile(tile)
            if normalized in value_honors:
                counts[normalized] = counts.get(normalized, 0) + 1
        bonus = 0
        for count in counts.values():
            if count >= 3:
                bonus += 1400
            elif count == 2:
                bonus += 450
        return bonus

    def _flush_value_bonus(self, tiles: list[str]) -> int:
        suited_tiles = [tile for tile in tiles if is_suited(tile)]
        if len(suited_tiles) < 7:
            return 0
        suit_counts: dict[str, int] = {}
        for tile in suited_tiles:
            suit = tile_suit(tile)
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        main_count = max(suit_counts.values(), default=0)
        off_suit = len(suited_tiles) - main_count
        honor_count = sum(1 for tile in tiles if tile in HONORS)
        if off_suit == 0 and honor_count <= 5:
            return 2600
        if main_count >= 8 and off_suit <= 2:
            return 1400
        return 0

    def _toitoi_value_bonus(self, tiles: list[str]) -> int:
        counts: dict[str, int] = {}
        for tile in tiles:
            normalized = normalize_tile(tile)
            counts[normalized] = counts.get(normalized, 0) + 1
        pairish = sum(1 for count in counts.values() if count >= 2)
        triplets = sum(1 for count in counts.values() if count >= 3)
        if triplets >= 3:
            return 1800
        if pairish >= 4:
            return 900
        return 0

    def _public_visible_kind_count(self, normalized_tile: str, state: dict[str, Any]) -> int:
        count = 0
        for discards in (state.get("discards") or {}).values():
            count += sum(1 for tile in discards if normalize_tile(tile) == normalized_tile)
        for melds in (state.get("melds") or {}).values():
            for meld in melds:
                count += sum(1 for tile in meld.get("tiles", []) if normalize_tile(tile) == normalized_tile)
        return count

    def _is_suji_like_safe(self, normalized_tile: str, threat_seat: str, state: dict[str, Any]) -> bool:
        if not is_suited(normalized_tile):
            return False
        rank = int(normalized_tile[0])
        suit = normalized_tile[1]
        suji_partners = {
            1: (4,),
            2: (5,),
            3: (6,),
            4: (1, 7),
            5: (2, 8),
            6: (3, 9),
            7: (4,),
            8: (5,),
            9: (6,),
        }[rank]
        discards = {normalize_tile(discard) for discard in (state.get("discards") or {}).get(threat_seat, [])}
        return any(f"{partner}{suit}" in discards for partner in suji_partners)


class MjaiManueAgent(ExpectedValueTeacherAgent):
    """mjai-manue-style expected-points teacher."""

    def __init__(self, call_min_ev_delta: float | None = None) -> None:
        super().__init__(MJAI_MANUE_PROFILE, call_min_ev_delta=call_min_ev_delta)


class AkochanAgent(ExpectedValueTeacherAgent):
    """Akochan-style placement and round-EV teacher."""

    def __init__(self, call_min_ev_delta: float | None = None) -> None:
        super().__init__(AKOCHAN_PROFILE, call_min_ev_delta=call_min_ev_delta)


class MahjongAIHeuristicAgent(ExpectedValueTeacherAgent):
    """MahjongAI-style rule heuristic teacher with stricter closed-hand protection."""

    def __init__(self, call_min_ev_delta: float | None = None) -> None:
        super().__init__(MAHJONG_AI_PROFILE, call_min_ev_delta=call_min_ev_delta)
