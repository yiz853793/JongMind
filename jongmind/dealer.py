"""Process-safe Mahjong Soul style dealer/referee service."""

from __future__ import annotations

from itertools import combinations
from multiprocessing import Queue
from pathlib import Path
from random import Random
from typing import Any, Mapping

from jongmind.action_space import legal_action_ids, legal_action_mask, legal_action_specs, public_observation
from jongmind.game import DealerCommand, DealerState, Meld, PendingReaction, Phase, Seat
from jongmind.recording import HandHistoryRecorder
from jongmind.rules import MAHJONG_SOUL_4P_RANKED, RuleSet
from jongmind.scoring import HandValue, MahjongSoulScoring, WinContext
from jongmind.tiles import (
    build_wall,
    count_tiles,
    is_suited,
    is_terminal_or_honor,
    normalize_tile,
    sort_tiles,
    tile_rank,
    tile_suit,
)


class MahjongDealer:
    """Owns one Mahjong Soul style hand and handles dealer commands."""

    def __init__(
        self,
        seed: int | None = None,
        rules: RuleSet = MAHJONG_SOUL_4P_RANKED,
        hand_log_path: str | Path | None = None,
        record_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.seed = seed
        self.random = Random(seed)
        self.rules = rules
        self.scoring = MahjongSoulScoring(rules)
        self.state = DealerState(scores={seat: rules.starting_points for seat in Seat})
        self.hand_recorder = (
            HandHistoryRecorder(hand_log_path, seed=seed, context=record_context)
            if hand_log_path is not None
            else None
        )

    def start_hand(self) -> dict[Seat, dict[str, Any]]:
        hand_number = self.state.hand_number
        scores = self.state.scores.copy()
        honba = self.state.honba
        riichi_sticks = self.state.riichi_sticks
        max_hand_number = self.state.max_hand_number
        round_wind = Seat.EAST if hand_number < 4 else Seat.SOUTH
        wall = build_wall(red_fives=self.rules.red_fives)
        self.random.shuffle(wall)
        dead_wall = wall[-14:]
        live_wall = wall[:-14]
        dealer = Seat(hand_number % 4)

        hands = {seat: [] for seat in Seat}
        for _ in range(13):
            for seat in Seat:
                hands[seat].append(live_wall.pop())
        hands[dealer].append(live_wall.pop())

        self.state = DealerState(
            hands={seat: sort_tiles(hand) for seat, hand in hands.items()},
            discards={seat: [] for seat in Seat},
            melds={seat: [] for seat in Seat},
            live_wall=live_wall,
            dead_wall=dead_wall,
            dora_indicators=[dead_wall[4]],
            ura_dora_indicators=[dead_wall[5]],
            scores=scores,
            current_turn=dealer,
            dealer=dealer,
            phase=Phase.DISCARD,
            round_wind=round_wind,
            hand_number=hand_number,
            max_hand_number=max_hand_number,
            honba=honba,
            riichi_sticks=riichi_sticks,
        )
        views = {seat: self.get_state(seat) for seat in Seat}
        self._record_start_hand(views)
        return views

    def draw(self, seat: Seat) -> dict[str, Any]:
        self._require_active_turn(seat, Phase.DRAW)
        command = DealerCommand(kind="draw", seat=seat)
        before_view = self.get_state(seat)
        if not self.state.live_wall:
            result = self._finish_exhaustive_draw()
            return self._record_command_result(command, result, before_view=before_view)

        tile = self.state.live_wall.pop()
        self._add_drawn_tile(seat, tile, rinshan=False)
        result = {"event": "draw", "actor": seat.name, "tile": tile, **self.get_state(seat)}
        return self._record_command_result(command, result, before_view=before_view)

    def discard(self, seat: Seat, tile: str, riichi: bool = False) -> dict[str, Any]:
        self._require_active_turn(seat, Phase.DISCARD)
        if tile not in self.state.hands[seat]:
            raise ValueError(f"{seat.name} cannot discard {tile}: tile is not in hand")
        if self.state.riichi_declared[seat]:
            forced_tile = self.state.last_draw_by_seat[seat]
            if tile != forced_tile:
                raise ValueError(f"{seat.name} has declared riichi and must discard the drawn tile")

        before_view = self.get_state(seat)
        if riichi:
            self._validate_riichi_discard(seat, tile)
            self.state.pending_riichi_seat = seat

        command = DealerCommand(kind="discard", seat=seat, tile=tile, riichi=riichi)
        was_ippatsu = self.state.ippatsu_active[seat]
        self.state.hands[seat].remove(tile)
        self.state.discards[seat].append(tile)
        from_draw = self.state.last_draw_by_seat[seat] == tile
        self.state.last_draw_by_seat[seat] = None
        self.state.last_draw = None
        self.state.last_draw_was_rinshan = False
        self.state.turn_count += 1
        self.state.first_turn[seat] = False
        if was_ippatsu and self.state.pending_riichi_seat != seat:
            self.state.ippatsu_active[seat] = False

        pending_reaction = self._build_pending_reaction(seat, tile, from_draw=from_draw)
        if pending_reaction is not None:
            self.state.pending_reaction = pending_reaction
            self.state.phase = Phase.REACTION
            result = {
                "event": "discard_reaction",
                "actor": seat.name,
                "tile": tile,
                "discard_type": "tsumogiri" if from_draw else "tedashi",
                **self.get_state(seat),
            }
            return self._record_command_result(command, result, before_view=before_view)

        result = self._finish_discard_without_call(seat, tile, from_draw=from_draw)
        return self._record_command_result(command, result, before_view=before_view)

    def pass_reaction(self, seat: Seat) -> dict[str, Any]:
        pending = self._require_pending_reaction(seat)
        before_view = self.get_state(seat)
        pending.responses[seat] = DealerCommand(kind="pass", seat=seat)
        result = self._resolve_reaction_if_ready()
        return self._record_command_result(
            DealerCommand(kind="pass", seat=seat),
            result,
            before_view=before_view,
        )

    def call(self, seat: Seat, action: str, tiles: tuple[str, ...]) -> dict[str, Any]:
        pending = self._require_pending_reaction(seat)
        if action not in pending.options[seat]:
            raise ValueError(f"{seat.name} cannot call {action}")
        if action == "ron":
            raise ValueError("use win command for ron")
        before_view = self.get_state(seat)
        self._validate_call_tiles(seat, action, tiles, pending.tile, pending.discarder)
        pending.responses[seat] = DealerCommand(
            kind="call",
            seat=seat,
            action=action,
            tiles=tiles,
        )
        result = self._resolve_reaction_if_ready()
        command = DealerCommand(kind="call", seat=seat, action=action, tiles=tiles)
        return self._record_command_result(command, result, before_view=before_view)

    def win(self, seat: Seat, action: str | None = None) -> dict[str, Any]:
        if self.state.phase == Phase.DISCARD and seat == self.state.current_turn:
            before_view = self.get_state(seat)
            result = self._apply_tsumo(seat)
            command = DealerCommand(kind="win", seat=seat, action=action or "tsumo")
            return self._record_command_result(command, result, before_view=before_view)
        if self.state.phase == Phase.REACTION:
            pending = self._require_pending_reaction(seat)
            if "ron" not in pending.options[seat]:
                raise ValueError(f"{seat.name} cannot ron")
            before_view = self.get_state(seat)
            pending.responses[seat] = DealerCommand(kind="win", seat=seat, action="ron")
            result = self._resolve_reaction_if_ready()
            command = DealerCommand(kind="win", seat=seat, action="ron")
            return self._record_command_result(command, result, before_view=before_view)
        raise RuntimeError("win is not legal in the current phase")

    def declare_closed_kan(self, seat: Seat, tiles: tuple[str, ...]) -> dict[str, Any]:
        self._require_active_turn(seat, Phase.DISCARD)
        if len(tiles) != 4:
            raise ValueError("closed kan requires four tiles")
        self._require_tiles_in_hand(seat, tiles)
        if len({normalize_tile(tile) for tile in tiles}) != 1:
            raise ValueError("closed kan tiles must all match")
        before_view = self.get_state(seat)
        result = self._apply_own_kan(seat, "closed_kan", tiles, open_meld=False)
        command = DealerCommand(kind="closed_kan", seat=seat, tiles=tiles)
        return self._record_command_result(command, result, before_view=before_view)

    def declare_added_kan(self, seat: Seat, tile: str) -> dict[str, Any]:
        self._require_active_turn(seat, Phase.DISCARD)
        if tile not in self.state.hands[seat]:
            raise ValueError(f"{seat.name} cannot add kan with {tile}: tile is not in hand")
        self._find_added_kan_meld_index(seat, tile)
        command = DealerCommand(kind="added_kan", seat=seat, tile=tile)
        before_view = self.get_state(seat)

        pending_reaction = self._build_chankan_reaction(seat, tile)
        if pending_reaction is not None:
            self.state.pending_reaction = pending_reaction
            self.state.phase = Phase.REACTION
            result = {
                "event": "added_kan_reaction",
                "actor": seat.name,
                "tile": tile,
                "kan_tile": normalize_tile(tile),
                **self.get_state(seat),
            }
            return self._record_command_result(command, result, before_view=before_view)

        result = self._complete_added_kan(seat, tile)
        return self._record_command_result(command, result, before_view=before_view)

    def abortive_draw(self, seat: Seat) -> dict[str, Any]:
        self._require_active_turn(seat, Phase.DISCARD)
        if not self._can_kyuushu_kyuuhai(seat):
            raise ValueError(f"{seat.name} cannot declare kyuushu kyuuhai")
        before_view = self.get_state(seat)
        result = self._finish_draw("kyuushu_kyuuhai")
        return self._record_command_result(
            DealerCommand(kind="abortive_draw", seat=seat),
            result,
            before_view=before_view,
        )

    def get_state(self, seat: Seat | None = None) -> dict[str, Any]:
        view = self.state.public_view(seat)
        if seat is None or self.state.phase == Phase.FINISHED:
            return view

        legal_actions = list(view["legal_actions"])
        action_hints: dict[str, Any] = {}
        if self.state.phase == Phase.DISCARD and seat == self.state.current_turn:
            if self._can_tsumo(seat):
                legal_actions.append("tsumo")
                action_hints["tsumo"] = {"tile": self.state.last_draw}
            if self.state.riichi_declared[seat]:
                forced_tile = self.state.last_draw_by_seat[seat]
                if forced_tile is not None:
                    view["legal_discard_tiles"] = [forced_tile]
                    action_hints["discard"] = {
                        "tiles": [forced_tile],
                        "forced_tsumogiri": True,
                    }
            else:
                if self._can_kyuushu_kyuuhai(seat):
                    legal_actions.append("abortive_draw")
                    action_hints["abortive_draw"] = {"reason": "kyuushu_kyuuhai"}
                if self._can_any_closed_kan(seat):
                    legal_actions.append("closed_kan")
                    action_hints["closed_kan"] = self._closed_kan_candidates(seat)
                if self._can_any_added_kan(seat):
                    legal_actions.append("added_kan")
                    action_hints["added_kan"] = self._added_kan_candidates(seat)
                riichi_discards = self._riichi_discard_candidates(seat)
                if riichi_discards:
                    legal_actions.append("riichi")
                    action_hints["riichi"] = {"discard_tiles": riichi_discards}
        elif self.state.phase == Phase.REACTION and self.state.pending_reaction is not None:
            action_hints = self._reaction_action_hints(seat)
        view["legal_actions"] = sorted(set(legal_actions), key=legal_actions.index)
        view["action_hints"] = action_hints
        view["legal_action_ids"] = legal_action_ids(view)
        view["legal_action_mask"] = legal_action_mask(view)
        view["legal_action_specs"] = [spec.public_view() for spec in legal_action_specs(view)]
        view["obs_public"] = public_observation(view)
        return view

    def handle(self, command: DealerCommand) -> dict[str, Any] | dict[Seat, dict[str, Any]]:
        if command.kind == "start":
            return self.start_hand()
        if command.kind == "state":
            return self.get_state(command.seat)
        if command.kind == "draw":
            if command.seat is None:
                raise ValueError("draw requires a seat")
            return self.draw(command.seat)
        if command.kind == "discard":
            if command.seat is None or command.tile is None:
                raise ValueError("discard requires a seat and tile")
            return self.discard(command.seat, command.tile, riichi=command.riichi)
        if command.kind == "pass":
            if command.seat is None:
                raise ValueError("pass requires a seat")
            return self.pass_reaction(command.seat)
        if command.kind == "call":
            if command.seat is None or command.action is None:
                raise ValueError("call requires a seat and action")
            return self.call(command.seat, command.action, command.tiles)
        if command.kind == "win":
            if command.seat is None:
                raise ValueError("win requires a seat")
            return self.win(command.seat, command.action)
        if command.kind == "closed_kan":
            if command.seat is None:
                raise ValueError("closed_kan requires a seat")
            return self.declare_closed_kan(command.seat, command.tiles)
        if command.kind == "added_kan":
            if command.seat is None or command.tile is None:
                raise ValueError("added_kan requires a seat and tile")
            return self.declare_added_kan(command.seat, command.tile)
        if command.kind == "abortive_draw":
            if command.seat is None:
                raise ValueError("abortive_draw requires a seat")
            return self.abortive_draw(command.seat)
        if command.kind == "stop":
            before_view = self.get_state(command.seat) if command.seat is not None else self.get_state()
            self.state.phase = Phase.FINISHED
            result = {"event": "stopped", **self.get_state(command.seat)}
            return self._record_command_result(command, result, before_view=before_view)
        raise ValueError(f"unknown command: {command.kind}")

    def close_hand_log(self, reason: str = "closed") -> None:
        if self.hand_recorder is not None:
            self.hand_recorder.close(self.state, reason=reason)

    def _record_start_hand(self, views: Mapping[Seat, dict[str, Any]]) -> None:
        if self.hand_recorder is not None:
            self.hand_recorder.start_hand(self.state, self.rules, views)

    def _record_command_result(
        self,
        command: DealerCommand,
        result: dict[str, Any],
        before_view: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.hand_recorder is not None:
            self.hand_recorder.record_command(command, result, self.state, before_view=before_view)
        return result

    def _require_active_turn(self, seat: Seat, phase: Phase) -> None:
        if self.state.phase == Phase.FINISHED:
            raise RuntimeError("hand is already finished")
        if self.state.current_turn != seat:
            raise RuntimeError(f"it is {self.state.current_turn.name}'s turn, not {seat.name}'s")
        if self.state.phase != phase:
            raise RuntimeError(f"dealer expects {self.state.phase.value}, not {phase.value}")

    def _add_drawn_tile(self, seat: Seat, tile: str, rinshan: bool) -> None:
        self.state.hands[seat].append(tile)
        self.state.hands[seat] = sort_tiles(self.state.hands[seat])
        self.state.last_draw = tile
        self.state.last_draw_by_seat[seat] = tile
        self.state.last_draw_was_rinshan = rinshan
        self.state.phase = Phase.DISCARD

    def _build_pending_reaction(self, discarder: Seat, tile: str, from_draw: bool = False) -> PendingReaction | None:
        options: dict[Seat, list[str]] = {}
        for seat in Seat:
            if seat == discarder:
                continue
            seat_options: list[str] = []
            if self._can_ron(seat, tile, discarder):
                seat_options.append("ron")
            if self.state.riichi_declared[seat]:
                if seat_options:
                    options[seat] = seat_options
                continue
            if self._can_daiminkan(seat, tile):
                seat_options.append("kan")
            if self._can_pon(seat, tile):
                seat_options.append("pon")
            if seat == discarder.next and self._can_chii(seat, tile):
                seat_options.append("chii")
            if seat_options:
                options[seat] = seat_options
        if not options:
            return None
        return PendingReaction(discarder=discarder, tile=tile, options=options, from_draw=from_draw)

    def _build_chankan_reaction(self, kan_seat: Seat, tile: str) -> PendingReaction | None:
        options: dict[Seat, list[str]] = {}
        for seat in Seat:
            if seat == kan_seat:
                continue
            if self._can_ron(seat, tile, kan_seat, is_chankan=True):
                options[seat] = ["ron"]
        if not options:
            return None
        return PendingReaction(
            discarder=kan_seat,
            tile=tile,
            options=options,
            reaction_type="added_kan",
        )

    def _finish_discard_without_call(self, discarder: Seat, tile: str, from_draw: bool = False) -> dict[str, Any]:
        self._accept_pending_riichi(discarder)
        self.state.pending_reaction = None
        if self._is_four_winds_draw():
            return self._finish_draw("four_winds")
        if self._is_four_riichi_draw():
            return self._finish_draw("four_riichi")
        self.state.current_turn = discarder.next
        self.state.phase = Phase.DRAW
        if not self.state.live_wall:
            return self._finish_exhaustive_draw()
        return {
            "event": "discard",
            "actor": discarder.name,
            "tile": tile,
            "discard_type": "tsumogiri" if from_draw else "tedashi",
            **self.get_state(discarder),
        }

    def _require_pending_reaction(self, seat: Seat) -> PendingReaction:
        if self.state.phase != Phase.REACTION or self.state.pending_reaction is None:
            raise RuntimeError("dealer is not waiting for reactions")
        pending = self.state.pending_reaction
        if seat not in pending.options:
            raise RuntimeError(f"{seat.name} has no legal reaction")
        if seat in pending.responses:
            raise RuntimeError(f"{seat.name} has already responded")
        return pending

    def _resolve_reaction_if_ready(self) -> dict[str, Any]:
        pending = self.state.pending_reaction
        if pending is None:
            raise RuntimeError("dealer is not waiting for reactions")
        if pending.waiting_seats:
            return {"event": "reaction_waiting", "actor": "DEALER", **self.get_state()}

        ron_responses = [
            response
            for response in pending.responses.values()
            if response.kind == "win" and response.action == "ron"
        ]
        if ron_responses:
            winners = [response.seat for response in ron_responses if response.seat is not None]
            return self._apply_ron(
                winners,
                pending.discarder,
                pending.tile,
                is_chankan=pending.reaction_type == "added_kan",
            )

        calls = [
            response
            for response in pending.responses.values()
            if response.kind == "call" and response.action is not None
        ]
        if not calls:
            if pending.reaction_type == "added_kan":
                self.state.pending_reaction = None
                return self._complete_added_kan(pending.discarder, pending.tile)
            return self._finish_discard_without_call(pending.discarder, pending.tile, from_draw=pending.from_draw)

        self._accept_pending_riichi(pending.discarder)
        self._clear_all_ippatsu()
        selected = min(
            calls,
            key=lambda response: (
                -self._call_priority(response.action or ""),
                self._turn_distance(pending.discarder, response.seat or pending.discarder),
            ),
        )
        assert selected.seat is not None
        assert selected.action is not None
        if selected.action == "kan":
            return self._apply_daiminkan(selected.seat, selected.tiles, pending)
        return self._apply_call(selected.seat, selected.action, selected.tiles, pending)

    def _apply_call(
        self,
        caller: Seat,
        action: str,
        tiles: tuple[str, ...],
        pending: PendingReaction,
    ) -> dict[str, Any]:
        discarder = pending.discarder
        called_tile = pending.tile
        self._remove_last_discard(discarder, called_tile)
        for tile in tiles:
            self.state.hands[caller].remove(tile)

        meld_tiles = sort_tiles((*tiles, called_tile))
        self.state.melds[caller].append(
            Meld(
                kind=action,
                owner=caller,
                from_seat=discarder,
                called_tile=called_tile,
                tiles=tuple(meld_tiles),
            )
        )
        self.state.hands[caller] = sort_tiles(self.state.hands[caller])
        self.state.current_turn = caller
        self.state.phase = Phase.DISCARD
        self.state.pending_reaction = None
        self.state.any_call_made = True
        self.state.last_draw = None
        self.state.last_draw_was_rinshan = False
        return {
            "event": "call",
            "action": action,
            "actor": caller.name,
            "tile": called_tile,
            "call": self.state.melds[caller][-1].public_view(),
            **self.get_state(caller),
        }

    def _apply_daiminkan(
        self,
        caller: Seat,
        tiles: tuple[str, ...],
        pending: PendingReaction,
    ) -> dict[str, Any]:
        discarder = pending.discarder
        called_tile = pending.tile
        self._remove_last_discard(discarder, called_tile)
        for tile in tiles:
            self.state.hands[caller].remove(tile)
        meld_tiles = sort_tiles((*tiles, called_tile))
        self.state.melds[caller].append(
            Meld(
                kind="kan",
                owner=caller,
                from_seat=discarder,
                called_tile=called_tile,
                tiles=tuple(meld_tiles),
                open=True,
            )
        )
        self.state.current_turn = caller
        self.state.pending_reaction = None
        self.state.any_call_made = True
        result = self._after_kan(caller, event="kan")
        result["action"] = "kan"
        result["called_tile"] = called_tile
        result["call"] = self.state.melds[caller][-1].public_view()
        result["kan_tile"] = normalize_tile(called_tile)
        result["rinshan_tile"] = result.get("tile")
        return result

    def _complete_added_kan(self, seat: Seat, tile: str) -> dict[str, Any]:
        meld_index = self._find_added_kan_meld_index(seat, tile)
        meld = self.state.melds[seat][meld_index]
        self.state.hands[seat].remove(tile)
        new_tiles = sort_tiles((*meld.tiles, tile))
        self.state.melds[seat][meld_index] = Meld(
            kind="added_kan",
            owner=seat,
            from_seat=meld.from_seat,
            called_tile=meld.called_tile,
            tiles=tuple(new_tiles),
            open=True,
        )
        result = self._after_kan(seat, event="added_kan")
        return self._annotate_own_kan_result(seat, result)

    def _find_added_kan_meld_index(self, seat: Seat, tile: str) -> int:
        normalized = normalize_tile(tile)
        for index, meld in enumerate(self.state.melds[seat]):
            if meld.kind == "pon" and normalize_tile(meld.called_tile) == normalized:
                return index
        raise ValueError("added kan requires an existing pon")

    def _apply_own_kan(
        self,
        seat: Seat,
        kind: str,
        tiles: tuple[str, ...],
        open_meld: bool,
    ) -> dict[str, Any]:
        for tile in tiles:
            self.state.hands[seat].remove(tile)
        self.state.melds[seat].append(
            Meld(
                kind=kind,
                owner=seat,
                from_seat=seat,
                called_tile=tiles[0],
                tiles=tuple(sort_tiles(tiles)),
                open=open_meld,
            )
        )
        result = self._after_kan(seat, event=kind)
        return self._annotate_own_kan_result(seat, result)

    def _after_kan(self, seat: Seat, event: str) -> dict[str, Any]:
        if self.state.rinshan_draws_used >= 4:
            raise RuntimeError("no rinshan tiles left")
        self.state.kan_count += 1
        self._reveal_next_kan_dora()
        if self.state.live_wall:
            self.state.live_wall.pop(0)
        tile = self.state.dead_wall[self.state.rinshan_draws_used]
        self.state.rinshan_draws_used += 1
        self._add_drawn_tile(seat, tile, rinshan=True)
        if self._is_four_kans_draw():
            return self._finish_draw("four_kans")
        return {"event": event, "actor": seat.name, "tile": tile, **self.get_state(seat)}

    def _annotate_own_kan_result(self, seat: Seat, result: dict[str, Any]) -> dict[str, Any]:
        if result.get("event") == "draw_result":
            return result
        meld = self.state.melds[seat][-1]
        result["action"] = meld.kind
        result["kan_tile"] = normalize_tile(meld.called_tile)
        result["call"] = meld.public_view()
        result["rinshan_tile"] = result.get("tile")
        return result

    def _apply_tsumo(self, winner: Seat) -> dict[str, Any]:
        value = self._score_win(winner, self.state.last_draw, is_tsumo=True)
        if not value.ok or value.cost is None:
            raise ValueError(value.error or "tsumo is not a winning hand")

        dealer_win = winner == self.state.dealer
        if dealer_win:
            payment = int(value.cost["main"]) + int(value.cost.get("main_bonus", 0))
            for seat in Seat:
                if seat != winner:
                    self.state.scores[seat] -= payment
                    self.state.scores[winner] += payment
        else:
            dealer_payment = int(value.cost["main"]) + int(value.cost.get("main_bonus", 0))
            child_payment = int(value.cost["additional"]) + int(value.cost.get("additional_bonus", 0))
            for seat in Seat:
                if seat == winner:
                    continue
                payment = dealer_payment if seat == self.state.dealer else child_payment
                self.state.scores[seat] -= payment
                self.state.scores[winner] += payment

        self.state.scores[winner] += int(value.cost.get("kyoutaku_bonus", 0))
        self.state.riichi_sticks = 0
        return self._finish_win(
            winners=[winner],
            loser=None,
            win_type="tsumo",
            hand_values={winner: value},
        )

    def _apply_ron(
        self,
        winners: list[Seat],
        loser: Seat,
        tile: str,
        is_chankan: bool = False,
    ) -> dict[str, Any]:
        winners = sorted(winners, key=lambda seat: self._turn_distance(loser, seat))
        hand_values: dict[Seat, HandValue] = {}
        for index, winner in enumerate(winners):
            value = self._score_win(
                winner,
                tile,
                is_tsumo=False,
                loser=loser,
                kyoutaku_number=self.state.riichi_sticks if index == 0 else 0,
                is_chankan=is_chankan,
            )
            if not value.ok or value.cost is None:
                raise ValueError(value.error or f"{winner.name} ron is not a winning hand")
            hand_values[winner] = value
            payment = int(value.cost["main"]) + int(value.cost.get("main_bonus", 0))
            self.state.scores[loser] -= payment
            self.state.scores[winner] += payment + int(value.cost.get("kyoutaku_bonus", 0))

        self.state.riichi_sticks = 0
        return self._finish_win(
            winners=winners,
            loser=loser,
            win_type="ron",
            hand_values=hand_values,
        )

    def _finish_win(
        self,
        winners: list[Seat],
        loser: Seat | None,
        win_type: str,
        hand_values: dict[Seat, HandValue],
    ) -> dict[str, Any]:
        result = {
            "type": win_type,
            "winners": [seat.name for seat in winners],
            "loser": loser.name if loser is not None else None,
            "scores": {seat.name: score for seat, score in self.state.scores.items()},
            "hands": {
                seat.name: {
                    "han": value.han,
                    "fu": value.fu,
                    "cost": value.cost,
                    "yaku": list(value.yaku),
                    "fu_details": list(value.fu_details),
                }
                for seat, value in hand_values.items()
            },
        }
        self.state.result = result
        self.state.phase = Phase.FINISHED
        self.state.pending_reaction = None
        return {
            "event": "win",
            "actor": winners[0].name,
            "winners": [seat.name for seat in winners],
            "loser": loser.name if loser is not None else None,
            "win_type": win_type,
            "result": result,
            **self.get_state(winners[0]),
        }

    def _finish_exhaustive_draw(self) -> dict[str, Any]:
        tenpai = [seat for seat in Seat if self._is_tenpai_for_draw(seat)]
        if 0 < len(tenpai) < 4:
            noten = [seat for seat in Seat if seat not in tenpai]
            gain = 3000 // len(tenpai)
            loss = 3000 // len(noten)
            for seat in tenpai:
                self.state.scores[seat] += gain
            for seat in noten:
                self.state.scores[seat] -= loss
        return self._finish_draw(
            "exhaustive",
            extra={"tenpai": [seat.name for seat in tenpai]},
        )

    def _finish_draw(self, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        result = {
            "type": "draw",
            "reason": reason,
            "scores": {seat.name: score for seat, score in self.state.scores.items()},
        }
        if extra:
            result.update(extra)
        self.state.result = result
        self.state.phase = Phase.FINISHED
        self.state.pending_reaction = None
        return {"event": "draw_result", "actor": "DEALER", "result": result, **self.get_state()}

    def _score_win(
        self,
        winner: Seat,
        win_tile: str | None,
        is_tsumo: bool,
        loser: Seat | None = None,
        kyoutaku_number: int | None = None,
        is_chankan: bool = False,
    ) -> HandValue:
        if win_tile is None:
            return HandValue(ok=False, error="missing_win_tile")
        context = WinContext(
            is_tsumo=is_tsumo,
            is_riichi=self.state.riichi_declared[winner],
            is_ippatsu=self.state.ippatsu_active[winner],
            is_rinshan=is_tsumo and self.state.last_draw_was_rinshan,
            is_chankan=not is_tsumo and is_chankan,
            is_haitei=is_tsumo and not self.state.live_wall,
            is_houtei=not is_tsumo and not is_chankan and not self.state.live_wall,
            is_daburu_riichi=False,
            is_tenhou=is_tsumo and winner == self.state.dealer and self.state.turn_count == 0,
            is_chiihou=(
                is_tsumo
                and winner != self.state.dealer
                and self.state.first_turn[winner]
                and not self.state.any_call_made
            ),
            is_renhou=(
                not is_tsumo
                and loser is not None
                and self.state.first_turn[winner]
                and not self.state.any_call_made
            ),
            kyoutaku_number=self.state.riichi_sticks if kyoutaku_number is None else kyoutaku_number,
            tsumi_number=self.state.honba,
        )
        tiles = self.state.hands[winner][:]
        if not is_tsumo:
            tiles.append(win_tile)
        return self.scoring.estimate_hand_value(
            concealed_tiles=sort_tiles(tiles),
            win_tile=win_tile,
            melds=self.state.melds[winner],
            seat_index=int(winner),
            dealer_index=int(self.state.dealer),
            round_wind_index=int(self.state.round_wind),
            dora_indicators=self.state.dora_indicators,
            context=context,
        )

    def _can_tsumo(self, seat: Seat) -> bool:
        value = self._score_win(seat, self.state.last_draw, is_tsumo=True)
        return value.ok

    def _can_ron(self, seat: Seat, tile: str, discarder: Seat, is_chankan: bool = False) -> bool:
        value = self._score_win(
            seat,
            tile,
            is_tsumo=False,
            loser=discarder,
            kyoutaku_number=0,
            is_chankan=is_chankan,
        )
        return value.ok

    def _can_pon(self, seat: Seat, tile: str) -> bool:
        return count_tiles(self.state.hands[seat])[normalize_tile(tile)] >= 2

    def _can_daiminkan(self, seat: Seat, tile: str) -> bool:
        return count_tiles(self.state.hands[seat])[normalize_tile(tile)] >= 3 and self.state.kan_count < 4

    def _can_chii(self, seat: Seat, tile: str) -> bool:
        if not is_suited(tile):
            return False
        counts = count_tiles(self.state.hands[seat])
        rank = tile_rank(tile)
        suit = tile_suit(tile)
        for start in range(rank - 2, rank + 1):
            if start < 1 or start > 7:
                continue
            needed = [f"{candidate}{suit}" for candidate in range(start, start + 3) if candidate != rank]
            if all(counts[needed_tile] > 0 for needed_tile in needed):
                return True
        return False

    def _can_any_closed_kan(self, seat: Seat) -> bool:
        counts = count_tiles(self.state.hands[seat])
        return self.state.kan_count < 4 and any(count == 4 for count in counts.values())

    def _can_any_added_kan(self, seat: Seat) -> bool:
        if self.state.kan_count >= 4:
            return False
        hand_counts = count_tiles(self.state.hands[seat])
        return any(
            meld.kind == "pon" and hand_counts[normalize_tile(meld.called_tile)] > 0
            for meld in self.state.melds[seat]
        )

    def _closed_kan_candidates(self, seat: Seat) -> list[list[str]]:
        candidates: list[list[str]] = []
        for normalized_tile, count in count_tiles(self.state.hands[seat]).items():
            if count == 4:
                tiles = [tile for tile in self.state.hands[seat] if normalize_tile(tile) == normalized_tile]
                candidates.append(sort_tiles(tiles))
        return candidates

    def _added_kan_candidates(self, seat: Seat) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        hand_counts = count_tiles(self.state.hands[seat])
        for meld in self.state.melds[seat]:
            normalized_tile = normalize_tile(meld.called_tile)
            if meld.kind == "pon" and hand_counts[normalized_tile] > 0:
                tiles = [
                    tile
                    for tile in self.state.hands[seat]
                    if normalize_tile(tile) == normalized_tile
                ]
                candidates.extend(
                    {"tile": tile, "upgrades_meld": meld.public_view()}
                    for tile in sort_tiles(tiles)
                )
        return candidates

    def _riichi_discard_candidates(self, seat: Seat) -> list[str]:
        if self.state.riichi_declared[seat]:
            return []
        if not self.scoring.is_closed_hand(self.state.melds[seat]):
            return []
        if self.state.scores[seat] < 1000:
            return []
        if len(self.state.hands[seat]) not in {2, 5, 8, 11, 14}:
            return []

        candidates: list[str] = []
        seen: set[str] = set()
        for tile in self.state.hands[seat]:
            if tile in seen:
                continue
            seen.add(tile)
            remaining = self.state.hands[seat][:]
            remaining.remove(tile)
            try:
                is_tenpai = self.scoring.is_tenpai(remaining)
            except ValueError:
                is_tenpai = False
            if is_tenpai:
                candidates.append(tile)
        return sort_tiles(candidates)

    def _reaction_action_hints(self, seat: Seat) -> dict[str, Any]:
        pending = self.state.pending_reaction
        if pending is None or seat not in pending.options or seat in pending.responses:
            return {}

        tile = pending.tile
        hints: dict[str, Any] = {
            "pass": True,
            "discarder": pending.discarder.name,
            "tile": tile,
        }
        if "ron" in pending.options[seat]:
            hints["ron"] = {"tile": tile, "from": pending.discarder.name}
        if "pon" in pending.options[seat]:
            hints["pon"] = [list(tiles) for tiles in self._matching_combinations(seat, tile, 2)]
        if "kan" in pending.options[seat]:
            hints["kan"] = [list(tiles) for tiles in self._matching_combinations(seat, tile, 3)]
        if "chii" in pending.options[seat]:
            hints["chii"] = [list(tiles) for tiles in self._chii_candidates(seat, tile, pending.discarder)]
        return hints

    def _matching_combinations(self, seat: Seat, tile: str, size: int) -> list[tuple[str, ...]]:
        matches = [
            hand_tile
            for hand_tile in self.state.hands[seat]
            if normalize_tile(hand_tile) == normalize_tile(tile)
        ]
        unique = {tuple(sort_tiles(candidate)) for candidate in combinations(matches, size)}
        return sorted(unique, key=lambda candidate: [sort_tiles([tile])[0] for tile in candidate])

    def _chii_candidates(self, seat: Seat, tile: str, discarder: Seat) -> list[tuple[str, ...]]:
        if seat != discarder.next:
            return []
        candidates: set[tuple[str, ...]] = set()
        for pair in combinations(self.state.hands[seat], 2):
            try:
                self._validate_chii_tiles(seat, tuple(pair), tile)
            except ValueError:
                continue
            candidates.add(tuple(sort_tiles(pair)))
        return sorted(candidates, key=lambda candidate: [sort_tiles([tile])[0] for tile in candidate])

    def _can_kyuushu_kyuuhai(self, seat: Seat) -> bool:
        if not self.state.first_turn[seat] or self.state.any_call_made:
            return False
        terminals = {normalize_tile(tile) for tile in self.state.hands[seat] if is_terminal_or_honor(tile)}
        return len(terminals) >= 9

    def _validate_riichi_discard(self, seat: Seat, tile: str) -> None:
        if self.state.riichi_declared[seat]:
            raise ValueError(f"{seat.name} has already declared riichi")
        if not self.scoring.is_closed_hand(self.state.melds[seat]):
            raise ValueError("riichi requires a closed hand")
        if self.state.scores[seat] < 1000:
            raise ValueError("riichi requires at least 1000 points")
        remaining = self.state.hands[seat][:]
        remaining.remove(tile)
        if not self.scoring.is_tenpai(remaining):
            raise ValueError("riichi discard must leave the hand in tenpai")

    def _accept_pending_riichi(self, discarder: Seat) -> None:
        if self.state.pending_riichi_seat != discarder:
            return
        self.state.scores[discarder] -= 1000
        self.state.riichi_sticks += 1
        self.state.riichi_declared[discarder] = True
        self.state.ippatsu_active[discarder] = True
        self.state.pending_riichi_seat = None

    def _clear_all_ippatsu(self) -> None:
        for seat in Seat:
            self.state.ippatsu_active[seat] = False

    def _validate_call_tiles(
        self,
        seat: Seat,
        action: str,
        tiles: tuple[str, ...],
        called_tile: str,
        discarder: Seat,
    ) -> None:
        if action == "pon":
            self._validate_pon_tiles(seat, tiles, called_tile)
            return
        if action == "chii":
            if seat != discarder.next:
                raise ValueError("chii is only legal from the player to the discarder next")
            self._validate_chii_tiles(seat, tiles, called_tile)
            return
        if action == "kan":
            self._validate_daiminkan_tiles(seat, tiles, called_tile)
            return
        raise ValueError(f"unsupported call action: {action}")

    def _validate_pon_tiles(self, seat: Seat, tiles: tuple[str, ...], called_tile: str) -> None:
        if len(tiles) != 2:
            raise ValueError("pon requires exactly two hand tiles")
        self._require_tiles_in_hand(seat, tiles)
        normalized = normalize_tile(called_tile)
        if any(normalize_tile(tile) != normalized for tile in tiles):
            raise ValueError("pon tiles must match the discarded tile")

    def _validate_daiminkan_tiles(self, seat: Seat, tiles: tuple[str, ...], called_tile: str) -> None:
        if len(tiles) != 3:
            raise ValueError("open kan requires exactly three hand tiles")
        self._require_tiles_in_hand(seat, tiles)
        normalized = normalize_tile(called_tile)
        if any(normalize_tile(tile) != normalized for tile in tiles):
            raise ValueError("kan tiles must match the discarded tile")

    def _validate_chii_tiles(self, seat: Seat, tiles: tuple[str, ...], called_tile: str) -> None:
        if len(tiles) != 2:
            raise ValueError("chii requires exactly two hand tiles")
        if not is_suited(called_tile):
            raise ValueError("chii requires a suited discarded tile")
        self._require_tiles_in_hand(seat, tiles)
        all_tiles = [normalize_tile(tile) for tile in (*tiles, called_tile)]
        if any(not is_suited(tile) for tile in all_tiles):
            raise ValueError("chii tiles must be suited")
        if len({tile_suit(tile) for tile in all_tiles}) != 1:
            raise ValueError("chii tiles must be in the same suit")
        ranks = sorted(tile_rank(tile) for tile in all_tiles)
        if ranks[1] != ranks[0] + 1 or ranks[2] != ranks[1] + 1:
            raise ValueError("chii tiles must form a sequence")

    def _require_tiles_in_hand(self, seat: Seat, tiles: tuple[str, ...]) -> None:
        hand_counts = {tile: self.state.hands[seat].count(tile) for tile in set(tiles)}
        requested_counts = {tile: tiles.count(tile) for tile in set(tiles)}
        missing = [
            tile
            for tile, count in requested_counts.items()
            if hand_counts.get(tile, 0) < count
        ]
        if missing:
            raise ValueError(f"{seat.name} cannot call with tiles not in hand: {missing}")

    def _remove_last_discard(self, seat: Seat, tile: str) -> None:
        if not self.state.discards[seat] or self.state.discards[seat][-1] != tile:
            raise RuntimeError("last discard does not match pending reaction")
        self.state.discards[seat].pop()

    def _reveal_next_kan_dora(self) -> None:
        indicator_index = 4 + len(self.state.dora_indicators) * 2
        ura_index = indicator_index + 1
        if indicator_index < len(self.state.dead_wall):
            self.state.dora_indicators.append(self.state.dead_wall[indicator_index])
        if ura_index < len(self.state.dead_wall):
            self.state.ura_dora_indicators.append(self.state.dead_wall[ura_index])

    def _is_tenpai_for_draw(self, seat: Seat) -> bool:
        return self.scoring.is_tenpai(self.state.hands[seat])

    def _is_four_winds_draw(self) -> bool:
        if self.state.turn_count != 4 or self.state.any_call_made:
            return False
        first_discards = [self.state.discards[seat][0] if self.state.discards[seat] else None for seat in Seat]
        return all(tile in ("E", "S", "W", "N") for tile in first_discards) and len(set(first_discards)) == 1

    def _is_four_riichi_draw(self) -> bool:
        return all(self.state.riichi_declared.values())

    def _is_four_kans_draw(self) -> bool:
        if self.state.kan_count < 4:
            return False
        kan_owners = {
            meld.owner
            for melds in self.state.melds.values()
            for meld in melds
            if "kan" in meld.kind
        }
        return len(kan_owners) > 1

    @staticmethod
    def _call_priority(action: str) -> int:
        priorities = {"ron": 4, "kan": 3, "pon": 2, "chii": 1}
        return priorities[action]

    @staticmethod
    def _turn_distance(from_seat: Seat, to_seat: Seat) -> int:
        return (int(to_seat) - int(from_seat)) % 4


def dealer_process(
    command_queue: Queue,
    response_queues: dict[Seat, Queue],
    seed: int | None = None,
    poll_timeout: float = 0.2,
    hand_log_path: str | Path | None = None,
) -> None:
    """Compatibility wrapper for ``jongmind.runtime.dealer_process``."""
    from jongmind.runtime import dealer_process as run_dealer_process

    run_dealer_process(
        command_queue=command_queue,
        response_queues=response_queues,
        seed=seed,
        poll_timeout=poll_timeout,
        hand_log_path=hand_log_path,
    )
