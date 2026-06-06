"""Hand history recording helpers for dealer replay logs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from jongmind.action_space import human_readable_to_action_id, public_observation
from jongmind.game import DealerCommand, DealerState, Phase, Seat
from jongmind.rules import RuleSet
from jongmind.tiles import sort_tiles


class HandHistoryRecorder:
    """Append one completed hand record per line to a JSONL file."""

    schema = "jongmind.hand_history.v1"

    def __init__(
        self,
        path: str | Path,
        seed: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.seed = seed
        self.context = dict(context or {})
        self._current: dict[str, Any] | None = None
        self._hand_index = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_hand(
        self,
        state: DealerState,
        rules: RuleSet,
        views: Mapping[Seat, dict[str, Any]],
    ) -> None:
        if self._current is not None:
            self._write_current(state, completed=False, close_reason="superseded_by_new_hand")

        self._current = {
            "schema": self.schema,
            "recorded_at": _timestamp(),
            "seed": self.seed,
            "context": _jsonable(self.context),
            "hand_index": self._hand_index,
            "hand": {
                "hand_number": state.hand_number,
                "max_hand_number": state.max_hand_number,
                "round_wind": state.round_wind.name,
                "dealer": state.dealer.name,
                "honba": state.honba,
                "riichi_sticks": state.riichi_sticks,
                "start_scores": {seat.name: score for seat, score in state.scores.items()},
            },
            "rules": _jsonable(asdict(rules)),
            "initial_state": _state_snapshot(state),
            "start_views": _jsonable(views),
            "events": [],
        }
        self._append_event(
            {
                "type": "start_hand",
                "public_state_after": _public_state_snapshot(state),
            }
        )

    def record_command(
        self,
        command: DealerCommand,
        result: dict[str, Any],
        state: DealerState,
        before_view: dict[str, Any] | None = None,
    ) -> None:
        if self._current is None:
            return

        public_after = _public_state_snapshot(state)
        before_public = public_observation(before_view) if before_view is not None else self._public_before_next_event()
        scores_before = {
            str(seat): int(score)
            for seat, score in (before_public.get("scores") or {}).items()
        }
        action_text = _command_action_text(command)
        self._append_event(
            {
                "type": "command",
                "round_id": state.hand_number,
                "step": before_public.get("turn_count", state.turn_count),
                "actor": command.seat.name if command.seat is not None else result.get("actor"),
                "phase": before_public.get("phase") or result.get("phase") or state.phase.value,
                "command": _jsonable(asdict(command)),
                "action": _command_action_id(action_text),
                "action_text": action_text,
                "scores_before": scores_before,
                "scores_after": {seat.name: score for seat, score in state.scores.items()},
                "obs_public": before_public,
                "legal_actions": list(before_public.get("legal_action_ids", [])),
                "visible_state": public_after,
                "hidden_state_for_review_only": _hidden_review_snapshot(state),
                "result": _jsonable(result),
                "public_state_after": public_after,
            }
        )
        if state.phase == Phase.FINISHED:
            self._write_current(state, completed=True, close_reason=None)

    def close(self, state: DealerState, reason: str = "closed") -> None:
        if self._current is not None:
            self._write_current(
                state,
                completed=state.phase == Phase.FINISHED,
                close_reason=reason,
            )

    def _append_event(self, event: dict[str, Any]) -> None:
        if self._current is None:
            return
        event = {
            "index": len(self._current["events"]),
            "recorded_at": _timestamp(),
            **event,
        }
        self._current["events"].append(event)

    def _public_before_next_event(self) -> dict[str, Any]:
        if self._current is None:
            return {}
        if self._current["events"]:
            previous = self._current["events"][-1].get("public_state_after") or {}
            if previous:
                return previous
        initial = self._current.get("initial_state", {}).get("public_view", {})
        return initial

    def _write_current(
        self,
        state: DealerState,
        completed: bool,
        close_reason: str | None,
    ) -> None:
        if self._current is None:
            return

        record = self._current
        record["completed"] = completed
        record["finished_at"] = _timestamp()
        record["final_state"] = _state_snapshot(state)
        record["result"] = _jsonable(state.result)
        if close_reason is not None:
            record["close_reason"] = close_reason

        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        self._current = None
        self._hand_index += 1


def _state_snapshot(state: DealerState) -> dict[str, Any]:
    snapshot = _jsonable(asdict(state))
    snapshot["public_view"] = _public_state_snapshot(state)
    return snapshot


def _public_state_snapshot(state: DealerState) -> dict[str, Any]:
    return _jsonable(state.public_view())


def _hidden_review_snapshot(state: DealerState) -> dict[str, Any]:
    return _jsonable(
        {
            "hands": state.hands,
            "live_wall": state.live_wall,
            "dead_wall": state.dead_wall,
            "ura_dora_indicators": state.ura_dora_indicators,
            "last_draw": state.last_draw,
            "last_draw_by_seat": state.last_draw_by_seat,
            "last_draw_was_rinshan": state.last_draw_was_rinshan,
            "first_turn": state.first_turn,
            "ippatsu_active": state.ippatsu_active,
            "pending_riichi_seat": state.pending_riichi_seat,
        }
    )


def _command_action_text(command: DealerCommand) -> str:
    if command.kind == "discard":
        if command.riichi:
            return "riichi"
        return f"discard_{command.tile}"
    if command.kind == "win":
        return command.action or "win"
    if command.kind == "call":
        return "_".join((command.action or "call", *sort_tiles(command.tiles)))
    if command.kind == "closed_kan":
        return "_".join(("closed_kan", *sort_tiles(command.tiles)))
    if command.kind == "added_kan":
        return f"added_kan_{command.tile}"
    return command.kind


def _command_action_id(action_text: str) -> int | None:
    try:
        return human_readable_to_action_id(action_text)
    except ValueError:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Seat):
        return value.name
    if isinstance(value, Phase):
        return value.value
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {_json_key(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def _json_key(key: Any) -> str:
    if isinstance(key, Seat):
        return key.name
    if isinstance(key, Phase):
        return key.value
    if isinstance(key, Enum):
        return key.name
    return str(key)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")
