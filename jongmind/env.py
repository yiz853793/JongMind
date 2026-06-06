"""Standard environment wrapper around ``MahjongDealer``."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from jongmind.action_space import ACTION_SPACE_SIZE, legal_action_mask, public_observation
from jongmind.dealer import MahjongDealer
from jongmind.game import DealerCommand, Phase, Seat
from jongmind.rules import MAHJONG_SOUL_4P_RANKED, RuleSet


class RiichiEnv:
    """A single-hand Riichi Mahjong environment with integer actions."""

    def __init__(
        self,
        seed: int | None = None,
        rules: RuleSet = MAHJONG_SOUL_4P_RANKED,
        hand_log_path: str | Path | None = None,
        record_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.seed = seed
        self.rules = rules
        self.hand_log_path = hand_log_path
        self.record_context = dict(record_context or {})
        self.dealer = self._new_dealer(seed)
        self._started = False
        self._done = True
        self._replay: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Start one hand and return the active player's public observation."""
        if seed is not None:
            self.seed = seed
        self.dealer = self._new_dealer(self.seed)
        views = self.dealer.start_hand()
        self._started = True
        self._done = False
        self._replay = {
            "seed": self.seed,
            "rules": self.rules.name,
            "events": [
                {
                    "type": "reset",
                    "obs_public": public_observation(views[self.dealer.state.current_turn]),
                    "hidden_state_for_review_only": self._hidden_state_for_review_only(),
                }
            ],
        }
        return self.get_observation()

    def step(self, action_id: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute one legal integer action and return ``obs, reward, done, info``."""
        if not self._started:
            raise RuntimeError("environment must be reset before step")
        if self._done:
            raise RuntimeError("hand is already finished")
        if not isinstance(action_id, int) or isinstance(action_id, bool):
            raise TypeError("step only accepts an integer action_id")

        state = self._active_state()
        mask = list(state.get("legal_action_mask") or legal_action_mask(state))
        if action_id < 0 or action_id >= len(mask) or not mask[action_id]:
            raise ValueError(f"illegal action_id: {action_id}")

        spec = self._action_spec(state, action_id)
        command = self._command_from_spec(spec, state)
        scores_before = dict(state.get("scores") or {})
        result = self.dealer.handle(command)
        assert isinstance(result, dict)

        self._done = self.dealer.state.phase == Phase.FINISHED
        obs = self.get_observation()
        info = {
            "action_spec": deepcopy(spec),
            "command": self._command_view(command),
            "result": deepcopy(result),
            "scores_before": scores_before,
            "scores_after": {seat.name: score for seat, score in self.dealer.state.scores.items()},
            "hidden_state_for_review_only": self._hidden_state_for_review_only(),
        }
        reward = self._reward(scores_before, command.seat) if self._done else 0.0
        self._append_replay_event(action_id, state, obs, reward, self._done, info)
        return obs, reward, self._done, info

    def legal_actions(self) -> list[int]:
        """Return the current legal action mask."""
        if not self._started or self._done:
            return [0] * ACTION_SPACE_SIZE
        state = self._active_state()
        return list(state.get("legal_action_mask") or legal_action_mask(state))

    def get_observation(self) -> dict[str, Any]:
        """Return only the current active player's public observation."""
        if not self._started:
            raise RuntimeError("environment must be reset before observation")
        return public_observation(self._active_state())

    def get_replay(self) -> dict[str, Any]:
        """Return the in-memory replay collected by this environment."""
        return deepcopy(self._replay)

    def _new_dealer(self, seed: int | None) -> MahjongDealer:
        return MahjongDealer(
            seed=seed,
            rules=self.rules,
            hand_log_path=self.hand_log_path,
            record_context=self.record_context,
        )

    def _active_state(self) -> dict[str, Any]:
        if self.dealer.state.phase == Phase.FINISHED:
            return self.dealer.get_state()
        seat = self._active_seat()
        return self.dealer.get_state(seat)

    def _active_seat(self) -> Seat:
        if self.dealer.state.phase in {Phase.DRAW, Phase.DISCARD}:
            return self.dealer.state.current_turn
        if self.dealer.state.phase == Phase.REACTION and self.dealer.state.pending_reaction is not None:
            waiting = self.dealer.state.pending_reaction.waiting_seats
            if waiting:
                return waiting[0]
        raise RuntimeError("no active player is available")

    @staticmethod
    def _action_spec(state: dict[str, Any], action_id: int) -> dict[str, Any]:
        for spec in state.get("legal_action_specs") or ():
            if spec.get("id") == action_id:
                return dict(spec)
        raise ValueError(f"action_id is legal in the mask but has no action spec: {action_id}")

    def _command_from_spec(self, spec: dict[str, Any], state: dict[str, Any]) -> DealerCommand:
        kind = spec.get("kind")
        seat = Seat[str(state["seat"])]
        if kind == "draw":
            return DealerCommand(kind="draw", seat=seat)
        if kind == "discard":
            return DealerCommand(kind="discard", seat=seat, tile=str(spec["tile"]))
        if kind == "riichi":
            discard_tiles = tuple(spec.get("riichi_discard_tiles") or ())
            if not discard_tiles:
                raise ValueError("riichi action has no discard tile candidate")
            return DealerCommand(kind="discard", seat=seat, tile=str(discard_tiles[0]), riichi=True)
        if kind == "tsumo":
            return DealerCommand(kind="win", seat=seat, action="tsumo")
        if kind == "ron":
            return DealerCommand(kind="win", seat=seat, action="ron")
        if kind == "pass":
            return DealerCommand(kind="pass", seat=seat)
        if kind in {"chii", "pon", "kan"}:
            return DealerCommand(kind="call", seat=seat, action=str(kind), tiles=tuple(spec.get("tiles") or ()))
        if kind == "closed_kan":
            return DealerCommand(kind="closed_kan", seat=seat, tiles=tuple(spec.get("tiles") or ()))
        if kind == "added_kan":
            return DealerCommand(kind="added_kan", seat=seat, tile=str(spec["tile"]))
        if kind == "abortive_draw":
            return DealerCommand(kind="abortive_draw", seat=seat)
        raise ValueError(f"unsupported action kind: {kind}")

    def _hidden_state_for_review_only(self) -> dict[str, Any]:
        state = self.dealer.state
        return {
            "hands": {seat.name: tiles[:] for seat, tiles in state.hands.items()},
            "live_wall": state.live_wall[:],
            "dead_wall": state.dead_wall[:],
            "ura_dora_indicators": state.ura_dora_indicators[:],
            "last_draw": state.last_draw,
            "last_draw_by_seat": {
                seat.name: tile for seat, tile in state.last_draw_by_seat.items()
            },
            "last_draw_was_rinshan": state.last_draw_was_rinshan,
            "first_turn": {seat.name: first for seat, first in state.first_turn.items()},
            "ippatsu_active": {seat.name: active for seat, active in state.ippatsu_active.items()},
            "pending_riichi_seat": (
                state.pending_riichi_seat.name if state.pending_riichi_seat is not None else None
            ),
        }

    @staticmethod
    def _command_view(command: DealerCommand) -> dict[str, Any]:
        return {
            "kind": command.kind,
            "seat": command.seat.name if command.seat is not None else None,
            "tile": command.tile,
            "action": command.action,
            "tiles": list(command.tiles),
            "riichi": command.riichi,
        }

    def _reward(self, scores_before: dict[str, Any], seat: Seat | None) -> float:
        if seat is None:
            return 0.0
        before = int(scores_before.get(seat.name, self.rules.starting_points))
        after = int(self.dealer.state.scores.get(seat, before))
        return float(after - before)

    def _append_replay_event(
        self,
        action_id: int,
        state_before: dict[str, Any],
        obs_after: dict[str, Any],
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self._replay.setdefault("events", []).append(
            {
                "type": "step",
                "actor": state_before.get("seat"),
                "phase": state_before.get("phase"),
                "action": action_id,
                "action_text": info["action_spec"].get("text"),
                "legal_actions": list(state_before.get("legal_action_ids") or []),
                "obs_public": public_observation(state_before),
                "next_obs_public": obs_after,
                "reward": reward,
                "done": done,
                "result": deepcopy(info.get("result")),
                "hidden_state_for_review_only": deepcopy(info["hidden_state_for_review_only"]),
            }
        )
