"""Runtime message helpers for dealer/player process loops."""

from __future__ import annotations

from multiprocessing import Queue
from queue import Empty
from typing import Any

from jongmind.game import DealerCommand, Seat


def dealer_process(
    command_queue: Queue,
    response_queues: dict[Seat, Queue],
    seed: int | None = None,
    poll_timeout: float = 0.2,
) -> None:
    """Run a dealer event loop in its own process."""
    from jongmind.dealer import MahjongDealer

    dealer = MahjongDealer(seed=seed)

    while True:
        try:
            command: DealerCommand = command_queue.get(timeout=poll_timeout)
        except Empty:
            continue

        target = command.seat
        try:
            result = dealer.handle(command)
            if command.kind == "start":
                assert isinstance(result, dict)
                for seat, queue in response_queues.items():
                    queue.put({"ok": True, **result[seat]})
            elif command.kind in {
                "draw",
                "discard",
                "pass",
                "call",
                "win",
                "closed_kan",
                "added_kan",
                "abortive_draw",
                "stop",
            }:
                assert isinstance(result, dict)
                broadcast_dealer_result(command, result, dealer, response_queues)
            elif target is None:
                for queue in response_queues.values():
                    queue.put({"ok": True, **result})
            else:
                response_queues[target].put({"ok": True, **result})
        except Exception as exc:  # pragma: no cover - process boundary reporting
            error = {"ok": False, "error": str(exc), "command": command.kind}
            if target is None:
                for queue in response_queues.values():
                    queue.put(error)
            else:
                response_queues[target].put(error)

        if command.kind == "stop":
            break


def broadcast_dealer_result(
    command: DealerCommand,
    result: dict[str, Any],
    dealer: Any,
    response_queues: dict[Seat, Queue],
) -> None:
    event = result.get("event")
    if event == "reaction_waiting":
        return
    actor = result.get("actor") or (command.seat.name if command.seat is not None else "DEALER")
    recipients = broadcast_recipients(command, event, dealer, response_queues)
    for seat, queue in recipients.items():
        state_view = dealer.get_state(seat)
        message_event = viewer_event(event, state_view)
        private_tile = (
            result.get("tile")
            if (
                (command.kind == "draw" and seat == command.seat)
                or (command.kind == "call" and result.get("event") == "kan" and seat == command.seat)
            )
            else None
        )
        message = {
            "ok": True,
            "event": message_event,
            "actor": actor,
            "public_event": public_event(command, result, viewer=seat, private_tile=private_tile),
            **state_view,
        }
        if event in {"discard", "discard_reaction"}:
            message["tile"] = result.get("tile") or command.tile
        elif command.kind == "draw" and seat == command.seat:
            message["tile"] = result.get("tile")
        elif command.kind == "call":
            message["tile"] = private_tile if result.get("event") == "kan" else result.get("tile")
            message["called_tile"] = result.get("called_tile")
            message["call"] = result.get("call")
            message["action"] = result.get("action") or command.action
        elif command.kind in {"closed_kan", "added_kan"}:
            message["tile"] = (
                result.get("tile")
                if event == "added_kan_reaction" or seat == command.seat
                else None
            )
            message["action"] = result.get("action") or command.kind
            message["kan_tile"] = result.get("kan_tile")
            message["call"] = result.get("call")
        queue.put(message)


def broadcast_recipients(
    command: DealerCommand,
    event: str | None,
    dealer: Any,
    response_queues: dict[Seat, Queue],
) -> dict[Seat, Queue]:
    if event == "discard" and command.kind == "pass":
        return {dealer.state.current_turn: response_queues[dealer.state.current_turn]}
    return response_queues


def viewer_event(event: str | None, state_view: dict[str, Any]) -> str | None:
    if event == "discard_reaction" and "pass" not in state_view.get("legal_actions", []):
        return "discard"
    if event == "added_kan_reaction" and "pass" not in state_view.get("legal_actions", []):
        return "added_kan"
    return event


def public_event(
    command: DealerCommand,
    result: dict[str, Any],
    viewer: Seat,
    private_tile: str | None,
) -> dict[str, Any]:
    actor = result.get("actor") or (command.seat.name if command.seat is not None else "DEALER")
    event = result.get("event")
    if event == "discard_reaction":
        event = "discard"
    elif event == "added_kan_reaction":
        event = "added_kan"
    public: dict[str, Any] = {"event": event, "actor": actor}
    if event == "draw":
        public["seat"] = actor
        public["tile"] = private_tile
        public["private_to"] = actor if private_tile is not None else None
    elif event in {"discard", "discard_reaction"}:
        public["seat"] = actor
        public["tile"] = result.get("tile") or command.tile
        public["discard_type"] = result.get("discard_type")
    elif event == "call":
        public["seat"] = actor
        public["action"] = result.get("action") or command.action
        public["called_tile"] = result.get("called_tile") or result.get("tile")
        public["meld"] = result.get("call")
    elif event in {"kan", "closed_kan", "added_kan"}:
        public["seat"] = actor
        public["action"] = result.get("action") or command.action or command.kind
        public["called_tile"] = result.get("called_tile")
        public["kan_tile"] = result.get("kan_tile")
        public["meld"] = result.get("call")
        public["rinshan_tile"] = private_tile
        public["rinshan_private_to"] = actor if private_tile is not None else None
    elif event == "win":
        public["winners"] = result.get("winners")
        public["loser"] = result.get("loser")
        public["win_type"] = result.get("win_type")
        public["result"] = result.get("result")
    elif event == "draw_result":
        public["result"] = result.get("result")
    return public
