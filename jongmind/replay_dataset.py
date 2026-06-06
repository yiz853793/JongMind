"""Convert replay logs into supervised learning dataset samples."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from jongmind.action_space import ACTION_SPACE_SIZE


DATASET_SCHEMA = "jongmind.replay_dataset.v1"
SCORE_SEATS = ("EAST", "SOUTH", "WEST", "NORTH")
TRAINING_SAMPLE_TYPES = ("discard", "reaction", "riichi", "win_pass")
REACTION_ACTION_PREFIXES = ("chii_", "pon_", "kan_", "closed_kan_", "added_kan_")
WIN_PASS_ACTIONS = {"tsumo", "ron", "pass"}


@dataclass(frozen=True)
class ReplaySample:
    sample_type: str
    obs_public: dict[str, Any]
    legal_actions: list[int]
    action: int
    action_text: str
    actor: str | None
    phase: str | None
    scores_before: dict[str, int]
    scores_after: dict[str, int]
    final_result: Any
    review: dict[str, Any]

    def to_raw_dict(self) -> dict[str, Any]:
        return {
            "sample_type": self.sample_type,
            "obs_public": self.obs_public,
            "legal_actions": self.legal_actions,
            "action": self.action,
            "action_text": self.action_text,
            "actor": self.actor,
            "phase": self.phase,
            "scores_before": self.scores_before,
            "scores_after": self.scores_after,
            "final_result": self.final_result,
            "review": self.review,
        }

    @classmethod
    def from_raw_dict(cls, data: Mapping[str, Any]) -> "ReplaySample":
        return cls(
            sample_type=str(data["sample_type"]),
            obs_public=dict(data["obs_public"]),
            legal_actions=[int(action) for action in data["legal_actions"]],
            action=int(data["action"]),
            action_text=str(data["action_text"]),
            actor=_optional_str(data.get("actor")),
            phase=_optional_str(data.get("phase")),
            scores_before=_score_dict(data.get("scores_before")),
            scores_after=_score_dict(data.get("scores_after")),
            final_result=data.get("final_result"),
            review=dict(data.get("review") or {}),
        )


def read_hand_history_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one hand replay record per JSONL line."""
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL replay at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"replay line {line_number} must decode to an object")
            yield record


def iter_replay_samples(records: Iterable[Mapping[str, Any]]) -> Iterator[ReplaySample]:
    """Yield supervised samples from hand history or env replay records."""
    for record_index, record in enumerate(records):
        yield from samples_from_record(record, record_index=record_index)


def samples_from_record(record: Mapping[str, Any], record_index: int = 0) -> list[ReplaySample]:
    final_result = _final_result(record)
    samples: list[ReplaySample] = []
    for event_index, event in enumerate(record.get("events") or ()):
        if not isinstance(event, Mapping):
            continue
        if event.get("type") not in {"command", "step"}:
            continue
        sample = sample_from_event(
            record=record,
            event=event,
            final_result=final_result,
            record_index=record_index,
            event_index=event_index,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def sample_from_event(
    record: Mapping[str, Any],
    event: Mapping[str, Any],
    final_result: Any,
    record_index: int = 0,
    event_index: int = 0,
) -> ReplaySample | None:
    action = event.get("action")
    if action is None:
        return None
    action = int(action)

    action_text = str(event.get("action_text") or "")
    sample_type = _sample_type(event, action_text)
    if sample_type is None:
        return None

    obs_public = _obs_public(event)
    legal_actions = _legal_actions(event, obs_public)
    if action not in legal_actions:
        raise ValueError(
            f"event action {action} is not in legal_actions at record={record_index} event={event_index}"
        )

    review = {
        "hidden_state_for_review_only": event.get("hidden_state_for_review_only"),
        "event_result": event.get("result"),
        "command": event.get("command"),
        "record_index": record_index,
        "event_index": int(event.get("index", event_index)),
        "hand_index": record.get("hand_index"),
        "source_schema": record.get("schema"),
    }
    return ReplaySample(
        sample_type=sample_type,
        obs_public=obs_public,
        legal_actions=legal_actions,
        action=action,
        action_text=action_text,
        actor=_optional_str(event.get("actor")),
        phase=_optional_str(event.get("phase")),
        scores_before=_score_dict(event.get("scores_before")),
        scores_after=_score_dict(event.get("scores_after")),
        final_result=final_result,
        review=review,
    )


def load_replay_samples(source: str | Path | Iterable[Mapping[str, Any]]) -> list[ReplaySample]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".pt":
            return _load_pt_samples(path)
        return list(iter_replay_samples(read_hand_history_jsonl(path)))
    return [
        item if isinstance(item, ReplaySample) else ReplaySample.from_raw_dict(item)
        for item in source
    ]


def build_dataset_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    samples = load_replay_samples(input_path)
    payload = {
        "schema": DATASET_SCHEMA,
        "source": str(input_path),
        "sample_count": len(samples),
        "sample_type_counts": sample_type_counts(samples),
        "samples": [sample.to_raw_dict() for sample in samples],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload


def sample_type_counts(samples: Sequence[ReplaySample]) -> dict[str, int]:
    counter = Counter(sample.sample_type for sample in samples)
    return {sample_type: counter.get(sample_type, 0) for sample_type in TRAINING_SAMPLE_TYPES}


class MahjongReplayDataset(Dataset):
    """PyTorch dataset for replay-derived imitation samples."""

    def __init__(self, source: str | Path | Iterable[Mapping[str, Any] | ReplaySample]) -> None:
        self.samples = load_replay_samples(source)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_type": sample.sample_type,
            "obs": _json_dumps(sample.obs_public),
            "obs_public": _json_dumps(sample.obs_public),
            "legal_actions": _legal_action_mask(sample.legal_actions),
            "legal_action_ids": _json_dumps(sample.legal_actions),
            "action": torch.tensor(sample.action, dtype=torch.long),
            "action_text": sample.action_text,
            "actor": sample.actor or "",
            "phase": sample.phase or "",
            "scores_before": _score_tensor(sample.scores_before),
            "scores_after": _score_tensor(sample.scores_after),
            "final_result": _json_dumps(sample.final_result),
            "review": _json_dumps(sample.review),
        }

    def raw_sample(self, index: int) -> dict[str, Any]:
        return self.samples[index].to_raw_dict()


def _sample_type(event: Mapping[str, Any], action_text: str) -> str | None:
    command = event.get("command") or {}
    command_kind = command.get("kind") if isinstance(command, Mapping) else None
    if action_text == "riichi" or (isinstance(command, Mapping) and command.get("riichi")):
        return "riichi"
    if action_text.startswith("discard_"):
        return "discard"
    if action_text in WIN_PASS_ACTIONS or command_kind == "win":
        return "win_pass"
    if action_text.startswith(REACTION_ACTION_PREFIXES):
        return "reaction"
    if str(event.get("phase") or "") == "reaction" and action_text != "draw":
        return "reaction"
    return None


def _obs_public(event: Mapping[str, Any]) -> dict[str, Any]:
    obs = event.get("obs_public") or {}
    if not isinstance(obs, Mapping):
        raise ValueError("event obs_public must be an object")
    obs = dict(obs)
    obs.pop("hidden_state_for_review_only", None)
    return obs


def _legal_actions(event: Mapping[str, Any], obs_public: Mapping[str, Any]) -> list[int]:
    actions = event.get("legal_actions") or obs_public.get("legal_action_ids") or []
    return [int(action) for action in actions]


def _final_result(record: Mapping[str, Any]) -> Any:
    if record.get("result") is not None:
        return record.get("result")
    final_state = record.get("final_state") or {}
    if isinstance(final_state, Mapping) and final_state.get("result") is not None:
        return final_state.get("result")
    for event in reversed(record.get("events") or ()):
        if isinstance(event, Mapping) and event.get("done") and event.get("result") is not None:
            return event.get("result")
    return {}


def _load_pt_samples(path: Path) -> list[ReplaySample]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    raw_samples = payload.get("samples", payload) if isinstance(payload, Mapping) else payload
    return [
        item if isinstance(item, ReplaySample) else ReplaySample.from_raw_dict(item)
        for item in raw_samples
    ]


def _legal_action_mask(legal_actions: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
    for action in legal_actions:
        if 0 <= int(action) < ACTION_SPACE_SIZE:
            mask[int(action)] = True
    return mask


def _score_tensor(scores: Mapping[str, int]) -> torch.Tensor:
    return torch.tensor([int(scores.get(seat, 0)) for seat in SCORE_SEATS], dtype=torch.long)


def _score_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(seat): int(score) for seat, score in value.items()}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
