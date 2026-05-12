"""Model registry used by evaluation and training tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jongmind_ai.base import DiscardAgent

ModelFactory = Callable[[int], DiscardAgent]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: ModelFactory
    description: str = ""


_MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    factory: ModelFactory,
    description: str = "",
    *,
    replace: bool = False,
) -> None:
    if not name:
        raise ValueError("model name cannot be empty")
    if name in _MODEL_REGISTRY and not replace:
        raise ValueError(f"model already registered: {name}")
    _MODEL_REGISTRY[name] = ModelSpec(name=name, factory=factory, description=description)


def create_model(name: str, seed: int = 0) -> DiscardAgent:
    try:
        spec = _MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown model: {name}; choose from {model_names()}") from exc
    return spec.factory(seed)


def model_names() -> tuple[str, ...]:
    return tuple(_MODEL_REGISTRY)


def available_models() -> dict[str, str]:
    return {name: spec.description for name, spec in _MODEL_REGISTRY.items()}
