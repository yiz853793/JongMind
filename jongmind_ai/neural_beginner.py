"""Neural beginner discard model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from jongmind_ai.base import CallDecision, DiscardDecision
from jongmind_ai.features import (
    FEATURE_SIZE,
    REACTION_ACTIONS,
    REACTION_FEATURE_SIZE,
    TILE_TYPES,
    encode_reaction_state,
    encode_state,
    legal_discard_mask,
    legal_reaction_mask,
)
from jongmind_ai.tile_efficiency import TileEfficiencyAgent


DEFAULT_CHECKPOINT = Path("outputs/checkpoints/neural_beginner.pt")


class ResMLPBlock(nn.Module):
    """Residual MLP block for flat Mahjong feature vectors."""

    def __init__(self, hidden_size: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(self.norm(features))


class ResMLPEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        num_blocks: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResMLPBlock(hidden_size, dropout=dropout) for _ in range(num_blocks))
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.input(features)
        encoded = self.blocks(encoded)
        return self.output_norm(encoded)


class _PolicyHeadAccessor:
    """Compatibility shim for old tests that adjust ``network[-1].bias``."""

    def __init__(self, policy_head: nn.Linear) -> None:
        self.policy_head = policy_head

    def __getitem__(self, index: int) -> nn.Linear:
        if index in (-1, 0):
            return self.policy_head
        raise IndexError(index)


class DiscardNet(nn.Module):
    def __init__(
        self,
        feature_size: int = FEATURE_SIZE,
        hidden_size: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder = ResMLPEncoder(
            feature_size=feature_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.policy_head = nn.Linear(hidden_size, len(TILE_TYPES))
        self.value_head = nn.Linear(hidden_size, 1)
        self.network = _PolicyHeadAccessor(self.policy_head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_critic(features)[0]

    def actor_critic(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return self.policy_head(encoded), self.value_head(encoded).squeeze(-1)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_critic(features)[1]


class CallNet(nn.Module):
    def __init__(
        self,
        feature_size: int = REACTION_FEATURE_SIZE,
        hidden_size: int = 192,
        num_blocks: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder = ResMLPEncoder(
            feature_size=feature_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.policy_head = nn.Linear(hidden_size, len(REACTION_ACTIONS))
        self.value_head = nn.Linear(hidden_size, 1)
        self.network = _PolicyHeadAccessor(self.policy_head)
        with torch.no_grad():
            self.policy_head.bias[REACTION_ACTIONS.index("pass")] = 1.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_critic(features)[0]

    def actor_critic(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return self.policy_head(encoded), self.value_head(encoded).squeeze(-1)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_critic(features)[1]




def load_state_dict_flexible(module: nn.Module, state_dict: dict[str, torch.Tensor], name: str = "model") -> bool:
    """Load matching tensors and copy overlapping input columns after feature upgrades.

    Old checkpoints remain useful when FEATURE_SIZE grows: hidden/output layers are
    loaded exactly, while the first Linear layer copies the old feature columns and
    leaves new feature columns at their random initialization.
    """
    own_state = module.state_dict()
    patched_state = {key: value.clone() for key, value in own_state.items()}
    fully_loaded = True
    copied: list[str] = []
    skipped: list[str] = []
    resized: list[str] = []

    for key, tensor in state_dict.items():
        if key not in patched_state:
            skipped.append(key)
            fully_loaded = False
            continue
        target = patched_state[key]
        if target.shape == tensor.shape:
            patched_state[key] = tensor.clone()
            copied.append(key)
            continue
        if key.endswith(".weight") and target.ndim == 2 and tensor.ndim == 2 and target.shape[0] == tensor.shape[0]:
            merged = target.clone()
            cols = min(target.shape[1], tensor.shape[1])
            merged[:, :cols] = tensor[:, :cols]
            patched_state[key] = merged
            resized.append(f"{key}:{tuple(tensor.shape)}->{tuple(target.shape)}")
            fully_loaded = False
            continue
        skipped.append(f"{key}:{tuple(tensor.shape)}->{tuple(target.shape)}")
        fully_loaded = False

    module.load_state_dict(patched_state)
    if resized or skipped:
        print(
            f"partially loaded {name}: copied={len(copied)} resized={resized} skipped={skipped}"
        )
    return fully_loaded

class NeuralBeginnerAgent:
    """Small neural discard model with a safe teacher fallback.

    Train it with ``python -m jongmind.train_policy_random``. Until a
    checkpoint exists, the agent falls back to tile efficiency instead of
    evaluating random neural weights.
    """

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.fallback = TileEfficiencyAgent()
        self.model: DiscardNet | None = None
        self.call_model: CallNet | None = None
        if self.checkpoint_path.exists():
            self.model = DiscardNet()
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model_state", checkpoint)
            load_state_dict_flexible(self.model, state_dict, name="discard_model")
            self.model.eval()
            call_state_dict = checkpoint.get("call_model_state")
            if call_state_dict is not None:
                self.call_model = CallNet()
                load_state_dict_flexible(self.call_model, call_state_dict, name="call_model")
                self.call_model.eval()

    def choose_discard(self, hand: list[str], state: dict[str, Any]) -> DiscardDecision:
        if self.model is None:
            decision = self.fallback.choose_discard(hand, state)
            return DiscardDecision(
                tile=decision.tile,
                shanten=decision.shanten,
                ukeire=decision.ukeire,
                reason=f"neural_checkpoint_missing_fallback:{decision.reason}",
            )

        with torch.no_grad():
            features = encode_state(hand, state).unsqueeze(0)
            logits = self.model(features).squeeze(0)
            mask = legal_discard_mask(hand, state)
            logits = logits.masked_fill(~mask, -1_000_000_000.0)
            tile = TILE_TYPES[int(torch.argmax(logits).item())]
        return DiscardDecision(tile=tile, shanten=-1, ukeire=-1, reason="neural_beginner")

    def choose_reaction(self, state: dict[str, Any]) -> CallDecision:
        if self.call_model is None:
            return CallDecision(action=None, reason="neural_call_checkpoint_missing_pass")

        hand = state.get("hand") or []
        with torch.no_grad():
            features = encode_reaction_state(hand, state).unsqueeze(0)
            logits = self.call_model(features).squeeze(0)
            mask = legal_reaction_mask(state)
            logits = logits.masked_fill(~mask, -1_000_000_000.0)
            action = REACTION_ACTIONS[int(torch.argmax(logits).item())]

        if action == "pass":
            return CallDecision(action=None, reason="neural_beginner_pass")

        candidates = (state.get("action_hints") or {}).get(action) or []
        if not candidates:
            return CallDecision(action=None, reason=f"neural_beginner_no_{action}_candidate")
        return CallDecision(
            action=action,
            tiles=tuple(candidates[0]),
            reason=f"neural_beginner_{action}",
        )
