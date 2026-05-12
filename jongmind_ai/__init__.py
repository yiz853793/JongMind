"""Playing models for JongMind."""

from jongmind_ai.base import CallDecision, DiscardAgent, DiscardDecision, ReactionAgent
from jongmind_ai.beginner import BeginnerAgent
from jongmind_ai.neural_beginner import NeuralBeginnerAgent
from jongmind_ai.open_call import OpenCallAgent
from jongmind_ai.random_discard import RandomDiscardAgent
from jongmind_ai.registry import (
    ModelFactory,
    ModelSpec,
    available_models,
    create_model,
    model_names,
    register_model,
)
from jongmind_ai.rewards import (
    EffectiveDraw,
    HandMetrics,
    MatchRewardWeights,
    RewardBreakdown,
    RewardWeights,
    analyze_hand,
    match_terminal_reward,
    shape_potential_reward,
    step_shape_reward,
    terminal_reward,
)
from jongmind_ai.shanten import ShantenAgent
from jongmind_ai.tile_efficiency import (
    AkochanAgent,
    MahjongAIHeuristicAgent,
    MjaiManueAgent,
    TileEfficiencyAgent,
)


register_model("beginner", lambda seed: BeginnerAgent(), "closed tile-efficiency baseline")
register_model("neural_beginner", lambda seed: NeuralBeginnerAgent(), "small neural discard model")
register_model("tile_efficiency", lambda seed: TileEfficiencyAgent(), "shanten plus ukeire baseline")
register_model(
    "tile_efficiency_call",
    lambda seed: TileEfficiencyAgent(allow_calls=True),
    "shanten plus ukeire baseline with tile-efficiency calls",
)
register_model(
    "mjai_manue",
    lambda seed: MjaiManueAgent(),
    "mjai-manue-style expected-points teacher",
)
register_model(
    "akochan",
    lambda seed: AkochanAgent(),
    "Akochan-style placement and round-EV teacher",
)
register_model(
    "mahjong_ai",
    lambda seed: MahjongAIHeuristicAgent(),
    "MahjongAI-style heuristic call teacher",
)
register_model(
    "expected_value_call",
    lambda seed: MjaiManueAgent(),
    "alias for the mjai-manue-style expected-value call teacher",
)
register_model("shanten", lambda seed: ShantenAgent(), "shanten-only baseline")
register_model("open_call", lambda seed: OpenCallAgent(), "tile-efficiency baseline with open calls")
register_model("random", lambda seed: RandomDiscardAgent(seed=seed), "random discard baseline")

MODEL_NAMES = model_names()


__all__ = [
    "CallDecision",
    "AkochanAgent",
    "BeginnerAgent",
    "DiscardAgent",
    "DiscardDecision",
    "MODEL_NAMES",
    "ModelFactory",
    "ModelSpec",
    "NeuralBeginnerAgent",
    "OpenCallAgent",
    "ReactionAgent",
    "EffectiveDraw",
    "HandMetrics",
    "MahjongAIHeuristicAgent",
    "MatchRewardWeights",
    "RewardBreakdown",
    "RewardWeights",
    "MjaiManueAgent",
    "RandomDiscardAgent",
    "ShantenAgent",
    "TileEfficiencyAgent",
    "analyze_hand",
    "available_models",
    "create_model",
    "match_terminal_reward",
    "model_names",
    "register_model",
    "shape_potential_reward",
    "step_shape_reward",
    "terminal_reward",
]
