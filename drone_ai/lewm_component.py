"""Metadata describing how LeWM is used inside the Drone AI stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeWMComponentSpec:
    """Repository-local description of the LeWM component boundary."""

    name: str
    role: str
    adapter: str
    model_files: tuple[str, ...]
    training_entrypoint: str
    evaluation_entrypoint: str
    planner_methods: tuple[str, ...]
    blocked_responsibilities: tuple[str, ...]


def default_lewm_component_spec() -> LeWMComponentSpec:
    """Return the default LeWM component map for this repository."""

    return LeWMComponentSpec(
        name="LeWorldModel",
        role=(
            "learned pixel-to-latent world model for short-horizon candidate "
            "scoring, latent rollout, and surprise/anomaly estimation"
        ),
        adapter="drone_ai.LeWMWorldModelPlanner",
        model_files=("jepa.py", "module.py", "train.py", "eval.py"),
        training_entrypoint="train.py",
        evaluation_entrypoint="eval.py",
        planner_methods=("score_action_candidates", "rollout", "surprise"),
        blocked_responsibilities=(
            "flight stabilization",
            "direct motor control",
            "safety interlocks",
            "operator authorization",
            "payload release",
            "weapon release",
            "target engagement",
        ),
    )
