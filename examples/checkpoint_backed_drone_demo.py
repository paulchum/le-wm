"""Run the drone autonomy kit with a real LeWM checkpoint.

Run from the repository root after installing the optional LeWM dependencies:

    python3 examples/checkpoint_backed_drone_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drone_ai import (  # noqa: E402
    AutonomyKit,
    AutonomyKitConfig,
    DroneActionCandidate,
    DroneObservation,
    LEWM_HF_MODEL_REPOS,
    LeWMCheckpointSpec,
    load_hf_lewm_world_model_planner,
)
from lewm_drone.interfaces import to_jsonable  # noqa: E402


def main() -> None:
    args = _parse_args()
    torch = _import_torch()
    _validate_device(torch, args.device)

    spec = LeWMCheckpointSpec(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
    )
    planner = load_hf_lewm_world_model_planner(spec, device=args.device)
    observation, goal = _demo_observations(torch, args.device)
    candidates = [
        DroneActionCandidate(0.0, 0.0, 0.0, 0.0, 1.0, source="demo_hover"),
        DroneActionCandidate(1.0, 0.0, 0.0, 0.0, 1.0, source="demo_forward"),
        DroneActionCandidate(0.0, 1.0, 0.0, 0.15, 1.0, source="demo_lateral_yaw"),
    ]

    kit = AutonomyKit(
        planner=planner,
        config=AutonomyKitConfig(max_candidates=len(candidates), log_all_candidates=True),
    )
    with torch.no_grad():
        decision = kit.step(observation, goal, candidates=candidates)

    costs = [cost for cost in decision.costs if cost is not None]
    model_metadata = getattr(planner.model, "lewm_checkpoint", {})
    print(
        json.dumps(
            {
                "status": decision.status,
                "repo_id": model_metadata.get("repo_id", args.repo_id),
                "checkpoint_dir": model_metadata.get("checkpoint_dir"),
                "model_action_dim": planner.config.model_action_dim,
                "candidate_count": decision.candidate_count,
                "accepted_candidate_count": decision.accepted_candidate_count,
                "costs": costs,
                "selected_action": to_jsonable(decision.selected_action),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=LEWM_HF_MODEL_REPOS["pusht"],
        help="HF model repo containing LeWM config.json and weights.pt.",
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache root.")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional local directory containing or receiving config.json and weights.pt.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for inference.")
    return parser.parse_args()


def _demo_observations(torch, device: str) -> tuple[DroneObservation, DroneObservation]:
    pixels = torch.zeros(1, 1, 3, 3, 224, 224, device=device)
    x = torch.linspace(0.0, 1.0, 224, device=device).view(1, 1, 1, 1, 1, 224)
    y = torch.linspace(0.0, 1.0, 224, device=device).view(1, 1, 1, 1, 224, 1)
    for step in range(3):
        pixels[:, :, step, 0] = (x + 0.04 * step).clamp(0.0, 1.0)
        pixels[:, :, step, 1] = y
        pixels[:, :, step, 2] = 0.2 + 0.1 * step

    goal_pixels = pixels.clone()
    goal_pixels[:, :, :, 0] = torch.roll(goal_pixels[:, :, :, 0], shifts=12, dims=-1)
    goal_pixels[:, :, :, 2] = 0.6

    observation = DroneObservation(
        timestamp_s=0.0,
        telemetry={
            "battery_percent": 90,
            "altitude_m": 20.0,
            "comms_link": True,
            "x_m": 0.0,
            "y_m": 0.0,
        },
        mission_state={"operator_authority": True},
        model_inputs={"pixels": pixels},
    )
    goal = DroneObservation(
        timestamp_s=1.0,
        telemetry={"x_m": 4.0, "y_m": 1.0, "altitude_m": 20.0},
        mission_state={"operator_authority": True},
        model_inputs={"pixels": goal_pixels},
    )
    return observation, goal


def _validate_device(torch, device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device!r}, but CUDA is not available.")


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "This demo requires PyTorch plus the LeWM optional dependencies. "
            "See the README checkpoint-backed demo section."
        ) from exc
    return torch


if __name__ == "__main__":
    main()
