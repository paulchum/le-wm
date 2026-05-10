"""Minimal autonomy-kit flow without loading a LeWM checkpoint.

Run from the repository root:

    python3 examples/canadian_defence_autonomy_kit.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drone_ai import (
    AutonomyKit,
    AutonomyKitConfig,
    AutonomyMode,
    DroneObservation,
    DryRunFlightController,
    HeuristicWorldModelPlanner,
    MissionLog,
    MissionEffectContext,
    OperatorInput,
    SafetyEnvelope,
    SafetySupervisor,
)


def main() -> None:
    observation = DroneObservation(
        timestamp_s=0.0,
        telemetry={
            "battery_percent": 82,
            "altitude_m": 20.0,
            "comms_link": True,
            "lat": 45.4215,
            "lon": -75.6972,
        },
        mission_state={"operator_authority": True, "mode": "advisory"},
        environment={"scenario": "simulation", "lighting": "day"},
    )

    goal = DroneObservation(
        timestamp_s=1.0,
        telemetry={"x_m": 8.0, "y_m": 1.0, "altitude_m": 20.0},
        mission_state={"operator_authority": True},
    )

    log = MissionLog(Path("artifacts") / "sample_mission.jsonl")
    kit = AutonomyKit(
        planner=HeuristicWorldModelPlanner(),
        safety_supervisor=SafetySupervisor(
            SafetyEnvelope(require_non_kinetic=False, allow_kinetic_context=True)
        ),
        flight_controller=DryRunFlightController(),
        mission_log=log,
        config=AutonomyKitConfig(
            mode=AutonomyMode.DRY_RUN,
            effect_context=MissionEffectContext.KINETIC_SUPPORT,
            log_all_candidates=True,
        ),
    )
    decision = kit.step(
        observation,
        goal,
        operator_input=OperatorInput("simulation-operator", "approve", True),
    )

    print(
        {
            "status": decision.status,
            "mode": decision.mode.value,
            "effect_context": decision.effect_context.value,
            "accepted": decision.accepted,
            "selected_action": decision.selected_action,
            "warnings": decision.safety_decision.warnings if decision.safety_decision else (),
            "reasons": decision.safety_decision.reasons if decision.safety_decision else (),
            "log_errors": log.verify(),
        }
    )


if __name__ == "__main__":
    main()
