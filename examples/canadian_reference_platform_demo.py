"""Reference-platform demo flow for the Canadian Reference Drone Platform.

Run from the repository root:

    python3 examples/canadian_reference_platform_demo.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drone_ai import (
    AutonomyKit,
    AutonomyKitConfig,
    AutonomyMode,
    DataArtifact,
    DroneObservation,
    DryRunFlightController,
    HeuristicWorldModelPlanner,
    MissionLog,
    OperatorInput,
    SecureDataPipeline,
    default_bvlos_l1c_evidence_pack,
    default_reference_platform,
    evidence_dashboard_summary,
    reference_platform_summary,
    validate_reference_platform,
    verify_mission_log,
)


def main() -> None:
    artifact_dir = REPO_ROOT / "artifacts" / "reference_platform"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    platform = default_reference_platform()
    evidence_pack = default_bvlos_l1c_evidence_pack()
    validation_errors = validate_reference_platform(platform)

    mission_log_path = artifact_dir / "mission.jsonl"
    mission_log = MissionLog(mission_log_path)
    kit = AutonomyKit(
        HeuristicWorldModelPlanner(),
        flight_controller=DryRunFlightController("px4-vtol-dry-run"),
        mission_log=mission_log,
        config=AutonomyKitConfig(mode=AutonomyMode.DRY_RUN),
    )

    observation = DroneObservation(
        timestamp_s=0.0,
        telemetry={
            "battery_percent": 76,
            "altitude_m": 85.0,
            "comms_link": True,
            "x_m": 0.0,
            "y_m": 0.0,
            "link_margin_db": 17.5,
            "airframe": platform.airframe.identifier,
        },
        mission_state={"operator_authority": True, "mode": "dry_run"},
        environment={"scenario": "contained_bvlos_test_range", "wind_mps": 6.0},
    )
    goal = DroneObservation(
        timestamp_s=1.0,
        telemetry={"x_m": 125.0, "y_m": 40.0, "altitude_m": 85.0},
        mission_state={"operator_authority": True},
    )

    decision = kit.step(
        observation,
        goal,
        operator_input=OperatorInput("demo-operator", "approve", True),
    )

    evidence_path = artifact_dir / "evidence_summary.json"
    evidence_path.write_text(
        json.dumps(evidence_dashboard_summary(evidence_pack), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pipeline = SecureDataPipeline(platform.data_policy)
    bundle = pipeline.create_bundle(
        (
            DataArtifact(
                name="mission-log",
                path=str(mission_log_path),
                artifact_type="mission_log",
                classification_label="protected_b",
                metadata={"hash_errors": verify_mission_log(mission_log_path)},
            ),
            DataArtifact(
                name="evidence-summary",
                path=str(evidence_path),
                artifact_type="qualification_evidence",
                classification_label="unclassified",
            ),
        ),
        bundle_id="crdp-demo-bundle-001",
        output_path=artifact_dir / "bundle_manifest.json",
    )

    print(
        json.dumps(
            {
                "platform": reference_platform_summary(platform),
                "validation_errors": validation_errors,
                "autonomy_status": decision.status,
                "selected_action": decision.selected_action.as_vector()
                if decision.selected_action
                else None,
                "mission_log_errors": verify_mission_log(mission_log_path),
                "evidence_status_counts": evidence_pack.status_counts(),
                "release_blockers": evidence_pack.release_blockers(),
                "bundle": {
                    "id": bundle.bundle_id,
                    "manifest_hash": bundle.manifest_hash,
                    "artifact_count": len(bundle.artifacts),
                    "export_status": bundle.export_status,
                    "review_gates": list(bundle.review_gates),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
