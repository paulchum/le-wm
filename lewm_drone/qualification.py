"""Qualification evidence matrix for the reference drone platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .interfaces import JsonValue, to_jsonable


@dataclass(frozen=True)
class EvidenceItem:
    """One qualification claim and its supporting artifacts."""

    identifier: str
    standard_ref: str
    title: str
    status: str
    artifacts: tuple[str, ...] = ()
    owner: str = "program_engineering"
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualificationEvidencePack:
    """Readiness matrix for BVLOS/L1C reference-platform evidence."""

    identifier: str
    target: str
    items: tuple[EvidenceItem, ...]
    approval_claim: str = "none"

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def open_items(self) -> list[EvidenceItem]:
        return [item for item in self.items if item.status not in {"complete", "not_applicable"}]

    def release_blockers(self) -> list[str]:
        blockers = []
        for item in self.open_items():
            if item.status in {"missing", "blocked"}:
                blockers.append(item.identifier)
        return blockers


def default_bvlos_l1c_evidence_pack() -> QualificationEvidencePack:
    """Return the selected public-evidence scaffold for BVLOS/L1C readiness."""

    items = (
        EvidenceItem(
            identifier="922-08-containment",
            standard_ref="Transport Canada Standard 922.08",
            title="Containment and operational volume evidence",
            status="draft",
            artifacts=("geofence_config", "lost_link_procedure", "flight_test_report"),
            notes="Document operational volume, containment assumptions, and emergency boundaries.",
        ),
        EvidenceItem(
            identifier="922-09-c2-link",
            standard_ref="Transport Canada Standard 922.09",
            title="Command-and-control link and lost-link behavior",
            status="draft",
            artifacts=("c2_link_budget", "lost_link_test", "operator_override_log"),
            notes="Show primary/backup C2 behavior and deterministic lost-link procedures.",
        ),
        EvidenceItem(
            identifier="922-10-daa",
            standard_ref="Transport Canada Standard 922.10",
            title="Detect, alert, and avoid assumptions",
            status="scaffold",
            artifacts=("payload_interface_matrix", "daa_assumption_register"),
            notes="Depth payload supports assumptions, but v1 does not claim certified DAA.",
        ),
        EvidenceItem(
            identifier="922-11-control-station",
            standard_ref="Transport Canada Standard 922.11",
            title="Control station design and operator awareness",
            status="draft",
            artifacts=("ground_station_demo", "operator_workflow_review", "human_factors_notes"),
            notes="React/Vite demo covers the unclassified operator-interface scaffold.",
        ),
        EvidenceItem(
            identifier="922-12-environmental-envelope",
            standard_ref="Transport Canada Standard 922.12",
            title="Environmental envelope and reliability limits",
            status="missing",
            artifacts=("environmental_test_plan", "reliability_summary"),
            notes="Requires physical test data from the selected airframe and payload suite.",
        ),
        EvidenceItem(
            identifier="classified-path-gates",
            standard_ref="Program security posture",
            title="Controlled Goods, export-control, and classified handling gates",
            status="draft",
            artifacts=("security_gate_policy", "export_review_record", "data_residency_attestation"),
            notes="Repo must not hold classified or controlled technical data.",
        ),
        EvidenceItem(
            identifier="non-kinetic-assurance",
            standard_ref="Program safety posture",
            title="Non-kinetic autonomy assurance",
            status="complete",
            artifacts=("safety_supervisor_tests", "mission_log_hash_chain"),
            notes="Safety supervisor blocks payload release and target-engagement metadata.",
        ),
    )
    return QualificationEvidencePack(
        identifier="crdp-bvlos-l1c-evidence-v1",
        target="small_vtol_under_25kg_bvlos_l1c_readiness",
        items=items,
    )


def evidence_dashboard_summary(pack: QualificationEvidencePack) -> dict[str, JsonValue]:
    """Return compact readiness data for the ground station."""

    return {
        "identifier": pack.identifier,
        "target": pack.target,
        "approval_claim": pack.approval_claim,
        "status_counts": pack.status_counts(),
        "release_blockers": pack.release_blockers(),
        "items": [
            {
                "identifier": item.identifier,
                "standard_ref": item.standard_ref,
                "title": item.title,
                "status": item.status,
                "artifact_count": len(item.artifacts),
            }
            for item in pack.items
        ],
    }
