"""Reference UAS platform contracts for the Canadian drone package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .interfaces import JsonValue, to_jsonable


@dataclass(frozen=True)
class AirframeProfile:
    """Vendor-neutral airframe baseline for qualification planning."""

    identifier: str
    name: str
    category: str
    max_takeoff_weight_kg: float
    propulsion: str
    endurance_min: float
    cruise_speed_mps: float
    flight_controller: str
    operating_modes: tuple[str, ...]
    payload_capacity_kg: float
    bvlos_evidence_target: str
    notes: str = ""


@dataclass(frozen=True)
class PayloadBay:
    """Mechanical/electrical/data constraints for one payload slot."""

    identifier: str
    max_mass_kg: float
    max_power_w: float
    interfaces: tuple[str, ...]
    isolated_power: bool = True


@dataclass(frozen=True)
class PayloadProfile:
    """Payload module carried by the reference platform."""

    identifier: str
    name: str
    payload_type: str
    mass_kg: float
    power_w: float
    interfaces: tuple[str, ...]
    data_products: tuple[str, ...]
    autonomy_roles: tuple[str, ...]
    classification_ceiling: str = "protected_b"
    non_kinetic_only: bool = True


@dataclass(frozen=True)
class GroundStationProfile:
    """Operator ground-station baseline."""

    identifier: str
    operator_roles: tuple[str, ...]
    command_links: tuple[str, ...]
    displays: tuple[str, ...]
    required_features: tuple[str, ...]
    audit_features: tuple[str, ...]


@dataclass(frozen=True)
class DataResidencyPolicy:
    """Data handling boundary for unclassified repo workflows."""

    residency: str = "canada_by_default"
    classification_path: str = "unclassified_scaffold_classified_external"
    allowed_labels: tuple[str, ...] = ("unclassified", "protected_a", "protected_b")
    blocked_labels: tuple[str, ...] = (
        "classified",
        "controlled_goods",
        "secret",
        "top_secret",
    )
    encryption_required: bool = True
    audit_required: bool = True
    export_review_required: bool = True


@dataclass(frozen=True)
class SupportModel:
    """Support commitments needed for a reference prototype trial."""

    tiers: tuple[str, ...]
    training: tuple[str, ...]
    maintenance_intervals: tuple[str, ...]
    spares: tuple[str, ...]
    incident_response: tuple[str, ...]


@dataclass(frozen=True)
class ReferencePlatformManifest:
    """Complete Canadian Reference Drone Platform manifest."""

    identifier: str
    name: str
    version: str
    posture: str
    airframe: AirframeProfile
    payload_bays: tuple[PayloadBay, ...]
    payloads: tuple[PayloadProfile, ...]
    ground_station: GroundStationProfile
    data_policy: DataResidencyPolicy
    support_model: SupportModel
    qualification_targets: tuple[str, ...]
    safety_envelope_ref: str = "canadian-defence-rd-v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]

    def payload_mass_kg(self) -> float:
        return sum(payload.mass_kg for payload in self.payloads)

    def payload_power_w(self) -> float:
        return sum(payload.power_w for payload in self.payloads)


def default_reference_platform() -> ReferencePlatformManifest:
    """Return the selected v1 small-VTOL BVLOS/L1C reference baseline."""

    airframe = AirframeProfile(
        identifier="crdp-vtol-small-v1",
        name="Canadian Reference VTOL UAS",
        category="small_rpas_under_25kg",
        max_takeoff_weight_kg=24.9,
        propulsion="electric_vtol_fixed_wing_transition",
        endurance_min=90.0,
        cruise_speed_mps=22.0,
        flight_controller="px4_or_ardupilot",
        operating_modes=("manual", "assisted", "advisory", "safety_filtered_autonomy"),
        payload_capacity_kg=2.2,
        bvlos_evidence_target="standard_922_l1_complex_readiness",
        notes="Reference baseline for contained BVLOS/L1C evidence planning, not approval.",
    )
    payload_bays = (
        PayloadBay(
            identifier="nose-gimbal",
            max_mass_kg=1.2,
            max_power_w=35.0,
            interfaces=("ethernet", "serial", "isolated_12v"),
        ),
        PayloadBay(
            identifier="belly-depth",
            max_mass_kg=0.8,
            max_power_w=25.0,
            interfaces=("usb3", "ethernet", "isolated_12v"),
        ),
    )
    payloads = (
        PayloadProfile(
            identifier="eo-ir-gimbal",
            name="EO/IR stabilized payload",
            payload_type="configurable_sensor",
            mass_kg=0.95,
            power_w=24.0,
            interfaces=("ethernet", "serial", "isolated_12v"),
            data_products=("rgb_video", "thermal_video", "payload_health"),
            autonomy_roles=("operator_awareness", "anomaly_review", "non_kinetic_isr"),
        ),
        PayloadProfile(
            identifier="depth-range-module",
            name="Depth/range perception module",
            payload_type="configurable_sensor",
            mass_kg=0.45,
            power_w=14.0,
            interfaces=("usb3", "ethernet", "isolated_12v"),
            data_products=("range_points", "obstacle_metadata", "sensor_health"),
            autonomy_roles=("containment_monitoring", "daa_assumption_support"),
        ),
    )
    ground_station = GroundStationProfile(
        identifier="crdp-ground-station-v1",
        operator_roles=("pilot_in_command", "payload_operator", "safety_observer"),
        command_links=("primary_c2", "backup_c2", "lte_or_satcom_optional"),
        displays=("mission_map", "aircraft_health", "payload_health", "evidence_readiness"),
        required_features=(
            "mission_planning",
            "live_status",
            "recommended_action",
            "operator_override",
            "lost_link_status",
            "audit_log_review",
        ),
        audit_features=("hash_chain_verification", "model_version_trace", "export_gate_status"),
    )
    support_model = SupportModel(
        tiers=("field_operator_support", "autonomy_engineering_support", "secure_program_support"),
        training=("operator_ground_school", "maintenance_orientation", "data_handling_briefing"),
        maintenance_intervals=("preflight", "postflight", "25_flight_hours", "payload_change"),
        spares=("propulsion_set", "payload_mount", "c2_radio", "edge_compute_module"),
        incident_response=("flight_anomaly_triage", "data_spill_review", "model_regression_review"),
    )
    return ReferencePlatformManifest(
        identifier="canadian-reference-drone-platform-v1",
        name="Canadian Reference Drone Platform",
        version="0.1.0",
        posture="non_kinetic_classified_path_scaffold",
        airframe=airframe,
        payload_bays=payload_bays,
        payloads=payloads,
        ground_station=ground_station,
        data_policy=DataResidencyPolicy(),
        support_model=support_model,
        qualification_targets=(
            "standard_922_08_containment",
            "standard_922_09_c2_link_lost_link",
            "standard_922_10_detect_alert_avoid_assumptions",
            "standard_922_11_control_station",
            "standard_922_12_environmental_envelope",
        ),
        metadata={
            "market": "canadian_government_defence",
            "reference_platform_role": "autonomy_kit_integration_target",
            "approval_claim": "none",
        },
    )


def validate_reference_platform(manifest: ReferencePlatformManifest) -> list[str]:
    """Return manifest validation errors."""

    errors: list[str] = []
    if manifest.airframe.max_takeoff_weight_kg >= 25.0:
        errors.append("airframe_not_small_rpas_under_25kg")
    if manifest.airframe.payload_capacity_kg < manifest.payload_mass_kg():
        errors.append("payload_mass_exceeds_airframe_capacity")
    if not manifest.payload_bays:
        errors.append("payload_bay_required")
    for payload in manifest.payloads:
        if not payload.non_kinetic_only:
            errors.append(f"{payload.identifier}:non_kinetic_required")
        if payload.classification_ceiling in manifest.data_policy.blocked_labels:
            errors.append(f"{payload.identifier}:blocked_classification_ceiling")
        if not _fits_any_payload_bay(payload, manifest.payload_bays):
            errors.append(f"{payload.identifier}:no_compatible_payload_bay")
    required_ground_station = {
        "mission_planning",
        "live_status",
        "operator_override",
        "audit_log_review",
    }
    missing = required_ground_station.difference(manifest.ground_station.required_features)
    for feature in sorted(missing):
        errors.append(f"ground_station_missing:{feature}")
    return errors


def payload_compatibility(
    payload: PayloadProfile,
    bays: Sequence[PayloadBay],
) -> dict[str, JsonValue]:
    """Summarize whether a payload can fit at least one available bay."""

    compatible_bays = [
        bay.identifier for bay in bays if _payload_fits_bay(payload, bay)
    ]
    return {
        "payload": payload.identifier,
        "compatible": bool(compatible_bays),
        "compatible_bays": compatible_bays,
        "mass_kg": payload.mass_kg,
        "power_w": payload.power_w,
    }


def reference_platform_summary(manifest: ReferencePlatformManifest) -> dict[str, JsonValue]:
    """Return the dashboard-friendly platform summary."""

    return {
        "identifier": manifest.identifier,
        "name": manifest.name,
        "version": manifest.version,
        "posture": manifest.posture,
        "airframe": manifest.airframe.name,
        "airframe_category": manifest.airframe.category,
        "payload_mass_kg": round(manifest.payload_mass_kg(), 3),
        "payload_power_w": round(manifest.payload_power_w(), 3),
        "payloads": [payload.identifier for payload in manifest.payloads],
        "qualification_targets": list(manifest.qualification_targets),
        "validation_errors": validate_reference_platform(manifest),
    }


def _fits_any_payload_bay(payload: PayloadProfile, bays: Sequence[PayloadBay]) -> bool:
    return any(_payload_fits_bay(payload, bay) for bay in bays)


def _payload_fits_bay(payload: PayloadProfile, bay: PayloadBay) -> bool:
    if payload.mass_kg > bay.max_mass_kg or payload.power_w > bay.max_power_w:
        return False
    return set(payload.interfaces).issubset(set(bay.interfaces))
