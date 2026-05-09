"""Safety supervisor for learned drone action proposals."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any, Iterable, Mapping

from .interfaces import DroneActionCandidate, DroneObservation, SafetyDecision


@dataclass(frozen=True)
class GeoFence:
    """Simple latitude/longitude bounding box for early validation."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, telemetry: Mapping[str, Any]) -> bool:
        lat = telemetry.get("lat")
        lon = telemetry.get("lon")
        if lat is None or lon is None:
            return True
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            return False
        return self.min_lat <= lat_f <= self.max_lat and self.min_lon <= lon_f <= self.max_lon


@dataclass(frozen=True)
class SafetyEnvelope:
    """Hard limits for non-kinetic Canadian defence R&D flight trials."""

    min_altitude_m: float = 1.0
    max_altitude_m: float = 120.0
    max_horizontal_speed_mps: float = 8.0
    max_vertical_speed_mps: float = 3.0
    max_yaw_rate_rps: float = 1.5
    max_duration_s: float = 2.0
    min_battery_percent: float = 25.0
    require_comms_link: bool = True
    require_operator_authority: bool = True
    require_non_kinetic: bool = True
    allow_kinetic_context: bool = False
    allow_clipping: bool = True
    geofence: GeoFence | None = None
    constraints_version: str = "canadian-defence-rd-v1"


class SafetySupervisor:
    """Reject or constrain learned action candidates before flight control."""

    def __init__(self, envelope: SafetyEnvelope | None = None):
        self.envelope = envelope or SafetyEnvelope()

    def evaluate(
        self,
        observation: DroneObservation,
        candidate: DroneActionCandidate,
    ) -> SafetyDecision:
        reasons: list[str] = []
        warnings: list[str] = []
        action = candidate

        if not self._finite_action(candidate):
            return self._reject("non_finite_action", candidate)

        if candidate.duration_s <= 0:
            reasons.append("duration_not_positive")
        elif candidate.duration_s > self.envelope.max_duration_s:
            if self.envelope.allow_clipping:
                action = action.with_updates(duration_s=self.envelope.max_duration_s)
                warnings.append("duration_clipped")
            else:
                reasons.append("duration_exceeds_limit")

        action, speed_warning, speed_reason = self._clip_horizontal_speed(action)
        if speed_warning:
            warnings.append(speed_warning)
        if speed_reason:
            reasons.append(speed_reason)

        if abs(action.vz_mps) > self.envelope.max_vertical_speed_mps:
            if self.envelope.allow_clipping:
                clipped_vz = self._clip(action.vz_mps, self.envelope.max_vertical_speed_mps)
                action = action.with_updates(vz_mps=clipped_vz)
                warnings.append("vertical_speed_clipped")
            else:
                reasons.append("vertical_speed_exceeds_limit")

        if abs(action.yaw_rate_rps) > self.envelope.max_yaw_rate_rps:
            if self.envelope.allow_clipping:
                clipped_yaw = self._clip(action.yaw_rate_rps, self.envelope.max_yaw_rate_rps)
                action = action.with_updates(yaw_rate_rps=clipped_yaw)
                warnings.append("yaw_rate_clipped")
            else:
                reasons.append("yaw_rate_exceeds_limit")

        self._check_state_constraints(observation, action, reasons)
        self._check_effect_constraints(observation, candidate, reasons)

        accepted = not reasons
        return SafetyDecision(
            accepted=accepted,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            action=action if accepted else None,
            constraints_version=self.envelope.constraints_version,
        )

    def filter_candidates(
        self,
        observation: DroneObservation,
        candidates: Iterable[DroneActionCandidate],
    ) -> list[SafetyDecision]:
        return [self.evaluate(observation, candidate) for candidate in candidates]

    def _check_state_constraints(
        self,
        observation: DroneObservation,
        action: DroneActionCandidate,
        reasons: list[str],
    ) -> None:
        telemetry = observation.telemetry
        battery = telemetry.get("battery_percent")
        if battery is not None:
            try:
                if float(battery) < self.envelope.min_battery_percent:
                    reasons.append("battery_below_minimum")
            except (TypeError, ValueError):
                reasons.append("battery_invalid")

        if self.envelope.require_comms_link and telemetry.get("comms_link", True) is False:
            reasons.append("comms_link_required")

        if self.envelope.require_operator_authority:
            authority = observation.mission_state.get("operator_authority", True)
            if authority is not True:
                reasons.append("operator_authority_required")

        if self.envelope.geofence and not self.envelope.geofence.contains(telemetry):
            reasons.append("outside_geofence")

        current_altitude = telemetry.get("altitude_m")
        target_altitude = action.target_altitude_m
        if target_altitude is None and current_altitude is not None:
            try:
                target_altitude = float(current_altitude) + action.vz_mps * action.duration_s
            except (TypeError, ValueError):
                reasons.append("altitude_invalid")

        if target_altitude is not None:
            try:
                target_altitude_f = float(target_altitude)
            except (TypeError, ValueError):
                reasons.append("altitude_invalid")
                return
            if not isfinite(target_altitude_f):
                reasons.append("altitude_invalid")
            elif target_altitude_f < self.envelope.min_altitude_m:
                reasons.append("altitude_below_floor")
            elif target_altitude_f > self.envelope.max_altitude_m:
                reasons.append("altitude_above_ceiling")

    def _check_effect_constraints(
        self,
        observation: DroneObservation,
        candidate: DroneActionCandidate,
        reasons: list[str],
    ) -> None:
        metadata = candidate.metadata
        mission_state = observation.mission_state

        effect_command_flags = (
            "effect_command",
            "kinetic_effect",
            "payload_release",
            "weapon_release",
            "target_engagement",
        )
        if any(metadata.get(flag) is True for flag in effect_command_flags):
            reasons.append("effect_command_blocked")

        kinetic_context = (
            metadata.get("kinetic_context") is True
            or mission_state.get("kinetic_context") is True
        )
        if kinetic_context and self.envelope.require_non_kinetic:
            reasons.append("non_kinetic_constraint")
        elif kinetic_context and not self.envelope.allow_kinetic_context:
            reasons.append("kinetic_context_not_allowed")

    def _clip_horizontal_speed(
        self,
        candidate: DroneActionCandidate,
    ) -> tuple[DroneActionCandidate, str | None, str | None]:
        speed = hypot(candidate.vx_mps, candidate.vy_mps)
        limit = self.envelope.max_horizontal_speed_mps
        if speed <= limit:
            return candidate, None, None
        if not self.envelope.allow_clipping:
            return candidate, None, "horizontal_speed_exceeds_limit"
        if speed == 0:
            return candidate, None, None
        scale = limit / speed
        return (
            candidate.with_updates(vx_mps=candidate.vx_mps * scale, vy_mps=candidate.vy_mps * scale),
            "horizontal_speed_clipped",
            None,
        )

    def _reject(self, reason: str, candidate: DroneActionCandidate) -> SafetyDecision:
        return SafetyDecision(
            accepted=False,
            reasons=(reason,),
            action=None,
            constraints_version=self.envelope.constraints_version,
        )

    @staticmethod
    def _clip(value: float, absolute_limit: float) -> float:
        return max(-absolute_limit, min(absolute_limit, value))

    @staticmethod
    def _finite_action(candidate: DroneActionCandidate) -> bool:
        values = (
            candidate.vx_mps,
            candidate.vy_mps,
            candidate.vz_mps,
            candidate.yaw_rate_rps,
            candidate.duration_s,
        )
        if candidate.target_altitude_m is not None:
            values = (*values, candidate.target_altitude_m)
        try:
            return all(isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            return False
