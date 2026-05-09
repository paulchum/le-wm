"""Pure conversion helpers for the AAS PX4 VTOL ROS bridge."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Any

from drone_ai import DroneActionCandidate, DroneObservation


@dataclass(frozen=True)
class LocalNedVelocity:
    """PX4 local NED velocity setpoint."""

    north_mps: float
    east_mps: float
    down_mps: float
    yawspeed_rps: float


def body_velocity_to_local_ned(
    vx_body_mps: float,
    vy_body_mps: float,
    vz_up_mps: float,
    heading_rad: float,
) -> tuple[float, float, float]:
    """Rotate body-frame forward/right/up velocity into PX4 local NED."""

    north_mps = vx_body_mps * cos(heading_rad) - vy_body_mps * sin(heading_rad)
    east_mps = vx_body_mps * sin(heading_rad) + vy_body_mps * cos(heading_rad)
    down_mps = -vz_up_mps
    return north_mps, east_mps, down_mps


def candidate_to_local_ned(
    candidate: DroneActionCandidate,
    heading_rad: float,
) -> LocalNedVelocity:
    north_mps, east_mps, down_mps = body_velocity_to_local_ned(
        candidate.vx_mps,
        candidate.vy_mps,
        candidate.vz_mps,
        heading_rad,
    )
    return LocalNedVelocity(
        north_mps=north_mps,
        east_mps=east_mps,
        down_mps=down_mps,
        yawspeed_rps=candidate.yaw_rate_rps,
    )


def is_fresh(now_s: float, updated_at_s: float | None, max_age_s: float) -> bool:
    if updated_at_s is None:
        return False
    return updated_at_s <= now_s + 1e-6 and (now_s - updated_at_s) <= max_age_s


def approval_is_active(
    *,
    approved: bool,
    now_s: float,
    expires_at_s: float = 0.0,
) -> bool:
    if not approved:
        return False
    if expires_at_s <= 0.0:
        return True
    return now_s <= expires_at_s


def should_publish_executable_setpoint(
    *,
    decision_accepted: bool,
    selected_action: Any | None,
    goal_fresh: bool,
    local_position_fresh: bool,
    approval_active: bool,
) -> bool:
    return (
        decision_accepted
        and selected_action is not None
        and goal_fresh
        and local_position_fresh
        and approval_active
    )


def build_px4_observation(
    *,
    timestamp_s: float,
    north_m: float,
    east_m: float,
    down_m: float,
    heading_rad: float,
    velocity_north_mps: float | None = None,
    velocity_east_mps: float | None = None,
    velocity_down_mps: float | None = None,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
    altitude_m: float | None = None,
    airspeed_mps: float | None = None,
    vehicle_status: dict[str, Any] | None = None,
    operator_authority: bool = True,
) -> DroneObservation:
    """Build the public drone_ai observation boundary from PX4 state."""

    resolved_altitude_m = altitude_m if altitude_m is not None else -down_m
    telemetry: dict[str, Any] = {
        "north_m": float(north_m),
        "east_m": float(east_m),
        "down_m": float(down_m),
        "altitude_m": float(resolved_altitude_m),
        "heading_rad": float(heading_rad),
        "autopilot": "px4",
        "airframe": "vtol",
    }
    optional_values = {
        "velocity_north_mps": velocity_north_mps,
        "velocity_east_mps": velocity_east_mps,
        "velocity_down_mps": velocity_down_mps,
        "lat": latitude_deg,
        "lon": longitude_deg,
        "airspeed_mps": airspeed_mps,
    }
    telemetry.update(
        {key: float(value) for key, value in optional_values.items() if _finite_or_none(value)}
    )

    mission_state = {
        "operator_authority": operator_authority,
        "autopilot": "px4",
        "airframe": "vtol",
    }
    if vehicle_status:
        mission_state["vehicle_status"] = vehicle_status

    return DroneObservation(
        timestamp_s=float(timestamp_s),
        telemetry=telemetry,
        mission_state=mission_state,
    )


def build_goal_observation(
    *,
    timestamp_s: float,
    north_m: float,
    east_m: float,
    down_m: float,
    goal_id: str = "",
    yaw_rad: float | None = None,
) -> DroneObservation:
    telemetry = {
        "north_m": float(north_m),
        "east_m": float(east_m),
        "down_m": float(down_m),
        "altitude_m": float(-down_m),
    }
    mission_state: dict[str, Any] = {"goal_id": goal_id}
    if yaw_rad is not None and isfinite(float(yaw_rad)):
        mission_state["yaw_rad"] = float(yaw_rad)
    return DroneObservation(
        timestamp_s=float(timestamp_s),
        telemetry=telemetry,
        mission_state=mission_state,
    )


def _finite_or_none(value: Any | None) -> bool:
    if value is None:
        return False
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False
