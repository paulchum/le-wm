"""Typed interfaces for guarded drone autonomy.

The classes in this module are deliberately small and serializable. They are
the boundary between the learned LeWM component, the safety supervisor, flight
control, operator tooling, and the flight-log pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except TypeError:
        return None


def _model_input_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return sorted(str(k) for k in value.keys())
    return [type(value).__name__]


def to_jsonable(value: Any) -> JsonValue:
    """Convert dataclasses and common scalar containers into JSON-safe values."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    shape = _shape_of(value)
    if shape is not None:
        return {"shape": shape, "type": type(value).__name__}
    return str(value)


@dataclass(frozen=True)
class DroneObservation:
    """Timestamped state snapshot for planning, safety, and logging.

    `model_inputs` can carry framework-specific tensors for LeWM without making
    the public interface depend on PyTorch or NumPy.
    """

    timestamp_s: float
    pixels: Any | None = None
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    proprioception: Sequence[float] = field(default_factory=tuple)
    mission_state: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    model_inputs: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def summary(self) -> dict[str, JsonValue]:
        """Return a JSON-safe summary that excludes raw image/model tensors."""

        return {
            "timestamp_s": self.timestamp_s,
            "pixel_shape": _shape_of(self.pixels),
            "telemetry": to_jsonable(self.telemetry),
            "proprioception": to_jsonable(list(self.proprioception)),
            "mission_state": to_jsonable(self.mission_state),
            "environment": to_jsonable(self.environment),
            "model_input_keys": _model_input_keys(self.model_inputs),
        }


@dataclass(frozen=True)
class DroneActionCandidate:
    """Short-horizon command candidate proposed before safety filtering."""

    vx_mps: float
    vy_mps: float
    vz_mps: float
    yaw_rate_rps: float
    duration_s: float
    target_altitude_m: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "world_model"

    def as_vector(self, order: Sequence[str] | None = None) -> list[float]:
        order = order or ("vx_mps", "vy_mps", "vz_mps", "yaw_rate_rps")
        return [float(getattr(self, field_name)) for field_name in order]

    def with_updates(self, **updates: Any) -> "DroneActionCandidate":
        values = {
            "vx_mps": self.vx_mps,
            "vy_mps": self.vy_mps,
            "vz_mps": self.vz_mps,
            "yaw_rate_rps": self.yaw_rate_rps,
            "duration_s": self.duration_s,
            "target_altitude_m": self.target_altitude_m,
            "metadata": self.metadata,
            "source": self.source,
        }
        values.update(updates)
        return DroneActionCandidate(**values)


@dataclass(frozen=True)
class SafetyDecision:
    """Safety-supervisor decision for a proposed action."""

    accepted: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    action: DroneActionCandidate | None = None
    constraints_version: str = "canadian-defence-rd-v1"

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)


@dataclass(frozen=True)
class OperatorInput:
    """Operator authority and override metadata captured with each decision."""

    operator_id: str
    command: str
    authorized: bool
    notes: str = ""


@dataclass(frozen=True)
class MissionLogEntry:
    """Immutable append-only audit record."""

    sequence_id: int
    created_at_s: float
    observation: Mapping[str, JsonValue]
    proposed_action: Mapping[str, JsonValue]
    safety_decision: Mapping[str, JsonValue]
    operator_input: Mapping[str, JsonValue] | None
    executed_action: Mapping[str, JsonValue] | None
    outcome: Mapping[str, JsonValue]
    previous_hash: str
    entry_hash: str
