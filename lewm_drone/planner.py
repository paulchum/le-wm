"""World-model planner adapters for drone autonomy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import atan2, cos, hypot, sin
from typing import Any, Protocol, Sequence

from .interfaces import (
    DroneActionCandidate,
    DroneObservation,
    OperatorInput,
    SafetyDecision,
    to_jsonable,
)
from .geo_world_model import coerce_model_inputs
from .mission_log import MissionLog
from .safety import SafetySupervisor


class WorldModelPlanner(Protocol):
    """Planner boundary consumed by safety and operator layers."""

    def score_action_candidates(
        self,
        observation: DroneObservation,
        goal: DroneObservation,
        candidates: Sequence[DroneActionCandidate],
    ) -> list[float]:
        """Return lower-is-better costs for each candidate."""

    def rollout(
        self,
        observation: DroneObservation,
        candidates: Sequence[DroneActionCandidate],
    ) -> Any:
        """Return predicted latent rollouts for each candidate."""

    def surprise(
        self,
        observation: DroneObservation,
        action: DroneActionCandidate,
        next_observation: DroneObservation,
    ) -> float:
        """Return a scalar prediction-error signal for anomaly detection."""


@dataclass(frozen=True)
class LeWMPlannerConfig:
    """Configuration for adapting LeWM to drone action candidates."""

    history_size: int = 3
    action_steps: int = 5
    action_order: tuple[str, ...] = ("vx_mps", "vy_mps", "vz_mps", "yaw_rate_rps")
    model_action_dim: int | None = None
    device: str | None = None


class LeWMWorldModelPlanner:
    """Adapter from the LeWM cost-model API to the drone planner interface.

    The adapter expects `DroneObservation.model_inputs` to contain the tensors
    required by the trained LeWM checkpoint. At minimum this typically includes
    `pixels`; scoring also requires a goal observation with `pixels` or `goal`.
    """

    def __init__(self, model: Any, config: LeWMPlannerConfig | None = None):
        self.model = model
        self.config = config or LeWMPlannerConfig()
        self._model_action_dim = (
            self.config.model_action_dim or self._infer_model_action_dim(model)
        )
        if hasattr(self.model, "eval"):
            self.model.eval()
        if self.config.device and hasattr(self.model, "to"):
            self.model.to(self.config.device)

    def score_action_candidates(
        self,
        observation: DroneObservation,
        goal: DroneObservation,
        candidates: Sequence[DroneActionCandidate],
    ) -> list[float]:
        if not candidates:
            return []
        self._require_torch()
        info = self._info_dict(observation)
        goal_info = self._info_dict(goal)
        if "goal" not in info:
            info["goal"] = goal_info.get("goal", goal_info.get("pixels", goal.pixels))
        if info["goal"] is None:
            raise ValueError("goal observation must provide pixels, goal, or model_inputs['goal']")

        action_tensor = self._candidate_tensor(candidates)
        if "action" not in info:
            info["action"] = self._zero_history_action(action_tensor)
        costs = self.model.get_cost(info, action_tensor)
        return self._tensor_to_float_list(costs)

    def rollout(
        self,
        observation: DroneObservation,
        candidates: Sequence[DroneActionCandidate],
    ) -> Any:
        if not candidates:
            return []
        self._require_torch()
        info = self._info_dict(observation)
        action_tensor = self._candidate_tensor(candidates)
        if "action" not in info:
            info["action"] = self._zero_history_action(action_tensor)
        output = self.model.rollout(info, action_tensor, history_size=self.config.history_size)
        predicted = output.get("predicted_emb", output)
        if hasattr(predicted, "detach"):
            return predicted.detach().cpu()
        return predicted

    def surprise(
        self,
        observation: DroneObservation,
        action: DroneActionCandidate,
        next_observation: DroneObservation,
    ) -> float:
        self._require_torch()
        torch = self._torch()
        info = self._info_dict(observation)
        action_tensor = self._candidate_tensor([action])
        if "action" not in info:
            info["action"] = self._zero_history_action(action_tensor)
        rollout_output = self.model.rollout(info, action_tensor, history_size=self.config.history_size)
        predicted = rollout_output["predicted_emb"][0, 0, -1]

        next_info = self._info_dict(next_observation)
        encoded = self.model.encode(next_info)
        actual = encoded["emb"][0, -1]
        return float(torch.mean((predicted - actual) ** 2).detach().cpu().item())

    def _info_dict(self, observation: DroneObservation) -> dict[str, Any]:
        info = dict(coerce_model_inputs(observation.model_inputs))
        if "pixels" not in info and observation.pixels is not None:
            info["pixels"] = observation.pixels
        return info

    def _candidate_tensor(self, candidates: Sequence[DroneActionCandidate]) -> Any:
        torch = self._torch()
        rows = [
            self._fit_action_width(candidate.as_vector(self.config.action_order))
            for candidate in candidates
        ]
        tensor = torch.tensor(rows, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, self.config.action_steps, 1)
        if self.config.device:
            tensor = tensor.to(self.config.device)
        return tensor

    def _fit_action_width(self, values: list[float]) -> list[float]:
        if self._model_action_dim is None:
            return values
        if self._model_action_dim <= 0:
            raise ValueError("model_action_dim must be positive when provided")
        if len(values) > self._model_action_dim:
            return values[: self._model_action_dim]
        if len(values) < self._model_action_dim:
            return [*values, *([0.0] * (self._model_action_dim - len(values)))]
        return values

    def _zero_history_action(self, action_tensor: Any) -> Any:
        torch = self._torch()
        batch = action_tensor.shape[0]
        action_dim = action_tensor.shape[-1]
        zeros = torch.zeros(
            batch,
            self.config.history_size,
            action_dim,
            dtype=action_tensor.dtype,
            device=action_tensor.device,
        )
        return zeros

    @staticmethod
    def _tensor_to_float_list(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            flat = value.detach().cpu().reshape(-1).tolist()
            return [float(item) for item in flat]
        if isinstance(value, Sequence):
            return [float(item) for item in value]
        return [float(value)]

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("LeWMWorldModelPlanner requires PyTorch for tensor conversion") from exc
        return torch

    def _require_torch(self) -> None:
        self._torch()

    @staticmethod
    def _infer_model_action_dim(model: Any) -> int | None:
        action_encoder = getattr(model, "action_encoder", None)
        patch_embed = getattr(action_encoder, "patch_embed", None)
        in_channels = getattr(patch_embed, "in_channels", None)
        try:
            return int(in_channels) if in_channels is not None else None
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class CandidateSamplerConfig:
    """Velocity-lattice candidate generation limits."""

    forward_speeds_mps: tuple[float, ...] = (0.0, 1.5, 3.0)
    lateral_speeds_mps: tuple[float, ...] = (-1.0, 0.0, 1.0)
    vertical_speeds_mps: tuple[float, ...] = (-0.5, 0.0, 0.5)
    yaw_rates_rps: tuple[float, ...] = (-0.25, 0.0, 0.25)
    durations_s: tuple[float, ...] = (1.0,)
    include_hover: bool = True
    include_goal_directed: bool = True
    max_goal_speed_mps: float = 3.0
    source: str = "velocity_lattice"


class VelocityLatticeSampler:
    """Generate conservative short-horizon body-frame velocity candidates."""

    def __init__(self, config: CandidateSamplerConfig | None = None):
        self.config = config or CandidateSamplerConfig()

    def generate(
        self,
        observation: DroneObservation,
        goal: DroneObservation | None = None,
    ) -> list[DroneActionCandidate]:
        candidates: list[DroneActionCandidate] = []
        for vx, vy, vz, yaw_rate, duration in product(
            self.config.forward_speeds_mps,
            self.config.lateral_speeds_mps,
            self.config.vertical_speeds_mps,
            self.config.yaw_rates_rps,
            self.config.durations_s,
        ):
            candidates.append(
                DroneActionCandidate(
                    vx_mps=float(vx),
                    vy_mps=float(vy),
                    vz_mps=float(vz),
                    yaw_rate_rps=float(yaw_rate),
                    duration_s=float(duration),
                    source=self.config.source,
                    metadata={"sampler": self.config.source},
                )
            )

        if self.config.include_hover:
            for duration in self.config.durations_s:
                candidates.append(
                    DroneActionCandidate(
                        vx_mps=0.0,
                        vy_mps=0.0,
                        vz_mps=0.0,
                        yaw_rate_rps=0.0,
                        duration_s=float(duration),
                        source="hover",
                        metadata={"sampler": self.config.source},
                    )
                )

        if self.config.include_goal_directed and goal is not None:
            candidates.extend(self._goal_directed_candidates(observation, goal))

        return _dedupe_candidates(candidates)

    def _goal_directed_candidates(
        self,
        observation: DroneObservation,
        goal: DroneObservation,
    ) -> list[DroneActionCandidate]:
        current_xy = _xy_from_observation(observation)
        goal_xy = _xy_from_observation(goal)
        if current_xy is None or goal_xy is None:
            return []

        dx = goal_xy[0] - current_xy[0]
        dy = goal_xy[1] - current_xy[1]
        distance = hypot(dx, dy)
        if distance == 0:
            return []

        heading = atan2(dy, dx)
        speed = min(self.config.max_goal_speed_mps, distance)
        vx = speed * cos(heading)
        vy = speed * sin(heading)
        return [
            DroneActionCandidate(
                vx_mps=vx,
                vy_mps=vy,
                vz_mps=0.0,
                yaw_rate_rps=0.0,
                duration_s=float(duration),
                source="goal_directed",
                metadata={"sampler": self.config.source},
            )
            for duration in self.config.durations_s
        ]


class HeuristicWorldModelPlanner:
    """Dependency-free planner for kit demos and bench tests."""

    def score_action_candidates(
        self,
        observation: DroneObservation,
        goal: DroneObservation,
        candidates: Sequence[DroneActionCandidate],
    ) -> list[float]:
        current_xy = _xy_from_observation(observation) or (0.0, 0.0)
        goal_xy = _xy_from_observation(goal) or current_xy
        costs = []
        for candidate in candidates:
            next_x = current_xy[0] + candidate.vx_mps * candidate.duration_s
            next_y = current_xy[1] + candidate.vy_mps * candidate.duration_s
            distance_cost = hypot(goal_xy[0] - next_x, goal_xy[1] - next_y)
            motion_cost = 0.05 * hypot(candidate.vx_mps, candidate.vy_mps)
            yaw_cost = 0.1 * abs(candidate.yaw_rate_rps)
            costs.append(distance_cost + motion_cost + yaw_cost)
        return costs

    def rollout(
        self,
        observation: DroneObservation,
        candidates: Sequence[DroneActionCandidate],
    ) -> list[dict[str, float]]:
        current_xy = _xy_from_observation(observation) or (0.0, 0.0)
        return [
            {
                "x_m": current_xy[0] + candidate.vx_mps * candidate.duration_s,
                "y_m": current_xy[1] + candidate.vy_mps * candidate.duration_s,
            }
            for candidate in candidates
        ]

    def surprise(
        self,
        observation: DroneObservation,
        action: DroneActionCandidate,
        next_observation: DroneObservation,
    ) -> float:
        predicted = self.rollout(observation, [action])[0]
        actual = _xy_from_observation(next_observation)
        if actual is None:
            return 0.0
        return hypot(predicted["x_m"] - actual[0], predicted["y_m"] - actual[1])


@dataclass(frozen=True)
class FlightCommand:
    """Command handoff record for a flight controller."""

    controller: str
    issued_at_s: float
    action: DroneActionCandidate
    accepted_by_controller: bool
    message: str = ""

    def to_record(self) -> dict:
        return to_jsonable(self)


class FlightController(Protocol):
    """Minimal setpoint handoff expected by the Autonomy Kit."""

    def send_setpoint(
        self,
        action: DroneActionCandidate,
        observation: DroneObservation,
    ) -> FlightCommand:
        """Send one already safety-approved setpoint."""


class DryRunFlightController:
    """Flight-controller adapter for simulation, bench tests, and demos."""

    def __init__(self, name: str = "dry_run"):
        self.name = name
        self.commands: list[FlightCommand] = []

    def send_setpoint(
        self,
        action: DroneActionCandidate,
        observation: DroneObservation,
    ) -> FlightCommand:
        command = FlightCommand(
            controller=self.name,
            issued_at_s=time.time(),
            action=action,
            accepted_by_controller=True,
            message=f"dry-run setpoint at observation {observation.timestamp_s}",
        )
        self.commands.append(command)
        return command


class AutonomyMode(str, Enum):
    """Execution posture for one autonomy-kit decision."""

    ADVISORY = "advisory"
    DRY_RUN = "dry_run"
    CLOSED_LOOP = "closed_loop"


class MissionEffectContext(str, Enum):
    """Mission context, not an effects-command interface."""

    NON_KINETIC = "non_kinetic"
    KINETIC_SUPPORT = "kinetic_support"


@dataclass(frozen=True)
class AutonomyKitConfig:
    """Runtime settings for the Autonomy Kit."""

    mode: AutonomyMode = AutonomyMode.ADVISORY
    effect_context: MissionEffectContext = MissionEffectContext.NON_KINETIC
    max_candidates: int = 128
    require_operator_approval_for_execution: bool = True
    log_all_candidates: bool = False


@dataclass(frozen=True)
class AutonomyDecision:
    """Result of one guarded Autonomy Kit decision cycle."""

    mode: AutonomyMode
    effect_context: MissionEffectContext
    candidate_count: int
    accepted_candidate_count: int
    proposed_action: DroneActionCandidate | None
    selected_action: DroneActionCandidate | None
    executed_action: DroneActionCandidate | None
    safety_decision: SafetyDecision | None
    command: FlightCommand | None
    costs: tuple[float | None, ...]
    status: str
    mission_log_hash: str | None = None

    @property
    def accepted(self) -> bool:
        return self.safety_decision.accepted if self.safety_decision else False

    def to_record(self) -> dict:
        return to_jsonable(self)


class AutonomyKit:
    """End-to-end guarded autonomy cycle around LeWM and flight control."""

    def __init__(
        self,
        planner: WorldModelPlanner,
        *,
        safety_supervisor: SafetySupervisor | None = None,
        candidate_sampler: VelocityLatticeSampler | None = None,
        flight_controller: FlightController | None = None,
        mission_log: MissionLog | None = None,
        config: AutonomyKitConfig | None = None,
    ):
        self.planner = planner
        self.safety_supervisor = safety_supervisor or SafetySupervisor()
        self.candidate_sampler = candidate_sampler or VelocityLatticeSampler()
        self.flight_controller = flight_controller
        self.mission_log = mission_log
        self.config = config or AutonomyKitConfig()

    def step(
        self,
        observation: DroneObservation,
        goal: DroneObservation,
        *,
        candidates: Sequence[DroneActionCandidate] | None = None,
        operator_input: OperatorInput | None = None,
    ) -> AutonomyDecision:
        generated = list(candidates or self.candidate_sampler.generate(observation, goal))
        generated = [self._annotate_effect_context(candidate) for candidate in generated]
        generated = generated[: self.config.max_candidates]
        if not generated:
            return self._decision(
                candidate_count=0,
                accepted_candidate_count=0,
                proposed_action=None,
                selected_action=None,
                executed_action=None,
                safety_decision=None,
                command=None,
                costs=(),
                status="no_candidates",
            )

        safety_by_index = [
            self.safety_supervisor.evaluate(observation, candidate) for candidate in generated
        ]
        accepted_pairs = [
            (idx, decision.action, decision)
            for idx, decision in enumerate(safety_by_index)
            if decision.accepted and decision.action is not None
        ]

        if not accepted_pairs:
            decision = safety_by_index[0]
            return self._log_and_decide(
                observation,
                generated[0],
                decision,
                operator_input=operator_input,
                executed_action=None,
                command=None,
                costs=tuple(None for _ in generated),
                status="all_candidates_rejected",
                candidate_count=len(generated),
                accepted_candidate_count=0,
            )

        accepted_candidates = [pair[1] for pair in accepted_pairs if pair[1] is not None]
        accepted_costs = self.planner.score_action_candidates(observation, goal, accepted_candidates)
        if len(accepted_costs) != len(accepted_candidates):
            raise ValueError("planner returned a different number of costs than candidates")

        best_local_index = min(range(len(accepted_costs)), key=accepted_costs.__getitem__)
        _selected_global_index, selected_action, selected_decision = accepted_pairs[best_local_index]
        costs = [None for _ in generated]
        for (global_index, _, _), cost in zip(accepted_pairs, accepted_costs, strict=True):
            costs[global_index] = float(cost)

        command = None
        executed_action = None
        status = "recommended"
        if self._should_execute(operator_input):
            if self.flight_controller is None:
                status = "controller_unavailable"
            elif selected_decision.action is None:
                status = "selected_action_rejected"
            else:
                command = self.flight_controller.send_setpoint(selected_decision.action, observation)
                executed_action = selected_decision.action if command.accepted_by_controller else None
                status = "executed" if command.accepted_by_controller else "controller_rejected"
        elif self.config.mode != AutonomyMode.ADVISORY:
            status = "operator_approval_required"

        return self._log_and_decide(
            observation,
            selected_action,
            selected_decision,
            operator_input=operator_input,
            executed_action=executed_action,
            command=command,
            costs=tuple(costs),
            status=status,
            candidate_count=len(generated),
            accepted_candidate_count=len(accepted_candidates),
        )

    def _annotate_effect_context(
        self,
        candidate: DroneActionCandidate,
    ) -> DroneActionCandidate:
        if self.config.effect_context != MissionEffectContext.KINETIC_SUPPORT:
            return candidate
        metadata = dict(candidate.metadata)
        metadata["kinetic_context"] = True
        metadata["effect_command"] = False
        return candidate.with_updates(metadata=metadata)

    def _should_execute(self, operator_input: OperatorInput | None) -> bool:
        if self.config.mode == AutonomyMode.ADVISORY:
            return False
        if self.config.require_operator_approval_for_execution:
            return operator_input is not None and operator_input.authorized
        return True

    def _log_and_decide(
        self,
        observation: DroneObservation,
        proposed_action: DroneActionCandidate,
        safety_decision: SafetyDecision,
        *,
        operator_input: OperatorInput | None,
        executed_action: DroneActionCandidate | None,
        command: FlightCommand | None,
        costs: tuple[float | None, ...],
        status: str,
        candidate_count: int,
        accepted_candidate_count: int,
    ) -> AutonomyDecision:
        mission_log_hash = None
        if self.mission_log is not None:
            entry = self.mission_log.append(
                observation,
                proposed_action,
                safety_decision,
                operator_input=operator_input,
                executed_action=executed_action,
                outcome={
                    "status": status,
                    "mode": self.config.mode.value,
                    "effect_context": self.config.effect_context.value,
                    "candidate_count": candidate_count,
                    "accepted_candidate_count": accepted_candidate_count,
                    "command": command.to_record() if command else None,
                    "costs": list(costs) if self.config.log_all_candidates else None,
                },
            )
            mission_log_hash = entry.entry_hash

        return self._decision(
            candidate_count=candidate_count,
            accepted_candidate_count=accepted_candidate_count,
            proposed_action=proposed_action,
            selected_action=safety_decision.action,
            executed_action=executed_action,
            safety_decision=safety_decision,
            command=command,
            costs=costs,
            status=status,
            mission_log_hash=mission_log_hash,
        )

    def _decision(self, **kwargs) -> AutonomyDecision:
        return AutonomyDecision(
            mode=self.config.mode,
            effect_context=self.config.effect_context,
            **kwargs,
        )


def _xy_from_observation(observation: DroneObservation) -> tuple[float, float] | None:
    telemetry = observation.telemetry
    for x_key, y_key in (("x_m", "y_m"), ("north_m", "east_m")):
        if x_key in telemetry and y_key in telemetry:
            try:
                return float(telemetry[x_key]), float(telemetry[y_key])
            except (TypeError, ValueError):
                return None
    return None


def _dedupe_candidates(candidates: Sequence[DroneActionCandidate]) -> list[DroneActionCandidate]:
    seen: set[tuple[float, float, float, float, float]] = set()
    unique: list[DroneActionCandidate] = []
    for candidate in candidates:
        key = (
            round(candidate.vx_mps, 4),
            round(candidate.vy_mps, 4),
            round(candidate.vz_mps, 4),
            round(candidate.yaw_rate_rps, 4),
            round(candidate.duration_s, 4),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
