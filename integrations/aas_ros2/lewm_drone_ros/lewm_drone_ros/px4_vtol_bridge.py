"""Operator-gated ROS2 bridge from drone_ai decisions to AAS PX4 VTOL setpoints."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from autopilot_interface_msgs.action import Offboard
from lewm_drone_ros_msgs.msg import NavigationGoal, OperatorApproval
from px4_msgs.msg import (
    AirspeedValidated,
    TrajectorySetpoint,
    VehicleGlobalPosition,
    VehicleLocalPosition,
    VehicleStatus,
)
from std_msgs.msg import String

from drone_ai import (
    AutonomyKit,
    AutonomyKitConfig,
    AutonomyMode,
    HeuristicWorldModelPlanner,
    MissionLog,
    OperatorInput,
)

from .conversions import (
    approval_is_active,
    build_goal_observation,
    build_px4_observation,
    candidate_to_local_ned,
    is_fresh,
    should_publish_executable_setpoint,
)


class PX4VtolAutonomyBridge(Node):
    """Run the Autonomy Kit from PX4 state and publish approved VTOL setpoints."""

    def __init__(self) -> None:
        super().__init__("lewm_drone_px4_vtol_bridge")

        self.declare_parameter("decision_hz", 5.0)
        self.declare_parameter("max_input_age_s", 1.0)
        self.declare_parameter("offboard_action_duration_s", 3.0)
        self.declare_parameter("max_candidates", 128)
        self.declare_parameter("mission_log_path", "")
        self.declare_parameter("local_position_topic", "fmu/out/vehicle_local_position")
        self.declare_parameter("global_position_topic", "fmu/out/vehicle_global_position")
        self.declare_parameter("airspeed_topic", "fmu/out/airspeed_validated")
        self.declare_parameter("vehicle_status_topic", "fmu/out/vehicle_status_v1")
        self.declare_parameter("goal_topic", "/lewm_drone/goal")
        self.declare_parameter("approval_topic", "/lewm_drone/operator_approval")
        self.declare_parameter("decision_topic", "/lewm_drone/decision_json")
        self.declare_parameter("setpoint_topic", "/lewm_drone/px4_trajectory_setpoint")
        self.declare_parameter("offboard_action_name", "offboard_action")

        mission_log_path = str(self.get_parameter("mission_log_path").value)
        mission_log = MissionLog(Path(mission_log_path)) if mission_log_path else None
        self.kit = AutonomyKit(
            planner=HeuristicWorldModelPlanner(),
            mission_log=mission_log,
            config=AutonomyKitConfig(
                mode=AutonomyMode.ADVISORY,
                max_candidates=int(self.get_parameter("max_candidates").value),
            ),
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._local_position: VehicleLocalPosition | None = None
        self._local_position_seen_s: float | None = None
        self._global_position: VehicleGlobalPosition | None = None
        self._global_position_seen_s: float | None = None
        self._airspeed: AirspeedValidated | None = None
        self._vehicle_status: VehicleStatus | None = None
        self._goal: NavigationGoal | None = None
        self._goal_seen_s: float | None = None
        self._approval: OperatorApproval | None = None
        self._approval_seen_s: float | None = None
        self._offboard_goal_active = False

        self.create_subscription(
            VehicleLocalPosition,
            str(self.get_parameter("local_position_topic").value),
            self._local_position_callback,
            qos,
        )
        self.create_subscription(
            VehicleGlobalPosition,
            str(self.get_parameter("global_position_topic").value),
            self._global_position_callback,
            qos,
        )
        self.create_subscription(
            AirspeedValidated,
            str(self.get_parameter("airspeed_topic").value),
            self._airspeed_callback,
            qos,
        )
        self.create_subscription(
            VehicleStatus,
            str(self.get_parameter("vehicle_status_topic").value),
            self._vehicle_status_callback,
            qos,
        )
        self.create_subscription(
            NavigationGoal,
            str(self.get_parameter("goal_topic").value),
            self._goal_callback,
            10,
        )
        self.create_subscription(
            OperatorApproval,
            str(self.get_parameter("approval_topic").value),
            self._approval_callback,
            10,
        )

        self._decision_pub = self.create_publisher(
            String,
            str(self.get_parameter("decision_topic").value),
            10,
        )
        self._setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            str(self.get_parameter("setpoint_topic").value),
            10,
        )
        self._offboard_client = ActionClient(
            self,
            Offboard,
            str(self.get_parameter("offboard_action_name").value),
        )

        decision_hz = float(self.get_parameter("decision_hz").value)
        self.create_timer(1.0 / max(decision_hz, 0.1), self._decision_timer_callback)

    def _local_position_callback(self, msg: VehicleLocalPosition) -> None:
        self._local_position = msg
        self._local_position_seen_s = self._now_s()

    def _global_position_callback(self, msg: VehicleGlobalPosition) -> None:
        self._global_position = msg
        self._global_position_seen_s = self._now_s()

    def _airspeed_callback(self, msg: AirspeedValidated) -> None:
        self._airspeed = msg

    def _vehicle_status_callback(self, msg: VehicleStatus) -> None:
        self._vehicle_status = msg

    def _goal_callback(self, msg: NavigationGoal) -> None:
        self._goal = msg
        self._goal_seen_s = self._now_s()

    def _approval_callback(self, msg: OperatorApproval) -> None:
        self._approval = msg
        self._approval_seen_s = self._now_s()

    def _decision_timer_callback(self) -> None:
        now_s = self._now_s()
        max_age_s = float(self.get_parameter("max_input_age_s").value)
        local_fresh = is_fresh(now_s, self._local_position_seen_s, max_age_s)
        goal_fresh = is_fresh(now_s, self._goal_seen_s, max_age_s)
        if not local_fresh or not goal_fresh:
            return
        if self._local_position is None or self._goal is None:
            return

        approval_active = self._approval_active(now_s)
        observation = self._build_observation(now_s)
        goal = build_goal_observation(
            timestamp_s=now_s,
            north_m=self._goal.north_m,
            east_m=self._goal.east_m,
            down_m=self._goal.down_m,
            goal_id=self._goal.goal_id,
            yaw_rad=self._goal.yaw_rad,
        )
        operator_input = self._operator_input(approval_active)

        decision = self.kit.step(
            observation,
            goal,
            operator_input=operator_input,
        )
        self._publish_decision(decision)

        if should_publish_executable_setpoint(
            decision_accepted=decision.accepted,
            selected_action=decision.selected_action,
            goal_fresh=goal_fresh,
            local_position_fresh=local_fresh,
            approval_active=approval_active,
        ):
            self._publish_setpoint(decision.selected_action)
            self._ensure_offboard_action()

    def _build_observation(self, now_s: float):
        assert self._local_position is not None
        local = self._local_position
        global_position = self._global_position
        airspeed = self._airspeed
        status = self._vehicle_status
        vehicle_status = None
        if status is not None:
            vehicle_status = {
                "arming_state": int(getattr(status, "arming_state", -1)),
                "vehicle_type": int(getattr(status, "vehicle_type", -1)),
                "is_vtol": bool(getattr(status, "is_vtol", False)),
                "in_transition_mode": bool(getattr(status, "in_transition_mode", False)),
                "in_transition_to_fw": bool(getattr(status, "in_transition_to_fw", False)),
            }
        return build_px4_observation(
            timestamp_s=now_s,
            north_m=float(local.x),
            east_m=float(local.y),
            down_m=float(local.z),
            heading_rad=float(local.heading),
            velocity_north_mps=_finite_attr(local, "vx"),
            velocity_east_mps=_finite_attr(local, "vy"),
            velocity_down_mps=_finite_attr(local, "vz"),
            latitude_deg=_finite_attr(global_position, "lat"),
            longitude_deg=_finite_attr(global_position, "lon"),
            altitude_m=_finite_attr(global_position, "alt"),
            airspeed_mps=_finite_attr(airspeed, "true_airspeed_m_s"),
            vehicle_status=vehicle_status,
            operator_authority=True,
        )

    def _operator_input(self, approval_active: bool) -> OperatorInput | None:
        if self._approval is None:
            return None
        return OperatorInput(
            operator_id=self._approval.operator_id or "unknown",
            command=self._approval.command or "approve",
            authorized=approval_active,
            notes=self._approval.notes,
        )

    def _approval_active(self, now_s: float) -> bool:
        if self._approval is None:
            return False
        max_age_s = float(self.get_parameter("max_input_age_s").value)
        if not is_fresh(now_s, self._approval_seen_s, max_age_s):
            return False
        return approval_is_active(
            approved=bool(self._approval.approved),
            now_s=now_s,
            expires_at_s=float(self._approval.expires_at_s),
        )

    def _publish_decision(self, decision: Any) -> None:
        msg = String()
        msg.data = json.dumps(decision.to_record(), sort_keys=True)
        self._decision_pub.publish(msg)

    def _publish_setpoint(self, action: Any) -> None:
        assert self._local_position is not None
        local_velocity = candidate_to_local_ned(action, float(self._local_position.heading))
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        nan = float("nan")
        msg.position = [nan, nan, nan]
        msg.velocity = [
            float(local_velocity.north_mps),
            float(local_velocity.east_mps),
            float(local_velocity.down_mps),
        ]
        msg.acceleration = [nan, nan, nan]
        msg.jerk = [nan, nan, nan]
        msg.yaw = nan
        msg.yawspeed = float(local_velocity.yawspeed_rps)
        self._setpoint_pub.publish(msg)

    def _ensure_offboard_action(self) -> None:
        if self._offboard_goal_active:
            return
        if not self._offboard_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warning("Offboard action server is not available")
            return
        goal = Offboard.Goal()
        goal.offboard_setpoint_type = int(getattr(Offboard.Goal, "TRAJECTORY", 2))
        goal.max_duration_sec = float(self.get_parameter("offboard_action_duration_s").value)
        self._offboard_goal_active = True
        future = self._offboard_client.send_goal_async(goal)
        future.add_done_callback(self._offboard_goal_response_callback)

    def _offboard_goal_response_callback(self, future: Any) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._offboard_goal_active = False
            self.get_logger().warning("Offboard action goal was rejected")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._offboard_result_callback)

    def _offboard_result_callback(self, future: Any) -> None:
        try:
            result = future.result().result
            if not result.success:
                self.get_logger().warning(f"Offboard action completed without success: {result.message}")
        finally:
            self._offboard_goal_active = False

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def _finite_attr(obj: Any | None, attr: str) -> float | None:
    if obj is None:
        return None
    value = getattr(obj, attr, None)
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if isfinite(value_f) else None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PX4VtolAutonomyBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
