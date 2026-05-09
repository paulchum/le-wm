import math
import sys
import unittest
from pathlib import Path

from drone_ai import DroneActionCandidate

ROS_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "aas_ros2"
    / "lewm_drone_ros"
)
sys.path.insert(0, str(ROS_PACKAGE_ROOT))

from lewm_drone_ros.conversions import (  # noqa: E402
    approval_is_active,
    body_velocity_to_local_ned,
    build_goal_observation,
    build_px4_observation,
    candidate_to_local_ned,
    is_fresh,
    should_publish_executable_setpoint,
)


class AASRos2ConversionTest(unittest.TestCase):
    def test_body_velocity_maps_to_ned_at_north_heading(self):
        north, east, down = body_velocity_to_local_ned(2.0, 1.0, 0.5, 0.0)

        self.assertAlmostEqual(north, 2.0)
        self.assertAlmostEqual(east, 1.0)
        self.assertAlmostEqual(down, -0.5)

    def test_body_velocity_rotates_with_heading(self):
        north, east, down = body_velocity_to_local_ned(2.0, 1.0, -0.5, math.pi / 2.0)

        self.assertAlmostEqual(north, -1.0)
        self.assertAlmostEqual(east, 2.0)
        self.assertAlmostEqual(down, 0.5)

    def test_candidate_to_local_ned_preserves_yaw_rate(self):
        setpoint = candidate_to_local_ned(
            DroneActionCandidate(3.0, 0.0, 0.25, -0.4, 1.0),
            0.0,
        )

        self.assertEqual(setpoint.north_mps, 3.0)
        self.assertEqual(setpoint.down_mps, -0.25)
        self.assertEqual(setpoint.yawspeed_rps, -0.4)

    def test_freshness_and_approval_gates(self):
        self.assertTrue(is_fresh(10.0, 9.25, 1.0))
        self.assertFalse(is_fresh(10.0, 8.5, 1.0))
        self.assertFalse(is_fresh(10.0, None, 1.0))

        self.assertTrue(approval_is_active(approved=True, now_s=10.0, expires_at_s=0.0))
        self.assertTrue(approval_is_active(approved=True, now_s=10.0, expires_at_s=10.5))
        self.assertFalse(approval_is_active(approved=True, now_s=10.0, expires_at_s=9.9))
        self.assertFalse(approval_is_active(approved=False, now_s=10.0, expires_at_s=0.0))

    def test_executable_setpoint_requires_all_gates(self):
        candidate = DroneActionCandidate(1.0, 0.0, 0.0, 0.0, 1.0)

        self.assertTrue(
            should_publish_executable_setpoint(
                decision_accepted=True,
                selected_action=candidate,
                goal_fresh=True,
                local_position_fresh=True,
                approval_active=True,
            )
        )
        self.assertFalse(
            should_publish_executable_setpoint(
                decision_accepted=True,
                selected_action=candidate,
                goal_fresh=True,
                local_position_fresh=True,
                approval_active=False,
            )
        )
        self.assertFalse(
            should_publish_executable_setpoint(
                decision_accepted=True,
                selected_action=None,
                goal_fresh=True,
                local_position_fresh=True,
                approval_active=True,
            )
        )

    def test_observation_and_goal_mapping_use_local_ned_contract(self):
        observation = build_px4_observation(
            timestamp_s=12.0,
            north_m=100.0,
            east_m=25.0,
            down_m=-40.0,
            heading_rad=0.3,
            velocity_north_mps=15.0,
            velocity_east_mps=2.0,
            vehicle_status={"is_vtol": True},
        )
        goal = build_goal_observation(
            timestamp_s=13.0,
            north_m=250.0,
            east_m=25.0,
            down_m=-60.0,
            goal_id="leg-1",
            yaw_rad=1.0,
        )

        self.assertEqual(observation.telemetry["north_m"], 100.0)
        self.assertEqual(observation.telemetry["east_m"], 25.0)
        self.assertEqual(observation.telemetry["altitude_m"], 40.0)
        self.assertTrue(observation.mission_state["operator_authority"])
        self.assertEqual(observation.mission_state["vehicle_status"]["is_vtol"], True)
        self.assertEqual(goal.telemetry["north_m"], 250.0)
        self.assertEqual(goal.telemetry["altitude_m"], 60.0)
        self.assertEqual(goal.mission_state["goal_id"], "leg-1")


if __name__ == "__main__":
    unittest.main()
