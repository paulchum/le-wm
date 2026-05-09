import unittest

from drone_ai import (
    DroneActionCandidate,
    DroneObservation,
    GeoFence,
    SafetyEnvelope,
    SafetySupervisor,
)


def observation(**overrides):
    telemetry = {"battery_percent": 80, "altitude_m": 10.0, "comms_link": True}
    telemetry.update(overrides.pop("telemetry", {}))
    mission_state = {"operator_authority": True}
    mission_state.update(overrides.pop("mission_state", {}))
    return DroneObservation(
        timestamp_s=1.0,
        telemetry=telemetry,
        mission_state=mission_state,
        **overrides,
    )


def action(**overrides):
    values = {
        "vx_mps": 1.0,
        "vy_mps": 0.0,
        "vz_mps": 0.0,
        "yaw_rate_rps": 0.0,
        "duration_s": 1.0,
    }
    values.update(overrides)
    return DroneActionCandidate(**values)


class SafetySupervisorTest(unittest.TestCase):
    def test_accepts_safe_action(self):
        decision = SafetySupervisor().evaluate(observation(), action())

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reasons, ())
        self.assertIsNotNone(decision.action)

    def test_clips_action_inside_envelope_when_allowed(self):
        supervisor = SafetySupervisor(SafetyEnvelope(max_horizontal_speed_mps=2.0))
        decision = supervisor.evaluate(observation(), action(vx_mps=10.0, vy_mps=0.0))

        self.assertTrue(decision.accepted)
        self.assertIn("horizontal_speed_clipped", decision.warnings)
        self.assertAlmostEqual(decision.action.vx_mps, 2.0)

    def test_rejects_speed_when_clipping_disabled(self):
        supervisor = SafetySupervisor(
            SafetyEnvelope(max_horizontal_speed_mps=2.0, allow_clipping=False)
        )
        decision = supervisor.evaluate(observation(), action(vx_mps=10.0, vy_mps=0.0))

        self.assertFalse(decision.accepted)
        self.assertIn("horizontal_speed_exceeds_limit", decision.reasons)

    def test_rejects_low_battery_and_non_kinetic_violation(self):
        decision = SafetySupervisor().evaluate(
            observation(telemetry={"battery_percent": 10}),
            action(metadata={"payload_release": True}),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("battery_below_minimum", decision.reasons)
        self.assertIn("effect_command_blocked", decision.reasons)

    def test_allows_kinetic_context_when_configured_but_blocks_effect_command(self):
        supervisor = SafetySupervisor(
            SafetyEnvelope(require_non_kinetic=False, allow_kinetic_context=True)
        )

        context_decision = supervisor.evaluate(
            observation(),
            action(metadata={"kinetic_context": True, "effect_command": False}),
        )
        effect_decision = supervisor.evaluate(
            observation(),
            action(metadata={"kinetic_context": True, "weapon_release": True}),
        )

        self.assertTrue(context_decision.accepted)
        self.assertFalse(effect_decision.accepted)
        self.assertIn("effect_command_blocked", effect_decision.reasons)

    def test_rejects_outside_geofence(self):
        supervisor = SafetySupervisor(
            SafetyEnvelope(geofence=GeoFence(min_lat=40, max_lat=50, min_lon=-80, max_lon=-70))
        )
        decision = supervisor.evaluate(
            observation(telemetry={"lat": 55, "lon": -75}),
            action(),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("outside_geofence", decision.reasons)


if __name__ == "__main__":
    unittest.main()
