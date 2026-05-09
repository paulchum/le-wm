import unittest

from drone_ai import DroneActionCandidate, DroneObservation, GeoWorldModelInput


class _Pixels:
    shape = (1, 3, 224, 224)


class InterfacesTest(unittest.TestCase):
    def test_observation_summary_is_json_safe_and_excludes_raw_pixels(self):
        observation = DroneObservation(
            timestamp_s=10.5,
            pixels=_Pixels(),
            telemetry={"battery_percent": 80, "lat": 45.4},
            proprioception=(1.0, 2.0),
            mission_state={"operator_authority": True},
            environment={"lighting": "low"},
            model_inputs={"pixels": object()},
        )

        summary = observation.summary()

        self.assertEqual(summary["pixel_shape"], [1, 3, 224, 224])
        self.assertEqual(summary["model_input_keys"], ["pixels"])
        self.assertNotIn("pixels", summary)

    def test_action_vector_order(self):
        candidate = DroneActionCandidate(
            vx_mps=1.0,
            vy_mps=2.0,
            vz_mps=-0.5,
            yaw_rate_rps=0.1,
            duration_s=1.0,
        )

        self.assertEqual(candidate.as_vector(), [1.0, 2.0, -0.5, 0.1])
        self.assertEqual(candidate.as_vector(("yaw_rate_rps", "vx_mps")), [0.1, 1.0])

    def test_observation_summary_accepts_typed_geo_model_input(self):
        observation = DroneObservation(
            timestamp_s=10.5,
            model_inputs=GeoWorldModelInput(pixels=_Pixels()),
        )

        self.assertEqual(observation.summary()["model_input_keys"], ["GeoWorldModelInput"])


if __name__ == "__main__":
    unittest.main()
