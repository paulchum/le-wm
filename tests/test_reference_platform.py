import unittest

from lewm_drone import (
    default_reference_platform,
    payload_compatibility,
    reference_platform_summary,
    validate_reference_platform,
)


class ReferencePlatformTest(unittest.TestCase):
    def test_default_reference_platform_is_valid_small_vtol(self):
        platform = default_reference_platform()

        self.assertEqual(validate_reference_platform(platform), [])
        self.assertLess(platform.airframe.max_takeoff_weight_kg, 25.0)
        self.assertEqual(platform.airframe.category, "small_rpas_under_25kg")
        self.assertIn("standard_922_09_c2_link_lost_link", platform.qualification_targets)

    def test_payload_compatibility_reports_compatible_bays(self):
        platform = default_reference_platform()
        result = payload_compatibility(platform.payloads[0], platform.payload_bays)

        self.assertTrue(result["compatible"])
        self.assertIn("nose-gimbal", result["compatible_bays"])

    def test_summary_includes_validation_state(self):
        summary = reference_platform_summary(default_reference_platform())

        self.assertEqual(summary["validation_errors"], [])
        self.assertEqual(summary["airframe_category"], "small_rpas_under_25kg")


if __name__ == "__main__":
    unittest.main()
