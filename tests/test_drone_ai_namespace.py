import unittest

import drone_ai


class DroneAINamespaceTest(unittest.TestCase):
    def test_public_namespace_exposes_autonomy_kit(self):
        self.assertTrue(hasattr(drone_ai, "AutonomyKit"))
        self.assertTrue(hasattr(drone_ai, "LEWM_HF_MODEL_REPOS"))
        self.assertTrue(hasattr(drone_ai, "LeWMCheckpointSpec"))
        self.assertTrue(hasattr(drone_ai, "LeWMWorldModelPlanner"))
        self.assertTrue(hasattr(drone_ai, "load_hf_lewm_world_model_planner"))
        self.assertTrue(hasattr(drone_ai, "SafetySupervisor"))
        self.assertTrue(hasattr(drone_ai, "default_reference_platform"))
        self.assertTrue(hasattr(drone_ai, "SecureDataPipeline"))
        self.assertTrue(hasattr(drone_ai, "GeoDatasetManifest"))
        self.assertTrue(hasattr(drone_ai, "UAVObservation"))
        self.assertTrue(hasattr(drone_ai, "validate_geospatial_dataset_manifest"))

    def test_lewm_component_spec_identifies_component_boundary(self):
        spec = drone_ai.default_lewm_component_spec()

        self.assertEqual(spec.name, "LeWorldModel")
        self.assertEqual(spec.adapter, "drone_ai.LeWMWorldModelPlanner")
        self.assertIn("jepa.py", spec.model_files)
        self.assertIn("weapon release", spec.blocked_responsibilities)


if __name__ == "__main__":
    unittest.main()
