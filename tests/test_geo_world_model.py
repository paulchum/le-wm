import unittest

from drone_ai import (
    GeoRasterLayer,
    GeoWorldModelInput,
    GeoWorldModelInputAdapter,
    coerce_model_inputs,
)


try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GeoWorldModelInputAdapterTest(unittest.TestCase):
    def test_converts_synthetic_geo_input_to_model_tensors(self):
        pixels = torch.zeros(2, 3, 3, 8, 8)
        geo_input = GeoWorldModelInput(
            pixels=pixels,
            orthomosaic=GeoRasterLayer(torch.full((2, 1, 3, 8, 8), 0.5)),
            static_rasters={
                "elevation": GeoRasterLayer(
                    torch.ones(2, 1, 1, 8, 8),
                    valid_mask=torch.ones(2, 1, 8, 8, dtype=torch.bool),
                    uncertainty=torch.full((2, 1, 1, 8, 8), 0.25),
                    metadata={"source": "synthetic_dem"},
                ),
                "land_cover": GeoRasterLayer(torch.ones(2, 3, 1, 8, 8) * 2),
                "hydro": GeoRasterLayer(torch.ones(2, 3, 1, 8, 8) * 3),
            },
            dynamic_rasters={
                "weather": GeoRasterLayer(torch.ones(2, 3, 2, 8, 8)),
                "ice": GeoRasterLayer(torch.ones(2, 1, 1, 8, 8) * 4),
            },
            vector_layers={"hydro_lines": torch.ones(2, 3, 4, 2)},
            action_history=torch.zeros(2, 3, 4),
            trajectory_history=torch.zeros(2, 3, 6),
            proprioception_history=torch.zeros(2, 3, 5),
            temporal_metadata={"start_s": 10.0},
            geospatial_metadata={"crs": "EPSG:3978"},
        )

        model_inputs = GeoWorldModelInputAdapter().to_model_inputs(geo_input)

        self.assertIs(model_inputs["pixels"], pixels)
        self.assertEqual(tuple(model_inputs["action"].shape), (2, 3, 4))
        self.assertEqual(tuple(model_inputs["proprio"].shape), (2, 3, 5))
        self.assertEqual(tuple(model_inputs["trajectory"].shape), (2, 3, 6))
        self.assertEqual(tuple(model_inputs["geo_static_elevation"].shape), (2, 3, 1, 8, 8))
        self.assertEqual(tuple(model_inputs["geo_dynamic_weather"].shape), (2, 3, 2, 8, 8))
        self.assertEqual(tuple(model_inputs["geo_imagery_orthomosaic"].shape), (2, 3, 3, 8, 8))
        self.assertEqual(tuple(model_inputs["geo_mask_imagery_orthomosaic"].shape), (2, 3, 3, 8, 8))
        self.assertTrue(model_inputs["geo_present_static_elevation"].all())
        self.assertTrue(model_inputs["geo_mask_static_elevation"].all())
        self.assertEqual(model_inputs["geo_temporal_metadata"], {"start_s": 10.0})
        self.assertEqual(model_inputs["geo_geospatial_metadata"], {"crs": "EPSG:3978"})
        self.assertEqual(
            model_inputs["geo_layer_metadata"]["geo_static_elevation"],
            {"source": "synthetic_dem"},
        )
        self.assertIn("geo_vector_hydro_lines", model_inputs)
        self.assertTrue(model_inputs["geo_present_vector_hydro_lines"])

    def test_missing_default_layers_are_explicit(self):
        model_inputs = GeoWorldModelInputAdapter().to_model_inputs(
            GeoWorldModelInput(pixels=torch.zeros(1, 2, 3, 4, 4))
        )

        for key in (
            "geo_dynamic_weather",
            "geo_dynamic_ice",
            "geo_imagery_orthomosaic",
        ):
            self.assertIn(key, model_inputs)
            self.assertEqual(tuple(model_inputs[key].shape), (1, 2, 1, 4, 4))
            self.assertEqual(float(model_inputs[key].sum()), 0.0)

        for key in (
            "geo_present_dynamic_weather",
            "geo_present_dynamic_ice",
            "geo_present_imagery_orthomosaic",
        ):
            self.assertIn(key, model_inputs)
            self.assertFalse(model_inputs[key].any())

        self.assertFalse(model_inputs["geo_mask_dynamic_weather"].any())
        self.assertFalse(model_inputs["geo_mask_dynamic_ice"].any())
        self.assertFalse(model_inputs["geo_mask_imagery_orthomosaic"].any())

    def test_shape_validation_rejects_mismatched_layers(self):
        adapter = GeoWorldModelInputAdapter()

        with self.assertRaises(ValueError):
            adapter.to_model_inputs(
                GeoWorldModelInput(
                    pixels=torch.zeros(1, 3, 3, 8, 8),
                    action_history=torch.zeros(1, 2, 4),
                )
            )

        with self.assertRaises(ValueError):
            adapter.to_model_inputs(
                GeoWorldModelInput(
                    pixels=torch.zeros(1, 3, 3, 8, 8),
                    dynamic_rasters={
                        "weather": GeoRasterLayer(torch.zeros(1, 3, 1, 7, 8))
                    },
                )
            )

    def test_legacy_model_inputs_pass_through_unchanged(self):
        legacy = {"pixels": object(), "action": object()}

        self.assertIs(coerce_model_inputs(legacy), legacy)


if __name__ == "__main__":
    unittest.main()
