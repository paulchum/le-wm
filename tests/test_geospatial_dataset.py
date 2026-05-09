import json
import unittest
from dataclasses import replace

from lewm_drone import (
    AOIMetadata,
    BoundingBox,
    CRSDefinition,
    DataAge,
    DroneObservation,
    GeoDatasetManifest,
    GeoDatasetValidationError,
    GeoGrid,
    LabelTarget,
    LeWMTrainingBinding,
    MaskLayer,
    OrthomosaicTile,
    RasterLayer,
    SourceProvenance,
    TemporalWindow,
    TrajectoryStep,
    UAVObservation,
    UncertaintyLayer,
    VectorLayer,
    require_valid_geospatial_dataset_manifest,
    validate_geospatial_dataset_manifest,
)


BASE_TIME_S = 1_800_000_000.0


class _Pixels:
    shape = (1, 3, 224, 224)


def _provenance(source_id: str, method: str = "public_open_data") -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        source_name=f"{source_id} source",
        collection_method=method,
        source_uri=f"https://example.invalid/{source_id}",
        license="open-test-fixture",
        organization="Canadian public-data fixture",
        product_version="2026.1",
        classification_label="unclassified",
    )


def _age(observed_delta_s: float, max_age_s: float | None = None) -> DataAge:
    return DataAge(
        observed_at_s=BASE_TIME_S - observed_delta_s,
        ingested_at_s=BASE_TIME_S - min(observed_delta_s, 60.0),
        max_age_s=max_age_s,
    )


def _grid(crs: CRSDefinition, resolution_m: float = 0.25) -> GeoGrid:
    return GeoGrid(
        width_px=512,
        height_px=512,
        resolution_m=resolution_m,
        origin_x=1_500_000.0,
        origin_y=-200_000.0,
        crs=crs,
    )


def _sample_manifest() -> GeoDatasetManifest:
    crs = CRSDefinition(horizontal="EPSG:3978", vertical_datum="CGVD2013")
    window = TemporalWindow(start_s=BASE_TIME_S - 86_400.0, end_s=BASE_TIME_S)
    aoi = AOIMetadata(
        aoi_id="can-aoi-001",
        name="Canadian layered geo-AI fixture AOI",
        country="Canada",
        region="Atlantic test range",
        bounds=BoundingBox(
            min_x=1_500_000.0,
            min_y=-200_000.0,
            max_x=1_501_000.0,
            max_y=-199_000.0,
            crs=crs,
        ),
        temporal_window=window,
        crs=crs,
        metadata={"usage": "unclassified schema fixture"},
    )
    rasters = (
        RasterLayer(
            layer_id="dem-001",
            layer_type="dem",
            uri="s3://example-unclassified/dem/can-aoi-001.tif",
            grid=_grid(crs, 0.25),
            temporal_window=window,
            bands=("elevation_m",),
            dtype="float32",
            provenance=_provenance("dem-public"),
            data_age=_age(365 * 24 * 60 * 60),
            nodata=-9999.0,
            mask_ids=("valid-mask-001",),
            uncertainty_ids=("dem-uncertainty-001",),
        ),
        RasterLayer(
            layer_id="ice-001",
            layer_type="ice",
            uri="s3://example-unclassified/ice/can-aoi-001.tif",
            grid=_grid(crs, 0.25),
            temporal_window=window,
            bands=("ice_probability",),
            dtype="float32",
            provenance=_provenance("ice-public"),
            data_age=_age(10 * 60 * 60),
        ),
        RasterLayer(
            layer_id="weather-001",
            layer_type="weather",
            uri="s3://example-unclassified/weather/can-aoi-001.tif",
            grid=_grid(crs, 0.25),
            temporal_window=window,
            bands=("wind_u_mps", "wind_v_mps", "temperature_c"),
            dtype="float32",
            provenance=_provenance("weather-public"),
            data_age=_age(60 * 60),
        ),
        RasterLayer(
            layer_id="land-cover-001",
            layer_type="land_cover",
            uri="s3://example-unclassified/land-cover/can-aoi-001.tif",
            grid=_grid(crs, 0.25),
            temporal_window=window,
            bands=("land_cover_class",),
            dtype="uint16",
            provenance=_provenance("land-cover-public"),
            data_age=_age(365 * 24 * 60 * 60),
        ),
    )
    hydro = VectorLayer(
        layer_id="hydro-001",
        layer_type="hydro",
        uri="s3://example-unclassified/hydro/can-aoi-001.geojson",
        geometry_type="LineString",
        crs=crs,
        temporal_window=window,
        attributes=("feature_type", "flow_direction"),
        provenance=_provenance("hydro-public"),
        data_age=_age(365 * 24 * 60 * 60),
    )
    uav = UAVObservation(
        observation_id="uav-frame-001",
        timestamp_s=BASE_TIME_S - 30.0,
        platform_id="uav-public-test-01",
        frame_uri="s3://example-unclassified/uav/frame-001.jpg",
        crs=crs,
        pose={"x_m": 1_500_050.0, "y_m": -199_950.0, "altitude_m": 120.0},
        telemetry={"battery_percent": 82.0, "heading_deg": 91.0},
        camera={"sensor": "rgb", "width_px": 1920, "height_px": 1080},
        provenance=_provenance("uav-operator", "uav_frame_capture"),
        data_age=_age(30.0),
        environment={"weather_layer_id": "weather-001"},
        model_inputs={"state": (1.0, 2.0, 3.0)},
    )
    ortho = OrthomosaicTile(
        tile_id="ortho-001",
        uri="s3://example-unclassified/ortho/tile-001.tif",
        grid=_grid(crs, 0.25),
        temporal_window=window,
        bands=("red", "green", "blue"),
        provenance=_provenance("orthomosaic-public"),
        data_age=_age(90 * 24 * 60 * 60),
    )
    mask = MaskLayer(
        mask_id="valid-mask-001",
        mask_type="valid_data",
        uri="s3://example-unclassified/masks/valid-mask-001.tif",
        grid=_grid(crs, 0.25),
        applies_to=("dem-001", "ortho-001"),
        valid_values=(0, 1),
        provenance=_provenance("mask-public"),
        data_age=_age(30.0),
    )
    uncertainty = UncertaintyLayer(
        uncertainty_id="dem-uncertainty-001",
        uncertainty_type="vertical_rmse",
        uri="s3://example-unclassified/uncertainty/dem-rmse.tif",
        grid=_grid(crs, 0.25),
        applies_to=("dem-001",),
        value_units="m",
        provenance=_provenance("uncertainty-public"),
        data_age=_age(365 * 24 * 60 * 60),
    )
    label = LabelTarget(
        target_id="label-safe-route-001",
        target_type="navigation_target",
        applies_to=("uav-frame-001", "ortho-001"),
        value={"route_quality": "nominal", "target_xy_m": [1_500_100.0, -199_900.0]},
        provenance=_provenance("label-public", "human_review"),
        data_age=_age(60.0),
        temporal_window=window,
        crs=crs,
    )
    trajectory = (
        TrajectoryStep(
            step_id="trajectory-step-001",
            timestamp_s=BASE_TIME_S - 10.0,
            observation_id="uav-frame-001",
            action={"vx_mps": 2.0, "vy_mps": 0.0, "vz_mps": 0.0, "yaw_rate_rps": 0.02},
            state={"x_m": 1_500_050.0, "y_m": -199_950.0, "altitude_m": 120.0},
            provenance=_provenance("trajectory-public", "flight_log"),
            data_age=_age(10.0),
        ),
    )
    binding = LeWMTrainingBinding(
        binding_id="lewm-hdf5-v1",
        keys_to_assets={
            "pixels": ("ortho-001",),
            "action": ("trajectory-step-001",),
            "proprio": ("uav-frame-001",),
            "state": ("dem-001", "hydro-001"),
        },
        model_input_resolution_m=0.25,
    )
    return GeoDatasetManifest(
        dataset_id="canadian-layered-geo-ai-fixture",
        dataset_version="0.1.0",
        aoi=aoi,
        crs=crs,
        temporal_window=window,
        raster_layers=rasters,
        vector_layers=(hydro,),
        uav_observations=(uav,),
        trajectory=trajectory,
        labels=(label,),
        masks=(mask,),
        uncertainty=(uncertainty,),
        orthomosaic_tiles=(ortho,),
        training_bindings=(binding,),
        metadata={"classification": "unclassified"},
    )


class GeospatialDatasetTest(unittest.TestCase):
    def test_sample_manifest_covers_required_canadian_geo_ai_inputs(self):
        manifest = _sample_manifest()

        findings = validate_geospatial_dataset_manifest(manifest)

        self.assertEqual(findings, [])
        self.assertEqual(manifest.aoi.country, "Canada")
        self.assertEqual(
            {layer.layer_type for layer in manifest.raster_layers},
            {"dem", "ice", "weather", "land_cover"},
        )
        self.assertEqual(manifest.vector_layers[0].layer_type, "hydro")
        self.assertEqual(manifest.uav_observations[0].observation_id, "uav-frame-001")
        self.assertEqual(manifest.orthomosaic_tiles[0].tile_id, "ortho-001")
        self.assertIn("pixels", manifest.training_bindings[0].keys_to_assets)

    def test_manifest_record_is_json_safe(self):
        record = _sample_manifest().to_record()

        json.dumps(record, sort_keys=True)
        self.assertEqual(record["schema_version"], "canadian-layered-geo-ai-world-model/v1")
        self.assertEqual(record["crs"]["vertical_datum"], "CGVD2013")

    def test_missing_crs_is_reported_and_rejected(self):
        manifest = _sample_manifest()
        aoi = replace(
            manifest.aoi,
            crs=None,
            bounds=replace(manifest.aoi.bounds, crs=None),
        )
        invalid = replace(manifest, crs=None, aoi=aoi)

        findings = validate_geospatial_dataset_manifest(invalid)

        self.assertIn("missing_crs", {finding.code for finding in findings})
        with self.assertRaises(GeoDatasetValidationError):
            require_valid_geospatial_dataset_manifest(invalid)

    def test_missing_provenance_is_reported_and_rejected(self):
        manifest = _sample_manifest()
        bad_raster = replace(manifest.raster_layers[0], provenance=None)
        invalid = replace(manifest, raster_layers=(bad_raster, *manifest.raster_layers[1:]))

        findings = validate_geospatial_dataset_manifest(invalid)

        self.assertIn("missing_provenance", {finding.code for finding in findings})
        with self.assertRaises(GeoDatasetValidationError):
            require_valid_geospatial_dataset_manifest(invalid)

    def test_blocked_provenance_label_is_rejected(self):
        manifest = _sample_manifest()
        blocked_source = replace(
            manifest.raster_layers[0].provenance,
            classification_label="secret",
        )
        bad_raster = replace(manifest.raster_layers[0], provenance=blocked_source)
        invalid = replace(manifest, raster_layers=(bad_raster, *manifest.raster_layers[1:]))

        findings = validate_geospatial_dataset_manifest(invalid)

        self.assertIn("blocked_provenance_label", {finding.code for finding in findings})
        with self.assertRaises(GeoDatasetValidationError):
            require_valid_geospatial_dataset_manifest(invalid)

    def test_stale_timestamps_are_flagged_and_can_be_enforced(self):
        manifest = _sample_manifest()
        stale_weather = replace(
            manifest.raster_layers[2],
            data_age=DataAge(
                observed_at_s=BASE_TIME_S - 7 * 60 * 60,
                ingested_at_s=BASE_TIME_S - 6 * 60 * 60,
            ),
        )
        stale_uav = replace(
            manifest.uav_observations[0],
            data_age=DataAge(
                observed_at_s=BASE_TIME_S - 31 * 24 * 60 * 60,
                ingested_at_s=BASE_TIME_S - 31 * 24 * 60 * 60 + 60,
            ),
        )
        stale = replace(
            manifest,
            raster_layers=(
                manifest.raster_layers[0],
                manifest.raster_layers[1],
                stale_weather,
                manifest.raster_layers[3],
            ),
            uav_observations=(stale_uav,),
        )

        findings = validate_geospatial_dataset_manifest(stale, reference_time_s=BASE_TIME_S)
        stale_findings = [finding for finding in findings if finding.code == "stale_timestamp"]

        self.assertTrue(stale_findings)
        self.assertTrue(all(finding.severity == "warning" for finding in stale_findings))
        self.assertTrue(any("raster_layers[2]" in finding.path for finding in stale_findings))
        self.assertTrue(any("uav_observations[0]" in finding.path for finding in stale_findings))
        with self.assertRaises(GeoDatasetValidationError):
            require_valid_geospatial_dataset_manifest(
                stale,
                reference_time_s=BASE_TIME_S,
                enforce_freshness=True,
            )

    def test_incompatible_model_input_resolution_is_rejected(self):
        manifest = _sample_manifest()
        bad_binding = replace(manifest.training_bindings[0], model_input_resolution_m=1.0)
        invalid = replace(manifest, training_bindings=(bad_binding,))

        findings = validate_geospatial_dataset_manifest(invalid)

        self.assertIn("incompatible_resolution", {finding.code for finding in findings})
        with self.assertRaises(GeoDatasetValidationError):
            require_valid_geospatial_dataset_manifest(invalid)

    def test_uav_observation_converts_to_drone_observation(self):
        observation = _sample_manifest().uav_observations[0].to_drone_observation(
            pixels=_Pixels(),
            model_inputs={"pixels": object(), "action": (0.0, 0.0, 0.0, 0.0)},
        )

        summary = observation.summary()

        self.assertIsInstance(observation, DroneObservation)
        self.assertEqual(summary["pixel_shape"], [1, 3, 224, 224])
        self.assertEqual(summary["model_input_keys"], ["action", "pixels", "state"])
        self.assertEqual(summary["telemetry"]["frame_uri"], "s3://example-unclassified/uav/frame-001.jpg")
        self.assertEqual(summary["environment"]["observation_id"], "uav-frame-001")


if __name__ == "__main__":
    unittest.main()
