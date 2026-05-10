import unittest
from datetime import datetime, timezone

from lewm_drone import (
    AOIMetadata,
    BoundingBox,
    CanadianGeoWorldModelConfig,
    CRSDefinition,
    DataAge,
    GeoDatasetManifest,
    GeoGrid,
    GeoTrainingDatasetBuildConfig,
    GeoTrainingTileSpec,
    OrthomosaicTile,
    RasterLayer,
    SourceProvenance,
    TemporalWindow,
    TrajectoryStep,
    UAVObservation,
    VectorLayer,
    build_canadian_geo_world_model_foundation,
    canadian_geo_world_model_channel_schema,
    prepare_canadian_geo_world_model_manifest,
    validate_canadian_geo_world_model_manifest,
)


class CanadianGeoWorldModelFoundationTest(unittest.TestCase):
    def test_full_manifest_prepares_canonical_training_binding(self):
        manifest = _full_manifest()

        prepared = prepare_canadian_geo_world_model_manifest(manifest)

        binding = prepared.training_bindings[0]
        self.assertEqual(binding.binding_id, "canadian-layered-geo-ai-world-model-v1")
        self.assertEqual(binding.keys_to_assets["pixels"], ("uav-frame-001",))
        self.assertEqual(binding.keys_to_assets["orthomosaic"], ("ortho-001",))
        self.assertIn("hydro-001", binding.keys_to_assets["geo_layers"])
        self.assertEqual(validate_canadian_geo_world_model_manifest(prepared), ())

        schema = canadian_geo_world_model_channel_schema(prepared)
        self.assertEqual(
            {channel["modality"] for channel in schema},
            {"elevation", "ice", "weather", "land_cover", "hydro"},
        )

    def test_missing_modality_is_reported(self):
        manifest = _full_manifest()
        incomplete = GeoDatasetManifest(
            **{
                **manifest.__dict__,
                "vector_layers": (),
                "training_bindings": (),
            }
        )

        findings = validate_canadian_geo_world_model_manifest(incomplete)

        self.assertIn("missing_hydro", {finding.code for finding in findings})

    def test_foundation_build_produces_training_plan(self):
        foundation = build_canadian_geo_world_model_foundation(
            [_full_manifest()],
            foundation_config=CanadianGeoWorldModelConfig(),
            training_config=GeoTrainingDatasetBuildConfig(
                tile=GeoTrainingTileSpec(width_px=10, height_px=10, resolution_m=1.0),
                split_fractions={"train": 1.0, "val": 0.0, "test": 0.0},
                output_format="metadata-jsonl",
            ),
            dry_run=True,
        )

        self.assertTrue(foundation.ready)
        self.assertIsNotNone(foundation.training_plan)
        self.assertEqual(foundation.training_plan.split_counts["train"], 4)


def _full_manifest() -> GeoDatasetManifest:
    timestamp_s = datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp()
    crs = CRSDefinition(horizontal="EPSG:3978", vertical_datum="CGVD2013")
    window = TemporalWindow(start_s=timestamp_s - 60.0, end_s=timestamp_s + 60.0)
    grid = GeoGrid(
        width_px=20,
        height_px=20,
        resolution_m=1.0,
        origin_x=0.0,
        origin_y=0.0,
        crs=crs,
    )
    return GeoDatasetManifest(
        dataset_id="full-canadian-geo-world-model",
        dataset_version="v1",
        aoi=AOIMetadata(
            aoi_id="aoi-full",
            name="aoi-full",
            country="Canada",
            region="test",
            bounds=BoundingBox(0.0, 0.0, 20.0, 20.0, crs=crs),
            temporal_window=window,
            crs=crs,
        ),
        crs=crs,
        temporal_window=window,
        raster_layers=(
            _raster("dem-001", "dem", ("elevation_m",), grid, window, timestamp_s),
            _raster("ice-001", "ice", ("ice_probability",), grid, window, timestamp_s),
            _raster("weather-001", "weather", ("wind_u", "wind_v"), grid, window, timestamp_s),
            _raster("land-cover-001", "land_cover", ("class",), grid, window, timestamp_s),
        ),
        vector_layers=(
            VectorLayer(
                layer_id="hydro-001",
                layer_type="hydro",
                uri="memory://hydro-001",
                geometry_type="LineString",
                crs=crs,
                temporal_window=window,
                attributes=("feature_type",),
                provenance=_provenance("hydro-source"),
                data_age=_age(timestamp_s),
            ),
        ),
        uav_observations=(
            UAVObservation(
                observation_id="uav-frame-001",
                timestamp_s=timestamp_s,
                platform_id="uav-001",
                frame_uri="memory://uav-frame-001",
                crs=crs,
                pose={"x_m": 1.0, "y_m": 1.0, "z_m": 120.0},
                telemetry={},
                camera={"channels": 3},
                provenance=_provenance("uav-source"),
                data_age=_age(timestamp_s),
            ),
        ),
        trajectory=(
            TrajectoryStep(
                step_id="step-001",
                timestamp_s=timestamp_s,
                action={"vx_mps": 1.0},
                state={"mission_id": "mission-full"},
                observation_id="uav-frame-001",
                provenance=_provenance("trajectory-source"),
                data_age=_age(timestamp_s),
            ),
        ),
        labels=(),
        masks=(),
        uncertainty=(),
        orthomosaic_tiles=(
            OrthomosaicTile(
                tile_id="ortho-001",
                uri="memory://ortho-001",
                grid=grid,
                temporal_window=window,
                bands=("red", "green", "blue"),
                provenance=_provenance("ortho-source"),
                data_age=_age(timestamp_s),
            ),
        ),
    )


def _raster(
    layer_id: str,
    layer_type: str,
    bands: tuple[str, ...],
    grid: GeoGrid,
    window: TemporalWindow,
    timestamp_s: float,
) -> RasterLayer:
    return RasterLayer(
        layer_id=layer_id,
        layer_type=layer_type,
        uri=f"memory://{layer_id}",
        grid=grid,
        temporal_window=window,
        bands=bands,
        dtype="float32",
        provenance=_provenance(f"{layer_id}-source"),
        data_age=_age(timestamp_s),
    )


def _age(timestamp_s: float) -> DataAge:
    return DataAge(observed_at_s=timestamp_s, ingested_at_s=timestamp_s + 1.0)


def _provenance(source_id: str) -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        source_name=source_id,
        collection_method="synthetic",
        source_uri=f"memory://{source_id}",
    )


if __name__ == "__main__":
    unittest.main()
