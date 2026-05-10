import json
import shutil
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from lewm_drone import (
    AOIMetadata,
    BoundingBox,
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
    UncertaintyLayer,
    VectorLayer,
    build_geo_training_dataset,
    read_canonical_geospatial_manifest,
)


class GeoTrainingDatasetBuilderTest(unittest.TestCase):
    def setUp(self):
        self.workdir = Path("artifacts") / "test_geo_training_dataset"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_dry_run_plans_train_val_test_without_writes_or_leakage(self):
        manifests = (
            _synthetic_manifest(
                "geo-a",
                "aoi-a",
                x_offset=0.0,
                timestamp_s=_ts(2026, 1, 15),
                mission_id="mission-a",
                sensor_id="sensor-a",
            ),
            _synthetic_manifest(
                "geo-b",
                "aoi-b",
                x_offset=100.0,
                timestamp_s=_ts(2026, 4, 15),
                mission_id="mission-b",
                sensor_id="sensor-b",
            ),
            _synthetic_manifest(
                "geo-c",
                "aoi-c",
                x_offset=200.0,
                timestamp_s=_ts(2026, 7, 15),
                mission_id="mission-c",
                sensor_id="sensor-c",
            ),
        )
        output_path = self.workdir / "geo_samples.h5"

        plan = build_geo_training_dataset(
            manifests,
            config=GeoTrainingDatasetBuildConfig(
                tile=GeoTrainingTileSpec(width_px=10, height_px=10, resolution_m=1.0),
                split_fractions={"train": 1.0, "val": 1.0, "test": 1.0},
                leakage_guard_dimensions=("aoi", "time", "mission"),
                seed=13,
            ),
            output_path=output_path,
            dry_run=True,
        )

        self.assertTrue(plan.dry_run)
        self.assertEqual(sum(plan.split_counts.values()), len(plan.samples))
        self.assertEqual(set(plan.split_counts), {"train", "val", "test"})
        self.assertTrue(all(count > 0 for count in plan.split_counts.values()))
        self.assertFalse(any(Path(path).exists() for path in plan.output_paths.values()))

        sample = plan.samples[0]
        for key in (
            "pixels",
            "action",
            "proprioception",
            "geo_layers",
            "uncertainty",
            "timestamps",
            "source_ids",
        ):
            self.assertIn(key, sample.training_keys)

        self.assertFalse(_overlaps_across_splits(plan.samples, "aoi"))
        self.assertFalse(_overlaps_across_splits(plan.samples, "time"))
        self.assertFalse(_overlaps_across_splits(plan.samples, "mission"))

    def test_unavailable_layers_are_reported_with_masks(self):
        timestamp_s = _ts(2026, 6, 1)
        manifest = _synthetic_manifest(
            "geo-mask",
            "aoi-mask",
            x_offset=0.0,
            timestamp_s=timestamp_s,
            mission_id="mission-mask",
            sensor_id="sensor-mask",
            extra_stale_layer=True,
        )

        plan = build_geo_training_dataset(
            [manifest],
            config=GeoTrainingDatasetBuildConfig(
                tile=GeoTrainingTileSpec(width_px=10, height_px=10, resolution_m=1.0),
                split_fractions={"train": 1.0, "val": 0.0, "test": 0.0},
                seed=3,
            ),
            dry_run=True,
        )

        self.assertEqual(len(plan.samples), 4)
        sample = plan.samples[0]
        self.assertEqual(sample.training_keys["geo_layer_available"], [1, 0])
        self.assertEqual(sample.training_keys["unavailable_layer_mask"], [0, 1])
        self.assertIn(
            "timestamp_outside_layer_window",
            {alignment.reason for alignment in sample.alignments},
        )

    def test_reads_canonical_json_manifest_record(self):
        manifest = _synthetic_manifest(
            "geo-json",
            "aoi-json",
            x_offset=0.0,
            timestamp_s=_ts(2026, 9, 1),
            mission_id="mission-json",
            sensor_id="sensor-json",
        )
        manifest_path = self.workdir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_record(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        loaded = read_canonical_geospatial_manifest(manifest_path)

        self.assertEqual(loaded.dataset_id, manifest.dataset_id)
        self.assertEqual(loaded.aoi.aoi_id, manifest.aoi.aoi_id)
        self.assertEqual(loaded.orthomosaic_tiles[0].tile_id, "ortho-geo-json")

    def test_sparse_uav_and_hydro_vector_are_planned_as_training_layers(self):
        timestamp_s = _ts(2026, 10, 1)
        manifest = _synthetic_manifest(
            "geo-uav-hydro",
            "aoi-uav-hydro",
            x_offset=0.0,
            timestamp_s=timestamp_s,
            mission_id="mission-uav-hydro",
            sensor_id="sensor-uav-hydro",
        )
        hydro = VectorLayer(
            layer_id="hydro-vector",
            layer_type="hydro",
            uri="memory://hydro-vector",
            geometry_type="LineString",
            crs=manifest.crs,
            temporal_window=manifest.temporal_window,
            attributes=("feature_type",),
            provenance=_provenance("hydro-vector-source"),
            data_age=_age(timestamp_s),
        )
        uav = UAVObservation(
            observation_id="uav-frame-001",
            timestamp_s=timestamp_s,
            platform_id="sensor-uav-hydro",
            frame_uri="memory://uav-frame-001",
            crs=manifest.crs,
            pose={"x_m": 1.0, "y_m": 1.0, "z_m": 120.0},
            telemetry={"battery_percent": 90.0},
            camera={"channels": 3, "width_px": 640, "height_px": 480},
            provenance=_provenance("uav-frame-source"),
            data_age=_age(timestamp_s),
        )
        trajectory = replace(
            manifest.trajectory[0],
            observation_id="uav-frame-001",
        )
        manifest = replace(
            manifest,
            vector_layers=(hydro,),
            uav_observations=(uav,),
            trajectory=(trajectory,),
        )

        plan = build_geo_training_dataset(
            [manifest],
            config=GeoTrainingDatasetBuildConfig(
                tile=GeoTrainingTileSpec(width_px=10, height_px=10, resolution_m=1.0),
                split_fractions={"train": 1.0, "val": 0.0, "test": 0.0},
            ),
            dry_run=True,
        )

        sample = plan.samples[0]
        self.assertEqual(sample.training_keys["pixels"], ["uav-frame-001"])
        self.assertEqual(sample.training_keys["sparse_uav_frames"], ["uav-frame-001"])
        self.assertIn("hydro-vector", sample.training_keys["geo_layers"])
        self.assertEqual(sample.training_keys["vector_layers"], ["hydro-vector"])
        self.assertIn(
            "time_aligned_sparse_observation",
            {alignment.reason for alignment in sample.alignments},
        )


def _synthetic_manifest(
    dataset_id: str,
    aoi_id: str,
    *,
    x_offset: float,
    timestamp_s: float,
    mission_id: str,
    sensor_id: str,
    extra_stale_layer: bool = False,
) -> GeoDatasetManifest:
    crs = CRSDefinition(horizontal="3978", vertical_datum="CGVD2013")
    window = TemporalWindow(start_s=timestamp_s - 60.0, end_s=timestamp_s + 60.0)
    bounds = BoundingBox(
        min_x=x_offset,
        min_y=0.0,
        max_x=x_offset + 20.0,
        max_y=20.0,
        crs=crs,
    )
    grid = GeoGrid(
        width_px=20,
        height_px=20,
        resolution_m=1.0,
        origin_x=x_offset,
        origin_y=0.0,
        crs=crs,
    )
    stale_window = TemporalWindow(start_s=timestamp_s - 3600.0, end_s=timestamp_s - 1800.0)
    raster_layers = [
        RasterLayer(
            layer_id=f"dem-{dataset_id}",
            layer_type="dem",
            uri=f"memory://{dataset_id}/dem",
            grid=grid,
            temporal_window=window,
            bands=("elevation",),
            dtype="float32",
            provenance=_provenance(f"dem-source-{dataset_id}"),
            data_age=_age(timestamp_s),
        )
    ]
    if extra_stale_layer:
        raster_layers.append(
            RasterLayer(
                layer_id=f"weather-stale-{dataset_id}",
                layer_type="weather",
                uri=f"memory://{dataset_id}/weather-stale",
                grid=grid,
                temporal_window=stale_window,
                bands=("wind",),
                dtype="float32",
                provenance=_provenance(f"weather-source-{dataset_id}"),
                data_age=_age(timestamp_s),
            )
        )

    return GeoDatasetManifest(
        dataset_id=dataset_id,
        dataset_version="v1",
        aoi=AOIMetadata(
            aoi_id=aoi_id,
            name=aoi_id,
            country="CA",
            region="synthetic",
            bounds=bounds,
            temporal_window=window,
            crs=crs,
        ),
        crs=crs,
        temporal_window=window,
        raster_layers=tuple(raster_layers),
        vector_layers=(),
        uav_observations=(),
        trajectory=(
            TrajectoryStep(
                step_id=f"step-{dataset_id}",
                timestamp_s=timestamp_s,
                action={
                    "vx_mps": 1.0,
                    "vy_mps": 0.0,
                    "vz_mps": 0.0,
                    "yaw_rate_rps": 0.1,
                },
                state={
                    "x_m": x_offset,
                    "y_m": 1.0,
                    "z_m": 120.0,
                    "mission_id": mission_id,
                },
                provenance=_provenance(f"trajectory-source-{dataset_id}"),
                data_age=_age(timestamp_s),
            ),
        ),
        labels=(),
        masks=(),
        uncertainty=(
            UncertaintyLayer(
                uncertainty_id=f"uncertainty-{dataset_id}",
                uncertainty_type="raster_quality",
                uri=f"memory://{dataset_id}/uncertainty",
                grid=grid,
                applies_to=(f"dem-{dataset_id}",),
                value_units="probability",
                provenance=_provenance(f"uncertainty-source-{dataset_id}"),
                data_age=_age(timestamp_s),
            ),
        ),
        orthomosaic_tiles=(
            OrthomosaicTile(
                tile_id=f"ortho-{dataset_id}",
                uri=f"memory://{dataset_id}/ortho",
                grid=grid,
                temporal_window=window,
                bands=("red", "green", "blue"),
                provenance=_provenance(sensor_id),
                data_age=_age(timestamp_s),
            ),
        ),
        metadata={"mission_id": mission_id, "sensor_id": sensor_id},
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


def _ts(year: int, month: int, day: int) -> float:
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def _overlaps_across_splits(samples, dimension):
    values = {}
    for sample in samples:
        value = sample.split_axes[dimension]
        values.setdefault(value, set()).add(sample.split)
    return {value for value, splits in values.items() if len(splits) > 1}


if __name__ == "__main__":
    unittest.main()
