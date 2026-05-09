import importlib.util
import json
import shutil
import unittest
from pathlib import Path

from lewm_drone import (
    CanadianGeoIngestionPipeline,
    GeoIngestionRequest,
    RemoteSourceRef,
    SecurityGateError,
)

HAS_GEO_DEPS = all(
    importlib.util.find_spec(module_name)
    for module_name in ("numpy", "rasterio", "pyogrio", "pyproj")
)


@unittest.skipUnless(HAS_GEO_DEPS, "geospatial parser dependencies are not installed")
class CanadianGeoIngestionPipelineTest(unittest.TestCase):
    def setUp(self):
        self.workdir = Path("artifacts") / "test_geospatial_ingestion"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.raster_path = self.workdir / "tiny_dem.tif"
        self.vector_path = self.workdir / "tiny_hydro.geojson"
        self._write_tiny_geotiff(self.raster_path)
        self._write_tiny_geojson(self.vector_path)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_synthetic_ingestion_run_emits_valid_dataset_manifest(self):
        output_path = self.workdir / "dataset_manifest.json"
        manifest = CanadianGeoIngestionPipeline().create_dataset_manifest(
            (
                GeoIngestionRequest(
                    provider_id="nrcan_elevation",
                    layer_id="dem-001",
                    local_path=str(self.raster_path),
                    acquisition_time_s=1000.0,
                    remote_sources=(
                        RemoteSourceRef(
                            source_url="https://example.test/nrcan/dem",
                            collection_id="synthetic-dem",
                            product_id="tiny-dem",
                        ),
                    ),
                ),
                GeoIngestionRequest(
                    provider_id="canadian_hydro",
                    layer_id="hydro-001",
                    local_path=str(self.vector_path),
                    acquisition_time_s=1001.0,
                    nominal_resolution_m=30.0,
                    source_scale="synthetic-test-scale",
                ),
            ),
            dataset_id="synthetic-canadian-geo",
            output_path=output_path,
        )

        self.assertEqual(manifest.dataset_id, "synthetic-canadian-geo")
        self.assertTrue(manifest.manifest_hash)
        self.assertEqual(len(manifest.raster_layers), 1)
        self.assertEqual(len(manifest.vector_layers), 1)
        self.assertTrue(output_path.exists())

        record = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(record["manifest_hash"], manifest.manifest_hash)
        self.assertEqual(record["metadata"]["ingestion_layers"][0]["checksum_sha256"], manifest.raster_layers[0].metadata["ingestion"]["checksum_sha256"])

    def test_every_adapter_normalizes_to_layer_manifest_with_provenance(self):
        requests = (
            GeoIngestionRequest(
                provider_id="nrcan_elevation",
                layer_id="nrcan-dem",
                local_path=str(self.raster_path),
                acquisition_time_s=1000.0,
            ),
            GeoIngestionRequest(
                provider_id="eccc_msc_geomet",
                layer_id="eccc-weather",
                local_path=str(self.raster_path),
                layer_type="weather",
                acquisition_time_s=1001.0,
            ),
            GeoIngestionRequest(
                provider_id="canadian_ice_service",
                layer_id="cis-ice",
                local_path=str(self.vector_path),
                asset_kind="vector",
                acquisition_time_s=1002.0,
                nominal_resolution_m=100.0,
            ),
            GeoIngestionRequest(
                provider_id="canadian_hydro",
                layer_id="hydro",
                local_path=str(self.vector_path),
                acquisition_time_s=1003.0,
                nominal_resolution_m=50.0,
            ),
            GeoIngestionRequest(
                provider_id="canadian_land_cover",
                layer_id="land-cover",
                local_path=str(self.raster_path),
                acquisition_time_s=1004.0,
            ),
            GeoIngestionRequest(
                provider_id="uav_orthomosaic",
                layer_id="ortho",
                local_path=str(self.raster_path),
                acquisition_time_s=1005.0,
            ),
        )
        pipeline = CanadianGeoIngestionPipeline()

        layers = pipeline.ingest_layers(requests)

        self.assertEqual(
            pipeline.provider_ids(),
            (
                "canadian_hydro",
                "canadian_ice_service",
                "canadian_land_cover",
                "eccc_msc_geomet",
                "nrcan_elevation",
                "uav_orthomosaic",
            ),
        )
        self.assertEqual(len(layers), 6)
        for layer in layers:
            self.assertTrue(layer.source_name)
            self.assertTrue(layer.license_use_caveat)
            self.assertTrue(layer.acquisition_time_s)
            self.assertTrue(layer.processed_time_s)
            self.assertGreaterEqual(layer.processing_duration_s, 0.0)
            self.assertTrue(layer.crs.horizontal)
            self.assertTrue(layer.resolution)
            self.assertTrue(layer.checksum_sha256)
            self.assertGreater(layer.size_bytes, 0)
            self.assertEqual(layer.classification_label, "unclassified")
            self.assertIn("provider_id", layer.metadata)

    def test_blocked_classification_fails_before_manifest_entry(self):
        with self.assertRaises(SecurityGateError):
            CanadianGeoIngestionPipeline().create_dataset_manifest(
                (
                    GeoIngestionRequest(
                        provider_id="nrcan_elevation",
                        layer_id="blocked",
                        local_path=str(self.workdir / "does_not_exist.tif"),
                        classification_label="classified",
                    ),
                ),
                dataset_id="blocked-dataset",
            )

    def _write_tiny_geotiff(self, path: Path) -> None:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:3978",
            transform=from_origin(1000.0, 2000.0, 30.0, 30.0),
            nodata=0,
        ) as dataset:
            dataset.write(data, 1)

    def _write_tiny_geojson(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"name": "test-stream"},
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[-75.0, 45.0], [-74.99, 45.01]],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
