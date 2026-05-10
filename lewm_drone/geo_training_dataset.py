"""Build training-ready layered geo-AI samples from canonical manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .geospatial_dataset import (
    AOIMetadata,
    BoundingBox,
    CRSDefinition,
    DataAge,
    GEOSPATIAL_DATASET_SCHEMA_VERSION,
    GeoDatasetManifest,
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
)
from .interfaces import JsonValue, to_jsonable

DEFAULT_GEO_TRAINING_KEYS = (
    "pixels",
    "action",
    "proprio",
    "proprioception",
    "geo_layers",
    "uncertainty",
    "timestamps",
    "source_ids",
    "geo_layer_available",
    "unavailable_layer_mask",
)


@dataclass(frozen=True)
class GeoTrainingTileSpec:
    """Fixed-size geospatial tile geometry for sample generation."""

    width_px: int
    height_px: int
    resolution_m: float | None = None
    stride_x_px: int | None = None
    stride_y_px: int | None = None
    include_partial_tiles: bool = False


@dataclass(frozen=True)
class GeoTrainingDatasetBuildConfig:
    """Controls tiling, alignment, split assignment, and output format."""

    tile: GeoTrainingTileSpec
    split_fractions: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    split_dimensions: tuple[str, ...] = ("aoi", "season", "time", "sensor", "mission")
    leakage_guard_dimensions: tuple[str, ...] = ("aoi", "mission")
    seed: int = 0
    time_bucket_s: float = 30 * 24 * 60 * 60
    output_format: str = "hdf5"
    action_order: tuple[str, ...] = ("vx_mps", "vy_mps", "vz_mps", "yaw_rate_rps")
    proprio_order: tuple[str, ...] = (
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "vx_mps",
        "vy_mps",
        "vz_mps",
    )
    training_keys: tuple[str, ...] = DEFAULT_GEO_TRAINING_KEYS
    alignment_resolution_tolerance_m: float = 0.001


@dataclass(frozen=True)
class GeoLayerAlignment:
    """Per-sample alignment and availability for one manifest asset."""

    asset_id: str
    training_key: str
    layer_type: str
    source_id: str | None
    available: bool
    unavailable_mask_value: int
    reason: str
    resolution_m: float | None
    timestamp_s: float
    tile_extent: BoundingBox

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class GeoTrainingSampleMetadata:
    """One fixed-size sample planned for a split output."""

    sample_id: str
    dataset_id: str
    dataset_version: str
    aoi_id: str
    mission_id: str
    sensor_id: str
    season: str
    time_bucket: str
    split: str
    leakage_group: str
    tile_id: str
    tile_row: int
    tile_col: int
    tile_extent: BoundingBox
    timestamp_s: float
    step_id: str
    training_keys: Mapping[str, JsonValue]
    alignments: tuple[GeoLayerAlignment, ...]
    source_ids: tuple[str, ...]
    split_axes: Mapping[str, str]

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class GeoTrainingDatasetPlan:
    """Dry-run or materialized output plan for a geo training dataset build."""

    dataset_id: str
    dry_run: bool
    output_format: str
    output_paths: Mapping[str, str]
    split_counts: Mapping[str, int]
    split_dimensions: tuple[str, ...]
    leakage_guard_dimensions: tuple[str, ...]
    samples: tuple[GeoTrainingSampleMetadata, ...]
    warnings: tuple[str, ...] = ()

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]


class GeoTrainingDatasetBuilder:
    """Convert validated geospatial manifests into split training samples.

    The builder plans and validates samples using only canonical manifest
    metadata. HDF5 materialization is available when `h5py` and `numpy` are
    installed; raster-reader integration can be added behind the same plan.
    """

    def __init__(self, config: GeoTrainingDatasetBuildConfig):
        self.config = config

    def build(
        self,
        manifests: Iterable[GeoDatasetManifest | Mapping[str, Any] | str | Path],
        *,
        output_path: str | Path | None = None,
        dry_run: bool = True,
    ) -> GeoTrainingDatasetPlan:
        manifest_tuple = tuple(read_canonical_geospatial_manifest(item) for item in manifests)
        if not manifest_tuple:
            raise ValueError("at_least_one_manifest_required")

        validated = tuple(require_valid_geospatial_dataset_manifest(item) for item in manifest_tuple)
        samples = self._plan_samples(validated)
        samples = self._assign_splits(samples)
        output_paths = self._planned_output_paths(validated, output_path)
        split_counts = _count_by_split(samples, self.config.split_fractions)
        warnings = self._warnings_for(samples)

        plan = GeoTrainingDatasetPlan(
            dataset_id=_dataset_family(validated),
            dry_run=dry_run,
            output_format=self.config.output_format,
            output_paths=output_paths,
            split_counts=split_counts,
            split_dimensions=self.config.split_dimensions,
            leakage_guard_dimensions=self.config.leakage_guard_dimensions,
            samples=samples,
            warnings=tuple(warnings),
        )
        if not dry_run:
            self._write_outputs(plan, validated)
        return plan

    def _plan_samples(
        self,
        manifests: Sequence[GeoDatasetManifest],
    ) -> tuple[GeoTrainingSampleMetadata, ...]:
        samples: list[GeoTrainingSampleMetadata] = []
        for manifest in manifests:
            resolution_m = self._target_resolution_m(manifest)
            tiles = tuple(self._tile_aoi(manifest, resolution_m))
            steps = tuple(manifest.trajectory) or (
                TrajectoryStep(
                    step_id="synthetic_midpoint",
                    timestamp_s=_temporal_midpoint(manifest.temporal_window),
                    action={},
                    state={},
                ),
            )
            for tile_id, tile_row, tile_col, tile_extent in tiles:
                for step_index, step in enumerate(steps):
                    sample = self._sample_metadata(
                        manifest,
                        tile_id=tile_id,
                        tile_row=tile_row,
                        tile_col=tile_col,
                        tile_extent=tile_extent,
                        step=step,
                        step_index=step_index,
                        resolution_m=resolution_m,
                    )
                    samples.append(sample)
        return tuple(samples)

    def _target_resolution_m(self, manifest: GeoDatasetManifest) -> float:
        if self.config.tile.resolution_m is not None:
            return self.config.tile.resolution_m
        for binding in manifest.training_bindings:
            if binding.model_input_resolution_m is not None:
                return binding.model_input_resolution_m
        for tile in manifest.orthomosaic_tiles:
            return tile.grid.resolution_m
        for layer in manifest.raster_layers:
            return layer.grid.resolution_m
        for mask in manifest.masks:
            return mask.grid.resolution_m
        raise ValueError(f"cannot_infer_training_resolution:{manifest.dataset_id}")

    def _tile_aoi(
        self,
        manifest: GeoDatasetManifest,
        resolution_m: float,
    ) -> Iterable[tuple[str, int, int, BoundingBox]]:
        bounds = manifest.aoi.bounds
        tile_width_m = self.config.tile.width_px * resolution_m
        tile_height_m = self.config.tile.height_px * resolution_m
        stride_x_m = (self.config.tile.stride_x_px or self.config.tile.width_px) * resolution_m
        stride_y_m = (self.config.tile.stride_y_px or self.config.tile.height_px) * resolution_m
        if tile_width_m <= 0 or tile_height_m <= 0:
            raise ValueError("tile_dimensions_must_be_positive")
        if stride_x_m <= 0 or stride_y_m <= 0:
            raise ValueError("tile_stride_must_be_positive")

        x_values = _tile_starts(bounds.min_x, bounds.max_x, tile_width_m, stride_x_m, self.config.tile.include_partial_tiles)
        y_values = _tile_starts(bounds.min_y, bounds.max_y, tile_height_m, stride_y_m, self.config.tile.include_partial_tiles)
        for tile_row, min_y in enumerate(y_values):
            for tile_col, min_x in enumerate(x_values):
                max_x = min(min_x + tile_width_m, bounds.max_x)
                max_y = min(min_y + tile_height_m, bounds.max_y)
                tile_id = f"{manifest.aoi.aoi_id}_r{tile_row:04d}_c{tile_col:04d}"
                yield (
                    tile_id,
                    tile_row,
                    tile_col,
                    BoundingBox(
                        min_x=min_x,
                        min_y=min_y,
                        max_x=max_x,
                        max_y=max_y,
                        crs=bounds.crs or manifest.aoi.crs or manifest.crs,
                    ),
                )

    def _sample_metadata(
        self,
        manifest: GeoDatasetManifest,
        *,
        tile_id: str,
        tile_row: int,
        tile_col: int,
        tile_extent: BoundingBox,
        step: TrajectoryStep,
        step_index: int,
        resolution_m: float,
    ) -> GeoTrainingSampleMetadata:
        binding = manifest.training_bindings[0] if manifest.training_bindings else None
        pixel_assets = _assets_for_key(binding, "pixels") or tuple(
            tile.tile_id for tile in manifest.orthomosaic_tiles
        )
        geo_assets = _assets_for_key(binding, "geo_layers") or tuple(
            layer.layer_id for layer in manifest.raster_layers
        )
        uncertainty_assets = _assets_for_key(binding, "uncertainty") or tuple(
            layer.uncertainty_id for layer in manifest.uncertainty
        )

        alignments = []
        for asset_id in pixel_assets:
            asset = _asset_by_id(manifest, asset_id)
            alignments.append(
                self._align_asset(
                    manifest,
                    asset=asset,
                    asset_id=asset_id,
                    training_key="pixels",
                    tile_extent=tile_extent,
                    timestamp_s=step.timestamp_s,
                    target_resolution_m=resolution_m,
                    tolerance_m=(binding.resolution_tolerance_m if binding else self.config.alignment_resolution_tolerance_m),
                )
            )
        for asset_id in geo_assets:
            asset = _asset_by_id(manifest, asset_id)
            alignments.append(
                self._align_asset(
                    manifest,
                    asset=asset,
                    asset_id=asset_id,
                    training_key="geo_layers",
                    tile_extent=tile_extent,
                    timestamp_s=step.timestamp_s,
                    target_resolution_m=resolution_m,
                    tolerance_m=self.config.alignment_resolution_tolerance_m,
                )
            )
        for asset_id in uncertainty_assets:
            asset = _asset_by_id(manifest, asset_id)
            alignments.append(
                self._align_asset(
                    manifest,
                    asset=asset,
                    asset_id=asset_id,
                    training_key="uncertainty",
                    tile_extent=tile_extent,
                    timestamp_s=step.timestamp_s,
                    target_resolution_m=resolution_m,
                    tolerance_m=self.config.alignment_resolution_tolerance_m,
                )
            )

        alignments_tuple = tuple(alignments)
        geo_alignments = tuple(item for item in alignments_tuple if item.training_key == "geo_layers")
        source_ids = tuple(
            sorted({item.source_id for item in alignments_tuple if item.available and item.source_id})
        )
        mission_id = _mission_id(manifest)
        sensor_id = _sensor_id(manifest, alignments_tuple)
        season = _season_for(step.timestamp_s)
        time_bucket = _time_bucket_for(step.timestamp_s, self.config.time_bucket_s)
        split_axes = {
            "aoi": manifest.aoi.aoi_id,
            "season": season,
            "time": time_bucket,
            "sensor": sensor_id,
            "mission": mission_id,
        }
        unavailable_mask = tuple(1 if not item.available else 0 for item in geo_alignments)
        available_mask = tuple(1 if item.available else 0 for item in geo_alignments)
        sample_id = _sample_id(
            manifest.dataset_id,
            manifest.dataset_version,
            tile_id,
            step.step_id,
            step_index,
            step.timestamp_s,
        )
        action = tuple(float(step.action.get(field_name, 0.0)) for field_name in self.config.action_order)
        proprioception = tuple(_state_float(step.state, field_name) for field_name in self.config.proprio_order)
        training_keys: dict[str, JsonValue] = {
            "pixels": list(pixel_assets),
            "action": list(action),
            "proprio": list(proprioception),
            "proprioception": list(proprioception),
            "geo_layers": [item.asset_id for item in geo_alignments],
            "uncertainty": list(uncertainty_assets),
            "timestamps": step.timestamp_s,
            "source_ids": list(source_ids),
            "geo_layer_available": list(available_mask),
            "unavailable_layer_mask": list(unavailable_mask),
        }
        return GeoTrainingSampleMetadata(
            sample_id=sample_id,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.dataset_version,
            aoi_id=manifest.aoi.aoi_id,
            mission_id=mission_id,
            sensor_id=sensor_id,
            season=season,
            time_bucket=time_bucket,
            split="unassigned",
            leakage_group="unassigned",
            tile_id=tile_id,
            tile_row=tile_row,
            tile_col=tile_col,
            tile_extent=tile_extent,
            timestamp_s=step.timestamp_s,
            step_id=step.step_id,
            training_keys=training_keys,
            alignments=alignments_tuple,
            source_ids=source_ids,
            split_axes=split_axes,
        )

    def _align_asset(
        self,
        manifest: GeoDatasetManifest,
        *,
        asset: Any | None,
        asset_id: str,
        training_key: str,
        tile_extent: BoundingBox,
        timestamp_s: float,
        target_resolution_m: float,
        tolerance_m: float,
    ) -> GeoLayerAlignment:
        if asset is None:
            return GeoLayerAlignment(
                asset_id=asset_id,
                training_key=training_key,
                layer_type="missing",
                source_id=None,
                available=False,
                unavailable_mask_value=1,
                reason="missing_asset",
                resolution_m=None,
                timestamp_s=timestamp_s,
                tile_extent=tile_extent,
            )

        layer_type = _asset_layer_type(asset)
        provenance = getattr(asset, "provenance", None)
        source_id = provenance.source_id if provenance is not None else None
        grid = getattr(asset, "grid", None)
        if grid is None:
            return GeoLayerAlignment(
                asset_id=asset_id,
                training_key=training_key,
                layer_type=layer_type,
                source_id=source_id,
                available=False,
                unavailable_mask_value=1,
                reason="asset_has_no_grid",
                resolution_m=None,
                timestamp_s=timestamp_s,
                tile_extent=tile_extent,
            )

        asset_crs = grid.crs or manifest.crs
        tile_crs = tile_extent.crs or manifest.crs
        if not _same_crs(asset_crs, tile_crs):
            return _unavailable_alignment(asset_id, training_key, layer_type, source_id, grid, timestamp_s, tile_extent, "crs_mismatch")
        if abs(grid.resolution_m - target_resolution_m) > tolerance_m:
            return _unavailable_alignment(asset_id, training_key, layer_type, source_id, grid, timestamp_s, tile_extent, "resolution_mismatch")
        temporal_window = getattr(asset, "temporal_window", None)
        if temporal_window is not None and not (temporal_window.start_s <= timestamp_s <= temporal_window.end_s):
            return _unavailable_alignment(asset_id, training_key, layer_type, source_id, grid, timestamp_s, tile_extent, "timestamp_outside_layer_window")
        if not _intersects(tile_extent, _grid_extent(grid, asset_crs)):
            return _unavailable_alignment(asset_id, training_key, layer_type, source_id, grid, timestamp_s, tile_extent, "tile_outside_layer_extent")
        return GeoLayerAlignment(
            asset_id=asset_id,
            training_key=training_key,
            layer_type=layer_type,
            source_id=source_id,
            available=True,
            unavailable_mask_value=0,
            reason="aligned",
            resolution_m=grid.resolution_m,
            timestamp_s=timestamp_s,
            tile_extent=tile_extent,
        )

    def _assign_splits(
        self,
        samples: Sequence[GeoTrainingSampleMetadata],
    ) -> tuple[GeoTrainingSampleMetadata, ...]:
        if not samples:
            return ()
        component_ids = _leakage_components(samples, self.config.leakage_guard_dimensions)
        component_to_samples: dict[str, list[GeoTrainingSampleMetadata]] = {}
        for sample in samples:
            component_to_samples.setdefault(component_ids[sample.sample_id], []).append(sample)

        split_for_component = _assign_component_splits(
            component_to_samples,
            self.config.split_fractions,
            self.config.split_dimensions,
            self.config.seed,
        )
        assigned = []
        for sample in samples:
            leakage_group = component_ids[sample.sample_id]
            assigned.append(
                replace(
                    sample,
                    split=split_for_component[leakage_group],
                    leakage_group=leakage_group,
                )
            )
        return tuple(assigned)

    def _planned_output_paths(
        self,
        manifests: Sequence[GeoDatasetManifest],
        output_path: str | Path | None,
    ) -> dict[str, str]:
        suffix = ".h5" if self.config.output_format == "hdf5" else ".jsonl"
        base = Path(output_path) if output_path is not None else Path("artifacts") / "geo_training" / _dataset_family(manifests)
        if base.suffix:
            return {
                split: str(base.with_name(f"{base.stem}_{split}{suffix}"))
                for split in self.config.split_fractions
            }
        return {
            split: str(base / f"{_dataset_family(manifests)}_{split}{suffix}")
            for split in self.config.split_fractions
        }

    def _warnings_for(self, samples: Sequence[GeoTrainingSampleMetadata]) -> list[str]:
        warnings = []
        split_counts = _count_by_split(samples, self.config.split_fractions)
        missing_splits = [
            split
            for split, count in split_counts.items()
            if self.config.split_fractions.get(split, 0.0) > 0.0 and count == 0
        ]
        if missing_splits:
            warnings.append(f"empty_splits:{','.join(missing_splits)}")
        for dimension in self.config.leakage_guard_dimensions:
            overlaps = _split_overlaps(samples, dimension)
            if overlaps:
                warnings.append(f"leakage_guard_overlap:{dimension}:{','.join(sorted(overlaps))}")
        return warnings

    def _write_outputs(
        self,
        plan: GeoTrainingDatasetPlan,
        manifests: Sequence[GeoDatasetManifest],
    ) -> None:
        if self.config.output_format == "metadata-jsonl":
            _write_metadata_jsonl(plan)
            return
        if self.config.output_format != "hdf5":
            raise ValueError(f"unsupported_output_format:{self.config.output_format}")
        _write_hdf5(plan, manifests, self.config)


def build_geo_training_dataset(
    manifests: Iterable[GeoDatasetManifest | Mapping[str, Any] | str | Path],
    *,
    config: GeoTrainingDatasetBuildConfig,
    output_path: str | Path | None = None,
    dry_run: bool = True,
) -> GeoTrainingDatasetPlan:
    """Build or dry-run a layered geo-AI training dataset plan."""

    return GeoTrainingDatasetBuilder(config).build(
        manifests,
        output_path=output_path,
        dry_run=dry_run,
    )


def read_canonical_geospatial_manifest(
    manifest: GeoDatasetManifest | Mapping[str, Any] | str | Path,
) -> GeoDatasetManifest:
    """Read a canonical geospatial manifest dataclass or JSON record."""

    if isinstance(manifest, GeoDatasetManifest):
        return manifest
    if isinstance(manifest, str | Path):
        path = Path(manifest)
        record = json.loads(path.read_text(encoding="utf-8"))
        return _manifest_from_record(record)
    return _manifest_from_record(manifest)


def _manifest_from_record(record: Mapping[str, Any]) -> GeoDatasetManifest:
    return GeoDatasetManifest(
        dataset_id=str(record["dataset_id"]),
        dataset_version=str(record["dataset_version"]),
        aoi=_aoi(record["aoi"]),
        crs=_crs(record.get("crs")),
        temporal_window=_window(record["temporal_window"]),
        raster_layers=tuple(_raster(item) for item in record.get("raster_layers", ())),
        vector_layers=tuple(_vector(item) for item in record.get("vector_layers", ())),
        uav_observations=tuple(_uav_observation(item) for item in record.get("uav_observations", ())),
        trajectory=tuple(_trajectory_step(item) for item in record.get("trajectory", ())),
        labels=tuple(_label(item) for item in record.get("labels", ())),
        masks=tuple(_mask(item) for item in record.get("masks", ())),
        uncertainty=tuple(_uncertainty(item) for item in record.get("uncertainty", ())),
        orthomosaic_tiles=tuple(_orthomosaic_tile(item) for item in record.get("orthomosaic_tiles", ())),
        training_bindings=tuple(_training_binding(item) for item in record.get("training_bindings", ())),
        schema_version=str(record.get("schema_version", GEOSPATIAL_DATASET_SCHEMA_VERSION)),
        metadata=dict(record.get("metadata", {})),
    )


def _crs(record: Mapping[str, Any] | None) -> CRSDefinition | None:
    if record is None:
        return None
    return CRSDefinition(
        horizontal=record.get("horizontal"),
        vertical_datum=record.get("vertical_datum"),
        units=str(record.get("units", "m")),
        authority=str(record.get("authority", "EPSG")),
        epoch=record.get("epoch"),
    )


def _bbox(record: Mapping[str, Any]) -> BoundingBox:
    return BoundingBox(
        min_x=float(record["min_x"]),
        min_y=float(record["min_y"]),
        max_x=float(record["max_x"]),
        max_y=float(record["max_y"]),
        crs=_crs(record.get("crs")),
    )


def _window(record: Mapping[str, Any]) -> TemporalWindow:
    return TemporalWindow(start_s=float(record["start_s"]), end_s=float(record["end_s"]))


def _aoi(record: Mapping[str, Any]) -> AOIMetadata:
    return AOIMetadata(
        aoi_id=str(record["aoi_id"]),
        name=str(record["name"]),
        country=str(record["country"]),
        region=str(record["region"]),
        bounds=_bbox(record["bounds"]),
        temporal_window=_window(record["temporal_window"]),
        crs=_crs(record.get("crs")),
        metadata=dict(record.get("metadata", {})),
    )


def _grid(record: Mapping[str, Any]) -> GeoGrid:
    return GeoGrid(
        width_px=int(record["width_px"]),
        height_px=int(record["height_px"]),
        resolution_m=float(record["resolution_m"]),
        origin_x=float(record["origin_x"]),
        origin_y=float(record["origin_y"]),
        crs=_crs(record.get("crs")),
        tile_matrix=record.get("tile_matrix"),
    )


def _provenance(record: Mapping[str, Any] | None) -> SourceProvenance | None:
    if record is None:
        return None
    return SourceProvenance(
        source_id=str(record["source_id"]),
        source_name=str(record["source_name"]),
        collection_method=str(record["collection_method"]),
        source_uri=str(record["source_uri"]),
        license=str(record.get("license", "unspecified")),
        organization=str(record.get("organization", "")),
        product_version=str(record.get("product_version", "")),
        classification_label=str(record.get("classification_label", "unclassified")),
        notes=str(record.get("notes", "")),
    )


def _data_age(record: Mapping[str, Any]) -> DataAge:
    return DataAge(
        observed_at_s=float(record["observed_at_s"]),
        ingested_at_s=float(record["ingested_at_s"]),
        max_age_s=(None if record.get("max_age_s") is None else float(record["max_age_s"])),
    )


def _raster(record: Mapping[str, Any]) -> RasterLayer:
    return RasterLayer(
        layer_id=str(record["layer_id"]),
        layer_type=str(record["layer_type"]),
        uri=str(record["uri"]),
        grid=_grid(record["grid"]),
        temporal_window=_window(record["temporal_window"]),
        bands=tuple(str(item) for item in record.get("bands", ())),
        dtype=str(record["dtype"]),
        provenance=_provenance(record.get("provenance")),
        data_age=_data_age(record["data_age"]),
        nodata=record.get("nodata"),
        mask_ids=tuple(str(item) for item in record.get("mask_ids", ())),
        uncertainty_ids=tuple(str(item) for item in record.get("uncertainty_ids", ())),
        metadata=dict(record.get("metadata", {})),
    )


def _vector(record: Mapping[str, Any]) -> VectorLayer:
    return VectorLayer(
        layer_id=str(record["layer_id"]),
        layer_type=str(record["layer_type"]),
        uri=str(record["uri"]),
        geometry_type=str(record["geometry_type"]),
        crs=_crs(record.get("crs")),
        temporal_window=_window(record["temporal_window"]),
        attributes=tuple(str(item) for item in record.get("attributes", ())),
        provenance=_provenance(record.get("provenance")),
        data_age=_data_age(record["data_age"]),
        metadata=dict(record.get("metadata", {})),
    )


def _orthomosaic_tile(record: Mapping[str, Any]) -> OrthomosaicTile:
    return OrthomosaicTile(
        tile_id=str(record["tile_id"]),
        uri=str(record["uri"]),
        grid=_grid(record["grid"]),
        temporal_window=_window(record["temporal_window"]),
        bands=tuple(str(item) for item in record.get("bands", ())),
        provenance=_provenance(record.get("provenance")),
        data_age=_data_age(record["data_age"]),
        metadata=dict(record.get("metadata", {})),
    )


def _uav_observation(record: Mapping[str, Any]) -> UAVObservation:
    return UAVObservation(
        observation_id=str(record["observation_id"]),
        timestamp_s=float(record["timestamp_s"]),
        platform_id=str(record["platform_id"]),
        frame_uri=str(record["frame_uri"]),
        crs=_crs(record.get("crs")),
        pose=dict(record.get("pose", {})),
        telemetry=dict(record.get("telemetry", {})),
        camera=dict(record.get("camera", {})),
        provenance=_provenance(record.get("provenance")),
        data_age=_data_age(record["data_age"]),
        environment=dict(record.get("environment", {})),
        model_inputs=dict(record.get("model_inputs", {})),
        metadata=dict(record.get("metadata", {})),
    )


def _trajectory_step(record: Mapping[str, Any]) -> TrajectoryStep:
    return TrajectoryStep(
        step_id=str(record["step_id"]),
        timestamp_s=float(record["timestamp_s"]),
        action={str(key): float(value) for key, value in dict(record.get("action", {})).items()},
        state=dict(record.get("state", {})),
        observation_id=record.get("observation_id"),
        provenance=_provenance(record.get("provenance")),
        data_age=_data_age(record["data_age"]) if record.get("data_age") is not None else None,
    )


def _label(record: Mapping[str, Any]) -> LabelTarget:
    return LabelTarget(
        target_id=str(record["target_id"]),
        target_type=str(record["target_type"]),
        applies_to=tuple(str(item) for item in record.get("applies_to", ())),
        value=dict(record.get("value", {})),
        provenance=_provenance(record.get("provenance")),
        data_age=_data_age(record["data_age"]),
        temporal_window=_window(record["temporal_window"]) if record.get("temporal_window") else None,
        crs=_crs(record.get("crs")),
        metadata=dict(record.get("metadata", {})),
    )


def _mask(record: Mapping[str, Any]) -> MaskLayer:
    return MaskLayer(
        mask_id=str(record["mask_id"]),
        mask_type=str(record["mask_type"]),
        uri=str(record["uri"]),
        grid=_grid(record["grid"]),
        applies_to=tuple(str(item) for item in record.get("applies_to", ())),
        valid_values=tuple(int(item) for item in record.get("valid_values", ())),
        provenance=_provenance(record.get("provenance")),
        data_age=_data_age(record["data_age"]),
        metadata=dict(record.get("metadata", {})),
    )


def _uncertainty(record: Mapping[str, Any]) -> UncertaintyLayer:
    return UncertaintyLayer(
        uncertainty_id=str(record["uncertainty_id"]),
        uncertainty_type=str(record["uncertainty_type"]),
        uri=str(record["uri"]),
        grid=_grid(record["grid"]),
        applies_to=tuple(str(item) for item in record.get("applies_to", ())),
        value_units=str(record["value_units"]),
        provenance=_provenance(record.get("provenance")),
        data_age=_data_age(record["data_age"]),
        metadata=dict(record.get("metadata", {})),
    )


def _training_binding(record: Mapping[str, Any]) -> LeWMTrainingBinding:
    return LeWMTrainingBinding(
        binding_id=str(record["binding_id"]),
        keys_to_assets={
            str(key): tuple(str(item) for item in value)
            for key, value in dict(record.get("keys_to_assets", {})).items()
        },
        model_input_resolution_m=(
            None
            if record.get("model_input_resolution_m") is None
            else float(record["model_input_resolution_m"])
        ),
        resolution_tolerance_m=float(record.get("resolution_tolerance_m", 0.001)),
        required_keys=tuple(str(item) for item in record.get("required_keys", ("pixels", "action", "proprio", "state"))),
        notes=str(record.get("notes", "")),
    )


def _tile_starts(
    min_value: float,
    max_value: float,
    tile_size: float,
    stride: float,
    include_partial: bool,
) -> list[float]:
    extent = max_value - min_value
    if extent < tile_size and not include_partial:
        return []
    values = []
    current = min_value
    epsilon = max(stride, tile_size) * 1e-9
    while current + tile_size <= max_value + epsilon:
        values.append(current)
        current += stride
    if include_partial and (not values or values[-1] + tile_size < max_value - epsilon):
        values.append(max(min_value, max_value - tile_size))
    return values


def _temporal_midpoint(window: TemporalWindow) -> float:
    return (window.start_s + window.end_s) / 2.0


def _assets_for_key(binding: LeWMTrainingBinding | None, key: str) -> tuple[str, ...]:
    if binding is None:
        return ()
    return tuple(binding.keys_to_assets.get(key, ()))


def _asset_by_id(manifest: GeoDatasetManifest, asset_id: str) -> Any | None:
    for collection in (
        manifest.orthomosaic_tiles,
        manifest.raster_layers,
        manifest.masks,
        manifest.uncertainty,
    ):
        for asset in collection:
            if asset_id in (
                getattr(asset, "tile_id", None),
                getattr(asset, "layer_id", None),
                getattr(asset, "mask_id", None),
                getattr(asset, "uncertainty_id", None),
            ):
                return asset
    return None


def _asset_layer_type(asset: Any) -> str:
    return str(
        getattr(asset, "layer_type", None)
        or getattr(asset, "mask_type", None)
        or getattr(asset, "uncertainty_type", None)
        or "orthomosaic"
    )


def _same_crs(left: CRSDefinition | None, right: CRSDefinition | None) -> bool:
    if left is None or right is None:
        return False
    return (
        left.horizontal == right.horizontal
        and left.vertical_datum == right.vertical_datum
        and left.units == right.units
        and left.authority == right.authority
    )


def _grid_extent(grid: GeoGrid, crs: CRSDefinition | None) -> BoundingBox:
    return BoundingBox(
        min_x=grid.origin_x,
        min_y=grid.origin_y,
        max_x=grid.origin_x + grid.width_px * grid.resolution_m,
        max_y=grid.origin_y + grid.height_px * grid.resolution_m,
        crs=grid.crs or crs,
    )


def _intersects(left: BoundingBox, right: BoundingBox) -> bool:
    return not (
        left.max_x <= right.min_x
        or left.min_x >= right.max_x
        or left.max_y <= right.min_y
        or left.min_y >= right.max_y
    )


def _unavailable_alignment(
    asset_id: str,
    training_key: str,
    layer_type: str,
    source_id: str | None,
    grid: GeoGrid,
    timestamp_s: float,
    tile_extent: BoundingBox,
    reason: str,
) -> GeoLayerAlignment:
    return GeoLayerAlignment(
        asset_id=asset_id,
        training_key=training_key,
        layer_type=layer_type,
        source_id=source_id,
        available=False,
        unavailable_mask_value=1,
        reason=reason,
        resolution_m=grid.resolution_m,
        timestamp_s=timestamp_s,
        tile_extent=tile_extent,
    )


def _mission_id(manifest: GeoDatasetManifest) -> str:
    value = manifest.metadata.get("mission_id")
    if value is not None:
        return str(value)
    for step in manifest.trajectory:
        mission = step.state.get("mission_id") if isinstance(step.state, Mapping) else None
        if mission is not None:
            return str(mission)
    return manifest.dataset_id


def _sensor_id(
    manifest: GeoDatasetManifest,
    alignments: Sequence[GeoLayerAlignment],
) -> str:
    metadata_sensor = manifest.metadata.get("sensor_id")
    if metadata_sensor is not None:
        return str(metadata_sensor)
    for observation in manifest.uav_observations:
        return observation.platform_id
    for alignment in alignments:
        if alignment.training_key == "pixels" and alignment.source_id:
            return alignment.source_id
    return "unknown_sensor"


def _season_for(timestamp_s: float) -> str:
    month = datetime.fromtimestamp(timestamp_s, tz=timezone.utc).month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _time_bucket_for(timestamp_s: float, time_bucket_s: float) -> str:
    if time_bucket_s <= 0:
        raise ValueError("time_bucket_s_must_be_positive")
    return str(int(timestamp_s // time_bucket_s))


def _sample_id(
    dataset_id: str,
    dataset_version: str,
    tile_id: str,
    step_id: str,
    step_index: int,
    timestamp_s: float,
) -> str:
    raw = f"{dataset_id}|{dataset_version}|{tile_id}|{step_id}|{step_index}|{timestamp_s:.6f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _state_float(state: Mapping[str, Any], field_name: str) -> float:
    value = state.get(field_name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _leakage_components(
    samples: Sequence[GeoTrainingSampleMetadata],
    dimensions: Sequence[str],
) -> dict[str, str]:
    parent = {sample.sample_id: sample.sample_id for sample in samples}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners: dict[tuple[str, str], str] = {}
    for sample in samples:
        for dimension in dimensions:
            value = sample.split_axes.get(dimension)
            if not value:
                continue
            key = (dimension, value)
            if key in owners:
                union(owners[key], sample.sample_id)
            else:
                owners[key] = sample.sample_id

    component_members: dict[str, list[str]] = {}
    for sample in samples:
        component_members.setdefault(find(sample.sample_id), []).append(sample.sample_id)
    component_ids = {}
    for members in component_members.values():
        group = "lg_" + hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
        for member in members:
            component_ids[member] = group
    return component_ids


def _assign_component_splits(
    component_to_samples: Mapping[str, Sequence[GeoTrainingSampleMetadata]],
    split_fractions: Mapping[str, float],
    split_dimensions: Sequence[str],
    seed: int,
) -> dict[str, str]:
    components = list(component_to_samples)
    split_counts = _component_split_counts(len(components), split_fractions)
    ordered = sorted(
        components,
        key=lambda component: _stable_sort_key(
            _component_signature(component_to_samples[component], split_dimensions),
            seed,
        ),
    )
    split_for_component: dict[str, str] = {}
    index = 0
    for split, count in split_counts.items():
        for component in ordered[index : index + count]:
            split_for_component[component] = split
        index += count
    return split_for_component


def _component_signature(
    samples: Sequence[GeoTrainingSampleMetadata],
    split_dimensions: Sequence[str],
) -> str:
    values = []
    for dimension in split_dimensions:
        dimension_values = sorted({sample.split_axes.get(dimension, "") for sample in samples})
        values.append(f"{dimension}={','.join(dimension_values)}")
    return "|".join(values)


def _stable_sort_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _component_split_counts(
    component_count: int,
    split_fractions: Mapping[str, float],
) -> dict[str, int]:
    if component_count == 0:
        return {split: 0 for split in split_fractions}
    total = sum(max(float(value), 0.0) for value in split_fractions.values())
    if total <= 0:
        raise ValueError("split_fractions_must_sum_positive")
    normalized = {split: max(float(value), 0.0) / total for split, value in split_fractions.items()}
    raw = {split: normalized[split] * component_count for split in normalized}
    counts = {split: int(raw[split]) for split in normalized}
    positive = [split for split, fraction in normalized.items() if fraction > 0]
    minimum = {split: 0 for split in normalized}
    if component_count >= len(positive):
        for split in positive:
            minimum[split] = 1
            counts[split] = max(counts[split], 1)

    while sum(counts.values()) > component_count:
        candidates = [
            split
            for split in counts
            if counts[split] > minimum[split]
        ]
        split = max(candidates, key=lambda item: (counts[item], -raw[item]))
        counts[split] -= 1
    while sum(counts.values()) < component_count:
        split = max(normalized, key=lambda item: (raw[item] - counts[item], normalized[item]))
        counts[split] += 1
    return counts


def _count_by_split(
    samples: Sequence[GeoTrainingSampleMetadata],
    split_fractions: Mapping[str, float],
) -> dict[str, int]:
    counts = {split: 0 for split in split_fractions}
    for sample in samples:
        counts[sample.split] = counts.get(sample.split, 0) + 1
    return counts


def _split_overlaps(samples: Sequence[GeoTrainingSampleMetadata], dimension: str) -> set[str]:
    values_to_splits: dict[str, set[str]] = {}
    for sample in samples:
        value = sample.split_axes.get(dimension)
        if value:
            values_to_splits.setdefault(value, set()).add(sample.split)
    return {value for value, splits in values_to_splits.items() if len(splits) > 1}


def _dataset_family(manifests: Sequence[GeoDatasetManifest]) -> str:
    dataset_ids = sorted({manifest.dataset_id for manifest in manifests})
    if len(dataset_ids) == 1:
        return dataset_ids[0]
    digest = hashlib.sha256("|".join(dataset_ids).encode("utf-8")).hexdigest()[:12]
    return f"geo_ai_{digest}"


def _write_metadata_jsonl(plan: GeoTrainingDatasetPlan) -> None:
    by_split: dict[str, list[GeoTrainingSampleMetadata]] = {}
    for sample in plan.samples:
        by_split.setdefault(sample.split, []).append(sample)
    for split, output_path in plan.output_paths.items():
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for sample in by_split.get(split, ()):
                handle.write(json.dumps(sample.to_record(), sort_keys=True) + "\n")


def _write_hdf5(
    plan: GeoTrainingDatasetPlan,
    manifests: Sequence[GeoDatasetManifest],
    config: GeoTrainingDatasetBuildConfig,
) -> None:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "hdf5_output_requires_h5py_and_numpy; use dry_run=True or output_format='metadata-jsonl'"
        ) from exc

    manifest_by_dataset = {manifest.dataset_id: manifest for manifest in manifests}
    by_split: dict[str, list[GeoTrainingSampleMetadata]] = {}
    for sample in plan.samples:
        by_split.setdefault(sample.split, []).append(sample)

    for split, output_path in plan.output_paths.items():
        split_samples = sorted(
            by_split.get(split, []),
            key=lambda sample: (
                sample.leakage_group,
                sample.tile_id,
                sample.timestamp_s,
                sample.sample_id,
            ),
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        episodes = _materialize_hdf5_episodes(split_samples, manifest_by_dataset, config, np)
        with h5py.File(path, "w") as handle:
            _write_hdf5_episodes(handle, episodes, np, h5py)


def _materialize_hdf5_episodes(
    samples: Sequence[GeoTrainingSampleMetadata],
    manifest_by_dataset: Mapping[str, GeoDatasetManifest],
    config: GeoTrainingDatasetBuildConfig,
    np: Any,
) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    current_key: tuple[str, str] | None = None
    current_rows: list[dict[str, Any]] = []
    for sample in samples:
        key = (sample.leakage_group, sample.tile_id)
        if current_key is not None and key != current_key:
            episodes.append(current_rows)
            current_rows = []
        current_key = key
        current_rows.append(
            _materialize_sample(
                sample,
                manifest_by_dataset[sample.dataset_id],
                config,
                np,
                episode_index=len(episodes),
                step_index=len(current_rows),
            )
        )
    if current_rows:
        episodes.append(current_rows)
    return episodes


def _materialize_sample(
    sample: GeoTrainingSampleMetadata,
    manifest: GeoDatasetManifest,
    config: GeoTrainingDatasetBuildConfig,
    np: Any,
    *,
    episode_index: int,
    step_index: int,
) -> dict[str, Any]:
    tile = config.tile
    pixel_channels = _pixel_channels(manifest, sample)
    geo_channels = len([item for item in sample.alignments if item.training_key == "geo_layers"])
    uncertainty_channels = len([item for item in sample.alignments if item.training_key == "uncertainty"])
    return {
        "pixels": np.full(
            (tile.height_px, tile.width_px, pixel_channels),
            _constant_for_key(manifest, "pixels", 0),
            dtype=np.uint8,
        ),
        "action": np.asarray(sample.training_keys["action"], dtype=np.float32),
        "proprio": np.asarray(sample.training_keys["proprio"], dtype=np.float32),
        "proprioception": np.asarray(sample.training_keys["proprioception"], dtype=np.float32),
        "geo_layers": np.full(
            (tile.height_px, tile.width_px, geo_channels),
            _constant_for_key(manifest, "geo_layers", 0.0),
            dtype=np.float32,
        ),
        "uncertainty": np.full(
            (tile.height_px, tile.width_px, uncertainty_channels),
            _constant_for_key(manifest, "uncertainty", 0.0),
            dtype=np.float32,
        ),
        "timestamps": np.asarray(sample.timestamp_s, dtype=np.float64),
        "source_ids": "|".join(sample.source_ids),
        "geo_layer_available": np.asarray(sample.training_keys["geo_layer_available"], dtype=np.uint8),
        "unavailable_layer_mask": np.asarray(sample.training_keys["unavailable_layer_mask"], dtype=np.uint8),
        "ep_idx": np.asarray(episode_index, dtype=np.int32),
        "step_idx": np.asarray(step_index, dtype=np.int32),
    }


def _pixel_channels(manifest: GeoDatasetManifest, sample: GeoTrainingSampleMetadata) -> int:
    pixel_asset_ids = tuple(str(item) for item in sample.training_keys.get("pixels", []))
    for asset_id in pixel_asset_ids:
        asset = _asset_by_id(manifest, asset_id)
        bands = getattr(asset, "bands", ())
        if bands:
            return len(bands)
    return 3


def _constant_for_key(manifest: GeoDatasetManifest, key: str, default: float | int) -> float | int:
    metadata_key = f"synthetic_{key}_value"
    value = manifest.metadata.get(metadata_key, default)
    if isinstance(value, int | float):
        return value
    return default


def _write_hdf5_episodes(
    handle: Any,
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    np: Any,
    h5py: Any,
) -> None:
    rows = [row for episode in episodes for row in episode]
    if not rows:
        handle.create_dataset("ep_len", data=np.asarray([], dtype=np.int32), maxshape=(None,))
        handle.create_dataset("ep_offset", data=np.asarray([], dtype=np.int64), maxshape=(None,))
        return
    keys = tuple(rows[0])
    for key in keys:
        values = [row[key] for row in rows]
        if isinstance(values[0], str):
            dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(key, data=np.asarray(values, dtype=dtype))
        else:
            handle.create_dataset(key, data=np.stack(values, axis=0))
    ep_len = np.asarray([len(episode) for episode in episodes], dtype=np.int32)
    ep_offset = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(ep_len[:-1], dtype=np.int64)))
    handle.create_dataset("ep_len", data=ep_len, maxshape=(None,))
    handle.create_dataset("ep_offset", data=ep_offset, maxshape=(None,))
