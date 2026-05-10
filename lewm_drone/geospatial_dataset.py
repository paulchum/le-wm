"""Versioned geospatial training-data contracts for layered drone world models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .interfaces import DroneObservation, JsonValue, to_jsonable

GEOSPATIAL_DATASET_SCHEMA_VERSION = "canadian-layered-geo-ai-world-model/v1"

BLOCKED_PROVENANCE_LABELS = frozenset(
    {"classified", "controlled_goods", "secret", "top_secret"}
)

DEFAULT_MAX_AGE_S = {
    "weather": 6 * 60 * 60,
    "ice": 24 * 60 * 60,
    "uav": 30 * 24 * 60 * 60,
    "orthomosaic": 365 * 24 * 60 * 60,
    "dem": 10 * 365 * 24 * 60 * 60,
    "land_cover": 3 * 365 * 24 * 60 * 60,
    "hydro": 3 * 365 * 24 * 60 * 60,
}


@dataclass(frozen=True)
class CRSDefinition:
    """Horizontal CRS and vertical datum used by an AOI or geospatial asset."""

    horizontal: str | None
    vertical_datum: str | None
    units: str = "m"
    authority: str = "EPSG"
    epoch: str | None = None


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned AOI bounds in the declared CRS."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: CRSDefinition | None = None


@dataclass(frozen=True)
class TemporalWindow:
    """Inclusive temporal coverage as epoch seconds."""

    start_s: float
    end_s: float


@dataclass(frozen=True)
class AOIMetadata:
    """Dataset area of interest metadata."""

    aoi_id: str
    name: str
    country: str
    region: str
    bounds: BoundingBox
    temporal_window: TemporalWindow
    crs: CRSDefinition | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeoGrid:
    """Raster grid geometry for model-ready geospatial layers."""

    width_px: int
    height_px: int
    resolution_m: float
    origin_x: float
    origin_y: float
    crs: CRSDefinition | None = None
    tile_matrix: str | None = None


@dataclass(frozen=True)
class SourceProvenance:
    """Auditable source metadata for every dataset asset."""

    source_id: str
    source_name: str
    collection_method: str
    source_uri: str
    license: str = "unspecified"
    organization: str = ""
    product_version: str = ""
    classification_label: str = "unclassified"
    notes: str = ""


@dataclass(frozen=True)
class DataAge:
    """Age metadata for freshness validation."""

    observed_at_s: float
    ingested_at_s: float
    max_age_s: float | None = None


@dataclass(frozen=True)
class RasterLayer:
    """Raster source such as elevation, ice, weather, or land cover."""

    layer_id: str
    layer_type: str
    uri: str
    grid: GeoGrid
    temporal_window: TemporalWindow
    bands: tuple[str, ...]
    dtype: str
    provenance: SourceProvenance | None
    data_age: DataAge
    nodata: float | int | None = None
    mask_ids: tuple[str, ...] = ()
    uncertainty_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorLayer:
    """Vector source such as hydrography or AOI line/polygon context."""

    layer_id: str
    layer_type: str
    uri: str
    geometry_type: str
    crs: CRSDefinition | None
    temporal_window: TemporalWindow
    attributes: tuple[str, ...]
    provenance: SourceProvenance | None
    data_age: DataAge
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrthomosaicTile:
    """Orthomosaic tile metadata for model input or map context."""

    tile_id: str
    uri: str
    grid: GeoGrid
    temporal_window: TemporalWindow
    bands: tuple[str, ...]
    provenance: SourceProvenance | None
    data_age: DataAge
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UAVObservation:
    """Sparse UAV image observation and pose metadata."""

    observation_id: str
    timestamp_s: float
    platform_id: str
    frame_uri: str
    crs: CRSDefinition | None
    pose: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    camera: Mapping[str, Any]
    provenance: SourceProvenance | None
    data_age: DataAge
    environment: Mapping[str, Any] = field(default_factory=dict)
    model_inputs: Mapping[str, Any] = field(default_factory=dict, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_drone_observation(
        self,
        *,
        pixels: Any | None = None,
        model_inputs: Mapping[str, Any] | None = None,
    ) -> DroneObservation:
        """Return a `DroneObservation` compatible with planner and LeWM adapters."""

        merged_model_inputs = dict(self.model_inputs)
        if model_inputs:
            merged_model_inputs.update(model_inputs)
        telemetry = {
            **dict(self.telemetry),
            "platform_id": self.platform_id,
            "frame_uri": self.frame_uri,
            "pose": to_jsonable(self.pose),
        }
        environment = {
            **dict(self.environment),
            "observation_id": self.observation_id,
            "camera": to_jsonable(self.camera),
            "crs": to_jsonable(self.crs),
            "provenance": to_jsonable(self.provenance),
        }
        return DroneObservation(
            timestamp_s=self.timestamp_s,
            pixels=pixels,
            telemetry=telemetry,
            environment=environment,
            model_inputs=merged_model_inputs,
        )


@dataclass(frozen=True)
class TrajectoryStep:
    """Observed action/state history aligned with LeWM action inputs."""

    step_id: str
    timestamp_s: float
    action: Mapping[str, float]
    state: Mapping[str, Any]
    observation_id: str | None = None
    provenance: SourceProvenance | None = None
    data_age: DataAge | None = None


@dataclass(frozen=True)
class LabelTarget:
    """Training target or supervision label for an AOI or observation."""

    target_id: str
    target_type: str
    applies_to: tuple[str, ...]
    value: Mapping[str, Any]
    provenance: SourceProvenance | None
    data_age: DataAge
    temporal_window: TemporalWindow | None = None
    crs: CRSDefinition | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaskLayer:
    """Mask layer for valid pixels, cloud, water, occlusion, or train/test splits."""

    mask_id: str
    mask_type: str
    uri: str
    grid: GeoGrid
    applies_to: tuple[str, ...]
    valid_values: tuple[int, ...]
    provenance: SourceProvenance | None
    data_age: DataAge
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UncertaintyLayer:
    """Per-layer uncertainty or quality raster."""

    uncertainty_id: str
    uncertainty_type: str
    uri: str
    grid: GeoGrid
    applies_to: tuple[str, ...]
    value_units: str
    provenance: SourceProvenance | None
    data_age: DataAge
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LeWMTrainingBinding:
    """Mapping from manifest assets to existing LeWM/HDF5 training keys."""

    binding_id: str
    keys_to_assets: Mapping[str, tuple[str, ...]]
    model_input_resolution_m: float | None = None
    resolution_tolerance_m: float = 0.001
    required_keys: tuple[str, ...] = ("pixels", "action", "proprio", "state")
    notes: str = ""


@dataclass(frozen=True)
class GeoDatasetManifest:
    """Canonical manifest for a layered Canadian geo-AI training dataset."""

    dataset_id: str
    dataset_version: str
    aoi: AOIMetadata
    crs: CRSDefinition | None
    temporal_window: TemporalWindow
    raster_layers: tuple[RasterLayer, ...]
    vector_layers: tuple[VectorLayer, ...]
    uav_observations: tuple[UAVObservation, ...]
    trajectory: tuple[TrajectoryStep, ...]
    labels: tuple[LabelTarget, ...]
    masks: tuple[MaskLayer, ...]
    uncertainty: tuple[UncertaintyLayer, ...]
    orthomosaic_tiles: tuple[OrthomosaicTile, ...] = ()
    training_bindings: tuple[LeWMTrainingBinding, ...] = ()
    manifest_hash: str = ""
    schema_version: str = GEOSPATIAL_DATASET_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class GeoDatasetValidationFinding:
    """Structured validation finding for dataset manifests."""

    severity: str
    code: str
    path: str
    message: str

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]


class GeoDatasetValidationError(ValueError):
    """Raised when a geospatial dataset manifest has blocking validation errors."""

    def __init__(self, findings: Sequence[GeoDatasetValidationFinding]):
        self.findings = tuple(findings)
        codes = ", ".join(finding.code for finding in self.findings)
        super().__init__(f"geospatial_dataset_validation_failed:{codes}")


def validate_geospatial_dataset_manifest(
    manifest: GeoDatasetManifest,
    *,
    reference_time_s: float | None = None,
    enforce_freshness: bool = False,
) -> list[GeoDatasetValidationFinding]:
    """Return validation findings for a geospatial training dataset manifest."""

    findings: list[GeoDatasetValidationFinding] = []
    effective_reference_time_s = (
        manifest.temporal_window.end_s if reference_time_s is None else reference_time_s
    )

    if manifest.schema_version != GEOSPATIAL_DATASET_SCHEMA_VERSION:
        findings.append(
            _error(
                "unsupported_schema_version",
                "schema_version",
                f"expected {GEOSPATIAL_DATASET_SCHEMA_VERSION}",
            )
        )

    _validate_temporal_window(manifest.temporal_window, "temporal_window", findings)
    _validate_temporal_window(manifest.aoi.temporal_window, "aoi.temporal_window", findings)
    _validate_crs(manifest.crs, "crs", findings)
    _validate_crs(_effective_crs(manifest.aoi.crs, manifest.crs), "aoi.crs", findings)
    _validate_bounds(manifest.aoi.bounds, manifest.crs, findings)

    asset_resolutions: dict[str, float] = {}
    known_asset_ids: set[str] = set()
    for index, layer in enumerate(manifest.raster_layers):
        path = f"raster_layers[{index}]"
        known_asset_ids.add(layer.layer_id)
        asset_resolutions[layer.layer_id] = layer.grid.resolution_m
        _validate_temporal_window(layer.temporal_window, f"{path}.temporal_window", findings)
        _validate_grid(layer.grid, manifest.crs, f"{path}.grid", findings)
        _validate_provenance(layer.provenance, f"{path}.provenance", findings)
        _validate_data_age(
            layer.data_age,
            _max_age_for(layer.layer_type, layer.data_age),
            effective_reference_time_s,
            f"{path}.data_age",
            enforce_freshness,
            findings,
        )

    for index, layer in enumerate(manifest.vector_layers):
        path = f"vector_layers[{index}]"
        known_asset_ids.add(layer.layer_id)
        _validate_temporal_window(layer.temporal_window, f"{path}.temporal_window", findings)
        _validate_crs(_effective_crs(layer.crs, manifest.crs), f"{path}.crs", findings)
        _validate_provenance(layer.provenance, f"{path}.provenance", findings)
        _validate_data_age(
            layer.data_age,
            _max_age_for(layer.layer_type, layer.data_age),
            effective_reference_time_s,
            f"{path}.data_age",
            enforce_freshness,
            findings,
        )

    for index, tile in enumerate(manifest.orthomosaic_tiles):
        path = f"orthomosaic_tiles[{index}]"
        known_asset_ids.add(tile.tile_id)
        asset_resolutions[tile.tile_id] = tile.grid.resolution_m
        _validate_temporal_window(tile.temporal_window, f"{path}.temporal_window", findings)
        _validate_grid(tile.grid, manifest.crs, f"{path}.grid", findings)
        _validate_provenance(tile.provenance, f"{path}.provenance", findings)
        _validate_data_age(
            tile.data_age,
            _max_age_for("orthomosaic", tile.data_age),
            effective_reference_time_s,
            f"{path}.data_age",
            enforce_freshness,
            findings,
        )

    for index, observation in enumerate(manifest.uav_observations):
        path = f"uav_observations[{index}]"
        known_asset_ids.add(observation.observation_id)
        _validate_crs(_effective_crs(observation.crs, manifest.crs), f"{path}.crs", findings)
        _validate_provenance(observation.provenance, f"{path}.provenance", findings)
        _validate_data_age(
            observation.data_age,
            _max_age_for("uav", observation.data_age),
            effective_reference_time_s,
            f"{path}.data_age",
            enforce_freshness,
            findings,
        )

    for index, step in enumerate(manifest.trajectory):
        path = f"trajectory[{index}]"
        known_asset_ids.add(step.step_id)
        _validate_provenance(step.provenance, f"{path}.provenance", findings)
        if step.data_age is not None:
            _validate_data_age(
                step.data_age,
                _max_age_for("uav", step.data_age),
                effective_reference_time_s,
                f"{path}.data_age",
                enforce_freshness,
                findings,
            )

    for index, label in enumerate(manifest.labels):
        path = f"labels[{index}]"
        known_asset_ids.add(label.target_id)
        if label.temporal_window is not None:
            _validate_temporal_window(label.temporal_window, f"{path}.temporal_window", findings)
        if label.crs is not None:
            _validate_crs(_effective_crs(label.crs, manifest.crs), f"{path}.crs", findings)
        _validate_provenance(label.provenance, f"{path}.provenance", findings)
        _validate_data_age(
            label.data_age,
            _max_age_for(label.target_type, label.data_age),
            effective_reference_time_s,
            f"{path}.data_age",
            enforce_freshness,
            findings,
        )

    for index, mask in enumerate(manifest.masks):
        path = f"masks[{index}]"
        known_asset_ids.add(mask.mask_id)
        asset_resolutions[mask.mask_id] = mask.grid.resolution_m
        _validate_grid(mask.grid, manifest.crs, f"{path}.grid", findings)
        _validate_provenance(mask.provenance, f"{path}.provenance", findings)
        _validate_data_age(
            mask.data_age,
            _max_age_for("orthomosaic", mask.data_age),
            effective_reference_time_s,
            f"{path}.data_age",
            enforce_freshness,
            findings,
        )

    for index, uncertainty in enumerate(manifest.uncertainty):
        path = f"uncertainty[{index}]"
        known_asset_ids.add(uncertainty.uncertainty_id)
        asset_resolutions[uncertainty.uncertainty_id] = uncertainty.grid.resolution_m
        _validate_grid(uncertainty.grid, manifest.crs, f"{path}.grid", findings)
        _validate_provenance(uncertainty.provenance, f"{path}.provenance", findings)
        _validate_data_age(
            uncertainty.data_age,
            _max_age_for("orthomosaic", uncertainty.data_age),
            effective_reference_time_s,
            f"{path}.data_age",
            enforce_freshness,
            findings,
        )

    for index, binding in enumerate(manifest.training_bindings):
        _validate_training_binding(
            binding,
            asset_resolutions,
            known_asset_ids,
            f"training_bindings[{index}]",
            findings,
        )

    return findings


def require_valid_geospatial_dataset_manifest(
    manifest: GeoDatasetManifest,
    *,
    reference_time_s: float | None = None,
    enforce_freshness: bool = False,
) -> GeoDatasetManifest:
    """Return the manifest or raise when validation has hard errors."""

    findings = validate_geospatial_dataset_manifest(
        manifest,
        reference_time_s=reference_time_s,
        enforce_freshness=enforce_freshness,
    )
    errors = [finding for finding in findings if finding.severity == "error"]
    if errors:
        raise GeoDatasetValidationError(errors)
    return manifest


def _validate_bounds(
    bounds: BoundingBox,
    fallback_crs: CRSDefinition | None,
    findings: list[GeoDatasetValidationFinding],
) -> None:
    if bounds.min_x >= bounds.max_x or bounds.min_y >= bounds.max_y:
        findings.append(_error("invalid_bounds", "aoi.bounds", "min bounds must be below max bounds"))
    _validate_crs(_effective_crs(bounds.crs, fallback_crs), "aoi.bounds.crs", findings)


def _validate_grid(
    grid: GeoGrid,
    fallback_crs: CRSDefinition | None,
    path: str,
    findings: list[GeoDatasetValidationFinding],
) -> None:
    if grid.width_px <= 0 or grid.height_px <= 0:
        findings.append(_error("invalid_grid_shape", path, "grid dimensions must be positive"))
    if grid.resolution_m <= 0:
        findings.append(_error("invalid_grid_resolution", path, "resolution_m must be positive"))
    _validate_crs(_effective_crs(grid.crs, fallback_crs), f"{path}.crs", findings)


def _validate_temporal_window(
    window: TemporalWindow,
    path: str,
    findings: list[GeoDatasetValidationFinding],
) -> None:
    if window.start_s > window.end_s:
        findings.append(_error("invalid_temporal_window", path, "start_s must be <= end_s"))


def _validate_crs(
    crs: CRSDefinition | None,
    path: str,
    findings: list[GeoDatasetValidationFinding],
) -> None:
    if crs is None or not crs.horizontal:
        findings.append(_error("missing_crs", path, "horizontal CRS is required"))
        return
    if not crs.vertical_datum:
        findings.append(_error("missing_vertical_datum", path, "vertical datum is required"))


def _validate_provenance(
    provenance: SourceProvenance | None,
    path: str,
    findings: list[GeoDatasetValidationFinding],
) -> None:
    if provenance is None:
        findings.append(_error("missing_provenance", path, "source provenance is required"))
        return
    missing = []
    if not provenance.source_id:
        missing.append("source_id")
    if not provenance.source_name:
        missing.append("source_name")
    if not provenance.collection_method:
        missing.append("collection_method")
    if not provenance.source_uri:
        missing.append("source_uri")
    if missing:
        findings.append(
            _error(
                "missing_provenance",
                path,
                f"source provenance missing {', '.join(missing)}",
            )
        )
    label = provenance.classification_label.lower()
    if label in BLOCKED_PROVENANCE_LABELS:
        findings.append(
            _error(
                "blocked_provenance_label",
                path,
                f"classification_label {label} is not allowed in repo manifests",
            )
        )


def _validate_data_age(
    data_age: DataAge,
    max_age_s: float | None,
    reference_time_s: float,
    path: str,
    enforce_freshness: bool,
    findings: list[GeoDatasetValidationFinding],
) -> None:
    if data_age.ingested_at_s < data_age.observed_at_s:
        findings.append(
            _error("invalid_data_age", path, "ingested_at_s must be >= observed_at_s")
        )
    if max_age_s is None:
        return
    age_s = reference_time_s - data_age.observed_at_s
    if age_s > max_age_s:
        severity = "error" if enforce_freshness else "warning"
        findings.append(
            GeoDatasetValidationFinding(
                severity=severity,
                code="stale_timestamp",
                path=path,
                message=f"observed_at_s is older than max_age_s ({age_s:.1f}s > {max_age_s:.1f}s)",
            )
        )


def _validate_training_binding(
    binding: LeWMTrainingBinding,
    asset_resolutions: Mapping[str, float],
    known_asset_ids: set[str],
    path: str,
    findings: list[GeoDatasetValidationFinding],
) -> None:
    missing_keys = [
        key for key in binding.required_keys if key not in binding.keys_to_assets
    ]
    for key in missing_keys:
        findings.append(
            _error("missing_training_key", f"{path}.keys_to_assets", f"missing {key}")
        )

    for key, asset_ids in binding.keys_to_assets.items():
        for asset_id in asset_ids:
            if asset_id not in known_asset_ids:
                findings.append(
                    _error(
                        "unknown_training_asset",
                        path,
                        f"{key} references unknown asset {asset_id}",
                    )
                )

    pixel_assets = binding.keys_to_assets.get("pixels", ())
    for asset_id in pixel_assets:
        if asset_id not in known_asset_ids:
            continue
        resolution = asset_resolutions.get(asset_id)
        if resolution is None:
            continue
        if binding.model_input_resolution_m is None:
            continue
        if abs(resolution - binding.model_input_resolution_m) > binding.resolution_tolerance_m:
            findings.append(
                _error(
                    "incompatible_resolution",
                    path,
                    (
                        f"asset {asset_id} resolution {resolution}m does not match "
                        f"model_input_resolution_m {binding.model_input_resolution_m}m"
                    ),
                )
            )


def _max_age_for(kind: str, data_age: DataAge) -> float | None:
    if data_age.max_age_s is not None:
        return data_age.max_age_s
    return DEFAULT_MAX_AGE_S.get(kind.lower())


def _effective_crs(
    crs: CRSDefinition | None,
    fallback: CRSDefinition | None,
) -> CRSDefinition | None:
    return crs if crs is not None else fallback


def _error(code: str, path: str, message: str) -> GeoDatasetValidationFinding:
    return GeoDatasetValidationFinding(
        severity="error",
        code=code,
        path=path,
        message=message,
    )
