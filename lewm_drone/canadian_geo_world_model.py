"""Canadian layered geo-AI world-model foundation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping

from .geo_training_dataset import (
    GeoTrainingDatasetBuildConfig,
    GeoTrainingDatasetPlan,
    build_geo_training_dataset,
    read_canonical_geospatial_manifest,
)
from .geospatial_dataset import (
    GeoDatasetManifest,
    GeoDatasetValidationFinding,
    LeWMTrainingBinding,
    RasterLayer,
    VectorLayer,
    require_valid_geospatial_dataset_manifest,
    validate_geospatial_dataset_manifest,
)

ELEVATION_LAYER_TYPES = frozenset({"dem", "elevation", "dtm", "dsm"})
ICE_LAYER_TYPES = frozenset({"ice", "sea_ice", "lake_ice"})
WEATHER_LAYER_TYPES = frozenset({"weather", "meteorology", "meteo"})
LAND_COVER_LAYER_TYPES = frozenset({"land_cover", "landcover", "lulc"})
HYDRO_LAYER_TYPES = frozenset({"hydro", "hydrology", "hydrography", "water"})

CANADIAN_LAYERED_WORLD_MODEL_MODALITIES = (
    "elevation",
    "ice",
    "weather",
    "land_cover",
    "hydro",
    "sparse_uav_imagery",
    "orthomosaic",
)


@dataclass(frozen=True)
class CanadianGeoWorldModelConfig:
    """Policy for preparing full Canadian layered geo-AI training manifests."""

    binding_id: str = "canadian-layered-geo-ai-world-model-v1"
    model_input_resolution_m: float | None = None
    resolution_tolerance_m: float = 0.001
    require_modalities: tuple[str, ...] = CANADIAN_LAYERED_WORLD_MODEL_MODALITIES
    required_training_keys: tuple[str, ...] = (
        "pixels",
        "action",
        "proprio",
        "state",
        "geo_layers",
        "orthomosaic",
        "sparse_uav_frames",
    )
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CanadianGeoWorldModelFoundation:
    """Validation, binding, and training-plan output for a layered manifest set."""

    manifests: tuple[GeoDatasetManifest, ...]
    findings: tuple[GeoDatasetValidationFinding, ...]
    training_plan: GeoTrainingDatasetPlan | None = None

    @property
    def ready(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)


def prepare_canadian_geo_world_model_manifest(
    manifest: GeoDatasetManifest | Mapping[str, object] | str | Path,
    *,
    config: CanadianGeoWorldModelConfig | None = None,
) -> GeoDatasetManifest:
    """Return a validated manifest with a canonical layered LeWM binding."""

    resolved_config = config or CanadianGeoWorldModelConfig()
    base = require_valid_geospatial_dataset_manifest(
        read_canonical_geospatial_manifest(manifest)
    )
    findings = validate_canadian_geo_world_model_manifest(base, config=resolved_config)
    errors = [finding for finding in findings if finding.severity == "error"]
    if errors:
        raise ValueError(
            "canadian_geo_world_model_manifest_not_ready:"
            + ",".join(finding.code for finding in errors)
        )
    prepared = replace(
        base,
        training_bindings=(
            _canonical_training_binding(base, resolved_config),
            *tuple(
                binding
                for binding in base.training_bindings
                if binding.binding_id != resolved_config.binding_id
            ),
        ),
        metadata={
            **dict(base.metadata),
            **dict(resolved_config.metadata),
            "geo_world_model_foundation": "canadian-layered-v1",
            "geo_world_model_modalities": list(resolved_config.require_modalities),
        },
        manifest_hash="",
    )
    return replace(prepared, manifest_hash=_manifest_hash(prepared))


def build_canadian_geo_world_model_foundation(
    manifests: Iterable[GeoDatasetManifest | Mapping[str, object] | str | Path],
    *,
    training_config: GeoTrainingDatasetBuildConfig,
    foundation_config: CanadianGeoWorldModelConfig | None = None,
    output_path: str | Path | None = None,
    dry_run: bool = True,
) -> CanadianGeoWorldModelFoundation:
    """Validate manifests, attach bindings, and optionally write training outputs."""

    resolved_config = foundation_config or CanadianGeoWorldModelConfig()
    raw_manifests = tuple(read_canonical_geospatial_manifest(manifest) for manifest in manifests)
    findings = tuple(
        finding
        for manifest in raw_manifests
        for finding in validate_canadian_geo_world_model_manifest(
            manifest,
            config=resolved_config,
        )
    )
    errors = [finding for finding in findings if finding.severity == "error"]
    if errors:
        return CanadianGeoWorldModelFoundation(
            manifests=raw_manifests,
            findings=findings,
            training_plan=None,
        )

    prepared = tuple(
        prepare_canadian_geo_world_model_manifest(
            manifest,
            config=resolved_config,
        )
        for manifest in raw_manifests
    )
    plan = None
    plan = build_geo_training_dataset(
        prepared,
        config=training_config,
        output_path=output_path,
        dry_run=dry_run,
    )
    return CanadianGeoWorldModelFoundation(
        manifests=prepared,
        findings=findings,
        training_plan=plan,
    )


def validate_canadian_geo_world_model_manifest(
    manifest: GeoDatasetManifest,
    *,
    config: CanadianGeoWorldModelConfig | None = None,
) -> tuple[GeoDatasetValidationFinding, ...]:
    """Return readiness findings for the full layered Canadian world-model stack."""

    resolved_config = config or CanadianGeoWorldModelConfig()
    findings = list(validate_geospatial_dataset_manifest(manifest))
    available = _available_modalities(manifest)
    for modality in resolved_config.require_modalities:
        if not available.get(modality):
            findings.append(
                GeoDatasetValidationFinding(
                    severity="error",
                    code=f"missing_{modality}",
                    path="modalities",
                    message=f"required Canadian geo-AI modality is missing: {modality}",
                )
            )
    return tuple(findings)


def canadian_geo_world_model_channel_schema(
    manifest: GeoDatasetManifest,
) -> tuple[Mapping[str, object], ...]:
    """Return deterministic channel metadata for layered fusion encoders."""

    channels: list[Mapping[str, object]] = []
    for layer in manifest.raster_layers:
        role = _modality_for_layer(layer.layer_type)
        for band_index, band_name in enumerate(layer.bands or ("value",)):
            channels.append(
                {
                    "asset_id": layer.layer_id,
                    "layer_type": layer.layer_type,
                    "modality": role,
                    "band_index": band_index,
                    "band_name": band_name,
                    "source_id": layer.provenance.source_id if layer.provenance else None,
                }
            )
    for layer in manifest.vector_layers:
        channels.append(
            {
                "asset_id": layer.layer_id,
                "layer_type": layer.layer_type,
                "modality": _modality_for_layer(layer.layer_type),
                "band_index": 0,
                "band_name": "rasterized_geometry",
                "source_id": layer.provenance.source_id if layer.provenance else None,
            }
        )
    return tuple(channels)


def _canonical_training_binding(
    manifest: GeoDatasetManifest,
    config: CanadianGeoWorldModelConfig,
) -> LeWMTrainingBinding:
    sparse_uav_ids = tuple(observation.observation_id for observation in manifest.uav_observations)
    orthomosaic_ids = tuple(tile.tile_id for tile in manifest.orthomosaic_tiles)
    geo_layer_ids = tuple(
        [layer.layer_id for layer in manifest.raster_layers]
        + [layer.layer_id for layer in manifest.vector_layers]
    )
    static_ids = tuple(
        asset_id
        for asset_id in geo_layer_ids
        if _modality_for_asset(manifest, asset_id) in {"elevation", "land_cover", "hydro"}
    )
    dynamic_ids = tuple(
        asset_id
        for asset_id in geo_layer_ids
        if _modality_for_asset(manifest, asset_id) in {"ice", "weather"}
    )
    action_ids = tuple(step.step_id for step in manifest.trajectory)
    return LeWMTrainingBinding(
        binding_id=config.binding_id,
        keys_to_assets={
            "pixels": sparse_uav_ids or orthomosaic_ids,
            "orthomosaic": orthomosaic_ids,
            "sparse_uav_frames": sparse_uav_ids,
            "geo_layers": geo_layer_ids,
            "static_geo_layers": static_ids,
            "dynamic_geo_layers": dynamic_ids,
            "vector_layers": tuple(layer.layer_id for layer in manifest.vector_layers),
            "uncertainty": tuple(layer.uncertainty_id for layer in manifest.uncertainty),
            "action": action_ids,
            "proprio": sparse_uav_ids,
            "state": static_ids,
        },
        model_input_resolution_m=config.model_input_resolution_m,
        resolution_tolerance_m=config.resolution_tolerance_m,
        required_keys=config.required_training_keys,
        notes=(
            "Canonical binding for elevation, ice, weather, land cover, hydro, "
            "sparse UAV imagery, and orthomosaic fusion."
        ),
    )


def _available_modalities(manifest: GeoDatasetManifest) -> dict[str, bool]:
    modalities = {name: False for name in CANADIAN_LAYERED_WORLD_MODEL_MODALITIES}
    for layer in manifest.raster_layers:
        modality = _modality_for_layer(layer.layer_type)
        if modality:
            modalities[modality] = True
    for layer in manifest.vector_layers:
        modality = _modality_for_layer(layer.layer_type)
        if modality:
            modalities[modality] = True
    modalities["sparse_uav_imagery"] = bool(manifest.uav_observations)
    modalities["orthomosaic"] = bool(manifest.orthomosaic_tiles)
    return modalities


def _modality_for_asset(manifest: GeoDatasetManifest, asset_id: str) -> str | None:
    for layer in manifest.raster_layers:
        if layer.layer_id == asset_id:
            return _modality_for_layer(layer.layer_type)
    for layer in manifest.vector_layers:
        if layer.layer_id == asset_id:
            return _modality_for_layer(layer.layer_type)
    return None


def _modality_for_layer(layer_type: str) -> str | None:
    normalized = layer_type.lower()
    if normalized in ELEVATION_LAYER_TYPES:
        return "elevation"
    if normalized in ICE_LAYER_TYPES:
        return "ice"
    if normalized in WEATHER_LAYER_TYPES:
        return "weather"
    if normalized in LAND_COVER_LAYER_TYPES:
        return "land_cover"
    if normalized in HYDRO_LAYER_TYPES:
        return "hydro"
    return None


def _manifest_hash(manifest: GeoDatasetManifest) -> str:
    payload = manifest.to_record()
    if isinstance(payload, dict):
        payload["manifest_hash"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
