"""Canadian geospatial source ingestion for layered geo-AI datasets."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from .geospatial_dataset import (
    AOIMetadata,
    BoundingBox,
    CRSDefinition,
    DataAge,
    GeoDatasetManifest,
    GeoGrid,
    OrthomosaicTile,
    RasterLayer,
    SourceProvenance,
    TemporalWindow,
    VectorLayer,
    require_valid_geospatial_dataset_manifest,
)
from .interfaces import JsonValue, to_jsonable
from .reference_platform import DataResidencyPolicy
from .data_pipeline import SecurityGateError

RASTER_EXTENSIONS = frozenset(
    {".tif", ".tiff", ".geotiff", ".vrt", ".img", ".jp2", ".grib", ".grb", ".grib2"}
)
VECTOR_EXTENSIONS = frozenset({".geojson", ".json", ".gpkg", ".shp", ".fgb"})

__all__ = [
    "CanadianGeoIngestionPipeline",
    "CanadianHydroProvider",
    "CanadianIceServiceProvider",
    "CanadianLandCoverProvider",
    "ECCCMSCGeoMetProvider",
    "GeoDatasetManifest",
    "GeoIngestionError",
    "GeoIngestionRequest",
    "GeoLayerManifest",
    "GeoProvider",
    "NRCanElevationProvider",
    "RemoteSourceRef",
    "UAVOrthomosaicProvider",
    "default_canadian_geo_providers",
]


@dataclass(frozen=True)
class RemoteSourceRef:
    """Remote/source catalog metadata preserved for local-file ingestion."""

    source_url: str
    access_url: str | None = None
    collection_id: str | None = None
    product_id: str | None = None
    service: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeoIngestionRequest:
    """Request to ingest one local geospatial layer through a named provider."""

    provider_id: str
    layer_id: str
    local_path: str
    layer_type: str | None = None
    asset_kind: str = "auto"
    classification_label: str = "unclassified"
    acquisition_time_s: float | None = None
    remote_sources: tuple[RemoteSourceRef, ...] = ()
    license_use_caveat: str | None = None
    collection_method: str = "local_file"
    crs_override: str | None = None
    vertical_datum: str = "source_declared_or_not_applicable"
    nominal_resolution_m: float | None = None
    source_scale: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeoLayerManifest:
    """Canonical ingestion record for one parsed source layer."""

    layer_id: str
    layer_type: str
    asset_kind: str
    local_path: str
    source_name: str
    license_use_caveat: str
    acquisition_time_s: float
    processed_time_s: float
    processing_duration_s: float
    crs: CRSDefinition
    resolution: Mapping[str, JsonValue]
    bounds: BoundingBox
    format_driver: str
    checksum_sha256: str
    size_bytes: int
    classification_label: str
    remote_sources: tuple[RemoteSourceRef, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, JsonValue]:
        return to_jsonable(self)  # type: ignore[return-value]


class GeoIngestionError(ValueError):
    """Raised when a local geospatial source cannot be normalized."""


class GeoProvider(Protocol):
    """Adapter interface for authoritative or local geospatial sources."""

    provider_id: str
    source_name: str
    default_layer_type: str
    default_asset_kind: str
    default_license_use_caveat: str
    default_remote_sources: tuple[RemoteSourceRef, ...]

    def ingest(
        self,
        request: GeoIngestionRequest,
        *,
        policy: DataResidencyPolicy,
    ) -> GeoLayerManifest:
        """Validate, parse, and normalize one local layer."""


@dataclass(frozen=True)
class _ParsedGeoFile:
    asset_kind: str
    crs: CRSDefinition
    bounds: BoundingBox
    resolution: Mapping[str, JsonValue]
    format_driver: str
    metadata: Mapping[str, Any]


class _BaseGeoProvider:
    provider_id = ""
    source_name = ""
    default_layer_type = "context"
    default_asset_kind = "auto"
    default_license_use_caveat = "Verify source-specific licence, attribution, and use caveats."
    default_remote_sources: tuple[RemoteSourceRef, ...] = ()

    def ingest(
        self,
        request: GeoIngestionRequest,
        *,
        policy: DataResidencyPolicy,
    ) -> GeoLayerManifest:
        label = _validate_classification_label(request.classification_label, policy)
        started_at_s = time.time()
        path = Path(request.local_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise GeoIngestionError(f"local_source_not_file:{path}")

        parsed = _parse_local_file(path, request, self.default_asset_kind)
        processed_at_s = time.time()
        checksum_sha256, size_bytes = _sha256_file(path)

        remote_sources = request.remote_sources or self.default_remote_sources
        metadata = {
            **dict(request.metadata),
            **dict(parsed.metadata),
            "provider_id": self.provider_id,
            "collection_method": request.collection_method,
        }
        return GeoLayerManifest(
            layer_id=request.layer_id,
            layer_type=request.layer_type or self.default_layer_type,
            asset_kind=parsed.asset_kind,
            local_path=str(path),
            source_name=self.source_name,
            license_use_caveat=request.license_use_caveat
            or self.default_license_use_caveat,
            acquisition_time_s=request.acquisition_time_s or started_at_s,
            processed_time_s=processed_at_s,
            processing_duration_s=processed_at_s - started_at_s,
            crs=parsed.crs,
            resolution=parsed.resolution,
            bounds=parsed.bounds,
            format_driver=parsed.format_driver,
            checksum_sha256=checksum_sha256,
            size_bytes=size_bytes,
            classification_label=label,
            remote_sources=remote_sources,
            metadata=metadata,
        )


class NRCanElevationProvider(_BaseGeoProvider):
    provider_id = "nrcan_elevation"
    source_name = "Natural Resources Canada elevation products"
    default_layer_type = "dem"
    default_asset_kind = "raster"
    default_license_use_caveat = (
        "NRCan/Open Canada elevation product; verify Open Government Licence - "
        "Canada attribution, no-endorsement terms, and product-specific notices."
    )
    default_remote_sources = (
        RemoteSourceRef(
            source_url="https://open.canada.ca/data/en/dataset/957782bf-847c-4644-a757-e383c0057995",
            collection_id="nrcan-hrdem",
            product_id="high-resolution-digital-elevation-model",
            service="open_canada_catalog",
        ),
    )


class ECCCMSCGeoMetProvider(_BaseGeoProvider):
    provider_id = "eccc_msc_geomet"
    source_name = "Environment and Climate Change Canada MSC GeoMet"
    default_layer_type = "weather"
    default_asset_kind = "auto"
    default_license_use_caveat = (
        "ECCC MSC GeoMet/Data Server product; verify ECCC end-use licence, "
        "Open Government terms, attribution, freshness, and product caveats."
    )
    default_remote_sources = (
        RemoteSourceRef(
            source_url="https://eccc-msc.github.io/open-data/msc-geomet/ogc_api_en/",
            access_url="https://api.weather.gc.ca/?f=html",
            collection_id="msc-geomet",
            service="ogc_api",
        ),
        RemoteSourceRef(
            source_url="https://eccc-msc.github.io/open-data/licence/readme_en/",
            collection_id="eccc-open-data-licence",
            service="licence",
        ),
    )


class CanadianIceServiceProvider(_BaseGeoProvider):
    provider_id = "canadian_ice_service"
    source_name = "Canadian Ice Service ice products"
    default_layer_type = "ice"
    default_asset_kind = "auto"
    default_license_use_caveat = (
        "Canadian Ice Service ice product; verify chart/product terms, "
        "observation validity, operational caveats, and attribution before use."
    )
    default_remote_sources = (
        RemoteSourceRef(
            source_url="https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations/latest-conditions/archive-overview.html",
            collection_id="canadian-ice-service-archive",
            service="canada_ca_archive",
        ),
        RemoteSourceRef(
            source_url="https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations/latest-conditions/products-guides/chart-descriptions.html",
            collection_id="canadian-ice-service-chart-descriptions",
            service="canada_ca_product_guide",
        ),
    )


class CanadianHydroProvider(_BaseGeoProvider):
    provider_id = "canadian_hydro"
    source_name = "Canadian Hydrospatial Network / National Hydro Network"
    default_layer_type = "hydro"
    default_asset_kind = "vector"
    default_license_use_caveat = (
        "NRCan Canadian Hydrospatial/NHN open data; verify Open Government "
        "Licence - Canada attribution and dataset-specific hydrographic notices."
    )
    default_remote_sources = (
        RemoteSourceRef(
            source_url="https://natural-resources.canada.ca/science-data/data-analysis/geospatial-data-tools-services/hydrographic-networks",
            collection_id="canadian-hydrospatial-network",
            service="nrcan_hydrographic_networks",
        ),
    )


class CanadianLandCoverProvider(_BaseGeoProvider):
    provider_id = "canadian_land_cover"
    source_name = "AAFC or NRCan land-cover products"
    default_layer_type = "land_cover"
    default_asset_kind = "raster"
    default_license_use_caveat = (
        "AAFC/NRCan land-cover product; verify Open Government Licence - "
        "Canada attribution, product accuracy, and class semantics."
    )
    default_remote_sources = (
        RemoteSourceRef(
            source_url="https://open.canada.ca/data/en/dataset/dedb79e3-1397-4aff-9756-9e78904b11b0",
            collection_id="aafc-annual-crop-inventory",
            service="open_canada_catalog",
        ),
    )


class UAVOrthomosaicProvider(_BaseGeoProvider):
    provider_id = "uav_orthomosaic"
    source_name = "Local UAV / orthomosaic files"
    default_layer_type = "orthomosaic"
    default_asset_kind = "orthomosaic"
    default_license_use_caveat = (
        "Local UAV/orthomosaic operator-provided data; verify collection "
        "authority, privacy, site permissions, and downstream sharing limits."
    )
    default_remote_sources = ()


class CanadianGeoIngestionPipeline:
    """Normalize Canadian geospatial provider layers into dataset manifests."""

    def __init__(
        self,
        policy: DataResidencyPolicy | None = None,
        providers: Iterable[GeoProvider] | None = None,
    ):
        self.policy = policy or DataResidencyPolicy()
        active_providers = tuple(providers or default_canadian_geo_providers())
        self.providers: dict[str, GeoProvider] = {
            provider.provider_id: provider for provider in active_providers
        }

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.providers))

    def ingest_layers(
        self,
        requests: Iterable[GeoIngestionRequest],
    ) -> tuple[GeoLayerManifest, ...]:
        normalized_requests = tuple(requests)
        for request in normalized_requests:
            _validate_classification_label(request.classification_label, self.policy)
        return tuple(self._ingest_layer(request) for request in normalized_requests)

    def create_dataset_manifest(
        self,
        requests: Iterable[GeoIngestionRequest],
        *,
        dataset_id: str,
        dataset_version: str = "0.1.0",
        output_path: str | Path | None = None,
        aoi_id: str | None = None,
        aoi_name: str | None = None,
        region: str = "canada",
        metadata: Mapping[str, Any] | None = None,
    ) -> GeoDatasetManifest:
        layers = self.ingest_layers(requests)
        if not layers:
            raise GeoIngestionError("at_least_one_layer_required")

        manifest_crs = layers[0].crs
        temporal_window = _temporal_union(layers)
        aoi_bounds = _bounds_union(layers, manifest_crs)
        aoi = AOIMetadata(
            aoi_id=aoi_id or f"{dataset_id}-aoi",
            name=aoi_name or f"{dataset_id} area of interest",
            country="Canada",
            region=region,
            bounds=aoi_bounds,
            temporal_window=temporal_window,
            crs=manifest_crs,
            metadata={"source": "canadian_geo_ingestion"},
        )
        raster_layers = tuple(_to_raster_layer(layer) for layer in layers if layer.asset_kind == "raster")
        vector_layers = tuple(_to_vector_layer(layer) for layer in layers if layer.asset_kind == "vector")
        orthomosaic_tiles = tuple(
            _to_orthomosaic_tile(layer)
            for layer in layers
            if layer.asset_kind == "orthomosaic"
        )
        manifest_metadata = {
            **dict(metadata or {}),
            "ingestion_layers": [layer.to_record() for layer in layers],
            "ingestion_provider_ids": [layer.metadata.get("provider_id", "") for layer in layers],
        }
        manifest = GeoDatasetManifest(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            aoi=aoi,
            crs=manifest_crs,
            temporal_window=temporal_window,
            raster_layers=raster_layers,
            vector_layers=vector_layers,
            uav_observations=(),
            trajectory=(),
            labels=(),
            masks=(),
            uncertainty=(),
            orthomosaic_tiles=orthomosaic_tiles,
            training_bindings=(),
            metadata=manifest_metadata,
        )
        manifest = replace(manifest, manifest_hash=_manifest_hash(manifest))
        manifest = require_valid_geospatial_dataset_manifest(manifest)

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(manifest.to_record(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return manifest

    def _ingest_layer(self, request: GeoIngestionRequest) -> GeoLayerManifest:
        provider = self.providers.get(request.provider_id)
        if provider is None:
            raise GeoIngestionError(f"unknown_geo_provider:{request.provider_id}")
        return provider.ingest(request, policy=self.policy)


def default_canadian_geo_providers() -> tuple[GeoProvider, ...]:
    """Return the built-in Canadian geospatial provider adapters."""

    return (
        NRCanElevationProvider(),
        ECCCMSCGeoMetProvider(),
        CanadianIceServiceProvider(),
        CanadianHydroProvider(),
        CanadianLandCoverProvider(),
        UAVOrthomosaicProvider(),
    )


def _validate_classification_label(
    classification_label: str,
    policy: DataResidencyPolicy,
) -> str:
    label = classification_label.lower()
    if label in policy.blocked_labels:
        raise SecurityGateError(f"blocked_classification_label:{label}")
    if label not in policy.allowed_labels:
        raise SecurityGateError(f"unknown_classification_label:{label}")
    return label


def _parse_local_file(
    path: Path,
    request: GeoIngestionRequest,
    default_asset_kind: str,
) -> _ParsedGeoFile:
    asset_kind = _resolve_asset_kind(path, request.asset_kind, default_asset_kind)
    if asset_kind in {"raster", "orthomosaic"}:
        return _parse_raster(path, request, asset_kind)
    if asset_kind == "vector":
        return _parse_vector(path, request)
    raise GeoIngestionError(f"unsupported_asset_kind:{asset_kind}")


def _resolve_asset_kind(path: Path, requested: str, default_asset_kind: str) -> str:
    requested_kind = requested.lower()
    if requested_kind != "auto":
        return requested_kind
    if default_asset_kind != "auto":
        return default_asset_kind
    suffix = path.suffix.lower()
    if suffix in RASTER_EXTENSIONS:
        return "raster"
    if suffix in VECTOR_EXTENSIONS:
        return "vector"
    raise GeoIngestionError(f"cannot_infer_asset_kind:{path}")


def _parse_raster(
    path: Path,
    request: GeoIngestionRequest,
    asset_kind: str,
) -> _ParsedGeoFile:
    try:
        import rasterio
    except ImportError as exc:
        raise GeoIngestionError("rasterio_required_for_raster_ingestion") from exc

    with rasterio.open(path) as dataset:
        crs = _crs_from_rasterio(dataset.crs, request)
        bounds = BoundingBox(
            min_x=float(dataset.bounds.left),
            min_y=float(dataset.bounds.bottom),
            max_x=float(dataset.bounds.right),
            max_y=float(dataset.bounds.top),
            crs=crs,
        )
        resolution_x = abs(float(dataset.res[0]))
        resolution_y = abs(float(dataset.res[1]))
        resolution_m = request.nominal_resolution_m or _resolution_m_from_crs(
            crs,
            resolution_x,
            resolution_y,
        )
        resolution = {
            "x": resolution_x,
            "y": resolution_y,
            "unit": _axis_unit_from_crs(dataset.crs),
            "nominal_m": resolution_m,
            "source": "raster_transform",
        }
        metadata = {
            "width_px": int(dataset.width),
            "height_px": int(dataset.height),
            "band_count": int(dataset.count),
            "bands": [f"band_{index}" for index in range(1, dataset.count + 1)],
            "dtypes": list(dataset.dtypes),
            "nodata": to_jsonable(dataset.nodata),
            "transform": tuple(float(value) for value in dataset.transform),
        }
        return _ParsedGeoFile(
            asset_kind=asset_kind,
            crs=crs,
            bounds=bounds,
            resolution=resolution,
            format_driver=str(dataset.driver),
            metadata=metadata,
        )


def _parse_vector(path: Path, request: GeoIngestionRequest) -> _ParsedGeoFile:
    try:
        import pyogrio
    except ImportError as exc:
        raise GeoIngestionError("pyogrio_required_for_vector_ingestion") from exc

    info = pyogrio.read_info(path)
    crs_value = info.get("crs")
    if crs_value is None and path.suffix.lower() in {".geojson", ".json"}:
        crs_value = "EPSG:4326"
        crs_caveat = "GeoJSON CRS absent; defaulted to EPSG:4326 per RFC 7946."
    else:
        crs_caveat = None
    crs = _crs_from_text(str(crs_value) if crs_value else None, request)
    bounds_values = info.get("total_bounds")
    if bounds_values is None:
        bounds_values = info.get("bbox")
    if bounds_values is None:
        bounds_values = _read_vector_bounds(path, pyogrio)
    if bounds_values is None:
        raise GeoIngestionError(f"missing_vector_bounds:{path}")
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_values]
    metadata = {
        "geometry_type": to_jsonable(info.get("geometry_type")),
        "feature_count": int(info.get("features") or 0),
        "fields": list(_metadata_sequence(info.get("fields"))),
        "dtypes": [str(dtype) for dtype in _metadata_sequence(info.get("dtypes"))],
    }
    if crs_caveat:
        metadata["crs_caveat"] = crs_caveat
    resolution: dict[str, JsonValue] = {
        "source": "vector_metadata",
        "nominal_m": request.nominal_resolution_m,
        "source_scale": request.source_scale,
    }
    if request.nominal_resolution_m is None and request.source_scale is None:
        resolution["status"] = "not_declared"
    return _ParsedGeoFile(
        asset_kind="vector",
        crs=crs,
        bounds=BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, crs=crs),
        resolution=resolution,
        format_driver=str(info.get("driver") or path.suffix.lstrip(".").upper()),
        metadata=metadata,
    )


def _read_vector_bounds(path: Path, pyogrio_module: Any) -> tuple[float, float, float, float] | None:
    try:
        _, bounds = pyogrio_module.read_bounds(path)
    except Exception:
        return None


def _metadata_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    try:
        return tuple(value)
    except TypeError:
        return (value,)
    if bounds is None:
        return None
    try:
        if getattr(bounds, "ndim", 1) == 2:
            return (
                float(bounds[0].min()),
                float(bounds[1].min()),
                float(bounds[2].max()),
                float(bounds[3].max()),
            )
        return tuple(float(value) for value in bounds)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _crs_from_rasterio(crs_obj: Any, request: GeoIngestionRequest) -> CRSDefinition:
    crs_text = request.crs_override
    if crs_text is None and crs_obj is not None:
        epsg = crs_obj.to_epsg()
        crs_text = f"EPSG:{epsg}" if epsg else crs_obj.to_string()
    return _crs_from_text(crs_text, request)


def _crs_from_text(crs_text: str | None, request: GeoIngestionRequest) -> CRSDefinition:
    horizontal = request.crs_override or crs_text
    if not horizontal:
        raise GeoIngestionError("missing_crs")
    return CRSDefinition(
        horizontal=horizontal,
        vertical_datum=request.vertical_datum,
        units="m",
    )


def _axis_unit_from_crs(crs_obj: Any) -> str:
    if crs_obj is None:
        return "unknown"
    try:
        axis_info = crs_obj.axis_info
        if axis_info:
            return str(axis_info[0].unit_name)
    except (AttributeError, IndexError, TypeError):
        pass
    return "unknown"


def _resolution_m_from_crs(
    crs: CRSDefinition,
    resolution_x: float,
    resolution_y: float,
) -> float:
    horizontal = crs.horizontal or ""
    if horizontal.upper() == "EPSG:4326":
        raise GeoIngestionError("nominal_resolution_m_required_for_geographic_raster")
    return (resolution_x + resolution_y) / 2.0


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _temporal_union(layers: tuple[GeoLayerManifest, ...]) -> TemporalWindow:
    acquisition_times = [layer.acquisition_time_s for layer in layers]
    return TemporalWindow(start_s=min(acquisition_times), end_s=max(acquisition_times))


def _bounds_union(
    layers: tuple[GeoLayerManifest, ...],
    target_crs: CRSDefinition,
) -> BoundingBox:
    transformed = [_transform_bounds(layer.bounds, target_crs) for layer in layers]
    return BoundingBox(
        min_x=min(bounds.min_x for bounds in transformed),
        min_y=min(bounds.min_y for bounds in transformed),
        max_x=max(bounds.max_x for bounds in transformed),
        max_y=max(bounds.max_y for bounds in transformed),
        crs=target_crs,
    )


def _transform_bounds(bounds: BoundingBox, target_crs: CRSDefinition) -> BoundingBox:
    source_crs = bounds.crs or target_crs
    if source_crs.horizontal == target_crs.horizontal:
        return replace(bounds, crs=target_crs)
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise GeoIngestionError("pyproj_required_for_mixed_crs_bounds") from exc
    transformer = Transformer.from_crs(
        source_crs.horizontal,
        target_crs.horizontal,
        always_xy=True,
    )
    min_x, min_y, max_x, max_y = transformer.transform_bounds(
        bounds.min_x,
        bounds.min_y,
        bounds.max_x,
        bounds.max_y,
    )
    return BoundingBox(
        min_x=float(min_x),
        min_y=float(min_y),
        max_x=float(max_x),
        max_y=float(max_y),
        crs=target_crs,
    )


def _to_raster_layer(layer: GeoLayerManifest) -> RasterLayer:
    width_px = int(layer.metadata.get("width_px") or 1)
    height_px = int(layer.metadata.get("height_px") or 1)
    resolution_m = _canonical_resolution_m(layer)
    bands = tuple(str(band) for band in layer.metadata.get("bands", ("band_1",)))
    dtypes = tuple(str(dtype) for dtype in layer.metadata.get("dtypes", ("unknown",)))
    return RasterLayer(
        layer_id=layer.layer_id,
        layer_type=layer.layer_type,
        uri=layer.local_path,
        grid=GeoGrid(
            width_px=width_px,
            height_px=height_px,
            resolution_m=resolution_m,
            origin_x=layer.bounds.min_x,
            origin_y=layer.bounds.min_y,
            crs=layer.crs,
        ),
        temporal_window=TemporalWindow(
            start_s=layer.acquisition_time_s,
            end_s=layer.acquisition_time_s,
        ),
        bands=bands,
        dtype=dtypes[0],
        provenance=_source_provenance(layer),
        data_age=_data_age(layer),
        nodata=layer.metadata.get("nodata"),  # type: ignore[arg-type]
        metadata={"ingestion": layer.to_record()},
    )


def _to_vector_layer(layer: GeoLayerManifest) -> VectorLayer:
    return VectorLayer(
        layer_id=layer.layer_id,
        layer_type=layer.layer_type,
        uri=layer.local_path,
        geometry_type=str(layer.metadata.get("geometry_type") or "Unknown"),
        crs=layer.crs,
        temporal_window=TemporalWindow(
            start_s=layer.acquisition_time_s,
            end_s=layer.acquisition_time_s,
        ),
        attributes=tuple(str(field) for field in layer.metadata.get("fields", ())),
        provenance=_source_provenance(layer),
        data_age=_data_age(layer),
        metadata={
            "bounds": layer.bounds,
            "ingestion": layer.to_record(),
        },
    )


def _to_orthomosaic_tile(layer: GeoLayerManifest) -> OrthomosaicTile:
    width_px = int(layer.metadata.get("width_px") or 1)
    height_px = int(layer.metadata.get("height_px") or 1)
    bands = tuple(str(band) for band in layer.metadata.get("bands", ("band_1",)))
    return OrthomosaicTile(
        tile_id=layer.layer_id,
        uri=layer.local_path,
        grid=GeoGrid(
            width_px=width_px,
            height_px=height_px,
            resolution_m=_canonical_resolution_m(layer),
            origin_x=layer.bounds.min_x,
            origin_y=layer.bounds.min_y,
            crs=layer.crs,
        ),
        temporal_window=TemporalWindow(
            start_s=layer.acquisition_time_s,
            end_s=layer.acquisition_time_s,
        ),
        bands=bands,
        provenance=_source_provenance(layer),
        data_age=_data_age(layer),
        metadata={"ingestion": layer.to_record()},
    )


def _canonical_resolution_m(layer: GeoLayerManifest) -> float:
    value = layer.resolution.get("nominal_m")
    if isinstance(value, int | float) and value > 0:
        return float(value)
    raise GeoIngestionError(f"missing_positive_nominal_resolution_m:{layer.layer_id}")


def _source_provenance(layer: GeoLayerManifest) -> SourceProvenance:
    primary_source = layer.remote_sources[0].source_url if layer.remote_sources else layer.local_path
    return SourceProvenance(
        source_id=str(layer.metadata.get("provider_id") or layer.source_name),
        source_name=layer.source_name,
        collection_method=str(layer.metadata.get("collection_method") or "local_file"),
        source_uri=primary_source,
        license=layer.license_use_caveat,
        classification_label=layer.classification_label,
        notes=f"checksum_sha256={layer.checksum_sha256}",
    )


def _data_age(layer: GeoLayerManifest) -> DataAge:
    return DataAge(
        observed_at_s=layer.acquisition_time_s,
        ingested_at_s=layer.processed_time_s,
    )


def _manifest_hash(manifest: GeoDatasetManifest) -> str:
    payload = manifest.to_record()
    if isinstance(payload, dict):
        payload["manifest_hash"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
