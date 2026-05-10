# Canadian Layered Geo-AI World-Model Foundation

This repo now exposes a training-ready foundation layer for Canadian geospatial
world-model work. The foundation keeps the LeWM pixel model boundary intact and
adds canonical side channels for:

- elevation
- ice
- weather
- land cover
- hydro
- sparse UAV imagery
- orthomosaic data

## Code Path

Use `prepare_canadian_geo_world_model_manifest` when you already have a
canonical `GeoDatasetManifest`. It validates that all required modalities are
present and attaches a `LeWMTrainingBinding` with these model keys:

- `pixels`: sparse UAV frame IDs when available, otherwise orthomosaic IDs
- `orthomosaic`: orthomosaic tile IDs
- `geo_layers`: raster layers plus vector layers such as hydro
- `static_geo_layers`: elevation, land cover, and hydro
- `dynamic_geo_layers`: weather and ice
- `vector_layers`: vector assets that will be rasterized during materialization
- `geo_layer_channels`: deterministic per-band channel metadata
- `geo_layer_available` and `unavailable_layer_mask`: explicit missing-layer masks

Use `build_canadian_geo_world_model_foundation` to validate manifests, attach
bindings, assign leakage-safe train/validation/test splits, and build either a
dry-run plan, metadata JSONL, or HDF5 training output.

## Materialization

The HDF5 writer now attempts real local materialization when optional geospatial
dependencies are installed:

- sparse UAV frames are resized from local image files with Pillow
- orthomosaics, DEMs, weather rasters, ice rasters, land-cover rasters, masks,
  and uncertainty layers are cropped with rasterio windows
- hydro and other vector layers are rasterized into tile-aligned binary channels
- remote-only or missing assets produce zero-filled channels and explicit
  availability masks instead of silently disappearing

The dependency-light dry-run path remains available for planning and CI.

## Configuration

The default foundation recipe is in
`config/geo_world_model/canadian_layered_foundation.yaml`. It records the required
modalities, provider IDs, default train/validation/test split policy, tiling
defaults, HDF5 key layout, and public source catalog references.

## Public Canadian Sources

The built-in provider adapters are source-catalog adapters, not downloaders.
They preserve provenance and licence caveats while ingesting local files.

- Elevation: NRCan HRDEM / CanElevation Open Government data.
- Weather: ECCC MSC GeoMet, including the OGC API landing page and collections.
- Ice: Canadian Ice Service latest products and archive.
- Land cover: AAFC/NRCan land-cover products.
- Hydro: Canadian Hydrospatial Network, replacing the National Hydro Network as
  NRCan's high-resolution hydrospatial product.

Operators still need to verify product-specific licensing, attribution,
freshness, and mission suitability before training or deployment.
