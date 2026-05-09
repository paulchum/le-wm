"""Geo-world-model input boundary for future layered fusion encoders."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GeoRasterLayer:
    """Raster-like geo context aligned to the UAV frame history."""

    data: Any
    valid_mask: Any | None = None
    uncertainty: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeoWorldModelInput:
    """Typed model-side input for layered geo-AI fusion.

    `pixels` are the legacy LeWM UAV frame tensor. All geo layers are converted
    into explicit side-channel tensors that current LeWM checkpoints ignore but
    future fusion encoders can consume.
    """

    pixels: Any
    orthomosaic: GeoRasterLayer | Any | None = None
    static_rasters: Mapping[str, GeoRasterLayer | Any] = field(default_factory=dict)
    dynamic_rasters: Mapping[str, GeoRasterLayer | Any] = field(default_factory=dict)
    vector_layers: Mapping[str, Any] = field(default_factory=dict)
    action_history: Any | None = None
    trajectory_history: Any | None = None
    proprioception_history: Any | None = None
    temporal_metadata: Mapping[str, Any] = field(default_factory=dict)
    geospatial_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_model_inputs(
        self, adapter: "GeoWorldModelInputAdapter | None" = None
    ) -> dict[str, Any]:
        """Convert this input with the default or provided adapter."""

        return (adapter or GeoWorldModelInputAdapter()).to_model_inputs(self)


class GeoWorldModelInputAdapter:
    """Convert typed geo-world-model inputs into LeWM model input dictionaries."""

    DEFAULT_STATIC_RASTER_CHANNELS = {
        "elevation": 1,
        "land_cover": 1,
        "hydro": 1,
    }
    DEFAULT_DYNAMIC_RASTER_CHANNELS = {
        "weather": 1,
        "ice": 1,
    }
    DEFAULT_IMAGERY_CHANNELS = {
        "orthomosaic": 1,
    }

    def __init__(
        self,
        *,
        static_raster_channels: Mapping[str, int] | None = None,
        dynamic_raster_channels: Mapping[str, int] | None = None,
        imagery_channels: Mapping[str, int] | None = None,
    ):
        self.static_raster_channels = dict(
            static_raster_channels or self.DEFAULT_STATIC_RASTER_CHANNELS
        )
        self.dynamic_raster_channels = dict(
            dynamic_raster_channels or self.DEFAULT_DYNAMIC_RASTER_CHANNELS
        )
        self.imagery_channels = dict(imagery_channels or self.DEFAULT_IMAGERY_CHANNELS)

    def to_model_inputs(self, value: GeoWorldModelInput) -> dict[str, Any]:
        """Return a model-ready dictionary with explicit missing-layer tensors."""

        torch = _torch()
        pixels = _as_tensor(value.pixels, "pixels")
        batch, steps, _channels, height, width = _validate_pixels(pixels)

        base_shape = (batch, steps, height, width)
        output: dict[str, Any] = {
            "pixels": pixels,
            "geo_temporal_metadata": dict(value.temporal_metadata),
            "geo_geospatial_metadata": dict(value.geospatial_metadata),
            "geo_layer_metadata": {},
        }

        self._add_history(
            output, "action", value.action_history, base_shape, pixels.device
        )
        self._add_history(
            output, "trajectory", value.trajectory_history, base_shape, pixels.device
        )
        self._add_history(
            output, "proprio", value.proprioception_history, base_shape, pixels.device
        )

        self._add_layer(
            output,
            category="imagery",
            name="orthomosaic",
            layer=value.orthomosaic,
            missing_channels=self.imagery_channels.get("orthomosaic", 1),
            base_shape=base_shape,
            device=pixels.device,
        )

        for name in _ordered_layer_names(
            self.static_raster_channels, value.static_rasters
        ):
            self._add_layer(
                output,
                category="static",
                name=name,
                layer=value.static_rasters.get(name),
                missing_channels=self.static_raster_channels.get(name, 1),
                base_shape=base_shape,
                device=pixels.device,
            )

        for name in _ordered_layer_names(
            self.dynamic_raster_channels, value.dynamic_rasters
        ):
            self._add_layer(
                output,
                category="dynamic",
                name=name,
                layer=value.dynamic_rasters.get(name),
                missing_channels=self.dynamic_raster_channels.get(name, 1),
                base_shape=base_shape,
                device=pixels.device,
            )

        for name, vector in value.vector_layers.items():
            key = f"geo_vector_{_safe_name(name)}"
            output[key] = _as_tensor(vector, key).to(device=pixels.device)
            output[f"geo_present_vector_{_safe_name(name)}"] = torch.tensor(
                True, dtype=torch.bool, device=pixels.device
            )

        return output

    def _add_history(
        self,
        output: dict[str, Any],
        key: str,
        value: Any | None,
        base_shape: tuple[int, int, int, int],
        device: Any,
    ) -> None:
        if value is None:
            return
        output[key] = _validate_history(key, _as_tensor(value, key), base_shape).to(
            device=device
        )

    def _add_layer(
        self,
        output: dict[str, Any],
        *,
        category: str,
        name: str,
        layer: GeoRasterLayer | Any | None,
        missing_channels: int,
        base_shape: tuple[int, int, int, int],
        device: Any,
    ) -> None:
        key = f"geo_{category}_{_safe_name(name)}"
        metadata = output["geo_layer_metadata"]
        if layer is None:
            data, mask, uncertainty, present = _missing_layer(
                missing_channels, base_shape, device
            )
            metadata[key] = {}
        else:
            raster = layer if isinstance(layer, GeoRasterLayer) else GeoRasterLayer(layer)
            data = _validate_raster_data(f"{key}.data", raster.data, base_shape)
            data = data.to(device=device)
            mask = _validate_mask(f"{key}.valid_mask", raster.valid_mask, data, base_shape)
            uncertainty = _validate_uncertainty(
                f"{key}.uncertainty", raster.uncertainty, data, base_shape
            )
            present = _present_tensor(base_shape, device)
            metadata[key] = dict(raster.metadata)

        output[key] = data
        output[f"geo_mask_{category}_{_safe_name(name)}"] = mask
        output[f"geo_uncertainty_{category}_{_safe_name(name)}"] = uncertainty
        output[f"geo_present_{category}_{_safe_name(name)}"] = present


def coerce_model_inputs(value: Any) -> Any:
    """Return legacy mappings unchanged or convert `GeoWorldModelInput` values."""

    if isinstance(value, GeoWorldModelInput):
        return GeoWorldModelInputAdapter().to_model_inputs(value)
    if isinstance(value, Mapping):
        return value
    raise TypeError(
        "model inputs must be a mapping or GeoWorldModelInput, "
        f"got {type(value).__name__}"
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("GeoWorldModelInput conversion requires PyTorch") from exc
    return torch


def _as_tensor(value: Any, name: str) -> Any:
    torch = _torch()
    if torch.is_tensor(value):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a torch tensor") from exc


def _validate_pixels(pixels: Any) -> tuple[int, int, int, int, int]:
    shape = tuple(int(dim) for dim in pixels.shape)
    if len(shape) != 5:
        raise ValueError(f"pixels must have shape (B, T, C, H, W), got {shape}")
    batch, steps, channels, height, width = shape
    if min(batch, steps, channels, height, width) <= 0:
        raise ValueError(f"pixels dimensions must be positive, got {shape}")
    return batch, steps, channels, height, width


def _validate_history(
    name: str, value: Any, base_shape: tuple[int, int, int, int]
) -> Any:
    batch, steps, _height, _width = base_shape
    shape = tuple(int(dim) for dim in value.shape)
    if len(shape) != 3:
        raise ValueError(f"{name} must have shape (B, T, D), got {shape}")
    if shape[0] != batch or shape[1] != steps:
        raise ValueError(
            f"{name} batch/time must match pixels {(batch, steps)}, got {shape[:2]}"
        )
    if shape[2] <= 0:
        raise ValueError(f"{name} feature dimension must be positive, got {shape}")
    return value.float()


def _validate_raster_data(
    name: str, value: Any, base_shape: tuple[int, int, int, int]
) -> Any:
    tensor = _as_tensor(value, name)
    tensor = _expand_time(name, tensor, base_shape)
    batch, steps, height, width = base_shape
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) != 5:
        raise ValueError(f"{name} must have shape (B, T, C, H, W), got {shape}")
    if shape[0] != batch or shape[1] != steps or shape[3:] != (height, width):
        raise ValueError(
            f"{name} must match pixels batch/time/height/width "
            f"{(batch, steps, height, width)}, got "
            f"{(shape[0], shape[1], shape[3], shape[4])}"
        )
    if shape[2] <= 0:
        raise ValueError(f"{name} channel dimension must be positive, got {shape}")
    return tensor


def _validate_mask(
    name: str,
    value: Any | None,
    data: Any,
    base_shape: tuple[int, int, int, int],
) -> Any:
    torch = _torch()
    batch, steps, height, width = base_shape
    data_channels = int(data.shape[2])
    if value is None:
        return torch.ones(
            batch,
            steps,
            data_channels,
            height,
            width,
            dtype=torch.bool,
            device=data.device,
        )

    mask = _as_tensor(value, name)
    shape = tuple(int(dim) for dim in mask.shape)
    if len(shape) == 4:
        mask = mask.unsqueeze(2)
    elif len(shape) != 5:
        raise ValueError(f"{name} must have shape (B, T, H, W) or (B, T, 1, H, W)")

    mask = _expand_time(name, mask, base_shape)
    shape = tuple(int(dim) for dim in mask.shape)
    if shape[0] != batch or shape[1] != steps or shape[3:] != (height, width):
        raise ValueError(
            f"{name} must match pixels batch/time/height/width "
            f"{(batch, steps, height, width)}, got "
            f"{(shape[0], shape[1], shape[3], shape[4])}"
        )
    if shape[2] not in (1, data_channels):
        raise ValueError(
            f"{name} channels must be 1 or match data channels {data_channels}, "
            f"got {shape[2]}"
        )
    if shape[2] == 1 and data_channels != 1:
        mask = mask.expand(batch, steps, data_channels, height, width)
    return mask.to(device=data.device, dtype=torch.bool)


def _validate_uncertainty(
    name: str,
    value: Any | None,
    data: Any,
    base_shape: tuple[int, int, int, int],
) -> Any:
    torch = _torch()
    if value is None:
        return torch.zeros_like(data, dtype=torch.float32)

    uncertainty = _validate_raster_data(name, value, base_shape).to(device=data.device)
    data_channels = int(data.shape[2])
    uncertainty_channels = int(uncertainty.shape[2])
    if uncertainty_channels not in (1, data_channels):
        raise ValueError(
            f"{name} channels must be 1 or match data channels {data_channels}, "
            f"got {uncertainty_channels}"
        )
    return uncertainty.float()


def _expand_time(name: str, tensor: Any, base_shape: tuple[int, int, int, int]) -> Any:
    if len(tuple(tensor.shape)) < 2:
        raise ValueError(f"{name} must include batch and time dimensions")
    target_steps = base_shape[1]
    shape = tuple(int(dim) for dim in tensor.shape)
    if shape[1] == target_steps:
        return tensor
    if shape[1] != 1:
        raise ValueError(
            f"{name} time dimension must be 1 or {target_steps}, got {shape[1]}"
        )
    return tensor.expand(shape[0], target_steps, *shape[2:])


def _missing_layer(
    channels: int, base_shape: tuple[int, int, int, int], device: Any
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    if channels <= 0:
        raise ValueError(f"missing layer channel count must be positive, got {channels}")
    batch, steps, height, width = base_shape
    data_shape = (batch, steps, channels, height, width)
    mask_shape = (batch, steps, 1, height, width)
    present_shape = (batch, steps)
    return (
        torch.zeros(data_shape, dtype=torch.float32, device=device),
        torch.zeros(mask_shape, dtype=torch.bool, device=device),
        torch.zeros(data_shape, dtype=torch.float32, device=device),
        torch.zeros(present_shape, dtype=torch.bool, device=device),
    )


def _present_tensor(base_shape: tuple[int, int, int, int], device: Any) -> Any:
    torch = _torch()
    batch, steps, _height, _width = base_shape
    return torch.ones(batch, steps, dtype=torch.bool, device=device)


def _ordered_layer_names(
    expected_channels: Mapping[str, int],
    provided_layers: Mapping[str, GeoRasterLayer | Any],
) -> tuple[str, ...]:
    names = list(expected_channels.keys())
    names.extend(name for name in provided_layers.keys() if name not in expected_channels)
    return tuple(names)


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
