"""Checkpoint loading helpers for LeWM-backed drone demos."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .planner import LeWMPlannerConfig, LeWMWorldModelPlanner


LEWM_HF_MODEL_REPOS: dict[str, str] = {
    "pusht": "quentinll/lewm-pusht",
    "cube": "quentinll/lewm-cube",
    "tworooms": "quentinll/lewm-tworooms",
    "reacher": "quentinll/lewm-reacher",
}


@dataclass(frozen=True)
class LeWMCheckpointSpec:
    """Location of a Hugging Face mirrored LeWM checkpoint."""

    repo_id: str = LEWM_HF_MODEL_REPOS["pusht"]
    revision: str | None = None
    cache_dir: str | Path | None = None
    local_dir: str | Path | None = None
    config_filename: str = "config.json"
    weights_filename: str = "weights.pt"


def load_hf_lewm_checkpoint(
    spec: LeWMCheckpointSpec | None = None,
    *,
    device: str | None = None,
) -> Any:
    """Download and instantiate a LeWM model from the public HF mirror."""

    spec = spec or LeWMCheckpointSpec()
    checkpoint_dir = download_hf_lewm_checkpoint(spec)
    cfg = _read_checkpoint_config(checkpoint_dir, spec)
    model = _build_lewm_model(cfg)

    torch = _import_torch()
    weights_path = checkpoint_dir / spec.weights_filename
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility
        state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    if device:
        model.to(device)
    model.eval()
    if hasattr(model, "requires_grad_"):
        model.requires_grad_(False)

    model.lewm_checkpoint = {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "checkpoint_dir": str(checkpoint_dir),
        "action_dim": _action_dim_from_config(cfg),
    }
    return model


def load_hf_lewm_world_model_planner(
    spec: LeWMCheckpointSpec | None = None,
    *,
    device: str | None = None,
    planner_config: LeWMPlannerConfig | None = None,
) -> LeWMWorldModelPlanner:
    """Load a HF LeWM checkpoint and wrap it in the drone planner adapter."""

    spec = spec or LeWMCheckpointSpec()
    checkpoint_dir = download_hf_lewm_checkpoint(spec)
    cfg = _read_checkpoint_config(checkpoint_dir, spec)
    model = load_hf_lewm_checkpoint(spec, device=device)
    action_dim = _action_dim_from_config(cfg)

    config = planner_config or LeWMPlannerConfig()
    if config.model_action_dim is None and action_dim is not None:
        config = replace(config, model_action_dim=action_dim)
    if config.device is None and device is not None:
        config = replace(config, device=device)

    return LeWMWorldModelPlanner(model, config)


def download_hf_lewm_checkpoint(spec: LeWMCheckpointSpec | None = None) -> Path:
    """Ensure `config.json` and `weights.pt` are available locally."""

    spec = spec or LeWMCheckpointSpec()
    if spec.local_dir is not None:
        local_dir = Path(spec.local_dir).expanduser()
        config_path = local_dir / spec.config_filename
        weights_path = local_dir / spec.weights_filename
        if config_path.exists() and weights_path.exists():
            return local_dir

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Loading HF LeWM checkpoints requires huggingface_hub. "
            "Install the optional demo dependencies first."
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "cache_dir": str(Path(spec.cache_dir).expanduser()) if spec.cache_dir else None,
        "local_dir": str(Path(spec.local_dir).expanduser()) if spec.local_dir else None,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    config_path = Path(hf_hub_download(filename=spec.config_filename, **kwargs))
    weights_path = Path(hf_hub_download(filename=spec.weights_filename, **kwargs))
    if config_path.parent != weights_path.parent:
        raise RuntimeError(
            "Hugging Face returned checkpoint files in different directories: "
            f"{config_path.parent} and {weights_path.parent}"
        )
    return config_path.parent


def _read_checkpoint_config(checkpoint_dir: Path, spec: LeWMCheckpointSpec) -> dict[str, Any]:
    config_path = checkpoint_dir / spec.config_filename
    return json.loads(config_path.read_text())


def _build_lewm_model(cfg: dict[str, Any]) -> Any:
    torch = _import_torch()
    try:
        import stable_pretraining as spt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Loading a LeWM checkpoint requires stable_pretraining. "
            "Install stable-worldmodel or stable-pretraining in the active environment."
        ) from exc

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder_cfg = _clean_hydra_kwargs(cfg["encoder"])
    encoder = spt.backbone.utils.vit_hf(
        encoder_cfg["size"],
        patch_size=encoder_cfg["patch_size"],
        image_size=encoder_cfg["image_size"],
        pretrained=encoder_cfg.get("pretrained", False),
        use_mask_token=encoder_cfg.get("use_mask_token", False),
    )

    def mlp(name: str) -> MLP:
        mlp_cfg = _clean_hydra_kwargs(cfg[name])
        return MLP(
            input_dim=mlp_cfg["input_dim"],
            output_dim=mlp_cfg["output_dim"],
            hidden_dim=mlp_cfg["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(**_clean_hydra_kwargs(cfg["predictor"])),
        action_encoder=Embedder(**_clean_hydra_kwargs(cfg["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )


def _clean_hydra_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if not key.startswith("_") and key != "norm_fn"
    }


def _action_dim_from_config(cfg: dict[str, Any]) -> int | None:
    try:
        return int(cfg["action_encoder"]["input_dim"])
    except (KeyError, TypeError, ValueError):
        return None


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Loading LeWM checkpoints requires PyTorch.") from exc
    return torch
