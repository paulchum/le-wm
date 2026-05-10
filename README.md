# Drone AI Autonomy Kit

This repository is a drone AI autonomy-kit codebase that uses
**LeWorldModel (LeWM)** as a learned world-model component. The drone-facing API
is exposed through the `drone_ai` package; LeWM remains the internal component
for pixel-to-latent prediction, short-horizon candidate scoring, latent rollout,
and surprise/anomaly estimation.

The stack is intentionally layered:

- `drone_ai.AutonomyKit` orchestrates candidate generation, safety filtering,
  world-model scoring, optional flight-controller handoff, and mission logging.
- `drone_ai.LeWMWorldModelPlanner` adapts trained LeWM checkpoints to the drone
  planner interface.
- `drone_ai.SafetySupervisor` gates every proposed movement setpoint before it
  can reach PX4, ArduPilot, or another flight-controller adapter.
- `drone_ai.MissionLog` records auditable decision traces.
- `ground_station` provides a Vite/React operator dashboard for the reference
  platform and evidence pipeline.

LeWM does not own stabilization, direct motor control, operator authorization,
payload release, weapon release, target engagement, or safety interlocks.

## Quick Start

Run the source-tree Autonomy Kit demo without a LeWM checkpoint:

```bash
python3 examples/canadian_defence_autonomy_kit.py
```

Run the checkpoint-backed Autonomy Kit demo with the Hugging Face mirror:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]" huggingface_hub
python3 examples/checkpoint_backed_drone_demo.py --repo-id quentinll/lewm-pusht
```

Use the public package namespace:

```python
from drone_ai import (
    AutonomyKit,
    HeuristicWorldModelPlanner,
    LeWMCheckpointSpec,
    SafetySupervisor,
    load_hf_lewm_world_model_planner,
)
```

Architecture docs:

- `docs/repo_architecture.md` explains the drone AI repo layout.
- `docs/lewm_component.md` explains the LeWM component boundary.
- `docs/canadian_defence_full_stack.md` explains the Canadian defence Autonomy
  Kit and reference platform track.

Ground-station UI:

```bash
cd ground_station
npm install
npm run dev
```

PX4 VTOL simulation integration:

- `integrations/aas_ros2/` provides a ROS2 overlay for running `drone_ai`
  against a local `aerial-autonomy-stack` PX4 VTOL simulation.
- `scripts/prepare_aas_ros2_overlay.sh` applies the AAS dynamic-setpoint patch
  and links the overlay into the AAS aircraft workspace.
- `docs/aerial_autonomy_stack_integration.md` documents Ubuntu GPU setup,
  runtime topics, operator approval, and smoke tests.

## LeWM Component

LeWorldModel is retained in this repo as the learned world-model component.
The original LeWM implementation lives in `jepa.py`, `module.py`, `train.py`,
and `eval.py`, with training/evaluation configs under `config/train` and
`config/eval`.

Original LeWM authors:
[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/).

LeWM summary: Joint Embedding Predictive Architectures (JEPAs) learn compact
latent world models. LeWM trains end-to-end from raw pixels with a next-embedding
prediction loss and a Gaussian latent regularizer, then supports fast planning
and surprise detection in latent space.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the LeWM component
The LeWM component builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]"
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Drone autonomy integration

The `drone_ai` package productizes LeWM as a guarded component in a drone
autonomy stack. It adds:

- typed drone interfaces for observations, action candidates, safety decisions,
  operator inputs, and mission log records
- an `AutonomyKit` orchestration loop for candidate generation, safety filtering,
  world-model scoring, optional flight-controller handoff, and mission logging
- a `LeWMWorldModelPlanner` adapter exposing `score_action_candidates`,
  `rollout`, and `surprise`
- a `SafetySupervisor` that clips or rejects learned action proposals before
  PX4/ArduPilot receives setpoints
- `VelocityLatticeSampler`, `HeuristicWorldModelPlanner`, and
  `DryRunFlightController` helpers for source-tree demos and bench tests
- append-only `MissionLog` JSONL records with hash chaining for audit review
- drone AI and Canadian defence configurations at
  `config/drone_ai/autonomy_kit.yaml` and `config/drone/canadian_defence.yaml`

The integration layer is non-kinetic by default. Kinetic-support contexts can be
represented for mobility-only tasks, but this package blocks effects commands
such as autonomous target engagement, payload release, and weapon release. LeWM
remains advisory and does not replace the flight controller, safety supervisor,
operator, or mission system.

```bash
python3 examples/canadian_defence_autonomy_kit.py
```

See `docs/canadian_defence_full_stack.md` for the dual-track Autonomy Kit and
Canadian Reference Drone Platform architecture.

### Canadian Reference Drone Platform prototype

The reference-platform v1 packages the autonomy kit with a small VTOL under
25 kg, configurable payload slots, default EO/IR + depth payloads, a secure
data-pipeline manifest, support tiers, and BVLOS/L1C evidence scaffolding. It is
an evidence-readiness prototype only; it does not claim Transport Canada
approval and it must not store classified or controlled technical data in this
repo.

```bash
python3 examples/canadian_reference_platform_demo.py
```

The operator ground-station demo lives in `ground_station/`:

```bash
cd ground_station
npm install
npm run dev
```

If the local volume is low on free space, install/build may fail before the
frontend dependencies are available.

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

The drone adapter can load the same HF mirror directly for a source-tree smoke
test without converting to `_object.ckpt`:

```bash
python3 examples/checkpoint_backed_drone_demo.py \
  --repo-id quentinll/lewm-pusht \
  --local-dir $STABLEWM_HOME/hf_pusht
```

To run the integration test path that scores drone candidates with the real
checkpoint:

```bash
LEWM_RUN_CHECKPOINT_TEST=1 \
LEWM_CHECKPOINT_DIR=$STABLEWM_HOME/hf_pusht \
python3 -m unittest tests.test_checkpoint_backed_planner
```

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
