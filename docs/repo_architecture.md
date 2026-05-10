# Drone AI Repo Architecture

The repository is organized as a drone AI autonomy stack with LeWM as an
internal learned world-model component.

## Public Drone AI Layer

- `drone_ai`: public package namespace for product-facing imports
- `config/drone_ai/autonomy_kit.yaml`: default autonomy-kit component map
- `examples/canadian_defence_autonomy_kit.py`: source-tree dry-run example
- `integrations/aas_ros2`: optional ROS2 overlay for running the Autonomy Kit
  against aerial-autonomy-stack PX4 VTOL simulation

## Autonomy Implementation Layer

- `lewm_drone.interfaces`: drone observations, action candidates, safety
  decisions, operator input, and mission log records
- `lewm_drone.planner`: Autonomy Kit orchestration, candidate sampling, planner
  adapters, dry-run flight controller, and LeWM adapter
- `lewm_drone.safety`: hard safety envelope and effects-command blocking
- `lewm_drone.mission_log`: append-only JSONL audit logs with hash chaining
- `lewm_drone.reference_platform`: Canadian reference UAS profiles, payload
  compatibility, and platform validation
- `lewm_drone.qualification`: BVLOS/L1C-style evidence-pack tracking
- `lewm_drone.data_pipeline`: Canadian-resident data bundle manifests and
  release gates

## LeWM Component Layer

- `jepa.py`, `module.py`: model architecture
- `train.py`, `config/train`: training
- `eval.py`, `config/eval`: benchmark evaluation and planning

The public stack imports LeWM through `drone_ai.LeWMWorldModelPlanner`; direct
use of LeWM internals should stay limited to component training, evaluation, and
adapter development.

## AAS Runtime Integration

The AAS integration is selective. The local `aerial-autonomy-stack/` checkout is
ignored by git and used as the ROS2/Gazebo/PX4 runtime. Repo-owned integration
code lives under `integrations/aas_ros2/` and is overlaid into the AAS aircraft
workspace with `scripts/prepare_aas_ros2_overlay.sh`.

The first path is PX4 VTOL dynamic setpoints:

- `lewm_drone_ros_msgs` defines runtime goal and operator approval messages.
- `lewm_drone_ros` subscribes to PX4 state, runs `AutonomyKit`, publishes
  decision JSON, and publishes approved PX4 trajectory setpoints.
- The AAS `px4_offboard` patch consumes `/lewm_drone/px4_trajectory_setpoint`
  only in VTOL trajectory offboard mode.

See `docs/aerial_autonomy_stack_integration.md` for setup and smoke-test steps.
