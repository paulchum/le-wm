# AAS PX4 VTOL Dynamic-Setpoint Integration

This repo integrates with `aerial-autonomy-stack` (AAS) as an external ROS2,
Gazebo, PX4, and Docker runtime. The `drone_ai` package remains the autonomy,
safety, operator-gating, and audit layer. AAS remains the flight simulation and
autopilot interface layer.

The first supported path is PX4 VTOL dynamic trajectory setpoints. PX4 quad and
ArduPilot support are intentionally deferred until the VTOL path is proven.

## Runtime Boundary

- `drone_ai.AutonomyKit` builds and scores candidate movement setpoints.
- `drone_ai.SafetySupervisor` accepts, clips, or rejects candidates.
- `lewm_drone_ros` converts the selected safety-approved action into PX4 local
  NED trajectory velocity.
- AAS `px4_interface` owns PX4 mode/action state and publishes `/offboard_flag`.
- AAS `px4_offboard` publishes PX4 `TrajectorySetpoint` messages only after the
  `lewm_drone_ros` bridge has published a fresh approved setpoint.

The adapter never publishes direct motor commands and never bypasses PX4 or the
AAS action server.

## Local Checkout Shape

Keep the AAS clone local and ignored by git:

```bash
git clone https://github.com/paulchum/aerial-autonomy-stack.git aerial-autonomy-stack
```

Prepare the overlay:

```bash
scripts/prepare_aas_ros2_overlay.sh
```

The script applies `integrations/aas_ros2/patches/px4_offboard_dynamic_setpoint.patch`
to the local AAS checkout and symlinks these packages into
`aerial-autonomy-stack/aircraft/aircraft_ws/src/`:

- `lewm_drone_ros_msgs`
- `lewm_drone_ros`

Use `--force` to replace existing overlay links. The default symlink mode is
recommended because `lewm_drone_ros/setup.py` can package the parent
`drone_ai` and `lewm_drone` source packages during `colcon build`. If `--copy`
is used, the script also copies those two Python packages into the copied
`lewm_drone_ros` package so the ROS build still has the autonomy runtime.

## Ubuntu GPU Host Setup

AAS expects an Ubuntu GPU host with Docker and the NVIDIA container runtime.
From the AAS checkout:

```bash
cd aerial-autonomy-stack/scripts
./check_requirements.sh
./sim_build.sh
```

For a lower-cost smoke path, pull and tag the prebuilt images listed in the AAS
README instead of building locally.

## Build The Overlay

Inside the AAS aircraft workspace:

```bash
cd aerial-autonomy-stack/aircraft/aircraft_ws
colcon build --symlink-install
source install/setup.bash
```

The overlay package installs the `px4_vtol_bridge` console script.

## PX4 VTOL Runtime

Start AAS with one PX4 VTOL:

```bash
cd aerial-autonomy-stack/scripts
AUTOPILOT=px4 NUM_QUADS=0 NUM_VTOLS=1 WORLD=swiss_town HEADLESS=false ./sim_run.sh
```

In the aircraft container or a shell with the aircraft workspace sourced, run:

```bash
ros2 run lewm_drone_ros px4_vtol_bridge \
  --ros-args \
  -p mission_log_path:=/tmp/lewm_drone_px4_vtol.jsonl
```

Publish a runtime goal:

```bash
ros2 topic pub --once /lewm_drone/goal lewm_drone_ros_msgs/msg/NavigationGoal \
  "{goal_id: 'demo-leg-1', north_m: 250.0, east_m: 0.0, down_m: -80.0, yaw_rad: 0.0}"
```

Before approval, decisions are advisory and visible on:

```bash
ros2 topic echo /lewm_drone/decision_json
```

Authorize execution:

```bash
ros2 topic pub --once /lewm_drone/operator_approval lewm_drone_ros_msgs/msg/OperatorApproval \
  "{approved: true, operator_id: 'operator-1', command: 'approve', expires_at_s: 0.0, notes: 'PX4 VTOL sim smoke'}"
```

When the goal, PX4 local position, safety decision, and approval are fresh, the
bridge publishes:

```bash
ros2 topic echo /lewm_drone/px4_trajectory_setpoint
```

The AAS patch makes `px4_offboard` consume this topic when the PX4 interface is
in VTOL trajectory offboard mode.

## Smoke Checks

Run local repo tests:

```bash
python3 -m unittest
```

On the Ubuntu GPU host:

```bash
scripts/prepare_aas_ros2_overlay.sh --force
cd aerial-autonomy-stack/aircraft/aircraft_ws
colcon build --symlink-install
source install/setup.bash
python3 -c "import lewm_drone_ros; import drone_ai"
```

Expected behavior:

- No executable setpoints are published before `/lewm_drone/operator_approval`.
- `/lewm_drone/decision_json` contains `AutonomyDecision.to_record()` JSON.
- `/lewm_drone/px4_trajectory_setpoint` updates only while approval and inputs
  are fresh.
- The configured JSONL mission log contains the selected safety-approved action.
