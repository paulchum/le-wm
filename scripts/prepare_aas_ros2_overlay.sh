#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AAS_DIR="${AAS_DIR:-${REPO_ROOT}/aerial-autonomy-stack}"
MODE="symlink"
FORCE="false"

usage() {
  cat <<'USAGE'
Usage: scripts/prepare_aas_ros2_overlay.sh [--aas-dir PATH] [--symlink|--copy] [--force]

Applies the PX4 VTOL dynamic-setpoint patch to a local aerial-autonomy-stack
checkout and places the lewm_drone ROS2 overlay packages in aircraft_ws/src.

Default mode is --symlink so lewm_drone_ros setup.py can include the parent
drone_ai/lewm_drone packages during colcon builds.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --aas-dir)
      AAS_DIR="$2"
      shift 2
      ;;
    --symlink)
      MODE="symlink"
      shift
      ;;
    --copy)
      MODE="copy"
      shift
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PATCH_FILE="${REPO_ROOT}/integrations/aas_ros2/patches/px4_offboard_dynamic_setpoint.patch"
OVERLAY_ROOT="${REPO_ROOT}/integrations/aas_ros2"
WORKSPACE_SRC="${AAS_DIR}/aircraft/aircraft_ws/src"

if [ ! -d "$AAS_DIR" ]; then
  echo "AAS checkout not found: $AAS_DIR" >&2
  exit 1
fi
if [ ! -d "$WORKSPACE_SRC" ]; then
  echo "AAS aircraft workspace not found: $WORKSPACE_SRC" >&2
  exit 1
fi

if grep -q "lewm_trajectory_setpoint_sub_" "${AAS_DIR}/aircraft/aircraft_ws/src/offboard_control/src/px4_offboard.hpp"; then
  echo "AAS px4_offboard dynamic-setpoint patch already appears to be applied."
else
  echo "Applying AAS px4_offboard dynamic-setpoint patch..."
  git -C "$AAS_DIR" apply "$PATCH_FILE"
fi

for package_name in lewm_drone_ros lewm_drone_ros_msgs; do
  source_path="${OVERLAY_ROOT}/${package_name}"
  target_path="${WORKSPACE_SRC}/${package_name}"
  if [ -e "$target_path" ] || [ -L "$target_path" ]; then
    if [ "$FORCE" != "true" ]; then
      echo "Target exists, use --force to replace: $target_path" >&2
      exit 1
    fi
    rm -rf "$target_path"
  fi

  if [ "$MODE" = "copy" ]; then
    cp -R "$source_path" "$target_path"
    if [ "$package_name" = "lewm_drone_ros" ]; then
      cp -R "${REPO_ROOT}/drone_ai" "${target_path}/drone_ai"
      cp -R "${REPO_ROOT}/lewm_drone" "${target_path}/lewm_drone"
    fi
  else
    ln -s "$source_path" "$target_path"
  fi
  echo "Placed ${package_name} in AAS aircraft workspace using ${MODE} mode."
done

echo "AAS ROS2 overlay is prepared at: $WORKSPACE_SRC"
