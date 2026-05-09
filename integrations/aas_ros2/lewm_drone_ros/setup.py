from pathlib import Path

from setuptools import setup


def _find_project_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "drone_ai").is_dir() and (parent / "lewm_drone").is_dir():
            return parent
    return None


packages = ["lewm_drone_ros"]
package_dir = {"lewm_drone_ros": "lewm_drone_ros"}

project_root = _find_project_root()
if project_root is not None:
    packages.extend(["drone_ai", "lewm_drone"])
    package_dir["drone_ai"] = str(project_root / "drone_ai")
    package_dir["lewm_drone"] = str(project_root / "lewm_drone")

setup(
    name="lewm_drone_ros",
    version="0.1.0",
    packages=packages,
    package_dir=package_dir,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/lewm_drone_ros"]),
        ("share/lewm_drone_ros", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="paulchum",
    maintainer_email="paulchum1@gmail.com",
    description="ROS2 bridge from drone_ai AutonomyKit decisions to AAS PX4 VTOL setpoints.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "px4_vtol_bridge = lewm_drone_ros.px4_vtol_bridge:main",
        ],
    },
)
