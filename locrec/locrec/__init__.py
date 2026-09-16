"""locrec: localizability as a resource.

Milestone 1 core: procedural tunnel worlds, a MuJoCo raycast LiDAR,
scan-to-local-map odometry with an IMU-like prior, and a localizability
detector read off the registration Hessian.
"""
from .lidar import LIMITED_FOV, SPINNING_360, Lidar, LidarSpec, Scan
from .localizability import (
    Localizability,
    LocalizabilityConfig,
    analyse_hessian,
    calibrate_threshold,
)
from .odometry import (
    LocalMap,
    MotionPrior,
    MotionPriorSpec,
    Odometry,
    OdometryConfig,
    OdomStep,
)
from .landmarks import LandmarkBook, LandmarkSpec, measurement_information
from .policies import (
    DropOnFailure,
    LocalizabilityScheduler,
    NoMarkers,
    OraclePlacement,
    UniformSpacing,
)
from .runner import RunConfig, RunResult, StepContext, run_pass
from .sim import Drone, MarkerDetection, PlatformSpec, TunnelSim, UGV
from .worlds import TunnelWorld, WorldSpec, build_world

__version__ = "0.1.0"

__all__ = [
    "LIMITED_FOV",
    "SPINNING_360",
    "Drone",
    "Lidar",
    "LidarSpec",
    "LocalMap",
    "Localizability",
    "LocalizabilityConfig",
    "MotionPrior",
    "MotionPriorSpec",
    "OdomStep",
    "Odometry",
    "OdometryConfig",
    "PlatformSpec",
    "DropOnFailure",
    "LandmarkBook",
    "LandmarkSpec",
    "LocalizabilityScheduler",
    "MarkerDetection",
    "NoMarkers",
    "OraclePlacement",
    "RunConfig",
    "StepContext",
    "UniformSpacing",
    "measurement_information",
    "RunResult",
    "Scan",
    "TunnelSim",
    "TunnelWorld",
    "UGV",
    "WorldSpec",
    "build_world",
    "analyse_hessian",
    "calibrate_threshold",
    "run_pass",
]
