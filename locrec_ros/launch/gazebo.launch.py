"""The demonstration in Gazebo: the tunnel, the vehicles, their estimators, rviz and the viewer.

    ros2 launch locrec_ros gazebo.launch.py                      # the ground robot and its markers
    ros2 launch locrec_ros gazebo.launch.py vehicle:=drone       # two drones: forward gaze, glance gaze
    ros2 launch locrec_ros gazebo.launch.py vehicle:=team        # the robot mounts strips, the drone uses them
    ros2 launch locrec_ros gazebo.launch.py record:=$HOME/ugv.mp4
    ros2 launch locrec_ros gazebo.launch.py gazebo_gui:=false rviz:=false    # headless

``gz_driver`` starts a Gazebo server on the tunnel, the same one ``demo.launch.py`` runs in
MuJoCo, and every scan comes from Gazebo's GPU LiDAR. The estimators are the same nodes on
the same topics, and read their calibrated numbers from ``locrec/results/thresholds_gazebo.json``,
which ``locrec/experiments/calibrate_thresholds_gazebo.py`` measured on this sensor.

The ground robot runs two estimators on one stream of scans, one without markers and one
with them, and the robot mounts a strip wherever the second asks. The drones fly two copies
of the tunnel side by side, one looking forward and one allowed to glance back at structure
when its scans go degenerate, each steered by its own estimator.

``vehicle:=team`` is the two of them in one tunnel. The robot goes first and mounts strips where
its own registration goes degenerate, publishing each on ``/team/landmarks`` as it does: slot,
where it believes the strip is, the covariance of the observation that placed it, and the face
normal. The drone follows 30 m behind, mounts nothing, and runs two estimators on its one stream
of scans, one that ignores the strips and one that localizes against them in the robot's frame.
What that is worth is in ``locrec/results/team_pass.csv`` and it is a frame-consistency result: the
drone agrees with the robot, and does not become more right about the world than the robot was.

Windows: Gazebo, following the vehicle; rviz; and the viewer's rendering in rviz's image
panel. With ``record`` set the viewer writes the video and finishes by itself once the scans
stop, at the end of the tunnel; ctrl+c after that.
"""
import os
import pathlib

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
SHARE = pathlib.Path(get_package_share_directory("locrec_ros"))

SENSORS = {
    # the study's two LiDARs, as locrec.lidar defines SPINNING_360 and LIMITED_FOV
    "ugv": {"range": 10.0, "fov": 360.0, "azimuth_beams": 360, "elevation_beams": 16, "fov_elevation": 30.0},
    "drone": {"range": 30.0, "fov": 90.0, "azimuth_beams": 180, "elevation_beams": 112, "fov_elevation": 60.0},
}
DEFAULT_WORLD = {"ugv": "mixed", "drone": "junction", "team": "mixed"}
"""The team flies the mixed world, which is the ground robot's: the strips are only worth
anything where the robot's own registration went degenerate, and that is where the mixed world
puts its blind stretches."""


def default_thresholds() -> str:
    """The calibrated thresholds the locrec package installs, or empty if it has none.

    They are measured, not chosen (locrec/experiments/calibrate_thresholds*.py), and a threshold
    does not transfer between sensors, so each simulator has its own file."""
    try:
        found = pathlib.Path(get_package_share_directory("locrec")) / "results" / "thresholds_gazebo.json"
    except PackageNotFoundError:
        return ""
    return str(found) if found.is_file() else ""


def gui_env() -> dict:
    """Environment for the windows: X11 through XWayland, and the NVIDIA GPU where there is one."""
    env = {}
    if os.environ.get("DISPLAY"):
        env["QT_QPA_PLATFORM"] = "xcb"
    if any(pathlib.Path(d, "libGLX_nvidia.so.0").exists()
           for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib")):
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    return env


def estimator(name: str, vehicle: str, thresholds: str, *, use_markers: bool = False, gaze: str = "across",
              namespace: str = "", share_landmarks: bool = False,
              use_shared_landmarks: bool = False) -> Node:
    # The estimators run as locrec/experiments/gazebo_crosscheck.py ran run_pass, which is what the
    # demonstration seed was chosen from: the estimate starts where the driver's odometry frame
    # does, the world frame at the start, and registration is on one thread, because the result
    # of a 300 m run depends on the thread count (docs/failures.md number 30). The gaze policies
    # need the first of those too.
    params = {"thresholds_file": thresholds, "platform": vehicle, "use_markers": use_markers,
              "start_at_odometry": True, "registration_threads": 1,
              "share_landmarks": share_landmarks, "use_shared_landmarks": use_shared_landmarks,
              **SENSORS[vehicle]}
    if vehicle == "drone":
        params["gaze"] = gaze
    remaps = [
        ("/localizability", f"/{name}/localizability"),
        ("/localizability/recommended_action", f"/{name}/localizability/recommended_action"),
        ("/localizability/estimate", f"/{name}/localizability/estimate"),
    ]
    if namespace:
        remaps += [(t, f"{namespace}{t}") for t in ("/points", "/odom_prior", "/scan_report")]
    return Node(package="locrec_estimator", executable="localizability_node", name=f"locrec_{name}",
                output="screen", parameters=[params], remappings=remaps)


def follow_command(target: str, offset: tuple, gain: float = 0.1) -> list:
    """Ask the Gazebo window to follow a model, until it does.

    Not until the request is accepted: the window accepts a follow request for a model that is
    not in its scene yet, finds it missing on its next frame, and drops it (gz-gui 8,
    CameraTracking.cc), which is what the window does when it is up before the driver's world
    is. So the request is repeated until ``/gui/currently_tracked`` names the model. It goes on
    ``/gui/track``, which takes a gain: at the default 0.01 per frame the camera trailed a
    moving robot by tens of metres. That plugin reads the track gain for following, so both
    are set.
    """
    x, y, z = offset
    request = (f"track_mode: FOLLOW follow_target {{ name: \"{target}\" }} "
               f"follow_offset {{ x: {x} y: {y} z: {z} }} follow_pgain: {gain} track_pgain: {gain}")
    script = (
        f"for i in $(seq 120); do "
        f"gz topic -t /gui/track -m gz.msgs.CameraTrack -p '{request}' > /dev/null 2>&1; "
        f"sleep 3; "
        f"timeout 5 gz topic -e -t /gui/currently_tracked -n 1 2>/dev/null | grep -q 'name: \"{target}\"' && break; "
        f"done"
    )
    return ["bash", "-c", script]


def launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    # a Gazebo transport partition of this launch's own, shared by the driver, its server and the
    # window: two launches on one machine otherwise answer each other's step requests
    gz_env = {"GZ_PARTITION": os.environ.get("GZ_PARTITION") or f"locrec_{os.getpid()}"}

    vehicle = arg("vehicle")
    if vehicle not in DEFAULT_WORLD:
        raise ValueError(f"vehicle must be ugv, drone or team, got {vehicle!r}")
    world = arg("world") or DEFAULT_WORLD[vehicle]
    thresholds = arg("thresholds_file")
    truthy = ("true", "1", "yes")

    if vehicle == "ugv":
        estimators = [estimator("baseline", "ugv", thresholds),
                      estimator("markers", "ugv", thresholds, use_markers=True)]
        # steeply from above and behind: from the side at a shallow angle the tunnel's own
        # 2.6 m wall hides the robot in the middle of it
        follow = follow_command("ugv", (-3.0, 0.0, 8.5))
    elif vehicle == "team":
        # one stream of scans per vehicle, on the vehicle's own namespace, and three estimators:
        # the robot's, which mounts and shares, and the drone's two, which differ only in whether
        # they are told about the strips
        estimators = [
            estimator("markers", "ugv", thresholds, use_markers=True, namespace="/ugv",
                      share_landmarks=True),
            estimator("solo", "drone", thresholds, gaze="forward", namespace="/drone"),
            estimator("team", "drone", thresholds, gaze="forward", namespace="/drone",
                      use_shared_landmarks=True),
        ]
        follow = follow_command("ugv", (-3.0, 0.0, 8.5))
    else:
        estimators = [estimator(name, "drone", thresholds, gaze=name, namespace=f"/{name}")
                      for name in ("forward", "glance")]
        # the anchor gz_driver moves along the glance drone's track: following the drone itself
        # would swing the camera round with every glance, the offset being in the model's frame
        follow = follow_command("glance_track", (-3.0, 0.0, 8.5))

    driver = Node(
        package="locrec_ros", executable="gz_driver", output="screen", additional_env=gz_env,
        parameters=[{
            "vehicle": vehicle, "world": world, "length": float(arg("length")), "seed": int(arg("seed")),
            "rate_hz": float(arg("rate_hz")), "lockstep": arg("lockstep").lower() in truthy,
            "team_lag": int(arg("team_lag")),
        }],
    )
    viewer = Node(
        package="locrec_ros", executable="demo_viewer", output="screen",
        parameters=[{
            "mode": vehicle, "simulator": "Gazebo", "world_name": world, "seed": int(arg("seed")),
            "length": float(arg("length")),
            "record": arg("record"), "thresholds_file": thresholds, "snapshot_dir": arg("snapshot_dir"),
            # The driver legitimately pauses while it waits for estimates, up to lockstep_timeout
            # and longer just after a vehicle sets off. The viewer ends its recording after
            # idle_finish seconds without a scan, and at its default of 3 s it read one of those
            # pauses as the end of the run and closed the team video at 90 m of 300
            # (docs/failures.md number 37). This has to outlast the driver's worst pause.
            "idle_finish": 35.0,
            # the same number the driver holds the drone back by: the viewer's frame-gap panel
            # compares the drone against where the robot was, not against where it is now
            "team_lag": int(arg("team_lag")),
            "rate_hz": float(arg("rate_hz")), "playback_speed": float(arg("playback_speed")),
            "renderer": arg("renderer"), "render": arg("render").lower() in truthy,
        }],
    )
    actions = [*estimators, viewer]
    if arg("rviz").lower() in truthy:
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", str(SHARE / "rviz" / (arg("rviz_config") or f"gazebo_{vehicle}.rviz"))],
            additional_env=gui_env()))
    if arg("gazebo_gui").lower() in truthy:
        actions.append(ExecuteProcess(
            cmd=["gz", "sim", "-g", "-v", "1", "--gui-config", str(SHARE / "config" / arg("gazebo_gui_config"))],
            additional_env={**gui_env(), **gz_env}, output="log"))
        actions.append(ExecuteProcess(cmd=follow, additional_env=gz_env, output="log"))
    # the estimators and the viewer have to be listening before the first scan goes out
    actions.append(TimerAction(period=4.0, actions=[driver]))
    return actions


def generate_launch_description() -> LaunchDescription:
    gui_default = "true" if HAS_DISPLAY else "false"
    return LaunchDescription([
        DeclareLaunchArgument("vehicle", default_value="ugv", description="ugv, drone or team"),
        DeclareLaunchArgument("world", default_value="",
                              description="blind, mixed or junction; mixed for the ugv and "
                                          "junction for the drone when empty"),
        DeclareLaunchArgument("seed", default_value="1"),
        DeclareLaunchArgument("length", default_value="300.0"),
        DeclareLaunchArgument("rate_hz", default_value="4.0"),
        DeclareLaunchArgument("lockstep", default_value="true",
                              description="wait for every estimator after each scan, as run_pass does"),
        DeclareLaunchArgument("gazebo_gui", default_value=gui_default),
        DeclareLaunchArgument("rviz", default_value=gui_default),
        DeclareLaunchArgument("render", default_value="true"),
        DeclareLaunchArgument("record", default_value="", description="path of an mp4 to write, empty for none"),
        DeclareLaunchArgument("thresholds_file", default_value=default_thresholds()),
        DeclareLaunchArgument("snapshot_dir", default_value=""),
        DeclareLaunchArgument("playback_speed", default_value="2.75"),
        DeclareLaunchArgument("rviz_config", default_value="",
                              description="rviz layout; empty for gazebo_<vehicle>.rviz, "
                                          "gazebo_ugv_recording.rviz for a recording"),
        DeclareLaunchArgument("gazebo_gui_config", default_value="gazebo_gui.config",
                              description="the Gazebo window's layout; gazebo_gui_half.config for a recording"),
        DeclareLaunchArgument("team_lag", default_value="60",
                              description="scans the team's drone trails the robot by, 30 m at the 0.5 m step"),
        DeclareLaunchArgument("renderer", default_value="auto"),
        OpaqueFunction(function=launch_setup),
    ])
