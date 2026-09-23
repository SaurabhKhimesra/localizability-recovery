"""One command for the whole story, live on screen: a tunnel, two estimators, rviz, the simulator.

    ros2 launch locrec_ros demo.launch.py
    ros2 launch locrec_ros demo.launch.py seed:=1 record:=$HOME/locrec_demo.mp4
    ros2 launch locrec_ros demo.launch.py rviz:=false sim_window:=false    # headless

Both estimator nodes read the same /points and the same /odom_prior. One ignores
markers. The other asks for them, the simulator mounts them on the wall, and they come
back through /scan_report. Two windows open on a desktop: rviz2, drawing the tunnel,
the live scan (red while the robot is blind), the map, the true path against both
estimated ones, the markers and the numbers; and MuJoCo's own viewer on the simulation,
following the robot. The viewer node renders the same story on the GPU into rviz's image
panel, and into a video when ``record`` is set.

The calibrated numbers are read from locrec/results/thresholds.json by the nodes. Nothing here
carries a copy of them. With ``record`` set the video finishes by itself, with a results
card, once scans stop arriving at the end of the tunnel; ctrl+c after that.
"""
import os
import pathlib

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def default_thresholds() -> str:
    """The calibrated thresholds the locrec package installs, or empty if it has none.

    They are measured, not chosen (locrec/experiments/calibrate_thresholds*.py), and a threshold
    does not transfer between sensors, so each simulator has its own file."""
    try:
        found = pathlib.Path(get_package_share_directory("locrec")) / "results" / "thresholds.json"
    except PackageNotFoundError:
        return ""
    return str(found) if found.is_file() else ""


def gui_env() -> dict:
    """Environment for the two windows.

    rviz's OGRE renderer needs GLX, so on a Wayland desktop it goes through XWayland. On a
    machine with the NVIDIA driver both windows are offloaded to that GPU, which is harmless
    where it is already the primary one and is what uses it on a hybrid laptop.
    """
    env = {}
    if os.environ.get("DISPLAY"):
        env["QT_QPA_PLATFORM"] = "xcb"
        env["PYGLFW_LIBRARY_VARIANT"] = "x11"
    if any(pathlib.Path(d, "libGLX_nvidia.so.0").exists()
           for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib")):
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    return env


def estimator(name: str, use_markers: bool) -> Node:
    return Node(
        package="locrec_estimator",
        executable="localizability_node",
        name=f"locrec_{name}",
        output="screen",
        parameters=[{
            "thresholds_file": LaunchConfiguration("thresholds_file"),
            "range": 10.0, "fov": 360.0, "azimuth_beams": 360,
            "platform": "ugv", "use_markers": use_markers,
        }],
        remappings=[
            ("/localizability", f"/{name}/localizability"),
            ("/localizability/recommended_action", f"/{name}/localizability/recommended_action"),
            ("/localizability/estimate", f"/{name}/localizability/estimate"),
        ],
    )


def generate_launch_description() -> LaunchDescription:
    sim = Node(
        package="locrec_ros", executable="sim_publisher", output="screen",
        parameters=[{
            "platform": "ugv",
            "world": LaunchConfiguration("world"),
            "length": ParameterValue(LaunchConfiguration("length"), value_type=float),
            "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
            "rate_hz": ParameterValue(LaunchConfiguration("rate_hz"), value_type=float),
            "drop_markers": True,
            "action_topic": "/markers/localizability/recommended_action",
            "window": ParameterValue(LaunchConfiguration("sim_window"), value_type=bool),
        }],
        additional_env=gui_env(),
    )
    viewer = Node(
        package="locrec_ros", executable="demo_viewer", output="screen",
        parameters=[{
            "mode": "ugv", "simulator": "MuJoCo",
            "world_name": LaunchConfiguration("world"),
            "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
            "length": ParameterValue(LaunchConfiguration("length"), value_type=float),
            "record": LaunchConfiguration("record"),
            "thresholds_file": LaunchConfiguration("thresholds_file"),
            "snapshot_dir": LaunchConfiguration("snapshot_dir"),
            "rate_hz": ParameterValue(LaunchConfiguration("rate_hz"), value_type=float),
            "playback_speed": ParameterValue(LaunchConfiguration("playback_speed"), value_type=float),
            "renderer": LaunchConfiguration("renderer"),
            "render": ParameterValue(LaunchConfiguration("render"), value_type=bool),
        }],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", str(pathlib.Path(get_package_share_directory("locrec_ros")) / "rviz" / "demo.rviz")],
        additional_env=gui_env(),
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    gui_default = "true" if HAS_DISPLAY else "false"
    return LaunchDescription([
        DeclareLaunchArgument("seed", default_value="1"),
        DeclareLaunchArgument("world", default_value="mixed"),
        DeclareLaunchArgument("length", default_value="300.0"),
        DeclareLaunchArgument("rate_hz", default_value="4.0"),
        DeclareLaunchArgument("rviz", default_value=gui_default,
                              description="open rviz2; on by default when there is a display"),
        DeclareLaunchArgument("sim_window", default_value=gui_default,
                              description="open MuJoCo's live viewer on the simulation"),
        DeclareLaunchArgument("render", default_value="true",
                              description="render the story on the GPU for rviz's image panel and videos"),
        DeclareLaunchArgument("record", default_value="",
                              description="path of an mp4 to write, empty for none"),
        DeclareLaunchArgument("thresholds_file", default_value=default_thresholds()),
        DeclareLaunchArgument("snapshot_dir", default_value="",
                              description="directory for a png every 150 frames, empty for none"),
        DeclareLaunchArgument("playback_speed", default_value="2.75",
                              description="how many times faster than real time the video plays"),
        DeclareLaunchArgument("renderer", default_value="auto",
                              description="gpu, cpu, or auto: the GPU when an EGL context can be made"),
        estimator("baseline", use_markers=False),
        estimator("markers", use_markers=True),
        viewer,
        rviz,
        # the estimators, and rviz, have to be up before the first scan goes out
        TimerAction(period=5.0, actions=[sim]),
    ])
