from setuptools import find_packages, setup

package_name = "locrec_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/demo.launch.py", "launch/gazebo.launch.py"]),
        ("share/" + package_name + "/rviz", ["rviz/demo.rviz", "rviz/gazebo_ugv.rviz", "rviz/gazebo_drone.rviz",
                                                     "rviz/gazebo_team.rviz", "rviz/gazebo_ugv_recording.rviz",
                                                     "rviz/gazebo_team_recording.rviz"]),
        ("share/" + package_name + "/config", ["config/gazebo_gui.config", "config/gazebo_gui_half.config"]),
        # screen-recording helpers, run with `ros2 run locrec_ros record_windows.sh ...`
        ("lib/" + package_name, ["scripts/record_windows.sh", "scripts/rqt_plot_waiting.py",
                                 "scripts/xresize.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Saurabh Khimesra",
    maintainer_email="64199670+SaurabhKhimesra@users.noreply.github.com",
    description="Publishes localizability and a recommended action from a LiDAR stream.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_publisher = locrec_ros.sim_publisher:main",
            "demo_viewer = locrec_ros.demo_viewer:main",
            "gz_driver = locrec_ros.gz_driver:main",
        ],
    },
)
