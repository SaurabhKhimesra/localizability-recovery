from glob import glob

from setuptools import find_packages, setup

package_name = "locrec"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["locrec", "locrec.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # the calibrated thresholds the locrec_ros nodes read, written by experiments/calibrate_*.py
        ("share/" + package_name + "/results", glob("results/thresholds*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Saurabh Khimesra",
    maintainer_email="64199670+SaurabhKhimesra@users.noreply.github.com",
    description="Localizability-aware LiDAR odometry, marker estimation and scheduling policies.",
    license="MIT",
)
