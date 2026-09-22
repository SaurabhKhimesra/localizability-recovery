# locrec_estimator

The ROS 2 node, in C++ (rclcpp). It runs the localizability-aware odometry over incoming scans,
publishes the degeneracy signal and the estimate, recommends where to mount a marker or where to
look, and shares the strips it mounts with the rest of the team.

The algorithm is [`locrec_core`](../locrec_core), which has no ROS dependency. This package adds
the graph and nothing else: everything it decides lives in `src/detector.cpp`,
`include/locrec_estimator/prior_pairer.hpp` and `src/conversions.cpp`, none of which includes
rclcpp, so the logic is tested without a graph.

## Run

```bash
ros2 run locrec_estimator localizability_node --ros-args \
  -p thresholds_file:=$(ros2 pkg prefix locrec)/share/locrec/results/thresholds_gazebo.json \
  -p range:=10.0 -p fov:=360.0 -p platform:=ugv -p use_markers:=true
```

The launch files in [`locrec_ros`](../locrec_ros) start it with everything else.

## Topics

| Direction | Topic | Type |
|---|---|---|
| in | `/points` | `sensor_msgs/PointCloud2` |
| in, required | `/odom_prior` | `nav_msgs/Odometry`, cumulative, of the LiDAR frame |
| in, with `use_markers` or a `gaze` policy | `/scan_report` | `locrec_msgs/ScanReport`, one per scan, stamped like the scan: markers mounted and seen, and the track heading |
| out | `/localizability` | `locrec_msgs/Localizability` |
| out | `/localizability/recommended_action` | `std_msgs/String` |
| out | `/localizability/estimate` | `nav_msgs/Odometry`, the estimate for each scan, stamped like the scan |
| out, with `share_landmarks` | `/team/landmarks` | `locrec_msgs/SharedLandmark`, one per strip mounted: slot, position, the 2x2 that placed it, the face normal. Reliable, transient local, keep all, so a receiver that joins late still gets every one |
| in, with `use_shared_landmarks` | `/team/landmarks` | the same, from a teammate. Registered when it arrives, long before the strip comes into view; a repeat of a slot already held updates the position and nothing else |

`recommended_action` is one of `none`, `drop_marker`, `yaw_to:<deg>`.

Every scan waits for the `/odom_prior` at its own stamp, and with `use_markers` or a gaze policy
for the `/scan_report` carrying exactly that stamp. "The latest odometry message" is not the
prior: the pose is interpolated between the two samples either side of the scan, never
extrapolated, and a scan that cannot be paired is dropped and counted rather than passed on
without one.

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `thresholds_file` | empty | Path to `locrec/results/thresholds*.json`, written by `locrec/experiments/calibrate_thresholds_gazebo.py`. Every calibrated value below that is not given explicitly is read from it. |
| `ratio_threshold` | **required, no default** | Below this the scan is called degenerate. The node refuses to start without it, from the file or explicitly, and an explicit value wins. There is deliberately nothing to copy: the value depends on the sensor and the space, so a shipped default would be a number that looks calibrated and is not. From the file: `ratio_threshold` for a 360 degree scanner, `drone_ratio_threshold` for the drone. |
| `marker_reliable_range`, `scheduler_margin` | **required for the UGV, no default** | The two numbers the marker scheduler spaces by, `marker_reliable_range_m` and `scheduler_margin_m` in the file. The rule is `locrec::LocalizabilityScheduler` itself. |
| `use_markers` | false | Hold each scan for the `/scan_report` with its stamp, register the markers it says were mounted, and correct the estimate with the ones it saw. |
| `share_landmarks` | false | Publish every strip this vehicle mounts on `/team/landmarks`. Needs `use_markers`. |
| `use_shared_landmarks` | false | Localize against a teammate's strips, in the teammate's frame. A vehicle with this on mounts none of its own. |
| `gaze` | `across` | Drone only. `across` looks across the weak direction whenever the scan is degenerate. `forward` and `glance` are the study's policies (`locrec_core/gaze.hpp`); they need the track heading from `/scan_report` and `start_at_odometry`. |
| `registration_threads` | 4 | Threads small_gicp registers with. The result of a long run depends on it, not only the speed: the study's grids and `gazebo.launch.py` use 1 (`docs/failures.md` number 30). |
| `start_at_odometry` | false | Start the estimate at the first scan's `/odom_prior` pose instead of the identity, which is where the offline pass starts it when the odometry frame is the world frame at the start, as `gz_driver`'s is. |
| `min_lambda_per_point` | 0.0 | Absolute guard for the degenerate-but-tiny-scan case. |
| `range` | 10.0 | Sensor max range, metres. With `fov` and `azimuth_beams` it sets the registration range, by the rule the thresholds were calibrated with (`defaultRegistrationRange`). |
| `fov` | 360.0 | Horizontal field of view, degrees. |
| `azimuth_beams` | 360 | Horizontal beams across `fov`: 360 for a VLP-16 class scanner, 180 for the drone's 90 degree scanner. Overstating it registers past the range where the scan samples the map finely enough, `docs/failures.md` number 2. |
| `elevation_beams`, `fov_elevation` | 16, 30.0 | Vertical beams and opening, degrees: 112 over 60 for the drone. They reach the marker measurement model and what a gaze policy thinks a yaw would see. |
| `platform` | `ugv` | `ugv` recommends markers, `drone` recommends a yaw. |
| `max_points` | 60000 | Subsample above this, to bound per-scan cost. |

## Tests

```bash
colcon test --packages-select locrec_estimator && colcon test-result --all
```

They cover the wire-format conversion (extra fields packed after xyz, row padding, organised
clouds, non-finite returns, truncated data, a missing field, a zero quaternion), the calibration
file and its refusal to invent a default, the scan-to-prior pairing (either arrival order,
interpolated, never extrapolated, bounded, and the report join), and the detector: what it refuses
to start without, the registration range, the action rule for both platforms, and the team's
shared strips.

## Calibrating for a new sensor

`ratio_threshold` is not portable. The ratio depends on the sensor's field of view, its range and
the size of the space, so a number calibrated on a 10 m 360 degree scanner in a 3.2 m tunnel means
nothing on a 30 m 90 degree scanner or in a 6 m drive. Run the sensor down a stretch you know is
featureless, collect the ratio, and take a high quantile of that distribution. The procedure is in
`locrec/experiments/calibrate_thresholds_gazebo.py`.

That is why the parameter is required rather than defaulted: a default transcribed from one
calibration run goes stale the moment the calibration changes, and then two places claim different
numbers (`docs/failures.md` number 20).

Collect that distribution at more than one orientation of the corridor against your map's voxel
axes. The ratio in a featureless space is periodic in that angle over 90 degrees and moved by
19 percent across it here, and calibrating at a single orientation produced a threshold that could
only ever under-fire. `docs/failures.md` number 21.
