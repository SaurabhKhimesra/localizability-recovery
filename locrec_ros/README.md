# locrec_ros

The ROS 2 layer of the project: nodes, launch files, rviz layouts and the Gazebo driver. It
publishes localizability and a recommended action from a LiDAR stream, runs the marker estimator
in the loop, and drives the Gazebo demonstrations. Targets ROS 2 Jazzy.

The algorithm itself lives in the `locrec` package, which has no ROS dependency, and this package
adds none to it: everything the node decides lives in `locrec_ros/detector.py`,
`locrec_ros/prior.py` and `locrec_ros/conversions.py`, none of which imports rclpy, so the logic
is testable without a graph.

Built and tested with ROS 2 Jazzy (RoboStack, Python 3.12) and Gazebo Harmonic (gz-sim 8).

## Build

Build instructions for the whole workspace are in the top-level [README](../README.md). In short:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/SaurabhKhimesra/localizability-recovery.git
pip install mujoco small-gicp                       # the two dependencies rosdep cannot install
pip install moderngl imageio imageio-ffmpeg          # demo_viewer only: GPU drawing and video
cd ~/ros2_ws && rosdep install --from-paths src -y --ignore-src
colcon build && source install/setup.bash
```

## Run against the simulator

`sim_publisher` walks a generated tunnel at the platform's own speed and publishes each scan on
`/points`, the noisy dead-reckoned pose as ordinary cumulative odometry on `/odom_prior`, and the
true pose on `/tf` so rviz has a frame to draw in. It is a demonstration harness, not a benchmark:
the 8-seed results come from `locrec/experiments/`, which runs the same estimator offline.

```bash
# terminal 1
ros2 run locrec_ros sim_publisher --ros-args \
  -p platform:=ugv -p world:=mixed -p length:=300.0 -p rate_hz:=4.0

# terminal 2. Every calibrated number has one source and no default: the file
# locrec/experiments/calibrate_thresholds.py writes, installed with the locrec package.
ros2 run locrec_ros localizability_node --ros-args \
  -p thresholds_file:=$(ros2 pkg prefix locrec)/share/locrec/results/thresholds.json \
  -p range:=10.0 -p fov:=360.0 -p platform:=ugv

# terminal 3
ros2 topic echo /localizability/recommended_action
ros2 topic echo /localizability --field ratio
```

Worlds are `blind`, `mixed` and `junction`. For the drone, `-p platform:=drone` on both nodes,
and on the detector `-p range:=30.0 -p fov:=90.0 -p azimuth_beams:=180`. With `platform:=drone`
the node reads `drone_ratio_threshold` from the same file, because the threshold does not transfer
between sensors.


## The demonstration

One command runs the whole story live, in two windows:

```bash
ros2 launch locrec_ros demo.launch.py
```

**rviz2** draws the tunnel's plan, the live scan (white while the geometry constrains
every direction, red while it is degenerate), the map built so far, the true path against
the two estimated ones, every marker strip, a line along the direction the estimate is free
to slide, and a label over the robot with the ratio. Its image panel shows the viewer's
rendering: the same scene, and the plots.

**The simulator window** is MuJoCo's own viewer on the world the LiDAR is ray cast
against: the robot, its rays and their hits, and each marker strip as the simulator
mounts it on the wall. The camera follows the robot; drag to orbit, scroll to zoom. It
draws its own copy of the world with the ceiling hidden, because MuJoCo's ray caster
skips any geom whose alpha is zero, and hiding the ceiling in the simulation's model
would take 130 of 5522 returns out of a scan.

Both windows open by default when there is a display, `rviz:=false` and `sim_window:=false`
close them, and `record:=$HOME/locrec_demo.mp4` also writes the video.

```bash
ros2 launch locrec_ros demo.launch.py record:=$HOME/locrec_demo.mp4
```

A UGV drives a 300 m tunnel. Two estimator nodes read the same `/points` and the same
`/odom_prior`. One has registration and nothing else. The other asks for markers where
the detector says the geometry has gone blind, `sim_publisher` mounts them on the wall,
and they come back through `/scan_report` as absolute fixes. `demo_viewer` draws what
arrives on the topics and nothing else: the live scan, red while it is degenerate, the map
it has seen, the markers, the true pose and both estimates, and plots of the ratio, of each
estimate's along-track error (the direction a featureless tunnel cannot observe and the
quantity the grids report) and of the markers mounted. Ground truth is on `/tf` for display
and scoring only. Neither estimator reads it.

The look is deliberately plain: the 3D view drawn flat on rviz's grey, matplotlib plots with
axis labels and units, and standard colours. The video is the run, then a closing figure of
the whole run's error and ratio with a table of mean, worst and final error, which the
viewer writes by itself once scans stop arriving at the end of the tunnel. Frames are driven by scans, not by the wall clock: output frame f shows
simulation time `f * playback_speed / 30`, so the pace is exact however fast the
machine renders, and a slow machine takes longer to write the same video. Drawing is on
the GPU, headless through EGL, where a context can be made, and on the CPU otherwise.

| Argument | Default | |
|---|---|---|
| `seed` | 1 | world and noise seed |
| `world` | `mixed` | `blind`, `mixed` or `junction` |
| `length` | 300.0 | metres |
| `rate_hz` | 4.0 | scans per second |
| `rviz` | true with a display | open rviz2 with `rviz/demo.rviz` |
| `sim_window` | true with a display | open MuJoCo's viewer on the simulation |
| `render` | true | render on the GPU for rviz's image panel and videos; false keeps only rviz's own drawing |
| `record` | empty | path of an mp4 to write, 1920x1080 at 30 fps |
| `playback_speed` | 2.75 | times real time. A 300 m run at 4 Hz is 150 s, so about 55 s of run plus the closing figure |
| `renderer` | `auto` | `gpu`, `cpu`, or `auto`: the GPU when an EGL context can be made |
| `snapshot_dir` | empty | a png every 150 frames |
| `thresholds_file` | `locrec/results/thresholds.json` of the installed `locrec` | the one source of every calibrated number |

**Which seed, and why it is not the best one.** Markers do not help on every seed. On the
mixed world over seeds 0 to 7, final along-track error without markers against the
scheduler is 0.15 to 1.40, 1.72 to 0.56, 0.64 to 1.05, 7.86 to 0.63, 3.17 to 2.02, 3.14 to
1.72, 4.36 to 0.02 and 0.58 to 0.77 m: better on five of eight, median paired benefit
1.15 m, medians 2.43 m against 0.91 m. Those are the UGV grid rerun at the recalibrated
threshold for M11, `locrec/results/monte_carlo_ugv.csv` in that milestone, which is not committed
on this branch yet. Seed 3 is the flattering one and seed 0 gets worse. The default is
seed 1 because its benefit, 1.16 m, is one of the two middle values; seed 4's 1.14 m is
the other. Those are final errors at 300 m. The video's closing table leads with the mean
and the worst along-track error over the whole run instead, because the dead-end wall comes
into view in the last metres and moves the final number more than the run does, and the
viewer writes every scan's numbers to a CSV beside the video.

The closed loop has no marker model of its own. Mounting is
`TunnelSim.drop_marker_on_wall`, detection is `TunnelSim.marker_detections`, the rule that
asks for a marker is `locrec.policies.LocalizabilityScheduler`, and the fix is the
estimator's own absolute stage: the same four calls `run_pass` makes, in the same order.
The order is what the stamps are for. A report is published for every scan, before the
scan, and the node holds each scan until the report with its stamp has arrived, so a
marker is always registered from the estimate that held when it was mounted.

## The demonstration in Gazebo

The same story with Gazebo doing the simulating, and a second vehicle:

```bash
ros2 launch locrec_ros gazebo.launch.py                    # the ground robot and its markers
ros2 launch locrec_ros gazebo.launch.py vehicle:=drone     # two drones: forward gaze and glance gaze
ros2 launch locrec_ros gazebo.launch.py vehicle:=team      # robot mounts strips, drone behind uses them
ros2 launch locrec_ros gazebo.launch.py vehicle:=team seed:=1 record:=$HOME/team.mp4   # the recorded one
ros2 launch locrec_ros gazebo.launch.py record:=$HOME/locrec_gazebo_ugv.mp4
ros2 launch locrec_ros gazebo.launch.py gazebo_gui:=false rviz:=false    # headless
```

`gz_driver` writes the tunnel out as an SDF world (`locrec.gazebo`), starts a headless
`gz sim` server on it, and moves the vehicles through it. Every scan comes from Gazebo's GPU
LiDAR, every marker strip is spawned into the running world when it is mounted, and the
driver publishes the same topics `sim_publisher` does, so the estimators and the viewer are
unchanged. Three windows open when there is a display: Gazebo's, following the vehicle
through a see-through ceiling (the ceiling still stops the LiDAR, measured beam for beam);
rviz2; and the viewer's rendering in rviz's image panel.

**The ground robot** runs the MuJoCo demonstration's two estimators on one stream of scans,
one without markers and one with them, and mounts a strip wherever the second asks.

**The drones** fly two copies of the junction tunnel, side by side, with the same seed: the
same tunnel and the same odometry noise, which is how `locrec/experiments/drone_gaze.py` pairs its
policies. Each carries the study's 90 by 60 degree, 30 m LiDAR and has its own estimator.
One looks forward. The other runs `GlanceGaze`: when its scans go degenerate it may turn the
sensor back toward structure it has already mapped, for at most 3 s, and then must look
forward for 5 s. In the study's MuJoCo grid glance gaze beat forward gaze in the junction
world, 1.17 m against 1.59 m of total error (the README drone table's Total, the hypotenuse of
the median along-track and lateral errors), and did worse in the mixed world. On Gazebo's LiDAR
the two are level in the junction world, 1.89 m against 1.91 m, and no difference either way
survives 8 seeds, below. The video's closing figure says both.

**Lockstep.** After every scan the driver waits for each estimator's estimate for that scan
and the action after it, and only then moves the vehicles. A vehicle therefore acts on what
its estimator made of the last scan, as `run_pass` does, however long registration takes,
and a live run does not drift from the offline one by an action arriving a scan late.

**Gazebo's LiDAR is a second sensor, and it was measured as one.** Its range noise is the
configured 2 cm. Its ranges are not ray cast: they are read out of rendered depth textures
sized by the beam count, and at the study's 360 beams a texture texel is 0.70 degrees wide.
Beams meeting a wall at a grazing angle came back up to 36 cm off, the median ratio on an
80 m stretch of the mixed tunnel rose by 47 percent, and the MuJoCo threshold never fired (`docs/failures.md` number 26). The
Gazebo sensors are therefore rendered with 6 times (robot) and 3 times (drone) the
horizontal beams, keeping every 6th or 3rd, which are exactly the study's beams. The
thresholds were then calibrated again on this sensor, by the procedure
`locrec/experiments/calibrate_thresholds.py` declares, into `locrec/results/thresholds_gazebo.json`,
which the launch file reads by default: 2.109e-3 for the robot against MuJoCo's 2.068e-3, 3.040e-3
for the drone against 2.888e-3, and a marker reliable range of 7.0 m on both.

**On Gazebo's sensor, 8 seeds, 300 m** (`locrec/experiments/gazebo_crosscheck.py`, `locrec/results/gazebo_crosscheck_*.csv`):

| | Without | With | Paired, 95 percent CI | Seeds better |
|---|---|---|---|---|
| Robot, mixed world, final along-track error, median: no markers, scheduler | 2.21 m | 0.85 m | +1.43 m [-0.32, +2.92] | 5 of 8 |
| Drone, junction world, final along-track error, median: forward, glance | 1.79 m | 1.61 m | +0.66 m [-1.23, +1.04] | 5 of 8 |
| Drone, junction world, Total as the README reports it: forward, glance | 1.91 m | 1.89 m | | |

The robot's result is the MuJoCo rerun's (2.43 m against 0.91 m, 5 of 8), with the same three
seeds made worse by markers. The drone's is not the MuJoCo grid's, and it is fragile: the same
cross-check on the box-per-visual worlds, whose scans differ from the meshes' by 0.5 mm at the
99th percentile, gave a Total of 1.72 m against 2.11 m.

**The recorded runs are the rule's seeds, and the robot's is its offline pass.** The seeds come
from a rule declared in the cross-check before it ran: the seed whose paired difference in mean
along-track error is at the median, ties to the lower seed. That gives seed 1 for the robot and
seed 4 for the drones. The estimators register on one thread and the driver
discards its first render as the offline pass does, and with that the recorded robot run's marker
estimator matches the offline `run_pass` of seed 1 on all 600 scans (`docs/failures.md` number
30). The two drones share one Gazebo server and one noise stream, so their live run is a different
draw of seed 4 than the offline passes.

**Requirements.** Gazebo Harmonic (gz-sim 8) with its Python bindings `gz.transport13` and
`gz.msgs10`. Verified on 2026-09-17 with RoboStack's `ros-jazzy-ros-gz`, gz-sim 8.10.0, on a
GTX 1650 with the server rendering headless through EGL. Not yet tried on a binary Jazzy
install.

| Argument | Default | |
|---|---|---|
| `vehicle` | `ugv` | `ugv` or `drone` |
| `world` | `mixed` for the ugv, `junction` for the drone | `blind`, `mixed` or `junction` |
| `seed`, `length`, `rate_hz` | 1, 300.0, 4.0 | as in `demo.launch.py` |
| `lockstep` | true | wait for every estimator after each scan |
| `gazebo_gui`, `rviz` | true with a display | the Gazebo window and rviz2 |
| `render`, `record`, `playback_speed`, `renderer`, `snapshot_dir` | | as in `demo.launch.py` |
| `thresholds_file` | `locrec/results/thresholds_gazebo.json` of the installed `locrec` | calibrated on Gazebo's sensor |

Each launch gets a Gazebo transport partition of its own, shared by the driver, its server and
the window, so two launches on one machine do not answer each other's requests. ctrl+c stops
everything, the Gazebo server included; measured, every process of either setup gone within 2 s.

## Run against a bag

```bash
ros2 run locrec_ros localizability_node --ros-args \
  -p thresholds_file:=$PWD/results/thresholds.json \
  -p range:=10.0 -p fov:=360.0 -p azimuth_beams:=360 -p platform:=ugv
ros2 bag play your_bag.db3 --remap /your/lidar/topic:=/points /your/odom/topic:=/odom_prior
```

On a bag from your own sensor that value is wrong, and there is nothing to copy in
its place. See the porting note at the end.

The node processes nothing without odometry. Each scan waits for `/odom_prior` at
its own stamp, interpolated between the samples either side, and a scan that
cannot be paired is dropped and counted in a warning rather than assumed still.
The odometry has to describe the LiDAR frame, or a frame that coincides with it;
the node warns once if `child_frame_id` and the cloud's `frame_id` differ.

**The team** puts a ground robot and a drone in **one** tunnel, not two copies. The robot goes
first, mounting a strip wherever its own registration goes degenerate, and publishes each one on
`/team/landmarks` as it does. The drone follows 30 m behind, mounts nothing, sees the strips with
its own 90 by 60 degree LiDAR, and runs two estimators on its one stream of scans: `solo`, which
ignores them, and `team`, which localizes against them in the robot's frame. Neither vehicle can
see the other: vehicle visuals are hidden from every LiDAR by visibility flags, as they are in
the other two modes.

The drone waits at the portal for the first 60 scans and publishes nothing until it sets off, so
its first scan is its first moving scan and the pass is the one `locrec/experiments/team_pass_gazebo.py`
measured on this sensor. Over 8 seeds at 300 m that pass ends within 1 to 14 cm of the robot's
frame on every seed, against a median 1.85 m alone, paired +1.78 m [+0.73, +3.41], and the two
vehicles' frames stay a median 0.06 m apart over the whole run, worst seed 0.17 m. In the world
frame the drone inherits the robot's chain error and does not beat it. The closing figure says
both.

### Recording the windows themselves

The viewer's recording (`record:=`) is a composed figure: 3D views, plots, a results table and
paragraphs of explanation. For a video that looks like what it is, an engineer's own screen,
`locrec_ros/scripts/record_windows.sh` records the real windows instead, tiled: Gazebo and rviz across the top,
two live `rqt_plot` windows across the bottom.

The plots are ordinary ROS topics. The localizability ratio is each estimator's own
`/<name>/localizability/ratio`. The along-track error against ground truth, and in the team the
drone's distance from the robot's frame at the same place in the tunnel, are published by
`demo_viewer` on `/demo/error/<estimator>` and `/demo/frame_gap/<solo|team>` as `std_msgs/Float64`;
it is the only node with the ground truth to compute them from, and no estimator subscribes.

```bash
ros2 run locrec_ros record_windows.sh team 1 300 ~/videos/locrec_team_seed1.mp4 6
ros2 run locrec_ros record_windows.sh ugv  1 300 ~/videos/locrec_ugv_seed1.mp4  5
```

Arguments are vehicle, seed, length and output, then how much to speed the real-time capture up.
It launches with every window sized to its tile of a 1920x1080 frame (`config/gazebo_gui_half.config`,
`rviz/gazebo_<vehicle>_recording.rviz`), captures each with ffmpeg's `x11grab` and tiles them.
Nothing is drawn over them.

What had to be true for it to work, since each of these failed once:

* **GNOME on Wayland** will not let one app read another's pixels, but Gazebo and rviz run through
  XWayland, and `x11grab` reads a single X window with plain `GetImage`. GStreamer's `ximagesrc`
  uses shared memory, which XWayland refuses.
* **Pick the largest window with the name, not the newest.** Qt opens small helper windows that
  carry the application's name, and the first attempt grabbed a 3x3 one.
* **Wait for Gazebo's window to settle.** It opens a window, loads its plugins, and replaces it; a
  capture of the first one is black and ends after 1.5 s. The recorder waits until the same window
  has been the largest match for 8 s, and restarts any capture that still drops.
* **Crop to even dimensions.** The compositor sizes windows as it likes; a 1087 px tall window made
  x264 refuse to encode.
* **The recording rviz layouts** hide the composite image panel and the floating status sentences:
  the first is empty with `render:=false`, and the second is narration rather than data.
* **rqt_plot adds its topics before it has discovered them.** It asks the graph for each topic's type
  while its widget is being built, a moment after its node is created, gets "does not exist", and
  plots nothing, without a word in the log. `locrec_ros/scripts/rqt_plot_waiting.py` runs the unmodified rqt_plot
  with that one lookup retried until discovery catches up, which takes 0.2 to 0.8 s here.
* **rqt_plot shows the last second, forever.** Its Plot plugin turns x autoscaling off and scrolls
  a window as wide as the axis was at startup, the 0 to 1 s fallback. The same launcher turns x
  autoscaling back on, which is the toolbar's "home" button: the axis covers the whole run.
* **rqt_plot opens at 321x169 and takes no geometry option**; `locrec_ros/scripts/xresize.py` resizes it through
  libX11 the way `wmctrl` would. The two plots share a title, so each is found by its client
  window's class, the second skipping the first.

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
| in, with `use_shared_landmarks` | `/team/landmarks` | the same, from a teammate. Registered when it arrives, long before the strip comes into view; a repeat of a slot already held is dropped rather than re-registered |

`recommended_action` is one of `none`, `drop_marker`, `yaw_to:<deg>`.

`sim_publisher` and `gz_driver` also publish `/scan_report` (every scan, with `drop_markers`
mounting a marker whenever the action topic says `drop_marker`) and the tunnel's plan on
`/world/outline`, latched. `demo_viewer` publishes its rendering on `/demo/image` and what
rviz draws under `/demo/`, and the scores it computes against ground truth as `std_msgs/Float64`
for live plotting: `/demo/error/<estimator>` (|along-track error|, m) and, with `vehicle:=team`,
`/demo/frame_gap/<solo|team>` (the drone's distance from the robot's frame at the same place, m).
No estimator subscribes to them.

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `thresholds_file` | empty | Path to `locrec/results/thresholds.json`, written by `locrec/experiments/calibrate_thresholds.py`. Every calibrated value below that is not given explicitly is read from it. |
| `ratio_threshold` | **required, no default** | Below this the scan is called degenerate. The node refuses to start without it, from the file or explicitly, and an explicit value wins. There is deliberately nothing to copy: the value depends on the sensor and the space, so a shipped default would be a number that looks calibrated and is not. From the file: `ratio_threshold` for a 360 degree scanner, `drone_ratio_threshold` for the drone. |
| `marker_reliable_range`, `scheduler_margin` | **required for the UGV, no default** | The two numbers the marker scheduler spaces by, `marker_reliable_range_m` and `scheduler_margin_m` in the file. The rule is `locrec.policies.LocalizabilityScheduler` itself. |
| `use_markers` | false | Hold each scan for the `/scan_report` with its stamp, register the markers it says were mounted, and correct the estimate with the ones it saw. |
| `gaze` | `across` | Drone only. `across` looks across the weak direction whenever the scan is degenerate. `forward` and `glance` are the study's `ForwardGaze` and `GlanceGaze` (`locrec.gaze`), fed what `run_pass` feeds them; they need the track heading from `/scan_report` and `start_at_odometry`. |
| `registration_threads` | 4 | Threads small_gicp registers with. The result of a long run depends on it, not only the speed: the study's grids and `gazebo.launch.py` use 1 (`docs/failures.md` number 30). |
| `start_at_odometry` | false | Start the estimate at the first scan's `/odom_prior` pose instead of the identity, which is where `run_pass` starts it when the odometry frame is the world frame at the start, as `gz_driver`'s is. |
| `min_lambda_per_point` | 0.0 | Absolute guard for the degenerate-but-tiny-scan case. |
| `range` | 10.0 | Sensor max range, metres. With `fov` and `azimuth_beams` it sets the registration range, by the rule the thresholds were calibrated with (`default_registration_range` in `locrec.runner`). |
| `fov` | 360.0 | Horizontal field of view, degrees. |
| `azimuth_beams` | 360 | Horizontal beams across `fov`: 360 for a VLP-16 class scanner, 180 for the drone's 90 degree scanner. Overstating it registers past the range where the scan samples the map finely enough, `docs/failures.md` number 2. |
| `elevation_beams`, `fov_elevation` | 16, 30.0 | Vertical beams and opening, degrees: 112 over 60 for the drone. They reach the marker measurement model and what a gaze policy thinks a yaw would see. |
| `platform` | `ugv` | `ugv` recommends markers, `drone` recommends a yaw. |
| `max_points` | 60000 | Subsample above this, to bound per-scan cost. |

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest locrec_ros/test -q
```

They cover the wire-format conversion (extra fields packed after xyz, row
padding, big endian, float64 fields, non-finite returns, truncated data, a
missing field) and the action rule for both platforms. `test_prior.py` covers the
two things the node once got wrong: a scan without a prior is refused, the prior
is paired by stamp (either arrival order, interpolated, never extrapolated, with a
bounded wait), and the registration range is the calibrated rule for both study
sensors. The message-assembly test skips itself when rclpy is absent.

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because with pytest 9 the ROS pytest plugins
(`launch_testing`, `launch_testing_ros`) use a hook argument pytest 9 removed and stop the run
before collection; none of these tests needs them.

## The one thing to get right when you port this

`ratio_threshold` is not portable. The ratio depends on the sensor's field of
view, its range and the size of the space, so a number calibrated on a 10 m
360 degree scanner in a 3.2 m tunnel means nothing on a 30 m 90 degree scanner or
in a 6 m drive. Run the sensor down a stretch you know is featureless, collect the
ratio, and take a high quantile of that distribution. The procedure is in
`locrec/experiments/calibrate_thresholds.py`.

That is why the parameter is required rather than defaulted. It used to ship 1.9e-3,
which was transcribed from a calibration run and then went stale when the calibration
changed, leaving two places claiming different numbers. One constant copied by eye
into a second place is `docs/failures.md` number 20, the bug this package already had
once.

Collect that distribution at more than one orientation of the corridor against your
map's voxel axes. The ratio in a featureless space is periodic in that angle over
90 degrees and moved by 19 percent across it here, and calibrating at a single
orientation produced a threshold that could only ever under-fire. `docs/failures.md`
number 21.
