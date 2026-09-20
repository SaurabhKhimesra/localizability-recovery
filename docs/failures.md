# Every silent failure found, with the measurement

Silent is the operative word. None of these announced themselves. Each produced a
pipeline that logged converged, inlier-rich registrations and a detector that
looked healthy, while the number that mattered was wrong. They are written down
because anyone building the same thing will hit them, and because the interesting
part of this project turned out to be the debugging rather than the policies.

## 1. A local map shorter than the sensor range makes the robot stop moving

A sliding window of the last twelve scans is six metres of tunnel. A ten metre
sensor therefore produces a scan that sticks out past both ends of the map, and
the only axial correspondences left are at the map's boundary, so every new scan
is pulled back toward the map centroid. The estimate advanced at 78 percent of
true speed and reported converged, inlier-rich registrations throughout:
**17.8 m of lag over 80 m** of straight tunnel. Fixed by cropping the map by
distance rather than by scan count.

## 2. Registering past the Nyquist range of the scan

A voxel map of pitch `v` can only represent structure the scan samples at `v / 2`
or finer. Past that the nearest map point to a far scan point sits up to a beam
spacing away in the wrong direction. Both platforms bottom out where the beam
spacing at the registration limit is half a voxel and fall off a cliff above it:
the UGV goes from **1.48 m to 12.66 m** of drift between 5.7 m and 8.5 m of
registration range, the drone from **0.36 m to 11.83 m** between 11.5 m and
28.5 m. `locrec/experiments/registration_range_sweep.py` has the full grid.

## 3. The registration's along-track information is not real

The calibrated Hessian claims a 4 to 5 mm standard deviation along a tunnel with
no along-track structure at all. Rebuilding the local map from ground-truth poses,
so it is a fixed and correct reference, does not make the drift vanish: it halves,
from about **1.0 m to 0.4 m over 120 m**, and what is left is a systematic
backward pull rather than a lock. The number is an artefact of treating a thousand
correlated correspondences as independent evidence. Translational information
below a calibrated eigenvalue ratio is now discarded outright.
`locrec/experiments/gt_map_diagnostic.py`.

## 4. Moving the pose without moving the map

An absolute fix that moves the pose and leaves the local map where it was is
undone by the next registration, which pulls the pose straight back to where the
stale map says it should be. This is why the first attempt at the marker
milestone measured **no effect at all from forty markers**, with the oracle flat
too. The fix moves pose, map and recent landmarks together.

## 5. Correcting the anchor with its own fix

If the marker a fix was computed from is also moved by that fix, the whole thing
is circular and the marker tracks the drift instead of opposing it. With this bug
the markers were worse than useless: **0.92 m of error where no markers gave
0.19 m** over 60 m.

## 6. A measurement that is vertical-free but not exactly horizontal

The point registered when the strip was bolted on and the centroid of the beams
that later land on it are at different heights, by up to half the strip. With a
basis tilted by the ray's elevation, that vertical mismatch leaks into the
horizontal residual as its sine: a 10 degree elevation turns a 0.4 m height
difference into **7 cm of fictitious along-track error**, larger than the
measurement itself.

## 7. Believing the centroid of a few beams on a one metre strip

The centroid slides across the strip as the viewing geometry changes. It is a
geometry-dependent bias, not zero-mean noise. Forcing the measurement covariance
to 1 mm took drift from about **1 m to about 40 m**, because the pose was dragged
along behind a wandering centroid.

## 8. A point landmark allowed to touch attitude

One landmark constrains position through a lever arm and leaves the rotational
posterior near-singular. Chaining that through the anchor covariances sent the
correction to infinity within a few fixes. The unconstrained vertical did the same
more slowly, growing to an **80 m standard deviation** and making metre-scale
corrections in z off numerical coupling alone. The fix is translation only and
horizontal only.

## 9. Charging the same uncertainty twice

When the two-anchor yaw fix was added, each anchor's uncertainty was being counted
once in the posterior by the traverse rule and once through the baseline between
two anchors. With a fix on most steps that compounded until the gain reached one
and the filter tracked measurement noise: **0.106 rad of heading error and 28 m
of lateral error over 300 m**, against 0.008 rad with no markers at all. Anchor
uncertainty now lives in the measurement covariance and the posterior is floored,
never inflated.

## 10. A covariance that only ever grows

The pose covariance was integrated from the prior and reset at fixes, and never
reduced by the registration. The estimator therefore believed it was lost even
where the geometry had just told it exactly where it was, and a marker carrying a
stale drop-time error would win against a locally correct registration. Markers
made things **worse in structured tunnels, 1.3 m against 0.5 m over 300 m**. The
covariance now shrinks along the directions that survive the absolute guard.

## 11. Rebinning the map on every fix

The rigid shift was implemented by transforming the map's points and re-voxelising
them. With a fix on most steps that is hundreds of re-bins per run, each snapping
points to fresh voxel centroids. The map now carries its own frame and the shift
moves the frame, so the grid is never rebinned.

## 12. A threshold criterion that was answering the wrong question

The marker detection range was first calibrated as the distance at which a strip
is seen in 90 percent of scans, which gave 3.5 m and would have forced the
scheduler to spend markers faster than uniform spacing. The estimator does not
need a fix every scan; it needs one before drift reaches one measurement sigma,
which is every 1.25 m, so the criterion is a 0.4 per-scan rate and the answer is
7 m. The first criterion was wrong rather than inconvenient.

## 13. Not a bug, but the one that cost the most

`np.unique(..., axis=0)` on an (N, 3) voxel key array costs a lexicographic
argsort over rows, and it was the single most expensive call in the pipeline,
above both the ray casting and the registration. Packing three 21-bit indices into
one integer turned it into a 1-D unique and cut **map maintenance from 32 percent
of the run to 8 percent**, and the whole run by 26 percent.

## 14. An information-greedy gaze looks where the map already is, which is behind

The objective maximises `lambda_min(H_have + H_pred)` over candidate sensor
bearings, and `H_pred` is predicted from the local map. The local map is the part
of the world the robot has already seen, so the bearing that promises the most
information is almost always backwards. Over a 300 m blind run the sensor sat at a
median bearing of **150 degrees off the direction of travel, and beyond 90 degrees
on 90 percent of steps**, flying into unmapped tunnel while staring at mapped
tunnel. Registration reported success on **99.7 percent of steps with a median
1823 inliers**, the same as the forward-gaze run, while along-track error grew
**36.9 mm per step against 0.71 mm** and finished at 16.09 m against 0.43 m.
Nothing in the logs says anything is wrong. `locrec/experiments/gaze_diagnostic.py`.

The objective is not wrong about information; it is wrong about which information
is worth having. Predicting from the map you have rewards exploitation, and the
quantity the estimator actually needs is structure it has not yet used. Fixing it
means predicting against unseen geometry or restricting the candidates to the
forward hemisphere, which is a different policy rather than a retuning of this
one, and it is not in these numbers.

## 15. Sensitivity runs pooled into the headline they were supposed to be measured against

The Monte Carlo writes one CSV for both the default grid and the one-parameter
sweeps, with every parameter recorded in the row, and the summary filtered the
default set on sensor range and odometry scale but not on the per-platform swept
parameter. The mixed world is where the sweeps run, so the drone's mixed-world
headline was three fields of view pooled together: **10.93 m reported against
23.01 m at the default 90 degrees**, a factor of two, in the direction that
flattered the result. Found by recomputing the same number a second way while
writing the sensitivity table. The filter is now one function, `is_default`, used
by both the headline and the tornado.

## 16. A marker deep in a tunnel weighted as though it were lost

The absolute pose covariance grows with distance from the start of a run and never
comes back. Folding it into a marker's measurement covariance, which is what the
estimator did through milestone 7, therefore says that a marker dropped 250 m into
a tunnel is almost worthless, precisely where it is the only thing available. What
a marker actually constrains is where the robot is *relative to the drop*, and
that relation is only as uncertain as the odometry accumulated since.

Measured on a straight 200 m run with a 2 percent scale bias and markers every
10 m, `locrec/test/test_traverse.py`: the relative formulation finishes **0.26 m out
against 4.01 m for dead reckoning**, a per-leg error of 12.4 mm against a
measurement sigma of 10.0 mm. The old formulation cannot do this at any spacing,
because the weight it gives each fix decays with the distance already travelled.

The traverse does not, however, grow like the square root of the number of legs,
which is the thing every mine-survey textbook says it should. Twenty legs against
five gives four times the error, not two. The bias is systematic and the prior
models it as white noise, so the filter sits at a steady-state lag of about a
centimetre behind the anchor it is tracking, and the next anchor is dropped
carrying that lag. What the markers buy is the size of the per-leg error, not its
statistics: a centimetre per leg instead of 2 percent of every metre.

## 17. A threshold crossed 77 times in a tunnel with nothing in it

The scheduler drops on the falling edge of localizability, which assumes the
detector enters a blind stretch once. It does not. On a 300 m blind run only 54
percent of steps are below threshold and the ratio crosses it **77 times**, so the
falling edge fired 77 times and the policy spent **64 markers where its own spacing
rule wanted 22**. The refill rule fired twice. A policy built on an edge detector
has to say what it does about chatter, and this one said nothing.

## 18. A strip of known width hides its own far half

The centroid of the returns on a marker is not the point the marker was registered
at, and the difference is a bias rather than noise. At grazing incidence a 0.15 m
strip occludes its far half, so the returns pile up on the near edge and the range
comes back short. Measured against the true mounting point with the true pose, so
no estimator error is involved: a **radial bias of 4.3 cm at 4 to 5 m against a
modelled sigma of 1.0 cm**, and the direction it acts in rotates from lateral to
along-track as the robot drives past, so the fixes drag the pose along the tunnel
rather than scattering about it. Over a 300 m run the fixes summed to **-0.8 m of
along-track correction on 220 fixes** while the true error was -3.6 m in the same
direction.

This is what separated the bench test from the pipeline. With each marker's true
position substituted, the same 300 m run gives 0.16 m where the centroid gives
2.01 m. `locrec/experiments/marker_leak.py` for the ablation, `locrec/experiments/strip_fit.py`
for the residual distribution.

The fit now uses what the robot knows: the strip's width, its thickness, and the
beam spacing. Seeing the whole strip, the midpoint of the observed extremes is
unbiased because the two quantisation errors cancel. Seeing less than the known
width, the far edge is the one that is missing, so the fit anchors to the near
edge and steps half a width. What remains is charged to the range axis in
proportion to how much of the width was actually seen. Median absolute radial
residual over the modelled sigma: **1.99 before, 0.31 after**.

## 19. An ablation that switched off the thing it was measuring with

The first version of the no-registration ablation set the minimum scan count to
infinity so that no scan could ever be registered. That also removes the
localizability ratio, and the scheduler drops markers on the ratio, so the run it
measured had no markers in it at all. Its number was identical to dead reckoning
to fifteen significant figures, which is what gave it away.

## 20. A wrapper that dropped the prior and registered past the sampling limit

The ROS 2 node published a ratio that moved with the geometry, rose two orders of
magnitude at junctions, and never once crossed the calibrated threshold: **0
degenerate scans in 596** on a 300 m mixed run, so not one `drop_marker`, and
nothing logged an error. Two faults in the wrapper, neither of them enough on its
own to hide the signal.

`node.py` subscribed to `/odom_prior`, stored it, and then called the detector
without it, so every scan was registered as though the sensor had not moved since
the last one. The map is built from the estimate, so the estimate stops following
the robot and the map fills with scans laid down in the wrong place: **28 m of
estimated travel over a 300 m run**.

`detector.py` set its own registration range to a fixed 0.85 of the sensor range,
8.5 m for the UGV, where `run_pass` also caps it where the beam spacing reaches
half a voxel, at 5.73 m. That is number 2 above, reintroduced in the wrapper, and
the `fov` parameter the cap needs was accepted and never read.

Blind-stretch ratio against a threshold of 1.8e-3, mixed world, seed 0, 300 m, the
UGV parameters from `locrec_ros/README.md`, the same scans through all four:

| prior | registration range | blind median | below threshold | markers |
|---|---|---|---|---|
| dropped | 8.5 m | 4.45e-3 | 0 percent | 0 |
| used | 8.5 m | 2.28e-3 | 0 percent | 0 |
| dropped | 5.73 m | 2.46e-3 | 0 percent | 0 |
| used | 5.73 m | **1.74e-3** | **71 percent** | **34** |

The last row is `run_pass` on the same world to three digits, 1.74e-3 and 69
percent, which is the whole point: the threshold describes a ratio produced by one
particular registration, and a wrapper that configures that registration
differently is not measuring the same quantity. Through the graph after both
fixes, the same run gives a blind median of 1.75e-3 and **39 markers, none of them
on structure**. Both faults are pinned in `locrec_ros/test/test_prior.py`, and
the range rule now has one definition, `locrec.runner.default_registration_range`,
which `run_pass` and the detector both call.

## 21. A gate calibrated at the one orientation that could not show the problem

The localizability ratio in a featureless tunnel is not a property of the tunnel. It
is a property of the angle between the tunnel and the estimator's voxel grids, which
are axis aligned in its world frame and never rebinned: the local map's own grid, and
the one `small_gicp` downsamples both clouds onto. Sweep a straight blind tunnel
through a quadrant with the same scans, the same beams and the same prior noise at
every angle, and the median blind-stretch ratio runs from **1.739e-3 to 2.067e-3, a
spread of 19 percent** on a 2.5 degree grid, 17 percent as resolved on the 5 degree
grid the calibration now uses. It is periodic in 90 degrees: theta and 90 minus theta
agree to **0.8 percent**, which is the square lattice's signature measured rather than
assumed. Seed to seed spread at a fixed angle is under 2 percent, so this is not run
to run noise.

`blind_world` has no curvature, so its heading is identically zero and its corridor
lies exactly along a grid axis. Every one of the 7980 calibration samples behind the
old `locrec/results/thresholds.json` came from that one orientation, and that orientation is
the **minimum** of the curve. The threshold could therefore only ever come out too
low, and the gate could only ever under-fire. Never over-fire.

Half a degree of misalignment is enough to matter. Fraction of blind scans below the
old calibrated threshold, against the corridor's angle to the grid:

| angle | 0.0 | 0.5 | 1.5 | 2.5 | 3.0 | 5.0 to 12.5 | 17.5 | 25.0 | 45.0 |
|---|---|---|---|---|---|---|---|---|---|
| below threshold | 0.96 | 0.84 | 0.73 | 0.27 | 0.04 | 0.00 | 0.50 | 0.95 | 0.13 |

Averaged uniformly over orientation the gate fires on **0.58** of blind scans against
the 0.90 its quantile was chosen for, and across 22 percent of the quadrant it never
fires at all. The spike at exactly 45 degrees is narrower than half a degree, the
diagonal version of the same alignment.

The live ROS run had already paid for this, and nothing about it looked wrong. Its
mixed world holds four blind stretches, three lying along the grid axis and the last
one turned 12.9 degrees by the world's own curves:

| blind stretch | heading | n | median ratio | below threshold | markers |
|---|---|---|---|---|---|
| 10-64 m | 0.0 deg | 111 | 1.717e-3 | 0.94 | 12 |
| 94-116 m | 0.0 deg | 45 | 1.715e-3 | 1.00 | 5 |
| 146-192 m | 0.0 deg | 93 | 1.719e-3 | 0.89 | 11 |
| 234-280 m | **12.9 deg** | 91 | 1.958e-3 | **0.00** | **0** |

The 68 percent blind-stretch detection rate reported for that run is just that
mixture, 249 samples at 93 percent and 91 at zero. The last marker of the 300 m run
was dropped at 194 m, so the run ends on **105.5 m with no fix at all**, 46 m of it
blind tunnel the gate never called. The ratio trace looks healthy throughout: at
12.9 degrees it sits just above the threshold instead of just below it.

Fixed by giving the calibration the orientation it never had, not by moving the
number. `locrec/experiments/calibrate_thresholds.py` sweeps the blind world through 19
orientations, 0 to 90 degrees inclusive, and pools all 151,620 samples before taking
the same 0.9 quantile as before. The rule did not change.

| | old | new |
|---|---|---|
| `ratio_threshold` (UGV) | 1.844588e-3 | **2.068275e-3**, +12.1 percent |
| `drone_ratio_threshold` | 2.826411e-3 | **2.888263e-3**, +2.2 percent |
| `min_inliers` | 987 | 1016 |
| pooled samples | 7,980 at one angle | 151,620 across 19 |

The sweep contains its own check that nothing else moved: the 0 degree slice of it
reproduces the old calibration exactly, p90 of 1.8446e-3 for the UGV and 2.8264e-3
for the drone, which are the two old thresholds to five digits.

The drone barely has this problem. Its per-angle p90 spans 3.7 percent against the
UGV's 16.6 percent, which is why its threshold moves by 2.2 percent and the UGV's by
12.1. A 90 degree field of view at 30 m does not lie along one lattice plane the way
a 360 degree sweep of a 3.2 m corridor at 10 m does. `min_inliers` moved for the same
reason the ratio did: at exact alignment the wall collapses into a single voxel row
and the map keeps about 9 percent fewer points.

## 22. A false alarm rate that counted structure the sensor could not see

Number 21 measured what a higher threshold costs by classifying every scan by its
distance along the centreline to the nearest junction, niche, curve or end cap, and
counting a scan within a metre of one as a false alarm if the gate fired. For the
UGV's 360 degree scanner that is fair. For the drone it is not: the drone looks
forward through 90 degrees, so a niche it has just passed contributes nothing to its
scan while the metric still counts it as structure in hand.

What that produced, 12.8 percent of on-structure drone scans called degenerate at the
old threshold against 0.0 percent for the UGV, reads as a defect in the drone's gate.
It is not. The same scans, split by whether the structure is ahead or behind:

| group | n | drone | UGV |
|---|---|---|---|
| structure 1-5 m ahead | 111 | **0.036** | 0.225 |
| structure 1-5 m behind | 118 | **0.475** | 0.144 |
| within 1 m, ahead | 185 | 0.097 | 0.000 |
| within 1 m, behind | 142 | 0.176 | 0.000 |

Where the geometry is in the drone's view, one to five metres ahead, it fires on
**3.6 percent**, the lowest rate of any group and well under the UGV's 22.5 percent in
the same band. Where the geometry is behind it, 47.5 percent, and those scans really
are blind: the sensor is looking at plain corridor and calling it degenerate is
correct. The UGV shows no such asymmetry, 0.225 against 0.144, which is what a 360
degree scanner should show. Even the residual inside a metre fits: a niche one metre
ahead on a wall 1.6 m away sits 58 degrees off axis, outside a plus or minus 45 degree
cone, so "within a metre of structure" does not mean "structure in view" for this
sensor at all.

The defect is in the measurement, not the detector, and it is the same shape as number
19: an instrument that was not measuring the thing its name claimed. The right test for
a directional sensor is whether the structure falls inside the field of view, not how
far along the centreline it sits. `locrec/experiments/threshold_false_alarms.py` still reports
centreline distance, which is correct for the UGV and misleading for the drone, and its
drone rows should be read against this entry until the classifier is taught the field
of view.

## 23. A marker rule copied from before its own fix

Number 17 fixed the scheduler so its chain survives a recovery: the ratio does not
enter a blind stretch once, it chatters across the threshold, and a rule that
re-arms its falling edge on every recovery pays for each crossing with a marker. The
ROS wrapper did not call the scheduler. `detector.py` carried its own rule, a copy of
the scheduler as it stood before that fix, with a 5.5 m spacing of its own that zeroed
whenever a scan came back above the threshold.

Through the graph at commit 17cb594, on the mixed world at seed 0, the ratio crossed
the threshold **27 times inside blind stretches** and **17 of the 36 gaps between
markers were shorter than the rule's own 5.5 m spacing**. The M10 demonstration in
`docs/DECISIONS.md` saw the same thing from the other side, where a threshold the
stretches sat clearly below stopped the crossings and seed 2's markers fell from 58
to 40 while detecting more.

The wrapper now calls `locrec.policies.LocalizabilityScheduler` itself, with the
calibrated spacing from `locrec/results/thresholds.json` and no default, so there is no
second copy of the rule to go stale. `locrec_ros/test/test_markers_loop.py` pins
it: 30 m of tunnel that chatters every 2 m buys markers at 0, 12.5 and 25 m, and on
400 random scans the wrapper decides exactly what the scheduler decides.

## 24. An estimator that would have discarded every marker without a word

The detector built its estimator as `Odometry(np.eye(4), odom_cfg)`, without the
sensor model. The estimator needs that model to weight a marker observation, and
where it is missing `_landmark_terms` in `locrec/odometry.py` returns no terms at all.
Every marker would have been seen, reported and thrown away, and the estimate would
have been exactly the registration-only estimate, which reads as "markers do not help"
rather than as a fault.

It never showed because nothing fed markers into the node until the closed loop
existed, which is also why it would have been believed the first time it ran. Found by
reading the call path before wiring the loop, not by a symptom. The detector now passes
the `LidarSpec` it already builds for the registration range, and
`test_the_estimator_is_given_the_sensor_model_markers_need` fails if it stops.

## 25. A ctrl+c that was delivered twice and a node that would not die

Recording the demonstration, one node stayed alive after the launch had gone, at 98
percent of a core for 84 minutes, and ignored SIGINT until it was killed. Two separate
faults were behind shutdown going wrong, and neither showed when a node was stopped on its
own.

The first: a terminal ctrl+c reaches every process in the group, and `ros2 launch` then
forwards a second SIGINT to its children. rclpy's handler shuts the context down on the
first. The second landed in Python's own handler, in the middle of cleanup, as a
KeyboardInterrupt: every node exited with -2 and launch printed an error for each, and the
viewer could have been cut off before it finished writing the video. Each node now ignores
SIGINT at the Python level before `rclpy.init`, which leaves rclpy's handler as the only one.

The second: `Node.destroy_node` after a signal has already shut the context down can hang in
rclpy 7.1.11. It wakes the executor, whose guard condition the shutdown is tearing down on
another thread. Found by registering `faulthandler` on SIGUSR1 in every node, which prints a
stuck process's stack without a debugger, and repeating ctrl+c until a process outlived it:
**one of eight runs**, `sim_publisher` this time, stuck on the last line of `destroy_node`.
The nodes now destroy themselves only while the context is up, since the process is exiting
anyway. After the change, **16 runs and 64 processes, all finished cleanly**. That is
evidence and not proof: at the old rate, 16 clean runs in a row would still happen about one
time in eight.

## 26. A second sensor whose ranges came out of a texture

The first run on Gazebo's LiDAR looked like a calibration problem. On the mixed world,
seed 1, 80 m, with the scheduler and the MuJoCo thresholds, the robot mounted **no
markers** and finished 1.94 m off. The median ratio was **2.55e-3 against 1.74e-3** on
MuJoCo for the same tunnel, so the 2.07e-3 threshold called 0 percent of scans degenerate
where MuJoCo called 73 percent. Recalibrating would have moved the threshold up and made the
gate fire again, and it would have hidden what was wrong.

The sensor was measured before anything was recalibrated. Gazebo's range noise is what it
was configured to be: 2.01 cm against 2 cm, mean +0.006 cm, and two noiseless frames at one
pose agree to the bit. The noiseless ranges are not. Beam by beam against noiseless MuJoCo
at the same pose, the median error was 0.40 cm but the 95th percentile **13.6 cm**, the 99th
29.9 cm and the worst 36.5 cm, all on beams meeting a wall or the floor at a grazing angle.
Gazebo's GPU LiDAR does not cast rays. It renders depth into cube faces and reads each range
out of that texture, and it sizes each 90 degree face from the beam count, rounded up to a
power of two and clamped to 128..1024 texels (`Ogre2GpuRays.cc`, gz-rendering 8). At 360
beams a face is **128 texels, 0.70 degrees each**. Near grazing incidence a fraction of a
texel is tens of centimetres of range, and the error moves with the pose, so to the
registration it looked like texture on smooth walls: along-track information the tunnel
does not have. On that run it also cost accuracy: 1.94 m of final error against 0.43 m
for the same tunnel once fixed, one run each.

The fix is at the sensor. `locrec.gazebo` renders k times as many horizontal beams and keeps
every k-th, which are exactly the study's beam directions; k = 6 for the UGV and 3 for the
drone reach the 1024 texel cap. Noiseless errors after it: UGV median 0.05 cm, 95th
percentile **1.6 cm**, worst 8.3 cm; drone median 0.22 cm, 95th percentile 2.5 cm, worst
46 cm, where 30 m beams run nearly parallel to the walls and the texture cap binds. On the
same 80 m run the ratio then tracks MuJoCo step by step: Gazebo over MuJoCo **median 1.015**,
10th to 90th percentile 0.96 to 1.07, and 73 percent of scans degenerate on both. Only then
were the thresholds calibrated on Gazebo (`locrec/experiments/calibrate_thresholds_gazebo.py`), so
the calibration measures the sensor and not the artefact.

## 27. A client that deadlocked on its own sensor

The Gazebo calibration capture stopped on a step request that got no answer in 5 s, on the
first step after the capture loop. Nothing was wrong with the server. gz-transport's Python
`Node.request` holds the interpreter lock for as long as it waits for its reply: a
background thread that should have run about 150 times during a 1.5 s request ran once. A
subscription callback needs that lock, and the transport thread that runs the callback is
the one that would have delivered the reply. So a LiDAR frame arriving during a request
stalls both until the request times out. The proof of concept never hit it because it waited
for every frame before the next request; the capture waited for one sensor's frame while the
second sensor's was still on its way.

`GazeboLink` now receives frames in a child process that only subscribes and passes them up
a pipe, and the process that makes requests never subscribes. Neither can block the other.

## 28. Two views that shared one overlay

The drone viewer draws two views, one per drone, and gave each its own GPU context. The top
view then carried the bottom view's title, localizability line and legend over the top
drone's scene: every label on it named the other drone. moderngl does not switch between
standalone contexts for you, so the second context's overlay texture was written, and drawn,
through the first. The viewer now draws every view with one renderer, each vehicle's layers
named after it, and `render` draws the layers of one prefix; a test checks that a view draws
nothing of another's.

## 29. An estimator drawn as missing for a whole run

A short MuJoCo recording after the viewer was rewritten scored the marker estimator as
missing on every scan: no along-track error, no path, while its ratio arrived for every scan
and the robot mounted three markers on its word. The estimates were on the topic and the
viewer was subscribed. The viewer aligns each estimator's frame to the world once, and it did
that at the estimator's first estimate. That estimator had processed the simulator's very
first scan, and the viewer held no true pose for that stamp, so the alignment was retried and
refused on every scan of the run. The most likely reason the pose was missing is discovery:
the simulator's transform publisher is created when it starts publishing, while the
estimator's publisher had been matched for seconds. The other estimator skipped the first
scan, and it is why the same code had drawn both on earlier runs.

The viewer now aligns at the first scan for which both the estimate and the true pose have
arrived, and a test gives it an estimate with no true pose.

## 30. A live run that was not its offline pass

The Gazebo demonstration of seed 1 was chosen by a rule applied to offline passes, and live it
did not look like its offline pass. Offline, mean along-track error without markers was 0.85 m
and with them 0.36 m. The first live ROS run of the same seed on the same sensor gave 0.76 m and
0.61 m: the same 19 markers, a third of the benefit. Two differences were behind it, and neither
was in the loop.

**Thread count.** Every estimator setting matched `run_pass` but one: the offline passes register
on one thread, as the study's grids do, and the node used small_gicp's default of four. The
offline pass with markers, repeated on one thread, gave 0.355 m again to the millimetre; on four
threads, twice, it gave **0.553 m both times**. Both are deterministic, and they are different
runs: a 300 m chain of scan-to-map registrations carries a small difference in one registration
into every later one.

**One render of noise.** On one thread the live run still gave 0.61 m against 0.61 m. Gazebo's
range noise comes from one seeded stream, drawn at every render, and the offline pass throws its
first, discovery, frame away while the driver had kept it as the first scan. Every live scan
therefore carried the noise of the render before its offline twin's. Per scan, the no-marker
estimator stayed within a centimetre of its offline trace for 86 m and the marker estimator,
whose fixes amplify a small difference, parted from it at 7.5 m.

The node now takes `registration_threads`, still 4 by default, and `gazebo.launch.py` sets 1; the
driver primes and discards as the offline pass does. After both, **the live marker estimator
reproduced its offline pass on all 600 scans with a largest difference of 0.000 mm**, mean 0.356 m
on both, and again after the shell was merged into meshes, mean 0.461 m. The live LiDAR-only estimator still differs, median 35 mm per scan, mean 0.838 m
against 0.850 m, and should: live it registers the same scans as the marker estimator, strips
included, where its offline pass ran in a tunnel with none. The two drones share one Gazebo server
and one noise stream, so a live drone run is not its offline passes and is not claimed to be.

Re-checked after the changes in numbers 31 to 37, on the recorded UGV run (seed 1, 300 m, 600
scans): the marker estimator's final along-track error is 1.063 m live and 1.063 m offline, mean
0.468 m against 0.467 m, 20 markers both. The LiDAR-only estimator differs, 0.780 m against
0.763 m, for the reason above.

## 31. Six rings at one azimuth counted as six measurements

Markers made three of eight seeds worse. The chain of strips ran about linearly behind the
truth, six seeds of eight on the near side, and a 1 mm strip moved every seed by about +1.5 m
on the same 8 seeds, so the defect was in what the strips measured, not in how many there were.

It sat in `measurement_information`. The covariance is diagonal in the ray frame, and the
horizontal tangential axis is the tunnel axis, the one the odometry cannot see for itself.
Both terms on that axis, beam quantisation and the strip's own extent, were divided by the
number of returns:

```
var_h += (r * lidar.azimuth_step_rad) ** 2 / (12.0 * n)
var_h += marker_width**2 / (12.0 * n)
```

A LiDAR's returns come in columns. Every ring in a column is the same azimuth, so a strip
crossed by one column of six rings was measured once along the tunnel and six times across
the range. Dividing the horizontal by six claimed **sigma 1.9 cm at 4.4 m where the strip's own
width over root twelve is 4.3 cm**, and the estimator let those fixes pull six times harder in
variance than the evidence allowed. It showed in the fix log: beyond 5 m the residuals had
standard deviation 2.42 m against the 1.79 m the estimator assumed, while the same runs with
perfect detections had 1.07 m against 1.72 m, conservative the other way
(`locrec/experiments/marker_bias_ablation.py`).

Measured inside six real runs, on every one of their 1193 detections against the strip's true
position: **56 per cent of detections
are a single column**, at mean range 4.4 m, and those carry +3.12 cm of along-track fit error,
against +1.07 cm for two columns and -0.98 cm for four. Beyond 3 m every detection is a single
column.

The horizontal terms now divide by the number of distinct azimuths, the vertical by the number
of rings, and the range keeps dividing by all n, which it earns. `MarkerDetection.n_columns`
carries the count and `MarkerObservation.msg` carries it over ROS.

### What did not work, and why it could not

Before this, the bias was attacked at the measurement: `landmarks.strip_fit_bias` runs the real
`fit_strip_centre` on synthetic returns from the known strip box and subtracts the mean error.
On the bench it is exact, within two standard errors of zero at every range from 1 to 8.5 m. On
the grid it recovered 0.3 m of the 1.4 m. Conditioning it on the observed column count, which is
the information the detection actually carries, recovered 0.5 cm more of the 3.12 cm and no more.

The reason is that it averages over where the strip sits between two beams, and **the run does
not sample that uniformly**. The robot drops a strip beside itself at a fixed 1.53 m and then
steps a fixed 0.5 m, so the bearing to every strip runs through the same sequence of sub-beam
phases, and every strip in the tunnel meets the same few. Poses on that lattice reproduce the
in-situ bias to a tenth of a centimetre, +4.29 cm against +4.20 cm at 4 to 5 m and +2.63 against
+2.65 at 3 to 4 m, while the phase average over the same geometry is under a
centimetre. A phase-averaged correction cannot remove a phase-locked error, and no amount of
conditioning on observables recovers a phase that is not observed. What the estimator can do
honestly is know it does not know: with one column, where the strip sits along the wall is
uncertain by its own width over root twelve, and saying so is the fix.

## 32. A team result that was exactly the solo result

The first run of `locrec/experiments/team_pass.py` gave the drone the same final error with the robot's
strips as without, to the digit, on every seed. Not close: identical. A result that is *exactly*
the control is never a null, it is a wire that is not connected, and the thing to do with it is
look for the wire rather than write it up.

The wire was the sign of the lag. The drone trails the robot by 60 steps, so the drone's own
step k is the robot's step k plus 60: a strip the robot mounted at its step i has been on the
wall since the drone's step **i minus 60**, which is before the drone gets there. The hook placed
each strip at i **plus** 60 instead, which is 30 m after the drone had already flown past it, and
the drone looks forward and never looks back. Six strips were placed in the drone's world on
seed 0 and it detected none of them.

Worth stating because the same trap is in any follower experiment where both vehicles start their
own clock at zero: the follower's clock is behind, so everything the leader did is further in the
follower's past, not its future. After the fix, on the same seed, the drone ended 4 cm from the
robot's frame where alone it ended 1.04 m from it.

## 33. The team drone read its teammate's strips and threw away its own eyes

The live team run came out with the drone's two estimates identical to the digit, which
`docs/failures.md` number 32 had already named as the signature of a wire that is not connected.
The drone's `team` estimator logged every strip it was told about, five of them, and its fix count
was zero on every scan.

The node parsed the scan report like this:

```python
drops = [...] if self.use_markers else []
detections = [...] if self.use_markers else []
```

Both gated on the same flag, which was right while the only vehicle that saw strips was the
vehicle that mounted them. A team vehicle mounts nothing, so `use_markers` is false for it, and
the second line threw away every strip it could see. It held five anchors it was never allowed to
observe.

Drops stay gated on `use_markers`, because a vehicle that mounts nothing has none. Detections are
now read whenever `use_markers` **or** `use_shared_landmarks` is set, and `use_report` likewise,
or the node would not even subscribe. `test_a_team_vehicle_uses_the_strips_it_sees` holds it.

The general shape, for anyone adding a third kind of vehicle: mounting and seeing are separate
capabilities and had been carried by one flag. They are not the same question and the code should
not ask them together.

## 34. A correction that was exact on one sensor and wrong on the other

`landmarks.strip_fit_bias` removes the strip fit's expected error by running the real fit on
synthetic returns from the known strip box. On MuJoCo it is exact: the corrected along-track bias
is within two standard errors of zero at every range from 1 to 8.5 m. It was left on by default
and carried into the Gazebo cross-check, where it cost most of the marker benefit: scheduler final
along-track error 1.48 m with it on against 0.85 m with it off, better on 4 of 8 seeds against 5,
and mean over the run 0.74 m against 0.47 m.

The reason is in what the two sensors do, measured in situ on each (mixed world, seed 1, the
detections of a real run):

| at one beam column, where more than half of all detections are | bias | scatter | total |
|---|---|---|---|
| MuJoCo, exact ray casts | **+3.28 cm** | 2.67 cm | 4.23 cm |
| Gazebo, ranges from a rendered depth texture | **+0.43 cm** | 4.74 cm | 4.76 cm |

The same total error, near the strip's own width over root twelve, 4.33 cm, split completely
differently. MuJoCo's rays meet the strip at repeatable sub-beam phases and the fit's error is
mostly a fixed offset; Gazebo's rendered depth scatters instead. Subtracting MuJoCo's offset from
Gazebo's returns adds 3 cm of error to every far detection.

The Monte Carlo casts rays, so it models MuJoCo and only MuJoCo. `correct_strip_bias` now defaults
to **off** and the docstring says to turn it on only for a sensor whose strip bias has been
measured and matches the model.

The general shape: a correction derived from a sensor model is only as portable as that model. The
spread from the same Monte Carlo did transfer, predicting 4.66 cm where Gazebo measured 4.74 cm,
because scatter of that size is set by the beam geometry both sensors share. The mean was not, and
nothing in the code said which of the two was safe to carry across.

## 35. The scale filter stops two thirds of the way

The prior overstates every step by a scale drawn once per run, and `Odometry` estimates it from
the along-track residual over each leg and divides the prior by the estimate. Across the 8 seeds
its final estimate looked unbiased, mean error +0.00006. Regressed on the truth it is not:

    scale error = -0.3718 * (s - 1) + 0.0022,   R squared 0.943

It **recovers 63 per cent of the error and leaves 37 per cent, systematically**. Because the true
scale is drawn with mean 1, the leftover averages to zero across seeds, which is the only reason
the mean read clean. Per seed it is deterministic, it explains 94 per cent of the variance in the
scale error, and that error correlates -0.965 with the final along-track error. Over 300 m,
37 per cent of a 2 per cent bias is 2.2 m, the size of the final errors observed.

On the bare traverse bench, with no detection noise and a fixed 2 per cent bias, the estimate
settles at 1.0137 and stays: 1.0135 at 10 legs, 1.0140 at 80, spread 0.0002 across seeds. It does
not converge to 1.02 however long the run.

The cause is in `_update_scale`. Its predicted innovation is `L * (scale - 1)`, the residual a leg
would show if the prior were **not** being corrected, while `step` has already divided the prior
by `scale`. The filter subtracts a prediction of a different measurement, and the update stops
where the two happen to balance rather than at the truth. Solving for that point gives something
between the true scale and `2s/(s+1)`; the measured value sits about 60 per cent of the way there,
so the analysis has the direction and the order but not the exact value.

Not fixed: the innovation model is the estimator's core and correcting it moves every marker
number in the study. `test_the_scale_state_recovers_two_thirds_of_the_bias_and_no_more` pins the
present behaviour so a fix fails the test and has to be accepted deliberately. Separately and
smaller, the filter is over-confident by a factor of 2 in variance, claiming sd 0.00478 where the
spread is 0.00655, which `_update_scale`'s docstring already predicted from using the residual
twice.

## 36. The team video showed everything except the result

The first 300 m team recording, on the seed the declared rule picked, closed with this:

    robot, on its own strips  mean 0.71 m
    drone alone               mean 0.57 m
    drone on the robot's strips  mean 0.65 m

Read plainly, the strips made the drone worse. Every number is correct and the conclusion drawn
from them would be wrong, because the viewer plots **error against the world** and that is not
what the team buys. What it buys is agreement with the robot, and on this seed the solo drone
happens to be nearly right in the world, so the world axis shows the team giving up a little
accuracy to adopt a frame it did not need. The declared statistic, the gap between the drone's
error and the robot's at the same place in the tunnel, was nowhere on screen: the same run has the
team drone a mean 0.06 m from the robot's frame against the solo drone's 0.86 m over 8 seeds.

Two things had to be true at once for this to bite. The demonstration seed is chosen by a rule
fixed before the runs, "closest to the median paired difference, ties to the lower seed", which on
Gazebo picked seed 1, the seed where the solo drone does best of the eight. That is the rule
working as intended and it is not to be changed to get a better picture. And the viewer's third
panel in team mode was still the ground robot's "markers mounted", a number the closing table
already gives.

Fixed by making that panel carry the claim: in team mode it plots the drone's distance from the
robot's frame, for both estimators, labelled as the different quantity it is rather than sharing
an axis with the world-frame error. The closing table gained the mean and worst of it. The middle
plot is unchanged, so the world-frame result is still there to be read, including the part that is
unflattering on this seed.

The lesson is not about a plot. A demonstration is an argument, and an argument that shows a
quantity the claim is not about will be read as evidence against the claim. The statistic the
experiment declared is the statistic the video has to draw.

## 37. The viewer read a lockstep pause as the end of the run

The 300 m team recording closed itself at 180 scan intervals of 660, 90 m into the tunnel, with
8 of the 20 strips mounted. Nothing crashed: the video was written, the results card rendered,
the summary printed, and every number in it was correct for the 90 m it had seen. A recording that
stops early and says so in units nobody checks is worse than one that fails.

Two settings that had never met. The driver runs in lockstep: it publishes a scan and waits for
every estimator to answer before moving, up to `lockstep_timeout`, 20 s. The viewer decides the
run is over when no scan has arrived for `idle_finish`, **3 s**. For one vehicle the two never
collided, because an estimator that has been running all along answers in milliseconds. A team's
drone sets off 60 scans late with estimators that have received nothing at all, and they took
longer than 3 s to produce their first estimates, so the driver paused, the viewer saw silence and
finished.

Two fixes, because either alone leaves it fragile:

* The driver's grace after a vehicle sets off was one scan (`startup_timeout` applied when
  `start_tick == scan_index`). The drone's estimators were still cold several scans in, so every
  one of those paid the ordinary 20 s. It is now a window of `COLD_SCANS`, ten scans, and the
  grace itself is 25 s rather than 90, because it has to stay below the viewer's patience.
* `gazebo.launch.py` gives the viewer `idle_finish` 35 s, which outlasts the driver's worst
  legitimate pause. It costs 35 s of wall clock at the end of a recording and nothing in the video.

The general shape: two components each had a timeout that was correct on its own, and no test
covers "one waits longer than the other is willing to wait". Anything that infers a state from
silence has to know how long the other side is allowed to be quiet.

### And the recording after that one was worse, for an unrelated reason

The next attempt closed at 33 intervals with the ground robot's error reading a mean of 83.87 m
and a worst of 313.20 m, and every row of its CSV at `dist 0`: the truth never advanced. Two
launches were running at once. The previous one idles after its video closes, right through to its
own timeout, and it was still publishing when the next started.

`GZ_PARTITION` is what keeps two Gazebo servers from hearing each other, and it is per launch
already (`docs/failures.md` number 28). It does nothing for ROS. Two launches on one machine share
the ROS graph, so both drivers published `/ugv/points`, `/odom_prior` and `/tf`, and the viewer
scored one run's estimates against the other run's truth. The fix while recording is
`ROS_DOMAIN_ID`, set per launch, and the discipline is to kill the previous launch rather than
assume it exited when its video did.
