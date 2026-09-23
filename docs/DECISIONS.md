# Decisions

Every choice made without asking, with the measurement that motivated it. Newest
last within each milestone.

## Milestone 2c

**Covariance floor implemented on the anchor's eigenbasis.** The order allowed
either a common-basis floor or a diagonal max if the two matrices are near
aligned. They are not always near aligned: the anchor covariance is elongated
along the tunnel while the posterior is elongated along whatever the registration
last failed to observe, and on a curve those differ by tens of degrees. So the
floor uses the anchor's eigenbasis, projects the posterior onto it, and lifts only
the deficient directions.

**The floor applies to the translation block only.** An anchor is a point. It
carries a position uncertainty and nothing about heading, so flooring the yaw
entry with it would be inventing a number. The yaw entry keeps its plain
posterior, which is what stopped the divergence in milestone 2b.

**Matched drift is computed but not reported as a headline.** The metric needs a
uniform-spacing curve that falls as markers are added, so that a policy's drift
can be read back as a marker count. At 8 seeds and 300 m the curve is not
monotone in two of the three worlds: in the blind world no markers gives a median
2.92 m along-track and 20 markers gives 3.73 m. Interpolating a saving across
that is interpolating across noise, so the table prints the saving with a warning
above it and the paired per-seed difference against no markers is what the write-up
quotes. Rerunning at 50 seeds is the fix, and it is a laptop job, not a decision.

**No policy is reported as beating no markers.** Every paired interval in every
world straddles zero at 8 seeds. Reported as measured, with the plot.

## Milestone 3

**Map normals are estimated per step with a local PCA rather than reused from
small_gicp.** The registration's covariances are computed on the downsampled
source cloud, not on the map, and are not exposed per map point through the Python
bindings. A radius PCA over the confirmed map costs about 4 ms per step at this
map size, which is under one percent of the step.

**The greedy gaze result is reported, not repaired.** The measured cause, that
predicting information from the local map points the sensor at the map rather than
at new structure, suggests an obvious change: drop the rear hemisphere from the
candidate set. That change would be chosen after seeing the result, which is the
one thing the build order forbids, so it is written down as the next experiment
and the negative number stands as the milestone 3 result.

## Milestone 4

**The milestone 2 marker effect was attributed by ablation rather than argued
about.** `locrec/experiments/marker_ablation.py` turns each milestone 2b and 2c estimator
change off on the blind world with uniform 15 m spacing, same seeds. The yaw fix
and the covariance floor produce results identical to the shipped build, to every
digit: with 7 m of detection range and 15 m of spacing, two anchors are almost
never in view at once, so the yaw fix does not fire, and the floor never binds.
Anchor re-survey is worth 0.35 m of the 0.45 m paired reduction. The changes that
were expected to matter do not, in this world, and that is recorded rather than
assumed.

## Milestone 5

See `docs/REAL_DATA.md`.

## Milestone 8

**The scheduler keeps its chain across a structured stretch.** Fix C changed the
refill distance to twice the detection range less a margin, which should have put
about 22 markers in a 300 m blind run. It put 64. The refill rule fired twice; the
falling edge fired 77 times, because a blind world is not one blind stretch to the
detector. Only 54 percent of steps are below threshold and the ratio crosses it 77
times, so a policy that forgets its chain whenever the ratio recovers pays for
chatter in hardware. The chain now survives a recovery, and the falling edge is
subject to the same spacing as the refill: on entering a blind stretch it fires at
the edge if the last anchor is more than a reach behind, which is rule 1 intact.
The count is then 22 in the blind world and 17 in the mixed one, matching the
spec's expectation.

**Q_since is the prior only, as specified, and that is not obviously right.** The
order defines the relative covariance as the sum of per-step prior covariances
since the drop. The registration's own information is therefore not subtracted,
even where the geometry is good and the registration genuinely does constrain how
far the robot has come. The consequence is that a marker keeps full authority in a
structured stretch, which is the mechanism behind the milestone 2b result where
markers made things worse in tunnels that had structure in them. Implemented as
specified, and flagged as an open question at the time.

**The change of twist centre is dropped from the relative accumulator.** The
per-step covariance goes into the running sum as it stands. Over one step the
lever arm is the step length, so the coupling that drops is second order in the
step, and the translation block, which is the only part a horizontal fix reads, is
unaffected by it.

**R_drop is modelled, not observed.** A marker is bolted on rather than measured
at drop time, so there is no beam count to compute its covariance from. The strip's
angular extent divided by the beam spacing gives the count the same geometry would
produce, and the measurement model is evaluated at that: 105 returns at 1.5 m for
the UGV, 5 at 7 m. This is the floor on how well any later fix can place the robot
relative to that anchor.

**The old anchor weighting is kept behind a flag rather than deleted.**
`OdometryConfig.relative_anchors=False` restores the absolute-covariance weighting
and the posterior floor it needed, so `locrec/experiments/marker_ablation.py` can measure
what fix A was worth instead of asserting it.

**sweep_forward was not added.** The order said to skip it if the existing sweep
already oscillates between plus and minus 60 degrees. It does: `SweepGaze`'s
amplitude has been 60 degrees since milestone 3.

**The forward cone is 60 degrees for the oracle too, and the oracle now plans from
the estimated pose.** Both per the order. The second change makes it a bound on
gaze rather than on gaze plus estimator, which is what it is being used for.

## Milestone 9

**The leak is (a), detection, and the D3 row that proves it is perfect_detection.**
Substituting each marker's true position for the centroid of its returns, with the
beam count and therefore the weighting unchanged, takes the blind world from 2.01,
0.97 and 1.29 m on seeds 0, 2 and 7 to **0.16, 0.30 and 0.12 m**, which is the
bench scale. Association was never a candidate: detection is by MuJoCo geom id, so
it is already ground truth, and that row is reported as the tautology it is.

The mechanism is in `docs/failures.md` and in `locrec/experiments/strip_fit.py`. A strip
of known width occludes its own far half at grazing incidence, so the returns pile
up on the near edge and the range comes back short: a **4.3 cm radial bias at 4 to
5 m against a modelled 1.0 cm sigma**, systematic rather than zero mean, in a
direction that rotates from lateral to along-track as the robot drives past. The
fix is the one the rule asks for, a tightened strip fit plus the logged residual
distribution: use the known width, the known thickness and the beam spacing, and
charge the remaining foreshortening to the range axis. Median |radial| over the
modelled sigma goes from **1.99 to 0.31**.

**The D3 no_registration row was invalid as first run and is reported as such.**
Switching the registration off leaves the detector with no Hessian, so the
scheduler never fires and the row measured pure dead reckoning. It is fixed by
comparing against the no-marker run under the same setting, and what it then shows
is that markers alone cannot carry the run: the frontend is doing most of the work
on most seeds.

**Anchor re-survey is off by default.** It re-registers an anchor from an estimate
that is itself drifting, so the anchor tracks the drift instead of opposing it.
Blind world, scheduler, seeds 0, 2 and 7: 3.65, 1.66 and 2.76 m with it on against
2.01, 0.97 and 1.29 m with it off. The milestone 8 ablation had it helping by
0.13 m on uniform spacing, well inside the noise, and the scheduler numbers are
consistent and larger.

**Q_since is still the prior only, and there is now a flag for the alternative.**
`OdometryConfig.credit_registration` fuses the guarded registration information
into each step's increment before it is accumulated, so a marker stops outvoting a
frontend that genuinely knows where it is. It is off by default because the
milestone 8 order specifies the prior-only accumulator, and it is measured in the
ablation rather than argued about.

**The scale random walk is read as a standard deviation per metre, not a
variance.** At 1e-4 of variance per metre the walk would swamp the 2 percent
initial uncertainty in four metres, which cannot be the intent of a term meant to
let a slowly varying bias move.

**The scale observation uses the leg since the last fix, not since the drop.**
After a fix the pose is tied to the anchor, so the residual that follows measures
drift since that fix. Using the distance since the drop would divide by a lever arm
several times too long and under-estimate the bias. Captured before the anchor
records are reset, which is a bug this milestone had for one iteration: the reset
ran first and every leg measured zero.

**The glance policy's four numbers are pre-registered.** 60 degree cone, 3 s hold,
5 s cooldown, engaged on the calibrated ratio. Fixed before the grid ran and not
adjusted afterwards.

## ROS 2 wrapper

**`/odom_prior` is ordinary cumulative odometry, and scans are paired to it by
stamp.** The simulator used to publish a per-scan increment in an `Odometry`
message, which no bag will ever carry. Pairing is by stamp rather than by the
latest message because the simulator publishes a scan before the odometry that
shares its stamp, so the latest message is either one scan stale or racing the
scan depending on which callback runs first. Between two samples the pose is
interpolated, outside them nothing is extrapolated, and a scan that cannot be
paired is dropped and counted rather than processed without a prior. The publisher
still sends the scan first, so every run exercises the waiting path.

**The registration range rule has one definition,
`locrec.runner.default_registration_range`, and both callers use it.** It was
inline in `run_pass` and copied by eye into the detector, where the sampling cap
was lost. The arithmetic is unchanged, which is the honest claim to make rather
than bit-identical runs: `run_pass` is not bit-reproducible run to run in any case,
because the registration is multithreaded, and two runs of the same code differ by
about 3e-10 in every ratio.

**`azimuth_beams` is a parameter and the nominal step length is not.** The cap
needs the azimuth spacing, which is the field of view over the beam count, so the
beam count has to be told to the node. The step length enters the same rule, but
the sampling limit binds first for both study sensors, 5.73 m against 8.5 m for
the UGV and 11.46 m against 28.5 m for the drone, so it stays a config default
rather than another knob to set wrong.

**The detector floors the local map radius at twice the sensor range, as
`run_pass` does.** It changes nothing for the UGV, whose 30 m default already
clears a 10 m sensor. The drone's 30 m sensor was being registered against a 30 m
map, which is number 1 in `docs/failures.md`.

**The ratio's dependence on tunnel heading is its own milestone.** In a straight
blind tunnel expressed in a frame yawed by theta, with the same scans throughout,
the median blind ratio moves from 1.74e-3 at 0 degrees to 2.04e-3 at 10 degrees and
the fraction below the threshold goes from 84 percent to 0. Calibration seeds 100
to 119 are all straight tunnels at 0 degrees. That is a knife-edge threshold rather
than a broken map, the two are not the same claim, and nothing in this work moves
the threshold to cover it.

## Milestone 10, the orientation the calibration never had

**The calibration set covered one orientation, and that orientation was the floor of
the band.** `blind_world` has no curvature, so its heading is identically zero and the
corridor runs exactly along the estimator's voxel grid axes, which are axis aligned in
its world frame and never rebinned. The ratio in a featureless tunnel is periodic in
that angle with a 90 degree period and a 19 percent spread, and zero degrees is its
minimum, so the threshold could only ever come out too low and the gate could only
ever under-fire. `docs/failures.md` number 21 has the curve, the half degree cliff,
and the 46 m blind stretch a live run skipped because of it.

**The fix is the calibration's coverage, not the threshold's value.** The rule is
unchanged: same seeds 100 to 119, same 200 m, same 0.9 quantile of the pooled blind
distribution. What changed is that the distribution now contains the orientations a
real tunnel has. The number that came out is an output of that calibration, not a
value chosen to make a run look better, which was the whole reason not to touch it
when the dependence was first measured.

**The angle count was fixed before any swept output was looked at, and it is 19**,
0 to 90 degrees inclusive in 5 degree steps. The grids are square, so one quadrant
covers every orientation. 0 and 90 degrees are the same orientation under that
symmetry, so that alignment carries double weight in the pool; it is also the
orientation with the lowest ratios, so the bias it leaves is toward the old threshold
rather than away from it. Recorded in `locrec/results/thresholds.json` as
`calibration_angles_deg` so the set is reproducible rather than remembered.

**`WorldSpec.heading_offset_deg` is how the tunnel is rotated**, defaulting to zero,
where nothing is added to the heading. Checked rather than asserted: XML, centreline
and heading are byte-identical to the previous generator across three specs and three
seeds, so no existing world or experiment changes.

**Both platforms, same rule, same seeds, same angles, separate numbers.** The drone
sees 90 degrees at 30 m rather than 360 at 10, so its ratio distribution is its own.
It turns out to be nearly orientation free, 3.7 percent of spread in its per-angle p90
against the UGV's 16.6, and its threshold moves by 2.2 percent against the UGV's 12.1.

**The sweep carries its own regression check.** Its 0 degree slice is the old
calibration, and it reproduces both old thresholds to five digits: 1.8446e-3 for the
UGV, 2.8264e-3 for the drone. So the move in the pooled number is the coverage and
nothing else.

**What the higher threshold costs where the geometry is fine, measured rather than
assumed.** `locrec/experiments/threshold_false_alarms.py`, evaluation seeds only, the mixed
world at eight orientations:

Evaluation seeds 0 to 9, eight orientations, 47,920 scans per platform, classified by
true distance to the nearest junction, niche, curve or end cap. The headline false
alarm rate is the nearest bin, where the geometry genuinely constrains the estimate
and a marker would be wasted.

UGV, 1.844588e-3 to 2.068275e-3, plus 12.1 percent:

| distance to structure | n | degenerate, old | new |
|---|---|---|---|
| on it (<=1 m) | 9008 | 0.000 | **0.024** |
| 1-5 m | 6912 | 0.119 | 0.313 |
| 5-10 m | 6616 | 0.533 | 0.906 |
| blind (>10 m) | 25384 | 0.576 | 0.943 |

Drone, 2.826411e-3 to 2.888263e-3, plus 2.2 percent:

| distance to structure | n | degenerate, old | new |
|---|---|---|---|
| on it (<=1 m) | 9008 | 0.128 | 0.151 |
| 1-5 m | 6912 | 0.278 | 0.326 |
| 5-10 m | 6616 | 0.536 | 0.615 |
| blind (>10 m) | 25384 | 0.808 | 0.911 |

Only the nearest bin is a false alarm in any strict sense, and there it goes from
**0.0 to 2.4 percent** for the UGV. The 1-5 m bin, 11.9 to 31.3 percent, is the real
bill. The 5-10 m bin going 0.533 to 0.906 is not a regression and should not be read
as one: the UGV registers out to 5.73 m, so structure between 5 and 10 m away is
outside the range that feeds the Hessian and those scans are blind by any honest
definition. A gate that calls them degenerate is right. It is not 2.4 percent more markers,
because `_action` only drops on a falling edge or after 5.5 m of travel and the
spacing rule absorbs most of those scans, but it is not nothing either.

What it buys is the orientations that were failing. Blind-stretch hit rate, old to new:

| world orientation | 0 | 11.2 | 22.5 | 33.8 | 45 | 56.2 | 67.5 | 78.8 |
|---|---|---|---|---|---|---|---|---|
| old | 0.762 | 0.279 | 0.868 | 0.853 | 0.373 | 0.677 | 0.690 | 0.106 |
| new | 0.992 | 0.848 | 0.990 | 1.000 | 0.958 | 0.951 | 0.928 | 0.878 |

The drone's rows in that table read as though its gate fires on 12.8 percent of
on-structure scans, which looks like a defect and is not one. It is the classifier:
centreline distance counts structure behind a sensor that only looks forward through
90 degrees. Where the geometry is actually in view, 1 to 5 m ahead, the drone false
alarms on 3.6 percent. `docs/failures.md` number 22 has the split, and the drone rows
above should be read against it.

One number there is real and is not caused by this change. The UGV's new blind hit
rate, 0.943, overshoots the 0.90 the quantile intends by four points: the threshold is
the 0.9 quantile of a pure straight tunnel, and the mixed world's blind stretches are
slightly more degenerate than that. It belongs to whoever redoes the headline numbers.

**Demonstrated through the ROS graph, not only offline.** Six live runs of
`sim_publisher` into `localizability_node`, three seeds, each once at the old
threshold and once at the new one, with nothing changed but the node parameter. Every
process exited 0. Blind stretches and the fraction of their scans called degenerate:

| seed | stretch | heading to the grid | old | new |
|---|---|---|---|---|
| 0 | 234-280 m | 12.9 deg | **0.02** | **0.93** |
| 1 | 100-124 m | 13.4 deg | **0.04** | **1.00** |
| 1 | 168-200 m | 2.5 deg | **0.16** | **1.00** |
| 1 | 236-284 m | 14.4 deg | **0.09** | **0.99** |
| 2 | 240-268 m | 26.0 deg | 0.98 | 1.00 |
| 0, 1, 2 | the nine stretches at 0 deg | 0 deg | 0.86 to 0.97 | 1.00 |

Seed 1 is the clearest: three of its four blind stretches lie off the grid axis and
all three were being skipped. Markers over the whole 300 m: seed 0 goes 38 to 45,
seed 1 38 to 43, and seed 2 **58 down to 40**. The drop is not a loss of coverage,
seed 2 detects more than before; it is the chatter rule in `docs/failures.md` number
17 and the open item on `detector._action`, which re-arms its falling edge on every
recovery. A threshold the stretch sits clearly below stops the ratio crossing it
repeatedly, so the spacing rule governs instead of the chatter. Largest gap between
markers falls in every seed: 56.5 to 26.5 m, 49.0 to 30.0 m, 31.0 to 29.5 m.

These are wall-clock paced runs through a real graph, so they are not reproducible
scan for scan and the old-threshold runs here differ slightly from the committed one
in `ws/log`. The effect is two orders of magnitude larger than that variation.

**Every headline in the write-up was produced with the old threshold and is not
reproducible from this commit.** The Monte Carlo grids, the marker counts and the
drift numbers all gate on it, and redoing them is its own milestone. This commit
changes the calibration and nothing downstream of it, deliberately, so the threshold
change can be argued with on its own. The transcribed values in
the ROS detector and its README were downstream in the same sense
and were left alone, which meant they disagreed with `locrec/results/thresholds.json`. That
is now fixed rather than deferred: `ratio_threshold` is required and has no default,
the node refuses to start without it with a message naming the file to read, and the
`locrec_ros` README's examples read `locrec/results/thresholds.json` instead of quoting a number. A
constant transcribed into a second place and left to go stale is `docs/failures.md`
number 20, and there is now nothing to transcribe.

## ROS 2 demonstration

**The loop is closed through the graph, with no marker model of its own.** The simulator
mounts a marker with `TunnelSim.drop_marker_on_wall` when the estimator asks, detects
markers with `TunnelSim.marker_detections`, and publishes both in one report per scan
(`ScanReport`, first named `MarkerReport`), before the scan, stamped like it. The node holds each scan until the report with its
exact stamp has arrived and applies the drops before stepping, which is the order
`run_pass` uses. A marker is therefore registered from the estimate that held when it was
mounted, however the messages interleave.

**Seed 1, because its benefit sits in the middle.** Across seeds 0 to 7 of the UGV grid
rerun at the recalibrated threshold, markers helped on five, and the paired benefit ranges
from minus 1.25 m on seed 0 to plus 7.23 m on seed 3. The median is 1.15 m, between seed
4's 1.14 m and seed 1's 1.16 m, the two middle values. The
video's results card says in words that markers do not win on every seed.

**The viewer scores along-track error, not total position error.** Along-track is the
direction a featureless tunnel cannot observe and the quantity the grids report. The first
viewer drew total error, and a sentence in the write-up then set a total error from the graph next
to an along-track number from the grid as though they were the same quantity.

**The results card leads with the whole run, not the last scan.** On seed 1 the dead-end
wall enters registration range in the last seven metres. The markers estimate falls from
0.24 m to 0.04 m in one scan there and the walls-only one jumps from 1.23 m to 1.50 m, so
the last scan reads 38 times better where the run reads about five: a mean of 0.68 m
against 0.14 m, and a worst of 1.53 m against 0.39 m. The card shows the mean and the
worst, and the viewer writes every scan's numbers to a CSV beside the video. The grids
report final error, a different statistic, and the write-up says which is which.

**Frames are driven by scans, not by the wall clock.** The first viewer rendered on a timer
and wrote whatever it managed, which on this machine was about 11 frames a second, so a run
meant to play at 1.9 times real time came out at about 2.7 without anyone choosing that.
Now frame f shows simulation time `f * playback_speed / fps` and the pace is exact on any
machine.

**The video is 16:9 and about a minute long.** A
playback speed of 2.75 puts the 150 s run at about 55 s, and the title and results cards
make up the rest.

**Drawing is on the GPU through EGL, with the CPU as a fallback.** On the GTX 1650 an EGL
context draws 400 thousand points at 1080p with readback in 17.6 ms, where the CPU splatting
was the bottleneck. Inside the conda environment the EGL dispatcher finds no vendor unless
`__EGL_VENDOR_LIBRARY_DIRS` points at the system's directory, so the renderer sets it when
the caller has not. This driver ships no NVENC library, so video encoding stays on the CPU
with libx264.

**Live on screen in rviz2 and in MuJoCo's own viewer, not in Gazebo.** The demonstration
has to be watchable as it runs. rviz2 is the standard view, fed from topics the viewer
node already joins by stamp. The simulator window is MuJoCo's viewer on the same world the
LiDAR is ray cast against. Moving the world into Gazebo would have replaced the ray cast
sensor that every threshold in `locrec/results/thresholds.json` was calibrated on. Gazebo was later
added beside MuJoCo rather than in its place, with thresholds calibrated on its own sensor:
see "Gazebo, the reference sensor".

**The simulator window draws a copy of the world, never the simulation's model.** The
ceiling hides the robot from any camera above it, and hiding it by setting its alpha to
zero in the model the LiDAR uses is not a display change: MuJoCo's ray caster skips a
geom whose alpha is zero. Measured on the mixed world, a ray straight up stops hitting the
ceiling and one scan loses 130 of its 5522 returns. The window compiles the same XML again
for drawing and copies the marker positions across on every scan.

**The simulator node leaves with `os._exit(0)` after closing its window.** MuJoCo's passive
viewer closes cleanly and then segfaults in interpreter teardown, exit 139 on every run
measured, which launch reports as a crashed node. The window is closed, the node is
cleaned up and the streams are flushed first.


## Gazebo, the reference sensor

**Beside MuJoCo, not instead of it.** The user asked for the demonstration to run in
Gazebo. Every threshold in `locrec/results/thresholds.json` describes MuJoCo's ray cast sensor, so
**Gazebo governs.** Decided 2026-09-18: where the two simulators disagree, the Gazebo number is
the result and MuJoCo is the cross-check. MuJoCo is 8 times faster per pass and stays the place to
run a grid or a declared A/B first, but it does not settle a question the reference sensor answers
differently. The rule has already changed a default, `correct_strip_bias` (`docs/failures.md`
number 34), and it is why the team's headline is `locrec/results/team_pass_gazebo.csv`.

Gazebo came in as a second simulator with its own calibration, `locrec/results/thresholds_gazebo.json`,
and `demo.launch.py` still runs the MuJoCo loop unchanged.

**The tunnel is exported from the MuJoCo XML, not generated a second time.** Every shell box
becomes a Gazebo visual with the same pose and full size, so the two worlds cannot drift apart.
Gazebo's GPU LiDAR renders visuals, so there are no collision shapes and no physics acts on
anything: each vehicle is a model with gravity off and no collision shape, placed with
`set_pose_vector` as `run_pass` places the MuJoCo sensor. Measured at eight poses on an 80 m
stretch before anything was built on it: Gazebo returned 5520 beams where MuJoCo returned 5522,
137 ceiling hits against 138, and a median per-beam range difference of 2.5 cm, against
1.9 cm for two independent 2 cm noises alone; the rest turned out to be the texture error
below. Making the ceiling transparent for the Gazebo window (85 percent in that test, 80 in
the launch file) changed none of those numbers.

**The Gazebo sensors are rendered oversampled.** At the study's beam counts Gazebo's depth
textures are 0.70 degrees per texel for the robot and 0.35 for the drone, and grazing beams
came back tens of centimetres off, enough to raise the mixed tunnel's median ratio by 47
percent and switch the gate off (`docs/failures.md` number 26). Rendering 6 and 3 times the
horizontal beams reaches Gazebo's 1024 texel cap; every 6th or 3rd beam is exactly a study
beam, a test holds the angles to 1e-8, and the calibration checks the order of the beams in
the cloud on every run. After it the ratio tracks MuJoCo's step by step, median ratio of the
two 1.015. The drone's worst beam at a test pose, grazing a wall at up to 30 m, is still
46 cm off at the cap: that is a property of the second sensor and it is left in.

**The Gazebo calibration captures one pass and replays it.** 760 passes through a live server
would take hours. `blind_world` is the same geometry for every seed, rotating the tunnel rotates
the sensor with it, and Gazebo's range noise measured as configured (2.01 cm, mean +0.006 cm,
noiseless frames identical to the bit), so one noiseless pass per platform is captured and each
seed's noise is drawn from its own generator. The rotation is checked, not assumed: a copy of the
200 m tunnel rotated 45 degrees and stacked 10 m above returned the same sensor-frame ranges to
1.4 and 1.5 mm at the 99th percentile for the two sensors, worst 9.4 mm, against 2 cm of noise. Quantiles, seeds, angles and
rules are imported from `calibrate_thresholds.py`, not copied.

**The shell is drawn as three meshes, so the GPU does the simulating.** The LiDAR always rendered on
the GTX 1650 (the server holds 128 MiB there), but with every tunnel box its own visual Gazebo
submitted one draw call per box for every face of the LiDAR's cube map, and the CPU was the
limit: on the 300 m mixed tunnel, 1529 visuals, a step with both sensors took 62 ms and 83 ms of
server CPU while the GPU sat at 2 percent. Merged into one mesh per material (walls, floor,
ceiling), 22 visuals, the same step takes 22.5 ms and 12.8 ms of CPU. The mesh is the same sensor:
over 60 poses, 329357 robot beams and 1201481 drone beams, no beam hit in one world and missed in
the other, and the range difference was 0.02 mm at the median, 0.5 mm at the 99th percentile and
18 mm at worst, against 20 mm of range noise. The server also runs with `__GL_YIELD=USLEEP`, which
stops NVIDIA's driver spinning a core while it waits for the GPU: 26 percent less server CPU at
the same scan time. The thresholds, captured on the box world, stand on that equivalence; every
run after the change, cross-checks and videos included, is on the meshes.

**Marker returns are picked out by intensity.** A strip carries a `laser_retro` value and Gazebo
returns it on every beam that hits it; rock returns 0, and nothing in between was seen. Which
strip a return belongs to is ground truth, the nearest mounted strip, as MuJoCo's geom id is.
The sideways probe that measures the wall before a strip is mounted is still MuJoCo's ray
against the same geometry. The detection rate against distance is within one of nine offsets of
MuJoCo's at every half metre from 1 to 12 m, with the same beam counts and fit errors within
noise, and both give a reliable range of 7.0 m.

**The client never subscribes in the process that makes requests.** gz-transport's Python
`request` holds the interpreter lock while it waits, so a frame arriving during a request stalls
the reply (`docs/failures.md` number 27). Frames come up a pipe from a child process. The server
is started with a parent death signal, because a driver that died in its constructor once left
a server running with its topics and the GPU.

**The driver waits for the estimators.** It waits after every scan for each estimator's
estimate and action for that scan before moving, so the live loop acts when `run_pass` acts.
Between scans it moves each vehicle in ten 0.05 s steps so the Gazebo window shows motion; only
the last step renders a scan.

**Two drones fly side by side rather than one after the other.** The two copies are placed at
the smallest sideways offset that keeps them at least 4 m apart, with the same seed, so
forward and glance gaze see the same tunnel and the same odometry noise at the same moment and
can be watched against each other. The gaze policies are the study's classes, called inside the
estimator node with only what `run_pass` gives them (a slotted context that fails on anything
more), and the driver slews the sensor at the study's rate limit.

**The Gazebo window looks down steeply, and follows the track rather than the drone.** From
beside the tunnel at a shallow angle the tunnel's own 2.6 m wall hid the robot in its middle, in
a screenshot of the running window; from behind and above, through the see-through ceiling, it
is in view. Gazebo keeps a follow offset in the followed model's own frame (`CameraTracking.cc`,
gz-gui 8), so following the glance drone would swing the camera round with each glance. The
driver moves an empty model along each drone's position with its track heading, and the window
follows that.

**The estimate can start at the odometry's first pose, and the gaze needs it to.** A gaze policy
steers relative to the track heading, which is in the odometry frame, and plans on the estimator's
map. `gz_driver` starts its odometry at the true first pose, as `run_pass` starts its estimate,
so starting the estimate there puts all three in one frame and the estimator's voxel grid where
`run_pass` has it. It is an opt-in parameter, `start_at_odometry`, rather than a new default:
the node's estimate has started at the identity since it was written, and the drone's gaze
refuses to run without it.

**The look is plain on purpose.** The user found the first viewer's glow and badges looked
machine-made and chose engineering plots: the 3D view flat on rviz's grey, matplotlib with axis
labels and units in Liberation Sans, the standard tab10 colours, and a closing figure with a
table instead of a slogan card. For the drone the viewer measures along-track error along the
track heading; `run_pass` measures it along the sensor's x axis, and for a glancing drone the two
differ while it looks away.

**What the second sensor changed, measured.** Calibrated on Gazebo by the declared procedure, the
UGV threshold is 2.109e-3 against MuJoCo's 2.068e-3 (+2.0 percent), the drone's 3.040e-3 against
2.888e-3 (+5.3 percent), the marker reliable range 7.0 m on both, from the same 151620 samples per
platform. The orientation band reproduces: the UGV's 90th percentile runs from 1.88e-3 at 0 degrees
to 2.19e-3 at 80. `locrec/experiments/gazebo_crosscheck.py` then ran the study's two comparisons on
Gazebo's LiDAR, 8 seeds, 300 m, one thread, on the merged-mesh worlds, in
`locrec/results/gazebo_crosscheck_{ugv,drone}.csv`:

* UGV, mixed world, no markers against the scheduler. Median final along-track error **2.21 m
  against 0.85 m**, paired +1.43 m [-0.32, +2.92], better on 5 of 8 seeds, 18 to 21 markers, with
  the measurement model counting beam columns rather than returns and the strip-bias correction
  off for this sensor (`docs/failures.md` numbers 31 and 34). The same comparison before both
  changes gave 1.17 m, paired +1.42 m [-0.94, +2.51], and with the column fix but the bias
  correction still on it gave 1.48 m, paired +0.91 m, better on only 4 of 8; that arm is kept in
  `locrec/results/gazebo_crosscheck_ugv_strip_bias_on.csv`. Mean over the run, 1.20 m against 0.47 m.
  The measured spread model is part of that result and was A/B'd on this sensor too: with it off
  the scheduler's median is 1.14 m, paired +1.07 m, so it is kept
  (`locrec/results/gazebo_crosscheck_ugv_strip_spread_off.csv`). The
  MuJoCo rerun at the recalibrated threshold that `locrec_ros/README.md` cites, same statistic: 2.43 m
  against 0.91 m, 5 of 8. The three seeds where markers made it worse are the same three in both,
  0, 2 and 7, and without markers the two simulators agree seed by seed to within 0.29 m, from
  0.32 m against 0.15 m on seed 0 to 7.68 m against 7.86 m on seed 3. Final total error, which is
  not the study's declared metric and is reported rather than promoted: median 4.11 m against
  3.03 m, paired +0.37 m [-1.38, +3.30], 5 of 8. Before the two changes it was 2.86 m, paired
  +0.71 m [+0.09, +2.54], 7 of 8, and that was the one interval in this table that excluded zero.
  It no longer does, and the honest reading is that the marker benefit on this sensor is clear in
  the along-track statistic the study declared and not established in total error. The same cross-check on the box worlds, before the
  shell was merged, gave 2.64 m against 1.00 m with the same three seeds worse.
* Drone, junction world, forward against glance gaze. The write-up's drone table reports "Total" as
  the hypotenuse of the median along-track and median lateral errors; on MuJoCo that was 1.59 m
  forward against 1.17 m glance. On Gazebo the same statistic is **1.91 m forward against 1.89 m
  glance**: level. Glance's own final error is the larger on 5 of 8 seeds, and the paired
  along-track difference straddles zero, +0.66 m [-1.23, +1.04], 5 of 8, as it did on MuJoCo,
  +0.23 m [-0.59, 0.89], 5 of 8. The comparison is fragile rather than reversed: on the box worlds,
  whose scans differ from the meshes' by 0.5 mm at the 99th percentile, the same 8 seeds gave 1.72 m
  against 2.11 m. So the study's reading that glance helps in the junction world does not survive
  the second sensor, no difference either way survives 8 seeds, and the drone video says that. It
  is consistent with the write-up's own headline that looking for localizability does not pay.

**The demonstration seeds are the rule's, and the rule had a bug.** The rule, declared in the
cross-check script before it ran, takes the seed whose paired difference in mean along-track
error is closest to the median, ties to the lower seed. With 8 seeds the median is the midpoint
of the two middle values, so those two seeds are always exactly as close as each other and the
tie rule always decides. The first version compared raw floats and picked seed 5 over seed 1 on a
difference in the 17th digit. Fixed, it gives **seed 1** for the UGV, as the MuJoCo demonstration's
own rule did, and on the mesh worlds **seed 4** for the drones.

**The live Gazebo run is the offline run.** With registration on one thread and the first render
discarded as the offline pass discards it (`docs/failures.md` number 30), the recorded UGV
demonstration's marker estimator matches its offline `run_pass` on all 600 scans to 0.000 mm,
mean 0.461 m, on the mesh world as it had on the box world. The LiDAR-only estimator differs by
design, because live it registers the scans that include the strips. The two drones share one
server and one noise stream, so the drone video is a different draw from the offline passes: on
seed 4, forward gaze 1.49 m mean along-track error and glance 3.02 m. Every jump of more than
0.3 m in glance's error, at 4, 57, 89.5 and 197 m, came while it was looking away, though most of
its glances, 136 scans in all, cost nothing visible.

## The team: a drone that mounts nothing, in the robot's frame

**A marker's worth is its measurement, not its owner.** A robot's own strip is a relative anchor:
the fix is weighted by the odometry accumulated since the drop, and the measurement noise is the
drop-time observation plus the current one. A teammate's strip has no drop of this robot's to be
relative to, so the same two questions get different answers.

*What the measurement noise is*: the teammate's drop-time 2x2 and this robot's observation, and
nothing else. Not the teammate's absolute covariance. In the teammate's frame the strip's
coordinates are the definition of the frame, so there is nothing else to be uncertain about.
Weighting by the teammate's absolute covariance is the mistake `relative_anchors` exists to
avoid, and it bites harder here: a strip 250 m into a tunnel would look worthless exactly where
it is the only thing on offer.

*What the prior is*: this robot's odometry since it was last fixed **in the shared frame**, not
since it was told about this particular strip. All of a teammate's strips live in one frame, so
being fixed on any of them is being fixed against all of them, including the ones still out of
sight. `Odometry._frame_q_ref` and `_frame_rel` keep that one mark, and `_relative_covariance`
uses it for any landmark marked `foreign`. Keeping a mark per anchor instead would charge a strip
first seen at 200 m for every metre since the message arrived, although the robot had been fixed
at 190 m by the strip before it. `locrec/test/test_team.py` asserts the two are identical.

**A teammate's strip is never re-surveyed.** `resurvey_anchors` moves an anchor to the current
estimate when the estimate is the better of the two, which is what a survey crew does with a
control point. Doing that to a teammate's strip would redefine the shared frame without telling
the teammate, so `_resurvey` skips anything `foreign`.

**The message carries the face normal, because the fit's bias is mirrored.** The end face a strip
shows is the one toward the sensor, so the fit reads it short on the approach and long once it is
passed. A vehicle that inferred the facing from its own pose, the way it does for a strip it
mounted itself, would apply that correction backwards on strips someone else left facing the
other way, which is worse than not correcting at all. So the message is four things and no more:
slot, position, the 2x2 that placed it, and the normal.

**A strip is sent again when its owner moves it.** A strip mounted since the last fix drifts with
the estimate, so the next fix moves it (`Odometry._absolute_fix`), and the vehicle then knows
where it is better than it did when it bolted it on. Publishing once at the drop would leave the
team holding a position its owner has already disowned, so the node compares each shared strip
against what it last sent and sends the new answer when they differ, which in practice is once per
strip, at the first fix after the drop. The receiver replaces the position and the covariance and
touches nothing else: re-registering would reset the anchor's relative record and hand it a fix it
has not earned, which is worse than a stale position. Since the robot is 30 m ahead when it mounts
a strip, the correction reaches the drone long before the drone can see it, which is why the
offline pass, which uses the robot's settled belief, is a fair model of the live one.

**What the team result is, and is not.** It is a frame-consistency result. The drone agrees with
the ground robot, and it does not become more right about the world than the robot was: it
inherits the robot's chain error exactly, which is the floor and is why the marker work comes
first. Numbers in `locrec/results/team_pass_gazebo.csv`, which is the reference sensor, and
`locrec/results/team_pass.csv` for the MuJoCo cross-check: in the robot's frame, median 1.85 m solo
against 0.05 m team on Gazebo and 1.35 m against 0.04 m on MuJoCo, better on 8 of 8 seeds in
both.

## The runtime is C++, the study is Python

The estimator has to keep up with a LiDAR stream, so it is C++: `locrec_core` is the algorithm on
Eigen and small_gicp with no ROS dependency, and `locrec_estimator` is the rclcpp node that carries
it. Keeping the core free of ROS is what lets it be unit tested and reused without a graph, and
keeping the node thin is what keeps the decisions in one place, the detector, where the tests can
reach them.

The procedural worlds, the Gazebo driver, the viewer and the offline study stay in Python. That is
where the work is generating geometry, drawing, and running one process per seed across grids, and
none of it sits on the path from a scan to a pose.

small_gicp has no rosdep key, so `locrec_core` fetches the pinned release at configure time and
compiles the two translation units it needs straight into the library. Nothing extra has to be
installed at runtime, and `-DSMALL_GICP_SOURCE_DIR` takes a local checkout for an offline build.
