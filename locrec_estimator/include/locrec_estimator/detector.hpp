// The detector and the action rule, with no ROS include.
//
// Everything the node decides lives here and in prior_pairer.hpp so it can be tested without a
// graph. The node is then a thin shell: convert, pair each scan with its prior, call this,
// publish.
#pragma once

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "locrec_core/gaze.hpp"
#include "locrec_core/landmarks.hpp"
#include "locrec_core/lidar.hpp"
#include "locrec_core/localizability.hpp"
#include "locrec_core/odometry.hpp"
#include "locrec_core/policies.hpp"
#include "locrec_core/se3.hpp"

namespace locrec_estimator
{

/// The registration range every calibrated threshold was produced with: the smaller of the
/// geometric limit, which keeps the leading shell of the scan out of the registration, and the
/// sampling limit, where the scan's azimuth spacing exceeds half a map voxel
/// (docs/failures.md number 2).
double defaultRegistrationRange(
  const locrec::LidarSpec & lidar, const locrec::OdometryConfig & odometry, double step_length);

struct DetectorConfig
{
  /// Below this the scan is called degenerate. Required, with no default: the ratio depends on
  /// the sensor's field of view, its range and the size of the space, so a number calibrated on
  /// one is meaningless on another, and a default would be a number that looks calibrated and is
  /// not. It has one source, locrec/results/thresholds*.json.
  double ratio_threshold = 0.0;
  double min_lambda_per_point = 0.0;
  double max_range = 10.0;
  double fov_azimuth_deg = 360.0;
  int azimuth_beams = 360;
  int elevation_beams = 16;
  double fov_elevation_deg = 30.0;
  /// "ugv" recommends mounting a marker, "drone" recommends a yaw.
  std::string platform = "ugv";
  /// How the drone picks its yaw: "across" looks across the weak direction whenever the scan is
  /// degenerate, "forward" and "glance" are the study's policies. Both of those steer relative to
  /// a track heading, so they need start_at_odometry and a track yaw with every scan.
  std::string gaze = "across";
  /// Threads small_gicp registers with. Not only a speed setting: over a 300 m run the result
  /// depends on it, deterministically, so the launch files set 1 as the study's grids do
  /// (docs/failures.md number 30).
  int registration_threads = 4;
  /// Start the estimate at the odometry pose of the first scan instead of the identity.
  bool start_at_odometry = false;
  double map_resolution = 0.2;
  double map_radius = 30.0;
  /// Nominal travel between scans, metres, as the offline pass used it.
  double step_length = 0.5;
  /// Record every strip this vehicle mounts, for the node to tell the team about.
  bool share_landmarks = false;
  /// Localize against strips a teammate mounted, in the teammate's frame. A vehicle with this on
  /// mounts nothing itself: it inherits the frame of whoever placed the strips, and with it that
  /// vehicle's chain error, which is the floor on how right it can be about the world.
  bool use_shared_landmarks = false;
  /// The strip as the world builds it. The estimator needs all three: the width and height size
  /// the measurement covariance, and the thickness drives the fit's end-face correction.
  double marker_width = 0.15;
  double marker_height = 1.0;
  double marker_thickness = 0.02;
  /// Range at which a marker is still reliably detected, and the headroom the scheduler keeps.
  /// Required for the ground robot, with no default, for the same reason the threshold has none.
  std::optional<double> marker_reliable_range_m;
  std::optional<double> scheduler_margin_m;
};

struct DetectorOutput
{
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Vector3d weak_direction = Eigen::Vector3d::UnitX();
  double ratio = 0.0;
  double lambda_min_per_point = 0.0;
  bool is_degenerate = false;
  int n_points = 0;
  /// One of "none", "drop_marker", "yaw_to:<deg>".
  std::string action = "none";
  /// Known markers that corrected the estimate on this scan.
  int landmarks_used = 0;
};

/// What this vehicle tells the team about a strip it has just mounted, and nothing else.
struct SharedStrip
{
  int slot = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Matrix2d drop_covariance = Eigen::Matrix2d::Zero();
  Eigen::Vector2d normal = Eigen::Vector2d::UnitX();
};

/// A marker mounted since the last scan: its slot and where it went, in the sensor frame of the
/// pose that mounted it.
struct MarkerDrop
{
  int slot = 0;
  Eigen::Vector3d offset_sensor = Eigen::Vector3d::Zero();
};

/// Runs the locrec odometry over incoming scans and emits the signal.
///
/// The threshold is only meaningful for a ratio produced the way the offline pass produced it
/// during calibration, so the registration is configured exactly as that pass configured it: the
/// motion prior, the registration range, and the local map radius, floored at twice the sensor
/// range.
///
/// The motion prior: every scan comes with the odometry pose at its own stamp, and consecutive
/// poses are differenced into the increment Odometry::step expects. There is no default. An
/// earlier version assumed no motion when no prior was given, on the argument that the Hessian
/// describes the geometry rather than the pose. That argument does not survive the map being
/// built from the estimate: with the robot moving and the estimate standing still, scans are
/// written into the map in the wrong place and nothing is ever called degenerate.
///
/// The marker rule is not reimplemented here either: it is LocalizabilityScheduler itself, fed
/// the same two things the offline pass feeds it.
class LocalizabilityDetector
{
public:
  explicit LocalizabilityDetector(const DetectorConfig & config);

  /// One scan in, one signal out. Returns nothing until registration is possible.
  ///
  /// drops are the markers mounted since the last scan, registered from the estimate as it stands
  /// before this scan is stepped; detections are the markers seen in this one. That is the order
  /// the offline pass does it in.
  std::optional<DetectorOutput> process(
    const locrec::Points & points_sensor, const Eigen::Isometry3d & odom_pose,
    const std::vector<MarkerDrop> & drops = {},
    const std::vector<locrec::MarkerDetection> & detections = {},
    std::optional<double> track_yaw = std::nullopt);

  /// Take a strip a teammate mounted. Returns whether it was new. A teammate republishes a strip
  /// when its own estimate of it moves, so the same slot arrives more than once: the later answer
  /// replaces the position and the covariance and touches nothing else, because re-registering
  /// would reset the anchor's relative record and hand this vehicle a fix it has not earned.
  bool addSharedLandmark(
    int slot, const Eigen::Vector3d & position, const Eigen::Matrix2d & drop_covariance,
    const Eigen::Vector2d & normal);

  /// How the drone's yaw is decided. "across" is this node's own rule; the other two are the
  /// study's gaze policies and steer relative to the track heading.
  enum class GazeMode { across, forward, glance };

  /// Whether process needs the track heading with every scan.
  bool needsTrackYaw() const {return gaze_ != GazeMode::across;}
  Eigen::Isometry3d pose() const {return odometry_.pose();}
  double estimatedDistance() const {return estimated_distance_;}
  const DetectorConfig & config() const {return config_;}
  const locrec::Odometry & odometry() const {return odometry_;}
  /// Strips mounted since the node last drained this, each ready to send.
  std::vector<SharedStrip> & shared() {return shared_;}

private:
  SharedStrip share(int slot) const;
  void reshareMovedStrips();
  std::string action(const locrec::Localizability & loc, std::optional<double> track_yaw);

  DetectorConfig config_;
  locrec::LidarSpec lidar_spec_;
  locrec::Odometry odometry_;
  locrec::LocalizabilityConfig loc_config_;
  std::optional<locrec::LocalizabilityScheduler> scheduler_;
  std::optional<locrec::GlanceGaze> glance_;
  GazeMode gaze_ = GazeMode::across;
  std::vector<SharedStrip> shared_;
  std::map<int, Eigen::Vector3d> shared_at_;
  std::set<int> foreign_;
  double estimated_distance_ = 0.0;
  std::optional<Eigen::Isometry3d> last_odom_pose_;
};

}  // namespace locrec_estimator
