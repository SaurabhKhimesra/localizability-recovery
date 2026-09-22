#include "locrec_estimator/detector.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace locrec_estimator
{

namespace
{

std::string yawAction(double yaw_rad, int digits)
{
  std::ostringstream out;
  out.precision(digits);
  out << std::fixed << "yaw_to:" << yaw_rad * 180.0 / M_PI;
  return out.str();
}

}  // namespace

double defaultRegistrationRange(
  const locrec::LidarSpec & lidar, const locrec::OdometryConfig & odometry, double step_length)
{
  return std::min(
    lidar.max_range - odometry.max_correspondence_distance - step_length,
    lidar.nyquistRange(odometry.map_resolution));
}

namespace
{

locrec::LidarSpec lidarFrom(const DetectorConfig & config)
{
  locrec::LidarSpec lidar;
  lidar.n_azimuth = config.azimuth_beams;
  lidar.fov_azimuth_deg = config.fov_azimuth_deg;
  lidar.max_range = config.max_range;
  lidar.n_elevation = config.elevation_beams;
  lidar.fov_elevation_deg = config.fov_elevation_deg;
  return lidar;
}

locrec::OdometryConfig odometryConfigFrom(
  const DetectorConfig & config, const locrec::LidarSpec & lidar)
{
  locrec::OdometryConfig cfg;
  cfg.map_resolution = config.map_resolution;
  // the map has to reach further than the sensor does, or the scan sticks out past both ends of
  // it (docs/failures.md number 1)
  cfg.map_radius = std::max(config.map_radius, 2.0 * config.max_range);
  cfg.gicp_ratio_floor = config.ratio_threshold;
  cfg.num_threads = config.registration_threads;
  cfg.registration_range = defaultRegistrationRange(lidar, cfg, config.step_length);
  // correct_strip_bias is deliberately left off: the strip fit's bias is sensor-specific,
  // +3.28 cm on MuJoCo's ray casts against +0.43 cm on Gazebo's rendered depth, and this node
  // drives Gazebo (docs/failures.md number 34). measure_strip_spread stays on, because the spread
  // does transfer between the two sensors.
  return cfg;
}

}  // namespace

LocalizabilityDetector::LocalizabilityDetector(const DetectorConfig & config)
: config_(config), lidar_spec_(lidarFrom(config)),
  odometry_(
    Eigen::Isometry3d::Identity(), odometryConfigFrom(config, lidarFrom(config)),
    locrec::MotionPriorSpec(), 0.5, lidarFrom(config), locrec::LandmarkSpec(),
    config.marker_height, config.marker_width, config.marker_thickness)
{
  if (config.fov_azimuth_deg <= 0.0 || config.azimuth_beams < 1) {
    throw std::invalid_argument("fov_azimuth_deg must be positive and azimuth_beams at least 1");
  }
  if (config.platform != "ugv" && config.platform != "drone") {
    throw std::invalid_argument("platform must be ugv or drone, got '" + config.platform + "'");
  }
  if (config.gaze == "across") {
    gaze_ = GazeMode::across;
  } else if (config.gaze == "forward") {
    gaze_ = GazeMode::forward;
  } else if (config.gaze == "glance") {
    gaze_ = GazeMode::glance;
  } else {
    throw std::invalid_argument(
            "gaze must be across, forward or glance, got '" + config.gaze + "'");
  }
  if (config.platform == "drone" && gaze_ != GazeMode::across && !config.start_at_odometry) {
    throw std::invalid_argument(
            "gaze '" + config.gaze + "' steers relative to a track heading in the odometry frame, "
            "so the estimate has to start in that frame: set start_at_odometry");
  }
  if (config.platform == "ugv") {
    gaze_ = GazeMode::across;  // the ground robot's sensor yaw is its heading
  }

  loc_config_.ratio_threshold = config.ratio_threshold;
  loc_config_.min_lambda_per_point = config.min_lambda_per_point;

  if (config.platform == "ugv") {
    if (!config.marker_reliable_range_m.has_value() || !config.scheduler_margin_m.has_value()) {
      throw std::invalid_argument(
              "the ugv recommends markers, so marker_reliable_range_m and scheduler_margin_m are "
              "required: both are calibrated and live in locrec/results/thresholds.json");
    }
    scheduler_.emplace(
      config.ratio_threshold, *config.marker_reliable_range_m, *config.scheduler_margin_m);
  }
  if (gaze_ == GazeMode::glance) {
    // the policy as locrec/experiments/drone_gaze.py builds it, pre-registered there
    glance_.emplace(config.ratio_threshold, 60.0);
  }
}

SharedStrip LocalizabilityDetector::share(int slot) const
{
  const locrec::Landmark * lm = odometry_.landmarks().get(slot);
  SharedStrip out;
  out.slot = slot;
  if (lm == nullptr) {
    return out;
  }
  out.position = lm->position;
  Eigen::Vector2d normal = Eigen::Vector2d::UnitX();
  if (lm->T_drop.has_value() && lm->offset_drop.has_value()) {
    const Eigen::Vector2d back = -(lm->T_drop->linear() * *lm->offset_drop).head<2>();
    if (back.norm() > 1e-9) {
      normal = back / back.norm();
    }
  }
  out.normal = normal;
  const double floor = odometry_.landmarkSpec().min_sigma_m;
  out.drop_covariance = lm->R_drop.value_or(Eigen::Matrix2d(Eigen::Matrix2d::Identity() * floor * floor));
  return out;
}

void LocalizabilityDetector::reshareMovedStrips()
{
  // A strip mounted since the last fix drifted with the estimate, so the next fix moves it. The
  // vehicle knows where it is better afterwards than it did when it bolted it on, and a teammate
  // told the first answer and never the second would hold a position its owner has disowned.
  for (auto & entry : shared_at_) {
    const locrec::Landmark * lm = odometry_.landmarks().get(entry.first);
    if (lm == nullptr || (lm->position - entry.second).norm() < 1e-9) {
      continue;
    }
    shared_.push_back(share(entry.first));
    entry.second = lm->position;
  }
}

bool LocalizabilityDetector::addSharedLandmark(
  int slot, const Eigen::Vector3d & position, const Eigen::Matrix2d & drop_covariance,
  const Eigen::Vector2d & normal)
{
  if (!config_.use_shared_landmarks) {
    return false;
  }
  if (foreign_.count(slot) != 0) {
    if (locrec::Landmark * lm = odometry_.landmarks().get(slot)) {
      lm->position = position;
      lm->R_drop = drop_covariance;
    }
    return false;
  }
  odometry_.registerForeignLandmark(slot, position, drop_covariance, normal);
  foreign_.insert(slot);
  return true;
}

std::optional<DetectorOutput> LocalizabilityDetector::process(
  const locrec::Points & points_sensor, const Eigen::Isometry3d & odom_pose,
  const std::vector<MarkerDrop> & drops, const std::vector<locrec::MarkerDetection> & detections,
  std::optional<double> track_yaw)
{
  if (!odom_pose.matrix().allFinite()) {
    throw std::invalid_argument("odom_pose must be a finite pose");
  }
  if (needsTrackYaw() && (!track_yaw.has_value() || !std::isfinite(*track_yaw))) {
    throw std::invalid_argument(
            "gaze '" + config_.gaze + "' steers relative to the track heading: pass track_yaw");
  }

  Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
  if (!last_odom_pose_.has_value()) {
    if (config_.start_at_odometry) {
      // nothing has been stepped, so placing the estimate here is constructing it here
      odometry_.pose() = odom_pose;
    }
  } else {
    delta = last_odom_pose_->inverse() * odom_pose;
  }
  last_odom_pose_ = odom_pose;

  // a marker is registered from the estimate that held when it was mounted, which is the one
  // still standing here, before this scan moves it
  for (const auto & drop : drops) {
    odometry_.registerLandmark(drop.slot, drop.offset_sensor);
    if (config_.share_landmarks) {
      shared_.push_back(share(drop.slot));
      if (const locrec::Landmark * lm = odometry_.landmarks().get(drop.slot)) {
        shared_at_[drop.slot] = lm->position;
      }
    }
  }

  const Eigen::Vector3d before = odometry_.pose().translation();
  const locrec::OdomStep step = odometry_.step(points_sensor, delta, detections);
  estimated_distance_ += (odometry_.pose().translation() - before).norm();
  reshareMovedStrips();

  if (!step.localizability.has_value()) {
    return std::nullopt;
  }
  const locrec::Localizability & loc = *step.localizability;

  DetectorOutput out;
  out.eigenvalues = loc.eigenvalues;
  out.weak_direction = loc.weak_direction;
  out.ratio = loc.ratio;
  out.lambda_min_per_point = loc.lambda_min_per_point;
  out.is_degenerate = loc.isDegenerate(loc_config_);
  out.n_points = loc.n_points;
  out.action = action(loc, track_yaw);
  out.landmarks_used = step.landmarks_used;
  return out;
}

std::string LocalizabilityDetector::action(
  const locrec::Localizability & loc, std::optional<double> track_yaw)
{
  if (scheduler_.has_value()) {
    return (*scheduler_)(loc.ratio, estimated_distance_) ? "drop_marker" : "none";
  }
  if (gaze_ == GazeMode::forward) {
    // enough digits that the yaw a vehicle is sent is the yaw the offline pass would command
    return yawAction(locrec::ForwardGaze()(*track_yaw), 6);
  }
  if (gaze_ == GazeMode::glance) {
    const locrec::Points points = odometry_.map().points();
    const locrec::Points normals = locrec::mapNormals(points);
    // the scan interval the policy's hold and cooldown are counted in
    const double dt = config_.step_length / 1.0;
    const double yaw = (*glance_)(
      *track_yaw, odometry_.pose(), loc, points, normals, lidar_spec_, dt);
    return yawAction(yaw, 6);
  }
  if (!loc.isDegenerate(loc_config_)) {
    return "none";
  }
  // look across the weak direction: that is where the structure which would constrain it has to be
  const Eigen::Vector3d weak = loc.weak_direction;
  return yawAction(std::atan2(weak.x(), -weak.y()), 1);
}

}  // namespace locrec_estimator
