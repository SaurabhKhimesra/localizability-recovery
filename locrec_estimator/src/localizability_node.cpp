// The node. Everything it decides lives in detector.cpp and prior_pairer.hpp; this is the shell.
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>

#include <locrec_msgs/msg/localizability.hpp>
#include <locrec_msgs/msg/scan_report.hpp>
#include <locrec_msgs/msg/shared_landmark.hpp>

#include "locrec_estimator/calibration.hpp"
#include "locrec_estimator/conversions.hpp"
#include "locrec_estimator/detector.hpp"
#include "locrec_estimator/prior_pairer.hpp"

namespace locrec_estimator
{

/// Frame of the published estimate. Its origin is the sensor pose of the first scan the node
/// processed, which is all an odometry can ever know about where it started; with
/// start_at_odometry it is the frame of /odom_prior instead.
constexpr char kEstimateFrame[] = "locrec_odom";

namespace
{

struct ScanPayload
{
  std_msgs::msg::Header header;
  locrec::Points points;
};

struct SidePayload
{
  std::vector<MarkerDrop> drops;
  std::vector<locrec::MarkerDetection> detections;
  double track_yaw = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace

/// The estimate for one scan, stamped like the scan so it joins to the truth exactly.
nav_msgs::msg::Odometry buildEstimate(
  const std_msgs::msg::Header & header, const Eigen::Isometry3d & pose)
{
  nav_msgs::msg::Odometry msg;
  msg.header.stamp = header.stamp;
  msg.header.frame_id = kEstimateFrame;
  msg.child_frame_id = header.frame_id;
  msg.pose.pose.position.x = pose.translation().x();
  msg.pose.pose.position.y = pose.translation().y();
  msg.pose.pose.position.z = pose.translation().z();
  const Eigen::Quaterniond q(pose.linear());
  msg.pose.pose.orientation.x = q.x();
  msg.pose.pose.orientation.y = q.y();
  msg.pose.pose.orientation.z = q.z();
  msg.pose.pose.orientation.w = q.w();
  return msg;
}

locrec_msgs::msg::Localizability buildMessage(
  const std_msgs::msg::Header & header, const DetectorOutput & out)
{
  locrec_msgs::msg::Localizability msg;
  msg.header = header;
  for (int i = 0; i < 3; ++i) {
    msg.eigenvalues[i] = out.eigenvalues(i);
    msg.weak_direction[i] = out.weak_direction(i);
  }
  msg.ratio = out.ratio;
  msg.lambda_min_per_point = out.lambda_min_per_point;
  msg.is_degenerate = out.is_degenerate;
  msg.n_points = out.n_points;
  msg.landmarks_used = out.landmarks_used;
  return msg;
}

locrec_msgs::msg::SharedLandmark buildShared(
  const std_msgs::msg::Header & header, const SharedStrip & strip)
{
  locrec_msgs::msg::SharedLandmark msg;
  msg.header = header;
  msg.slot = strip.slot;
  msg.position.x = strip.position.x();
  msg.position.y = strip.position.y();
  msg.position.z = strip.position.z();
  for (int i = 0; i < 4; ++i) {
    msg.drop_covariance[i] = strip.drop_covariance(i / 2, i % 2);
  }
  msg.normal.x = strip.normal.x();
  msg.normal.y = strip.normal.y();
  msg.normal.z = 0.0;
  return msg;
}

class LocalizabilityNode : public rclcpp::Node
{
public:
  LocalizabilityNode()
  : rclcpp::Node("locrec_localizability")
  {
    // calibrated, so declared by type with no default: a default here would be a second source
    // for a number that has one. See calibration.hpp.
    declare_parameter("thresholds_file", "");
    declare_parameter("ratio_threshold", rclcpp::PARAMETER_DOUBLE);
    declare_parameter("marker_reliable_range", rclcpp::PARAMETER_DOUBLE);
    declare_parameter("scheduler_margin", rclcpp::PARAMETER_DOUBLE);
    declare_parameter("min_lambda_per_point", 0.0);
    declare_parameter("range", 10.0);
    declare_parameter("fov", 360.0);
    declare_parameter("azimuth_beams", 360);
    declare_parameter("elevation_beams", 16);
    declare_parameter("fov_elevation", 30.0);
    declare_parameter("platform", "ugv");
    declare_parameter("gaze", "across");
    declare_parameter("start_at_odometry", false);
    declare_parameter("registration_threads", 4);
    declare_parameter("max_points", 60000);
    declare_parameter("use_markers", false);
    declare_parameter("share_landmarks", false);
    declare_parameter("use_shared_landmarks", false);

    max_points_ = get_parameter("max_points").as_int();
    use_markers_ = get_parameter("use_markers").as_bool();
    share_landmarks_ = get_parameter("share_landmarks").as_bool();
    use_shared_ = get_parameter("use_shared_landmarks").as_bool();
    if (share_landmarks_ && !use_markers_) {
      throw std::runtime_error(
              "share_landmarks needs use_markers: a vehicle can only tell the team about strips "
              "it is mounting");
    }
    if (use_shared_ && use_markers_) {
      throw std::runtime_error(
              "use_shared_landmarks is for a vehicle that mounts nothing of its own; with "
              "use_markers the two sets of slots would collide");
    }

    const std::string platform = get_parameter("platform").as_string();
    const Calibration calibrated = resolveCalibration(
      platform, optionalDouble("ratio_threshold"), optionalDouble("marker_reliable_range"),
      optionalDouble("scheduler_margin"), get_parameter("thresholds_file").as_string());

    DetectorConfig config;
    config.ratio_threshold = calibrated.ratio_threshold;
    config.min_lambda_per_point = get_parameter("min_lambda_per_point").as_double();
    config.max_range = get_parameter("range").as_double();
    config.fov_azimuth_deg = get_parameter("fov").as_double();
    config.azimuth_beams = static_cast<int>(get_parameter("azimuth_beams").as_int());
    config.elevation_beams = static_cast<int>(get_parameter("elevation_beams").as_int());
    config.fov_elevation_deg = get_parameter("fov_elevation").as_double();
    config.platform = platform;
    config.gaze = get_parameter("gaze").as_string();
    config.start_at_odometry = get_parameter("start_at_odometry").as_bool();
    config.registration_threads =
      static_cast<int>(get_parameter("registration_threads").as_int());
    config.marker_reliable_range_m = calibrated.marker_reliable_range_m;
    config.scheduler_margin_m = calibrated.scheduler_margin_m;
    config.share_landmarks = share_landmarks_;
    config.use_shared_landmarks = use_shared_;
    detector_ = std::make_unique<LocalizabilityDetector>(config);

    // the report carries what markers and a gaze policy need, so either one holds each scan for
    // the report with its stamp
    use_report_ = use_markers_ || use_shared_ || detector_->needsTrackYaw();
    pairer_ = std::make_unique<PriorPairer<ScanPayload, SidePayload>>(10, 4096, use_report_);

    pub_ = create_publisher<locrec_msgs::msg::Localizability>("/localizability", 10);
    action_pub_ = create_publisher<std_msgs::msg::String>("/localizability/recommended_action", 10);
    estimate_pub_ = create_publisher<nav_msgs::msg::Odometry>("/localizability/estimate", 10);

    // Reliable, keep all, and latched. A strip is told of once and used for the rest of the run,
    // so a receiver that joins late, or blinks, must still get every one of them.
    rclcpp::QoS team_qos(rclcpp::KeepAll{});
    team_qos.reliable().transient_local();
    if (share_landmarks_) {
      shared_pub_ = create_publisher<locrec_msgs::msg::SharedLandmark>("/team/landmarks", team_qos);
    }
    if (use_shared_) {
      shared_sub_ = create_subscription<locrec_msgs::msg::SharedLandmark>(
        "/team/landmarks", team_qos,
        [this](locrec_msgs::msg::SharedLandmark::ConstSharedPtr msg) {onShared(*msg);});
    }
    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/points", 10, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {onPoints(*msg);});
    prior_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom_prior", 100, [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {onPrior(*msg);});
    if (use_report_) {
      report_sub_ = create_subscription<locrec_msgs::msg::ScanReport>(
        "/scan_report", 100,
        [this](locrec_msgs::msg::ScanReport::ConstSharedPtr msg) {onReport(*msg);});
    }

    RCLCPP_INFO(
      get_logger(),
      "locrec localizability node up: threshold %.4e, registration range %.2f m, every scan waits "
      "for /odom_prior at its stamp%s%s",
      calibrated.ratio_threshold,
      detector_->odometry().config().registration_range.value_or(0.0),
      use_report_ ? " and for its /scan_report" : "",
      platform == "drone" ? (" , gaze " + detector_->config().gaze).c_str() : "");
  }

private:
  std::optional<double> optionalDouble(const std::string & name) const
  {
    rclcpp::Parameter parameter;
    if (!get_parameter(name, parameter) ||
      parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
    {
      return std::nullopt;
    }
    return parameter.as_double();
  }

  void onPrior(const nav_msgs::msg::Odometry & msg)
  {
    Eigen::Isometry3d pose;
    try {
      pose = odometryToPose(msg);
    } catch (const std::invalid_argument & exc) {
      RCLCPP_WARN(get_logger(), "unusable odometry prior: %s", exc.what());
      return;
    }
    odom_child_frame_ = msg.child_frame_id;
    handle(pairer_->addOdometry(stampToNs(msg.header.stamp), pose));
  }

  void onReport(const locrec_msgs::msg::ScanReport & msg)
  {
    ++n_reports_;
    SidePayload side;
    if (use_markers_) {
      for (const auto & drop : msg.drops) {
        MarkerDrop out;
        out.slot = drop.slot;
        out.offset_sensor =
          Eigen::Vector3d(drop.offset_sensor.x, drop.offset_sensor.y, drop.offset_sensor.z);
        side.drops.push_back(out);
      }
    }
    // A vehicle using a teammate's strips mounts none of its own, so it has no drops to read, and
    // it still has to read the detections: those are the strips it can see. Gating both on
    // use_markers threw away every detection the team drone made (docs/failures.md number 33).
    if (use_markers_ || use_shared_) {
      for (const auto & obs : msg.detections) {
        locrec::MarkerDetection det;
        det.slot = obs.slot;
        det.point_sensor =
          Eigen::Vector3d(obs.point_sensor.x, obs.point_sensor.y, obs.point_sensor.z);
        det.n_beams = obs.n_beams;
        det.range_m = obs.range_m;
        if (std::isfinite(obs.seen_width_m)) {
          det.seen_width_m = obs.seen_width_m;
        }
        if (obs.n_columns != 0) {
          det.n_columns = obs.n_columns;
        }
        side.detections.push_back(det);
      }
    }
    side.track_yaw = msg.track_yaw;
    handle(pairer_->addSide(stampToNs(msg.header.stamp), side));
  }

  void onPoints(const sensor_msgs::msg::PointCloud2 & msg)
  {
    ScanPayload payload;
    payload.header = msg.header;
    try {
      payload.points = pointCloud2ToXyz(msg, static_cast<int>(max_points_));
    } catch (const std::invalid_argument & exc) {
      RCLCPP_WARN(get_logger(), "unusable point cloud: %s", exc.what());
      return;
    }
    handle(pairer_->addScan(stampToNs(msg.header.stamp), payload));
    if (pairer_->nOdometry() == 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "no /odom_prior received yet: scans are held for it, never processed without it");
    }
    if (use_report_ && n_reports_ == 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "no /scan_report has arrived: with use_markers or a gaze policy, scans are held for the "
        "report that carries their stamp");
    }
  }

  /// A strip a teammate mounted. Registered at once, well before it comes into view: the anchor
  /// is a statement about the shared frame, not an observation of this vehicle's, so there is no
  /// pairing with a scan stamp.
  void onShared(const locrec_msgs::msg::SharedLandmark & msg)
  {
    Eigen::Matrix2d cov;
    cov << msg.drop_covariance[0], msg.drop_covariance[1],
      msg.drop_covariance[2], msg.drop_covariance[3];
    bool is_new = false;
    try {
      is_new = detector_->addSharedLandmark(
        msg.slot, Eigen::Vector3d(msg.position.x, msg.position.y, msg.position.z), cov,
        Eigen::Vector2d(msg.normal.x, msg.normal.y));
    } catch (const std::invalid_argument & exc) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000, "unusable shared landmark %d: %s", msg.slot, exc.what());
      return;
    }
    if (is_new) {
      ++n_shared_;
      RCLCPP_INFO(get_logger(), "strip %d from the team, %d held", msg.slot, n_shared_);
    }
  }

  void handle(const std::vector<PriorPairer<ScanPayload, SidePayload>::Paired> & released)
  {
    for (const auto & item : released) {
      const std_msgs::msg::Header & header = item.payload.header;
      if (!warned_frames_ && !odom_child_frame_.empty() && odom_child_frame_ != header.frame_id) {
        warned_frames_ = true;
        RCLCPP_WARN(
          get_logger(),
          "/odom_prior describes '%s' but the cloud is in '%s': the prior is only right if the "
          "two coincide", odom_child_frame_.c_str(), header.frame_id.c_str());
      }
      const SidePayload side = item.side.value_or(SidePayload{});
      std::optional<double> track_yaw;
      if (item.side.has_value() && std::isfinite(side.track_yaw)) {
        track_yaw = side.track_yaw;
      }

      std::optional<DetectorOutput> out;
      try {
        // a marker mounted while this node still had nothing to register against would anchor to
        // a pose the estimator never held, so such a drop cannot happen: the node only recommends
        // one after it has processed a scan
        out = detector_->process(
          item.payload.points, item.pose, side.drops, side.detections, track_yaw);
      } catch (const std::invalid_argument & exc) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5000, "scan not processed: %s", exc.what());
        continue;
      }

      estimate_pub_->publish(buildEstimate(header, detector_->pose()));
      if (shared_pub_) {
        for (const auto & strip : detector_->shared()) {
          shared_pub_->publish(buildShared(header, strip));
        }
      }
      detector_->shared().clear();
      if (!out.has_value()) {
        continue;
      }
      pub_->publish(buildMessage(header, *out));
      std_msgs::msg::String action;
      action.data = out->action;
      action_pub_->publish(action);
    }

    if (pairer_->dropped() > dropped_reported_) {
      dropped_reported_ = pairer_->dropped();
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "%d scans dropped rather than processed on a guess (older than the odometry %d, waited "
        "too long %d, out of order %d, marker report never came %d)",
        pairer_->dropped(), pairer_->droppedStale(), pairer_->droppedOverflow(),
        pairer_->droppedOutOfOrder(), pairer_->droppedNoSide());
    }
  }

  std::unique_ptr<LocalizabilityDetector> detector_;
  std::unique_ptr<PriorPairer<ScanPayload, SidePayload>> pairer_;
  rclcpp::Publisher<locrec_msgs::msg::Localizability>::SharedPtr pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr action_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr estimate_pub_;
  rclcpp::Publisher<locrec_msgs::msg::SharedLandmark>::SharedPtr shared_pub_;
  rclcpp::Subscription<locrec_msgs::msg::SharedLandmark>::SharedPtr shared_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr prior_sub_;
  rclcpp::Subscription<locrec_msgs::msg::ScanReport>::SharedPtr report_sub_;

  int64_t max_points_ = 60000;
  bool use_markers_ = false;
  bool share_landmarks_ = false;
  bool use_shared_ = false;
  bool use_report_ = false;
  std::string odom_child_frame_;
  bool warned_frames_ = false;
  int dropped_reported_ = 0;
  int n_reports_ = 0;
  int n_shared_ = 0;
};

}  // namespace locrec_estimator

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<locrec_estimator::LocalizabilityNode>());
  } catch (const std::exception & exc) {
    RCLCPP_FATAL(rclcpp::get_logger("locrec_localizability"), "%s", exc.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
