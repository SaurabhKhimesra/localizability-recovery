#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "locrec_estimator/detector.hpp"

using locrec_estimator::DetectorConfig;
using locrec_estimator::LocalizabilityDetector;
using locrec_estimator::defaultRegistrationRange;

namespace
{

/// A straight tunnel in the sensor frame, seen from a sensor at the origin looking along +x.
locrec::Points tunnelScan(double reach = 9.0, double spacing = 0.12)
{
  locrec::Points pts;
  for (double x = -reach; x <= reach; x += spacing) {
    for (double z = -0.6; z <= 0.6; z += 0.3) {
      pts.emplace_back(x, -1.6, z);
      pts.emplace_back(x, 1.6, z);
    }
    pts.emplace_back(x, 0.0, -0.7);
  }
  return pts;
}

DetectorConfig ugvConfig()
{
  DetectorConfig config;
  config.ratio_threshold = 0.0021087425685472777;  // the calibrated Gazebo value
  config.marker_reliable_range_m = 7.0;
  config.scheduler_margin_m = 1.5;
  config.platform = "ugv";
  config.max_range = 10.0;
  config.registration_threads = 1;
  return config;
}

DetectorConfig droneConfig(const std::string & gaze)
{
  DetectorConfig config;
  config.ratio_threshold = 0.00303997335026656;
  config.platform = "drone";
  config.gaze = gaze;
  config.max_range = 30.0;
  config.fov_azimuth_deg = 90.0;
  config.azimuth_beams = 180;
  config.elevation_beams = 112;
  config.fov_elevation_deg = 60.0;
  config.registration_threads = 1;
  config.start_at_odometry = gaze != "across";
  return config;
}

Eigen::Isometry3d at(double x)
{
  return locrec::makeT(Eigen::Matrix3d::Identity(), Eigen::Vector3d(x, 0.0, 0.0));
}

}  // namespace

TEST(RegistrationRange, takes_the_smaller_of_the_geometric_and_sampling_limits) {
  // The sampling limit is where the scan's azimuth spacing exceeds half a map voxel. For the
  // ground robot's 1 degree scanner and a 0.2 m map that is 5.73 m, well inside the geometric
  // 10 - 1 - 0.5 = 8.5 m, and registering out to 8.5 m produced a ratio the calibrated threshold
  // was never meant for (docs/failures.md number 2).
  locrec::OdometryConfig cfg;
  const double range = defaultRegistrationRange(locrec::spinning360(), cfg, 0.5);
  EXPECT_NEAR(range, 5.73, 0.01);
  EXPECT_LT(range, 8.5);
}

TEST(Detector, a_ground_robot_needs_its_scheduler_numbers) {
  DetectorConfig config = ugvConfig();
  config.marker_reliable_range_m.reset();
  EXPECT_THROW(LocalizabilityDetector{config}, std::invalid_argument);
}

TEST(Detector, an_unknown_platform_or_gaze_is_rejected) {
  DetectorConfig config = ugvConfig();
  config.platform = "submarine";
  EXPECT_THROW(LocalizabilityDetector{config}, std::invalid_argument);

  DetectorConfig drone = droneConfig("sideways");
  EXPECT_THROW(LocalizabilityDetector{drone}, std::invalid_argument);
}

TEST(Detector, a_steering_gaze_needs_the_estimate_to_start_in_the_odometry_frame) {
  DetectorConfig config = droneConfig("forward");
  config.start_at_odometry = false;
  EXPECT_THROW(LocalizabilityDetector{config}, std::invalid_argument);
}

TEST(Detector, says_nothing_until_it_can_register) {
  LocalizabilityDetector detector{ugvConfig()};
  // the first scan has nothing to register against, so there is no Hessian and no signal
  EXPECT_FALSE(detector.process(tunnelScan(), at(0.0)).has_value());
}

TEST(Detector, calls_a_straight_tunnel_degenerate_and_asks_for_a_marker) {
  LocalizabilityDetector detector{ugvConfig()};
  std::optional<locrec_estimator::DetectorOutput> out;
  std::vector<std::string> actions;
  for (int i = 0; i < 4; ++i) {
    out = detector.process(tunnelScan(), at(i * 0.5));
    if (out.has_value()) {
      actions.push_back(out->action);
    }
  }
  ASSERT_TRUE(out.has_value());
  ASSERT_FALSE(actions.empty());
  EXPECT_GT(out->n_points, 0);
  EXPECT_TRUE(out->is_degenerate) << "ratio was " << out->ratio;
  // the falling edge: the marker goes on at the first degenerate scan, and the ones after it are
  // spaced by the duty cycle rather than paid for per scan
  EXPECT_EQ(actions.front(), "drop_marker");
  for (std::size_t i = 1; i < actions.size(); ++i) {
    EXPECT_EQ(actions[i], "none") << "within 12.5 m of the last anchor";
  }
  // the weak direction is the tunnel's own axis
  EXPECT_NEAR(std::abs(out->weak_direction.x()), 1.0, 0.1);
}

TEST(Detector, follows_the_odometry_prior_between_scans) {
  LocalizabilityDetector detector{ugvConfig()};
  for (int i = 0; i < 4; ++i) {
    detector.process(tunnelScan(), at(i * 0.5));
  }
  // the tunnel gives nothing along track, so the estimate rides on the prior, which moved 1.5 m
  EXPECT_NEAR(detector.pose().translation().x(), 1.5, 0.3);
  EXPECT_GT(detector.estimatedDistance(), 1.0);
}

TEST(Detector, a_mounted_marker_is_registered_and_offered_to_the_team) {
  DetectorConfig config = ugvConfig();
  config.share_landmarks = true;
  LocalizabilityDetector detector{config};

  locrec_estimator::MarkerDrop drop;
  drop.slot = 4;
  drop.offset_sensor = Eigen::Vector3d(0.0, 1.6, 0.0);
  detector.process(tunnelScan(), at(0.0), {drop});

  ASSERT_EQ(detector.shared().size(), 1u);
  const auto & strip = detector.shared().front();
  EXPECT_EQ(strip.slot, 4);
  EXPECT_NEAR(strip.position.y(), 1.6, 1e-9);
  EXPECT_GT(strip.drop_covariance.trace(), 0.0);
  // the face points back at the pose that mounted it
  EXPECT_NEAR(strip.normal.y(), -1.0, 1e-9);
}

TEST(Detector, a_vehicle_that_mounts_nothing_takes_its_teammates_strips_on_trust) {
  DetectorConfig config = droneConfig("across");
  config.use_shared_landmarks = true;
  LocalizabilityDetector detector{config};

  Eigen::Matrix2d cov = Eigen::Matrix2d::Identity() * 1e-4;
  EXPECT_TRUE(
    detector.addSharedLandmark(2, Eigen::Vector3d(5.0, 1.6, 0.0), cov, Eigen::Vector2d(0.0, -1.0)));
  // a teammate republishes a strip when its own estimate of it moves: the later answer replaces
  // the position, and registering it again would hand this vehicle a fix it has not earned
  EXPECT_FALSE(
    detector.addSharedLandmark(2, Eigen::Vector3d(5.1, 1.6, 0.0), cov, Eigen::Vector2d(0.0, -1.0)));
  ASSERT_NE(detector.odometry().landmarks().get(2), nullptr);
  EXPECT_NEAR(detector.odometry().landmarks().get(2)->position.x(), 5.1, 1e-9);
}

TEST(Detector, shared_landmarks_are_ignored_unless_the_vehicle_was_told_to_use_them) {
  LocalizabilityDetector detector{ugvConfig()};
  EXPECT_FALSE(
    detector.addSharedLandmark(
      1, Eigen::Vector3d(5.0, 1.6, 0.0), Eigen::Matrix2d::Identity(), Eigen::Vector2d(0.0, -1.0)));
  EXPECT_EQ(detector.odometry().landmarks().size(), 0u);
}

TEST(Detector, the_drone_recommends_a_yaw_rather_than_a_marker) {
  LocalizabilityDetector detector{droneConfig("across")};
  std::optional<locrec_estimator::DetectorOutput> out;
  for (int i = 0; i < 4; ++i) {
    out = detector.process(tunnelScan(25.0, 0.25), at(i * 0.5));
  }
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->action.rfind("yaw_to:", 0), 0u) << "action was " << out->action;
}

TEST(Detector, the_forward_gaze_commands_the_track_heading) {
  LocalizabilityDetector detector{droneConfig("forward")};
  EXPECT_TRUE(detector.needsTrackYaw());
  std::optional<locrec_estimator::DetectorOutput> out;
  for (int i = 0; i < 4; ++i) {
    out = detector.process(tunnelScan(25.0, 0.25), at(i * 0.5), {}, {}, 0.25);
  }
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->action, "yaw_to:14.323945") << "0.25 rad, to the digits the vehicle is sent";
}

TEST(Detector, a_steering_gaze_refuses_to_run_without_a_track_heading) {
  LocalizabilityDetector detector{droneConfig("forward")};
  EXPECT_THROW(detector.process(tunnelScan(), at(0.0)), std::invalid_argument);
}
