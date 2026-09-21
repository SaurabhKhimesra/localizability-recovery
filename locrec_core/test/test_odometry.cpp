#include <gtest/gtest.h>

#include <cmath>

#include "locrec_core/odometry.hpp"

using namespace locrec;  // NOLINT(build/namespaces)

namespace
{

Eigen::Isometry3d straightStep(double distance)
{
  return makeT(Eigen::Matrix3d::Identity(), Eigen::Vector3d(distance, 0.0, 0.0));
}

OdometryConfig markerConfig()
{
  OdometryConfig cfg;
  cfg.min_scan_points = 60;  // an empty scan skips registration and the prior carries the pose
  return cfg;
}

MarkerDetection detectionFrom(int slot, const Eigen::Vector3d & point_sensor)
{
  MarkerDetection det;
  det.slot = slot;
  det.point_sensor = point_sensor;
  det.n_beams = 12;
  det.range_m = point_sensor.norm();
  det.n_columns = 3;
  return det;
}

}  // namespace

TEST(VoxelGrid, packs_indices_and_separates_neighbouring_voxels) {
  EXPECT_EQ(voxelKey(Eigen::Vector3d(0.05, 0.05, 0.05), 0.2),
    voxelKey(Eigen::Vector3d(0.15, 0.15, 0.15), 0.2));
  EXPECT_NE(voxelKey(Eigen::Vector3d(0.05, 0.0, 0.0), 0.2),
    voxelKey(Eigen::Vector3d(0.35, 0.0, 0.0), 0.2));
}

TEST(VoxelGrid, downsample_gives_one_centroid_per_occupied_voxel) {
  const Points pts = {
    {0.01, 0.0, 0.0}, {0.03, 0.0, 0.0},  // same voxel
    {5.0, 0.0, 0.0}};
  const Points out = voxelDownsample(pts, 0.2);
  ASSERT_EQ(out.size(), 2u);
  // the centroids themselves, not only how many there are: an accumulator that does not start
  // from zero still gives the right count
  double near_x = 1e9;
  for (const auto & p : out) {
    near_x = std::min(near_x, p.x());
  }
  EXPECT_NEAR(near_x, 0.02, 1e-12);
}

TEST(LocalMapTest, a_voxel_holds_the_centroid_of_what_landed_in_it) {
  LocalMap map(30.0, 0.2, 80000, 1);
  map.add({{1.01, 0.05, 0.05}, {1.03, 0.07, 0.05}});
  const Points pts = map.points();
  ASSERT_EQ(pts.size(), 1u);
  EXPECT_NEAR(pts[0].x(), 1.02, 1e-12);
  EXPECT_NEAR(pts[0].y(), 0.06, 1e-12);
}

TEST(LocalMapTest, offers_a_voxel_only_once_two_scans_have_confirmed_it) {
  // At the far end of a long-range narrow-FOV scan a single sweep leaves beams further apart than
  // the voxel pitch, so a map built from one viewpoint is full of one-hit voxels whose nearest
  // neighbour is in the wrong place.
  LocalMap map(30.0, 0.2, 80000, 2);
  Points cloud;
  for (int i = 0; i < 50; ++i) {
    cloud.emplace_back(i * 0.5, 1.5, 0.0);
  }
  map.add(cloud, Eigen::Vector3d::Zero());
  EXPECT_EQ(map.size(), 0u) << "one observation is not a confirmed voxel";
  EXPECT_GT(map.nVoxels(), 0u);

  map.add(cloud, Eigen::Vector3d::Zero());
  EXPECT_GT(map.size(), 0u);
}

TEST(LocalMapTest, crops_by_distance_so_the_scan_never_sticks_out_past_the_map) {
  LocalMap map(5.0, 0.2, 80000, 1);
  Points cloud;
  for (int i = 0; i < 100; ++i) {
    cloud.emplace_back(i * 0.5, 0.0, 0.0);
  }
  map.add(cloud, Eigen::Vector3d::Zero());
  for (const auto & p : map.points()) {
    EXPECT_LE(p.norm(), 5.0 + 1e-9);
  }
}

TEST(LocalMapTest, a_rigid_transform_moves_the_frame_and_never_rebins) {
  LocalMap map(30.0, 0.2, 80000, 1);
  const Points cloud = {{1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
  map.add(cloud);
  const std::size_t before = map.nVoxels();

  map.applyTransform(Eigen::Matrix3d::Identity(), Eigen::Vector3d(10.0, 0.0, 0.0));
  const Points moved = map.points();
  ASSERT_EQ(moved.size(), before);
  EXPECT_EQ(map.nVoxels(), before) << "the grid itself must not be rebinned";
  double min_x = 1e9;
  for (const auto & p : moved) {
    min_x = std::min(min_x, p.x());
  }
  EXPECT_NEAR(min_x, 11.0, 1e-9);
}

TEST(PriorInformation, trusts_the_direction_of_travel_least) {
  // The scale error acts only along the direction of travel, the random walk on all three axes.
  // Treating the prior as isotropic would over-trust the along-track axis, which is the one the
  // tunnel cannot observe, and the whole study turns on telling those apart.
  MotionPriorSpec spec;
  const Matrix6d info = priorInformation(spec, 0.5, straightStep(0.5));
  const Eigen::Matrix3d trans = info.block<3, 3>(3, 3);
  const double along = Eigen::Vector3d::UnitX().transpose() * trans * Eigen::Vector3d::UnitX();
  const double across = Eigen::Vector3d::UnitY().transpose() * trans * Eigen::Vector3d::UnitY();
  EXPECT_LT(along, across);
}

TEST(PriorInformation, is_computed_from_the_increment_the_prior_produced) {
  MotionPriorSpec spec;
  const Matrix6d small_step = priorInformation(spec, 0.5, straightStep(0.1));
  const Matrix6d big_step = priorInformation(spec, 0.5, straightStep(2.0));
  const double along_small =
    Eigen::Vector3d::UnitX().transpose() * small_step.block<3, 3>(3, 3) * Eigen::Vector3d::UnitX();
  const double along_big =
    Eigen::Vector3d::UnitX().transpose() * big_step.block<3, 3>(3, 3) * Eigen::Vector3d::UnitX();
  EXPECT_GT(along_small, along_big) << "a longer step earns less trust along track";
}

TEST(OdometryTest, follows_the_prior_when_there_is_nothing_to_register_against) {
  Odometry odom(Eigen::Isometry3d::Identity(), markerConfig());
  for (int i = 0; i < 4; ++i) {
    const OdomStep step = odom.step({}, straightStep(0.5));
    EXPECT_FALSE(step.registered);
    EXPECT_FALSE(step.localizability.has_value());
  }
  EXPECT_NEAR(odom.pose().translation().x(), 2.0, 1e-9);
}

TEST(OdometryTest, a_marker_inherits_the_pose_the_estimate_held_when_it_went_on_the_wall) {
  Odometry odom(Eigen::Isometry3d::Identity(), markerConfig(), MotionPriorSpec(), 0.5,
    spinning360());
  odom.step({}, straightStep(3.0));
  odom.registerLandmark(7, Eigen::Vector3d(0.0, 1.5, 0.0));

  const Landmark * lm = odom.landmarks().get(7);
  ASSERT_NE(lm, nullptr);
  EXPECT_NEAR(lm->position.x(), 3.0, 1e-9);
  EXPECT_NEAR(lm->position.y(), 1.5, 1e-9);
  ASSERT_TRUE(lm->R_drop.has_value());
  EXPECT_GT(lm->R_drop->trace(), 0.0) << "the drop is an observation like any other";
  EXPECT_FALSE(lm->foreign);
}

TEST(OdometryTest, an_observed_marker_pulls_the_estimate_back_toward_its_anchor) {
  Odometry odom(Eigen::Isometry3d::Identity(), markerConfig(), MotionPriorSpec(), 0.5,
    spinning360());
  odom.registerLandmark(1, Eigen::Vector3d(0.0, 1.5, 0.0));  // anchor at the world origin, abeam

  // the prior says one metre; the robot really moved half of it
  odom.step({}, straightStep(1.0));
  ASSERT_NEAR(odom.pose().translation().x(), 1.0, 1e-9);

  // seen from where the robot really is, at x = 0.5, the strip is 0.5 m behind
  const OdomStep step = odom.step(
    {}, straightStep(0.0), {detectionFrom(1, Eigen::Vector3d(-0.5, 1.5, 0.0))});

  EXPECT_EQ(step.landmarks_used, 1);
  EXPECT_EQ(odom.nFixes(), 1);
  EXPECT_LT(odom.pose().translation().x(), 1.0) << "the fix should pull the estimate back";
  EXPECT_GT(odom.pose().translation().x(), 0.4) << "and not past the anchor's own accuracy";
  ASSERT_EQ(odom.fixLog().size(), 1u);
  EXPECT_EQ(odom.fixLog().front().n_terms, 1);
  EXPECT_FALSE(odom.fixLog().front().used_yaw) << "one anchor gives position, not heading";
}

TEST(OdometryTest, two_anchors_in_view_correct_heading_as_well_as_position) {
  Odometry odom(Eigen::Isometry3d::Identity(), markerConfig(), MotionPriorSpec(), 0.5,
    spinning360());
  odom.registerLandmark(1, Eigen::Vector3d(0.0, 1.5, 0.0));
  odom.step({}, straightStep(4.0));
  odom.registerLandmark(2, Eigen::Vector3d(0.0, 1.5, 0.0));
  odom.step({}, straightStep(1.0));

  const OdomStep step = odom.step(
    {}, straightStep(0.0),
    {detectionFrom(1, Eigen::Vector3d(-5.2, 1.5, 0.0)),
      detectionFrom(2, Eigen::Vector3d(-1.2, 1.5, 0.0))});

  EXPECT_EQ(step.landmarks_used, 2);
  EXPECT_EQ(step.yaw_fixed, 1);
  ASSERT_FALSE(odom.fixLog().empty());
  EXPECT_TRUE(odom.fixLog().back().used_yaw);
}

TEST(OdometryTest, the_map_moves_with_an_absolute_fix) {
  // The registration constraint is relative, so shifting pose and map together leaves it
  // satisfied; shifting the pose alone means the next registration drags it straight back, which
  // is why markers changed nothing before this existed.
  OdometryConfig cfg = markerConfig();
  // high enough that registration is always skipped here: this test is about the fix moving the
  // map, not about what GICP does with a wall
  cfg.min_scan_points = 100000;
  Odometry odom(Eigen::Isometry3d::Identity(), cfg, MotionPriorSpec(), 0.5, spinning360());
  odom.registerLandmark(1, Eigen::Vector3d(0.0, 1.5, 0.0));
  Points wall;
  for (int i = 0; i < 80; ++i) {
    wall.emplace_back(i * 0.25, 1.5, 0.0);
  }
  // twice, because a voxel is only confirmed once two scans have landed in it
  odom.step(wall, straightStep(0.5));
  odom.step(wall, straightStep(0.5));
  const Points before = odom.map().points();
  ASSERT_FALSE(before.empty());

  odom.step({}, straightStep(0.0), {detectionFrom(1, Eigen::Vector3d(-0.5, 1.5, 0.0))});
  const Points after = odom.map().points();
  ASSERT_EQ(after.size(), before.size());
  double moved = 0.0;
  for (std::size_t i = 0; i < after.size(); ++i) {
    moved = std::max(moved, (after[i] - before[i]).norm());
  }
  EXPECT_GT(moved, 1e-6) << "the map has to move rigidly with the pose";
}

TEST(OdometryTest, a_teammates_strip_is_taken_on_trust_and_never_re_surveyed) {
  OdometryConfig cfg = markerConfig();
  cfg.resurvey_anchors = true;  // even with re-survey on, a foreign anchor is left alone
  Odometry odom(Eigen::Isometry3d::Identity(), cfg, MotionPriorSpec(), 0.5, spinning360());

  Eigen::Matrix2d R_drop = Eigen::Matrix2d::Identity() * 1e-4;
  odom.registerForeignLandmark(5, Eigen::Vector3d(10.0, 1.5, 0.0), R_drop,
    Eigen::Vector2d(0.0, -1.0));
  const Landmark * lm = odom.landmarks().get(5);
  ASSERT_NE(lm, nullptr);
  EXPECT_TRUE(lm->foreign);
  EXPECT_FALSE(lm->T_drop.has_value()) << "there is no drop pose of this robot's";

  odom.step({}, straightStep(9.0));
  odom.step({}, straightStep(0.0), {detectionFrom(5, Eigen::Vector3d(1.2, 1.5, 0.0))});
  EXPECT_NEAR(odom.landmarks().get(5)->position.x(), 10.0, 1e-12)
    << "its coordinates are the shared frame's definition";
}

TEST(OdometryTest, a_foreign_landmark_needs_a_facing) {
  Odometry odom(Eigen::Isometry3d::Identity(), markerConfig(), MotionPriorSpec(), 0.5,
    spinning360());
  EXPECT_THROW(
    odom.registerForeignLandmark(
      1, Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero()),
    std::invalid_argument);
}

TEST(OdometryTest, the_scale_state_moves_toward_the_prior_bias_and_not_past_it) {
  // The prior overstates every step by 2 percent. A marker fix measures exactly that: the
  // along-track residual over a leg of length L is an observation of (s - 1) * L. The filter
  // recovers part of it and leaves the rest, systematically (docs/failures.md number 35).
  Odometry odom(Eigen::Isometry3d::Identity(), markerConfig(), MotionPriorSpec(), 0.5,
    spinning360());
  odom.registerLandmark(1, Eigen::Vector3d(0.0, 1.5, 0.0));
  for (int i = 0; i < 10; ++i) {
    odom.step({}, straightStep(1.02));
  }
  ASSERT_NEAR(odom.pose().translation().x(), 10.2, 1e-6);
  EXPECT_NEAR(odom.scale(), 1.0, 1e-12) << "nothing has measured the scale yet";

  // the robot is really at 10 m, so the strip it left at the origin is 10 m behind
  odom.step({}, straightStep(0.0), {detectionFrom(1, Eigen::Vector3d(-10.0, 1.5, 0.0))});

  EXPECT_GT(odom.scale(), 1.0) << "the residual says the prior is long";
  EXPECT_LT(odom.scale(), 1.02) << "one fix does not recover all of it";
  EXPECT_LT(odom.pose().translation().x(), 10.2);
}

TEST(OdometryTest, the_estimate_starts_where_it_is_told_to) {
  const Eigen::Isometry3d start = makeT(rotz(0.5), Eigen::Vector3d(3.0, -2.0, 1.0));
  Odometry odom(start, markerConfig());
  EXPECT_NEAR((odom.pose().translation() - start.translation()).norm(), 0.0, 1e-12);
  EXPECT_EQ(odom.nSteps(), 0);
  EXPECT_NEAR(odom.travelled(), 0.0, 1e-12);
}
