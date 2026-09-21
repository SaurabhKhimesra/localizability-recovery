#include <gtest/gtest.h>

#include <cmath>

#include "locrec_core/gaze.hpp"

using namespace locrec;  // NOLINT(build/namespaces)

namespace
{

/// A patch of wall at x = +/- offset, with normals pointing back at the origin.
void addWall(Points & pts, Points & normals, double x, double y_from, double y_to, double nx)
{
  for (double y = y_from; y <= y_to; y += 0.4) {
    for (double z = -0.5; z <= 0.5; z += 0.5) {
      pts.emplace_back(x, y, z);
      normals.emplace_back(nx, 0.0, 0.0);
    }
  }
}

Localizability degenerateAlongY()
{
  Matrix6d H = Matrix6d::Zero();
  H.block<3, 3>(3, 3).diagonal() << 100.0, 0.001, 100.0;  // y is the unobserved direction
  return analyseHessian(H, 500);
}

}  // namespace

TEST(Gaze, forward_is_the_track_heading) {
  ForwardGaze gaze;
  EXPECT_NEAR(gaze(0.7), 0.7, 1e-12);
}

TEST(Gaze, map_normals_are_unit_vectors_on_a_plane) {
  Points pts;
  for (double y = -2.0; y <= 2.0; y += 0.2) {
    for (double z = -1.0; z <= 1.0; z += 0.2) {
      pts.emplace_back(3.0, y, z);
    }
  }
  const Points normals = mapNormals(pts);
  ASSERT_EQ(normals.size(), pts.size());
  for (const auto & n : normals) {
    EXPECT_NEAR(n.norm(), 1.0, 1e-6);
    EXPECT_NEAR(std::abs(n.x()), 1.0, 1e-3) << "the plane's normal is along x";
  }
}

TEST(Gaze, map_normals_are_empty_for_too_few_points) {
  const Points normals = mapNormals({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}});
  ASSERT_EQ(normals.size(), 2u);
  EXPECT_NEAR(normals[0].norm(), 0.0, 1e-12);
}

TEST(Gaze, predicted_information_counts_only_what_the_sensor_would_see) {
  LidarSpec spec = limitedFov();
  Points pts;
  Points normals;
  addWall(pts, normals, 5.0, -1.0, 1.0, -1.0);  // a wall ahead, at +x

  const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d looking_at_it = predictedInformation(pts, normals, origin, 0.0, spec);
  const Eigen::Matrix3d looking_away = predictedInformation(pts, normals, origin, M_PI, spec);

  EXPECT_GT(looking_at_it.trace(), 0.0);
  EXPECT_NEAR(looking_away.trace(), 0.0, 1e-12);
  // the wall constrains the direction of its own normal, which is x
  EXPECT_GT(looking_at_it(0, 0), looking_at_it(1, 1));
}

TEST(Gaze, predicted_information_ignores_what_is_out_of_range) {
  LidarSpec spec = limitedFov();
  spec.max_range = 3.0;
  Points pts;
  Points normals;
  addWall(pts, normals, 5.0, -1.0, 1.0, -1.0);
  const Eigen::Matrix3d info =
    predictedInformation(pts, normals, Eigen::Vector3d::Zero(), 0.0, spec);
  EXPECT_NEAR(info.trace(), 0.0, 1e-12);
}

TEST(Gaze, glance_stays_forward_while_the_geometry_is_good) {
  GlanceGaze gaze(0.01);
  Matrix6d H = Matrix6d::Zero();
  H.block<3, 3>(3, 3).diagonal() << 100.0, 90.0, 100.0;  // well constrained everywhere
  const Localizability loc = analyseHessian(H, 500);

  Points pts;
  Points normals;
  addWall(pts, normals, -6.0, -2.0, 2.0, 1.0);
  for (int i = 0; i < 20; ++i) {
    EXPECT_NEAR(gaze(0.0, Eigen::Isometry3d::Identity(), loc, pts, normals, limitedFov(), 0.5),
      0.0, 1e-12);
  }
  EXPECT_NEAR(gaze.glanceFraction(), 0.0, 1e-12);
}

TEST(Gaze, glance_turns_to_structure_behind_then_returns_to_forward) {
  GlanceGaze gaze(0.01, 60.0, 15.0, 3.0, 5.0);
  const Localizability loc = degenerateAlongY();
  LidarSpec spec = limitedFov();

  // the only structure that would constrain y is behind the drone, off to one side
  Points pts;
  Points normals;
  for (double x = -12.0; x <= -6.0; x += 0.4) {
    for (double z = -0.5; z <= 0.5; z += 0.5) {
      pts.emplace_back(x, 4.0, z);
      normals.emplace_back(0.0, -1.0, 0.0);
    }
  }

  const double dt = 0.5;
  double yaw = gaze(0.0, Eigen::Isometry3d::Identity(), loc, pts, normals, spec, dt);
  EXPECT_GT(std::abs(yaw), 60.0 * M_PI / 180.0) << "a glance leaves the forward cone";

  // it holds for at most hold_s, then goes back to forward
  for (int i = 0; i < 6; ++i) {
    yaw = gaze(0.0, Eigen::Isometry3d::Identity(), loc, pts, normals, spec, dt);
  }
  EXPECT_NEAR(yaw, 0.0, 1e-9);
  EXPECT_GT(gaze.glanceFraction(), 0.0);
  EXPECT_LT(gaze.glanceFraction(), 1.0) << "the sensor is not allowed to stare backwards";
}

TEST(Gaze, glance_does_not_spend_slew_when_there_is_nothing_behind) {
  GlanceGaze gaze(0.01);
  const Localizability loc = degenerateAlongY();
  Points pts;
  Points normals;
  addWall(pts, normals, 6.0, -2.0, 2.0, -1.0);  // everything worth seeing is ahead
  const double yaw =
    gaze(0.0, Eigen::Isometry3d::Identity(), loc, pts, normals, limitedFov(), 0.5);
  EXPECT_NEAR(yaw, 0.0, 1e-12);
}
