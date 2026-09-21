#include <gtest/gtest.h>

#include "locrec_core/se3.hpp"

using namespace locrec;  // NOLINT(build/namespaces)

TEST(Se3, exp_and_log_are_inverses) {
  for (const Eigen::Vector3d & w :
    {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.1, -0.2, 0.3),
      Eigen::Vector3d(0.0, 0.0, 1.5)})
  {
    const Eigen::Vector3d back = so3Log(so3Exp(w));
    EXPECT_NEAR((back - w).norm(), 0.0, 1e-12) << "w = " << w.transpose();
  }
}

TEST(Se3, log_is_stable_near_pi) {
  // the antisymmetric part vanishes at pi, so the axis has to come from the symmetric part
  const Eigen::Vector3d w = Eigen::Vector3d(0.0, 0.0, 1.0) * (M_PI - 1e-9);
  const Eigen::Vector3d back = so3Log(so3Exp(w));
  EXPECT_NEAR(std::abs(back.z()), M_PI, 1e-6);
  EXPECT_NEAR(back.head<2>().norm(), 0.0, 1e-6);
}

TEST(Se3, rotz_turns_x_toward_y) {
  const Eigen::Vector3d turned = rotz(M_PI / 2.0) * Eigen::Vector3d::UnitX();
  EXPECT_NEAR((turned - Eigen::Vector3d::UnitY()).norm(), 0.0, 1e-12);
}

TEST(Se3, pose_error_is_zero_for_the_same_pose) {
  const Eigen::Isometry3d T = makeT(rotz(0.3), Eigen::Vector3d(1.0, 2.0, 3.0));
  const auto err = poseError(T, T);
  EXPECT_NEAR(err.first, 0.0, 1e-12);
  EXPECT_NEAR(err.second, 0.0, 1e-12);
}

TEST(Se3, pose_error_reports_translation_and_rotation) {
  const Eigen::Isometry3d truth = makeT(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  const Eigen::Isometry3d est = makeT(rotz(0.05), Eigen::Vector3d(0.3, 0.4, 0.0));
  const auto err = poseError(est, truth);
  EXPECT_NEAR(err.first, 0.5, 1e-12);
  EXPECT_NEAR(err.second, 0.05, 1e-12);
}

TEST(Se3, psd_clips_negative_eigenvalues_and_keeps_the_rest) {
  Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
  M.diagonal() << 4.0, -1e-9, 1.0;
  const Eigen::Matrix3d clipped = psd(M);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(clipped);
  EXPECT_GE(es.eigenvalues().minCoeff(), 0.0);
  EXPECT_NEAR(es.eigenvalues().maxCoeff(), 4.0, 1e-12);
}

TEST(Se3, transform_points_moves_into_the_world_frame) {
  const Eigen::Isometry3d T = makeT(rotz(M_PI / 2.0), Eigen::Vector3d(1.0, 0.0, 0.0));
  const Points out = transformPoints(T, {Eigen::Vector3d(1.0, 0.0, 0.0)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR((out[0] - Eigen::Vector3d(1.0, 1.0, 0.0)).norm(), 0.0, 1e-12);
}
