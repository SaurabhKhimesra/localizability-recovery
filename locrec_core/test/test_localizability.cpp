#include <gtest/gtest.h>

#include <cmath>

#include "locrec_core/localizability.hpp"
#include "locrec_core/registration.hpp"

using namespace locrec;  // NOLINT(build/namespaces)

namespace
{

/// A straight corridor: two walls and a floor, featureless along x.
Points corridor(double length = 20.0, double spacing = 0.1)
{
  Points pts;
  for (double x = -length; x <= length; x += spacing) {
    for (double z = -1.0; z <= 1.0; z += spacing * 2) {
      pts.emplace_back(x, -1.5, z);
      pts.emplace_back(x, 1.5, z);
    }
    for (double y = -1.5; y <= 1.5; y += spacing * 2) {
      pts.emplace_back(x, y, -1.0);
    }
  }
  return pts;
}

}  // namespace

TEST(Localizability, reports_eigenvalues_ascending_with_the_weak_direction_first) {
  Matrix6d H = Matrix6d::Zero();
  H.block<3, 3>(3, 3).diagonal() << 1.0, 10.0, 100.0;
  H.block<3, 3>(0, 0).diagonal() << 2.0, 4.0, 8.0;

  const Localizability loc = analyseHessian(H, 500);
  EXPECT_NEAR(loc.eigenvalues(0), 1.0, 1e-12);
  EXPECT_NEAR(loc.eigenvalues(2), 100.0, 1e-12);
  EXPECT_NEAR(loc.ratio, 0.01, 1e-12);
  EXPECT_NEAR(loc.lambda_min_per_point, 1.0 / 500.0, 1e-12);
  EXPECT_NEAR(std::abs(loc.weak_direction.x()), 1.0, 1e-12);
  EXPECT_NEAR(loc.rot_ratio, 0.25, 1e-12);
  EXPECT_EQ(loc.n_points, 500);
}

TEST(Localizability, the_weak_direction_sign_is_fixed_so_traces_are_readable) {
  Matrix6d H = Matrix6d::Zero();
  H.block<3, 3>(3, 3).diagonal() << 1.0, 10.0, 100.0;
  const Localizability loc = analyseHessian(H, 10);
  int k = 0;
  loc.weak_direction.cwiseAbs().maxCoeff(&k);
  EXPECT_GT(loc.weak_direction(k), 0.0);
}

TEST(Localizability, in_frame_re_expresses_the_directions) {
  Matrix6d H = Matrix6d::Zero();
  H.block<3, 3>(3, 3).diagonal() << 1.0, 10.0, 100.0;
  const Localizability loc = analyseHessian(H, 10);
  // a frame turned by 90 degrees about z sees the weak direction along its own -y
  const Localizability turned = loc.inFrame(rotz(M_PI / 2.0));
  EXPECT_NEAR(std::abs(turned.weak_direction.y()), 1.0, 1e-9);
}

TEST(Localizability, is_degenerate_uses_both_the_ratio_and_the_absolute_guard) {
  Matrix6d H = Matrix6d::Zero();
  H.block<3, 3>(3, 3).diagonal() << 1.0, 10.0, 100.0;
  const Localizability loc = analyseHessian(H, 500);

  LocalizabilityConfig cfg;
  cfg.ratio_threshold = 0.1;
  EXPECT_TRUE(loc.isDegenerate(cfg));  // ratio 0.01 is below 0.1

  cfg.ratio_threshold = 0.001;
  EXPECT_FALSE(loc.isDegenerate(cfg));

  // the ratio looks fine only because every direction is badly observed
  cfg.min_lambda_per_point = 1.0;
  EXPECT_TRUE(loc.isDegenerate(cfg));
}

// This is the test that pins small_gicp's Hessian layout, which its API does not document:
// [rotation(0:3), translation(3:6)] in the target frame. Everything downstream, from the
// detector's threshold to the guard in OdometryConfig, reads the translational block.
TEST(Localizability, a_corridor_is_degenerate_along_its_own_axis) {
  const Points cloud = corridor();
  ASSERT_GT(cloud.size(), 1000u);

  RegistrationConfig cfg;
  cfg.num_threads = 1;
  const RegistrationOutput res = align(cloud, cloud, cfg);
  ASSERT_GT(res.num_inliers, 100);

  const Localizability loc = analyseHessian(res.H, res.num_inliers);
  // the corridor runs along x, so that is the direction registration cannot constrain
  EXPECT_NEAR(std::abs(loc.weak_direction.x()), 1.0, 0.05);
  EXPECT_LT(loc.ratio, 0.05) << "a featureless corridor should not look well constrained";
}
