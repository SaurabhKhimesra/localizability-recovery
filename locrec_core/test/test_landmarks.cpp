#include <gtest/gtest.h>

#include <cmath>

#include "locrec_core/landmarks.hpp"

using namespace locrec;  // NOLINT(build/namespaces)

namespace
{

/// Returns on a strip of the given width, abeam of the sensor at range r, one column per beam.
Points stripReturns(double r, double width, const LidarSpec & lidar, int n_columns)
{
  Points pts;
  const double step = lidar.azimuthStepRad();
  for (int c = 0; c < n_columns; ++c) {
    const double az = M_PI / 2.0 + (c - 0.5 * (n_columns - 1)) * step;
    for (int ring = -2; ring <= 2; ++ring) {
      const double el = ring * lidar.elevationStepRad();
      pts.emplace_back(
        r * std::cos(el) * std::cos(az), r * std::cos(el) * std::sin(az), r * std::sin(el));
    }
  }
  (void)width;
  return pts;
}

}  // namespace

TEST(Landmarks, beam_columns_counts_distinct_azimuths_not_returns) {
  const LidarSpec lidar = spinning360();
  const Points pts = stripReturns(4.0, 0.15, lidar, 3);
  ASSERT_EQ(pts.size(), 15u);  // three columns of five rings
  EXPECT_EQ(beamColumns(pts, lidar), 3);
}

TEST(Landmarks, beam_columns_is_zero_without_returns) {
  EXPECT_EQ(beamColumns({}, spinning360()), 0);
}

TEST(Landmarks, the_measurement_says_nothing_about_height) {
  // The returns land wherever the elevation channels cross the strip, so the vertical component
  // of their centroid slides with the geometry. Believing it took drift from about 1 m to 40 m.
  const Eigen::Matrix3d info = measurementInformation(
    Eigen::Vector3d(0.0, 4.0, 0.3), 12, spinning360(), LandmarkSpec(), 1.0, 0.15);
  EXPECT_NEAR((info * Eigen::Vector3d::UnitZ()).norm(), 0.0, 1e-12);
  EXPECT_GT(info.trace(), 0.0);
}

TEST(Landmarks, one_column_is_worth_less_along_the_tunnel_than_four) {
  // Every ring in a column shares one azimuth, so it is the columns that say where the strip sits
  // across the line of sight. Averaging over all returns called a strip crossed by one column of
  // six rings six measurements along the tunnel when it is one (docs/failures.md number 31).
  const LidarSpec lidar = spinning360();
  const Eigen::Vector3d p(0.0, 4.4, 0.0);  // abeam, so the tunnel axis is the x direction
  const Eigen::Matrix3d one = measurementInformation(
    p, 6, lidar, LandmarkSpec(), 1.0, 0.15, std::nullopt, 1);
  const Eigen::Matrix3d four = measurementInformation(
    p, 6, lidar, LandmarkSpec(), 1.0, 0.15, std::nullopt, 4);
  const double along_one = Eigen::Vector3d::UnitX().transpose() * one * Eigen::Vector3d::UnitX();
  const double along_four = Eigen::Vector3d::UnitX().transpose() * four * Eigen::Vector3d::UnitX();
  EXPECT_LT(along_one, along_four);
}

TEST(Landmarks, a_foreshortened_view_is_trusted_less_in_range) {
  const LidarSpec lidar = spinning360();
  const Eigen::Vector3d p(0.0, 4.0, 0.0);
  const Eigen::Matrix3d full = measurementInformation(
    p, 8, lidar, LandmarkSpec(), 1.0, 0.15, 0.15, 3);
  const Eigen::Matrix3d partial = measurementInformation(
    p, 8, lidar, LandmarkSpec(), 1.0, 0.15, 0.05, 3);
  // range is the y axis here, since the strip is abeam
  const double radial_full = Eigen::Vector3d::UnitY().transpose() * full * Eigen::Vector3d::UnitY();
  const double radial_partial =
    Eigen::Vector3d::UnitY().transpose() * partial * Eigen::Vector3d::UnitY();
  EXPECT_LT(radial_partial, radial_full);
}

TEST(Landmarks, a_measured_fit_spread_replaces_the_modelled_terms) {
  const LidarSpec lidar = spinning360();
  const Eigen::Vector3d p(0.0, 2.0, 0.0);
  const Eigen::Matrix3d modelled = measurementInformation(
    p, 8, lidar, LandmarkSpec(), 1.0, 0.15, std::nullopt, 4);
  const Eigen::Matrix3d measured = measurementInformation(
    p, 8, lidar, LandmarkSpec(), 1.0, 0.15, std::nullopt, 4, std::make_pair(0.02, 0.0063));
  // 0.63 cm of measured spread against the 2.2 cm the width-over-root-twelve term charged
  const double along_modelled =
    Eigen::Vector3d::UnitX().transpose() * modelled * Eigen::Vector3d::UnitX();
  const double along_measured =
    Eigen::Vector3d::UnitX().transpose() * measured * Eigen::Vector3d::UnitX();
  EXPECT_GT(along_measured, along_modelled);
}

TEST(Landmarks, predicted_beam_count_is_at_least_one_and_grows_as_the_strip_nears) {
  const LidarSpec lidar = spinning360();
  const int near = predictedBeamCount(Eigen::Vector3d(0.0, 2.0, 0.0), lidar, 1.0, 0.15);
  const int far = predictedBeamCount(Eigen::Vector3d(0.0, 9.0, 0.0), lidar, 1.0, 0.15);
  EXPECT_GE(far, 1);
  EXPECT_GT(near, far);
}

TEST(Landmarks, the_fit_recovers_the_centre_of_a_fully_seen_strip) {
  const LidarSpec lidar = spinning360();
  // a strip abeam at 4 m, its width across the line of sight, seen by five columns
  Points pts;
  for (int i = -2; i <= 2; ++i) {
    const double x = i * 0.03;  // across the line of sight, within the 0.15 m width
    for (int ring = -1; ring <= 1; ++ring) {
      pts.emplace_back(x, 4.0, ring * 0.1);
    }
  }
  const auto fit = fitStripCentre(pts, lidar, 0.15, 0.0);
  EXPECT_NEAR(fit.first.x(), 0.0, 0.02);
  EXPECT_NEAR(fit.first.y(), 4.0, 0.02);
  EXPECT_GT(fit.second, 0.1);  // most of the width was in view
}

TEST(Landmarks, the_fit_steps_across_a_strip_whose_far_half_is_hidden) {
  // At grazing incidence the strip occludes its own far half, so the returns cluster on the near
  // edge and the centroid sits up to half a width short. The fit anchors to the near edge and
  // steps half a width toward the far one.
  const LidarSpec lidar = spinning360();
  Points pts;
  for (int i = 0; i < 3; ++i) {
    pts.emplace_back(i * 0.005, 4.0, 0.0);
  }
  const auto fit = fitStripCentre(pts, lidar, 0.15, 0.0);
  const double centroid_x = 0.005;
  EXPECT_GT(std::abs(fit.first.x() - centroid_x), 0.02) << "the fit should not be the centroid";
  EXPECT_LT(fit.second, 0.15);  // and it should report that it did not see the whole width
}

TEST(Landmarks, the_fit_spread_is_measured_and_finite) {
  // strip_fit_sigma runs the real fit at this geometry: it already contains the range noise, the
  // beam comb and the strip's shape, so it replaces the modelled terms rather than joining them.
  const LidarSpec lidar = spinning360();
  const auto sigma = stripFitSigma(
    Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(0.0, -1.0), 4.0, lidar, 0.15, 0.02, 1.0, 0.5, 1);
  EXPECT_GT(sigma.first, 0.0);
  EXPECT_GT(sigma.second, 0.0);
  EXPECT_LT(sigma.first, 0.5);
  EXPECT_LT(sigma.second, 0.5);
}

TEST(Landmarks, there_is_no_bias_to_correct_for_a_strip_with_no_thickness) {
  const LidarSpec lidar = spinning360();
  const Eigen::Vector2d bias = stripFitBias(
    Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(0.0, -1.0), 4.0, lidar, 0.15, 0.0, 1.0, 0.5,
    std::nullopt);
  EXPECT_NEAR(bias.norm(), 0.0, 1e-12);
}

TEST(LandmarkBook, keeps_one_record_per_slot) {
  LandmarkBook book;
  Landmark lm;
  lm.slot = 3;
  lm.position = Eigen::Vector3d(1.0, 2.0, 0.0);
  book.registerLandmark(lm);
  ASSERT_NE(book.get(3), nullptr);
  EXPECT_EQ(book.get(9), nullptr);
  EXPECT_EQ(book.size(), 1u);
  EXPECT_NEAR(book.get(3)->position.x(), 1.0, 1e-12);
}
