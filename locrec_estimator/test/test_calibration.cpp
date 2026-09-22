#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "locrec_estimator/calibration.hpp"

using locrec_estimator::readTopLevelNumbers;
using locrec_estimator::resolveCalibration;

namespace
{

// the shape of locrec/results/thresholds_gazebo.json: scalars at the top, capture metadata nested
const char * kThresholds = R"({
  "sensor": "gazebo gpu_lidar",
  "oversample": {"ugv": 6, "drone": 3},
  "capture": {"ugv": {"poses": 401, "beams": 5760}},
  "calibration_seeds": [100, 101, 102],
  "ratio_threshold": 0.0021087425685472777,
  "marker_reliable_range_m": 7.0,
  "scheduler_margin_m": 1.5,
  "drone_ratio_threshold": 0.00303997335026656
})";

std::string writeTemp(const std::string & contents)
{
  const std::string path = std::string(std::tmpnam(nullptr)) + ".json";
  std::ofstream out(path);
  out << contents;
  return path;
}

}  // namespace

TEST(Calibration, reads_top_level_scalars_and_ignores_nested_ones) {
  const auto values = readTopLevelNumbers(kThresholds);
  EXPECT_NEAR(values.at("ratio_threshold"), 0.0021087425685472777, 1e-18);
  EXPECT_NEAR(values.at("marker_reliable_range_m"), 7.0, 1e-12);
  EXPECT_NEAR(values.at("drone_ratio_threshold"), 0.00303997335026656, 1e-18);
  EXPECT_EQ(values.count("ugv"), 0u) << "that one is nested under oversample";
  EXPECT_EQ(values.count("poses"), 0u);
  EXPECT_EQ(values.count("sensor"), 0u) << "strings are not calibration values";
}

TEST(Calibration, the_ugv_needs_the_threshold_and_both_spacing_numbers) {
  const std::string path = writeTemp(kThresholds);
  const auto out = resolveCalibration("ugv", std::nullopt, std::nullopt, std::nullopt, path);
  EXPECT_NEAR(out.ratio_threshold, 0.0021087425685472777, 1e-18);
  ASSERT_TRUE(out.marker_reliable_range_m.has_value());
  EXPECT_NEAR(*out.marker_reliable_range_m, 7.0, 1e-12);
  ASSERT_TRUE(out.scheduler_margin_m.has_value());
  EXPECT_NEAR(*out.scheduler_margin_m, 1.5, 1e-12);
  std::remove(path.c_str());
}

TEST(Calibration, the_drone_recommends_a_yaw_so_it_needs_the_threshold_alone) {
  const std::string path = writeTemp(kThresholds);
  const auto out = resolveCalibration("drone", std::nullopt, std::nullopt, std::nullopt, path);
  EXPECT_NEAR(out.ratio_threshold, 0.00303997335026656, 1e-18)
    << "the drone's threshold is its own, not the ground robot's";
  EXPECT_FALSE(out.marker_reliable_range_m.has_value());
  std::remove(path.c_str());
}

TEST(Calibration, an_explicit_parameter_wins_over_the_file) {
  const std::string path = writeTemp(kThresholds);
  const auto out = resolveCalibration("ugv", 0.5, 3.0, 0.25, path);
  EXPECT_NEAR(out.ratio_threshold, 0.5, 1e-12);
  EXPECT_NEAR(*out.marker_reliable_range_m, 3.0, 1e-12);
  EXPECT_NEAR(*out.scheduler_margin_m, 0.25, 1e-12);
  std::remove(path.c_str());
}

TEST(Calibration, nothing_ships_a_default) {
  // a default would be a second source for a number that has one, and the second source is the
  // one that goes stale (docs/failures.md numbers 20 and 23)
  EXPECT_THROW(
    resolveCalibration("ugv", std::nullopt, std::nullopt, std::nullopt, ""), std::invalid_argument);
  try {
    resolveCalibration("ugv", std::nullopt, std::nullopt, std::nullopt, "");
  } catch (const std::invalid_argument & exc) {
    const std::string message = exc.what();
    EXPECT_NE(message.find("ratio_threshold"), std::string::npos);
    EXPECT_NE(message.find("thresholds_file"), std::string::npos) << "say how to supply it";
  }
}

TEST(Calibration, a_missing_file_is_an_error_rather_than_a_silent_default) {
  EXPECT_THROW(
    resolveCalibration("ugv", std::nullopt, std::nullopt, std::nullopt, "/no/such/file.json"),
    std::invalid_argument);
}

TEST(Calibration, an_unknown_platform_is_rejected) {
  EXPECT_THROW(
    resolveCalibration("submarine", 0.1, 1.0, 1.0, ""), std::invalid_argument);
}
