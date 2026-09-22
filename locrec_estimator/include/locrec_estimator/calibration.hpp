// Where the calibrated numbers come from, with no ROS include.
//
// There are three: the ratio threshold, and for the ground robot the two numbers the marker
// scheduler spaces by. None of them has a default anywhere in this package, because a default is
// a second source and the second source is the one that goes stale (docs/failures.md numbers 20
// and 23). They come from the file locrec/experiments/calibrate_thresholds_gazebo.py writes, or
// from a parameter given explicitly, which wins so a threshold can still be varied for an
// experiment without editing the file.
#pragma once

#include <map>
#include <optional>
#include <string>

namespace locrec_estimator
{

struct Calibration
{
  double ratio_threshold = 0.0;
  std::optional<double> marker_reliable_range_m;
  std::optional<double> scheduler_margin_m;
};

/// Top-level numeric entries of a JSON object. Only the scalars at depth one are read: the
/// thresholds file also carries nested capture metadata, which nothing here needs.
std::map<std::string, double> readTopLevelNumbers(const std::string & json);

/// Every calibrated value this platform needs, or a std::invalid_argument saying what is missing.
/// An explicit value wins over the file.
Calibration resolveCalibration(
  const std::string & platform, const std::optional<double> & ratio_threshold,
  const std::optional<double> & marker_reliable_range_m,
  const std::optional<double> & scheduler_margin_m, const std::string & thresholds_file);

}  // namespace locrec_estimator
