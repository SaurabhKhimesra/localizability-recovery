#include "locrec_estimator/calibration.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace locrec_estimator
{

namespace
{

/// Read a JSON string literal starting at the opening quote. Returns the contents and leaves
/// `i` on the closing quote.
std::string readString(const std::string & s, std::size_t & i)
{
  std::string out;
  ++i;  // opening quote
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) {
      ++i;
    }
    out.push_back(s[i]);
    ++i;
  }
  return out;
}

}  // namespace

std::map<std::string, double> readTopLevelNumbers(const std::string & json)
{
  std::map<std::string, double> out;
  int depth = 0;
  std::string pending_key;
  for (std::size_t i = 0; i < json.size(); ++i) {
    const char c = json[i];
    if (c == '{' || c == '[') {
      ++depth;
      continue;
    }
    if (c == '}' || c == ']') {
      --depth;
      continue;
    }
    if (c == '"') {
      const std::string text = readString(json, i);
      if (depth == 1) {
        pending_key = text;
      }
      continue;
    }
    if (depth == 1 && !pending_key.empty() && (std::isdigit(c) || c == '-' || c == '+')) {
      std::size_t used = 0;
      const double value = std::stod(json.substr(i), &used);
      out[pending_key] = value;
      pending_key.clear();
      i += used - 1;
      continue;
    }
    if (c == ',') {
      pending_key.clear();
    }
  }
  return out;
}

Calibration resolveCalibration(
  const std::string & platform, const std::optional<double> & ratio_threshold,
  const std::optional<double> & marker_reliable_range_m,
  const std::optional<double> & scheduler_margin_m, const std::string & thresholds_file)
{
  if (platform != "ugv" && platform != "drone") {
    throw std::invalid_argument("platform must be ugv or drone, got '" + platform + "'");
  }

  std::map<std::string, double> from_file;
  if (!thresholds_file.empty()) {
    std::ifstream in(thresholds_file);
    if (!in) {
      throw std::invalid_argument("thresholds_file '" + thresholds_file + "' does not exist");
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    from_file = readTopLevelNumbers(buffer.str());
  }

  // the drone recommends a yaw, not a marker, so it needs the threshold alone
  const std::string ratio_key = platform == "drone" ? "drone_ratio_threshold" : "ratio_threshold";

  const auto pick = [&](const std::optional<double> & explicit_value, const std::string & key)
    -> std::optional<double> {
      if (explicit_value.has_value()) {
        return explicit_value;
      }
      const auto it = from_file.find(key);
      return it == from_file.end() ? std::nullopt : std::optional<double>(it->second);
    };

  std::vector<std::string> missing;
  Calibration out;

  const auto ratio = pick(ratio_threshold, ratio_key);
  if (ratio.has_value()) {
    out.ratio_threshold = *ratio;
  } else {
    missing.push_back("ratio_threshold (thresholds.json key '" + ratio_key + "')");
  }

  if (platform == "ugv") {
    out.marker_reliable_range_m = pick(marker_reliable_range_m, "marker_reliable_range_m");
    if (!out.marker_reliable_range_m.has_value()) {
      missing.push_back("marker_reliable_range (thresholds.json key 'marker_reliable_range_m')");
    }
    out.scheduler_margin_m = pick(scheduler_margin_m, "scheduler_margin_m");
    if (!out.scheduler_margin_m.has_value()) {
      missing.push_back("scheduler_margin (thresholds.json key 'scheduler_margin_m')");
    }
  }

  if (!missing.empty()) {
    std::string message = "no value for ";
    for (std::size_t i = 0; i < missing.size(); ++i) {
      message += (i ? ", " : "") + missing[i];
    }
    message +=
      ". These are calibrated, they do not transfer between sensors or spaces, and nothing here "
      "ships a default. Pass thresholds_file:=<path to locrec/results/thresholds.json>, written "
      "by locrec/experiments/calibrate_thresholds_gazebo.py, or give each one as a parameter.";
    throw std::invalid_argument(message);
  }
  return out;
}

}  // namespace locrec_estimator
