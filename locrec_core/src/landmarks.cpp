#include "locrec_core/landmarks.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <random>
#include <tuple>

#include <Eigen/Eigenvalues>

namespace locrec
{

namespace
{

constexpr double kBiasGridM = 0.25;
const double kBiasGridRad = 2.5 * M_PI / 180.0;
constexpr int kBiasSamples = 1024;
constexpr int kBiasMinMatches = 24;

double roundTo(double v, int decimals)
{
  const double f = std::pow(10.0, decimals);
  return std::round(v * f) / f;
}

/// Mean of (fit - centre) in the strip's frame and the spread of the fit, both from one Monte
/// Carlo of the real fit at this geometry.
struct FitMoments
{
  double mean_along = 0.0;  ///< along the strip's width axis
  double mean_out = 0.0;    ///< along the outward normal
  double sd_radial = 0.0;
  double sd_tangential = 0.0;
};

/// Cache key: the grid cell the geometry falls in, plus everything the fit depends on.
using MomentKey = std::tuple<int, int, int, int, double, double, double, double, double, double, int>;

std::map<MomentKey, FitMoments> g_moment_cache;
std::mutex g_moment_mutex;

/// Simulate the fit at a known geometry: cast the azimuth comb at a random phase at the strip's
/// box (front face and the end toward the sensor), keep every ring that crosses it, add the
/// sensor's range noise, and run the real fit on the returns.
///
/// Nothing here models the fit by hand. An analytic model of the returns was right from 4 m out
/// and 0.5 to 0.8 cm wrong at 2 to 3 m, where the fit takes its other branch, so the fit itself
/// is what is run. The generator is seeded by the grid cell, so the same geometry always gives
/// the same correction.
FitMoments simulateFit(
  double distance, double ang, const LidarSpec & lidar, double width, double thickness,
  double height, double dz, std::optional<int> n_columns)
{
  FitMoments out;
  if (distance < 1e-6) {
    return out;
  }
  // seeded by the grid cell, so the same geometry always gets the same correction
  const uint64_t seed = static_cast<uint64_t>(std::llround(distance * 1000.0)) * 1000003ULL ^
    static_cast<uint64_t>(std::llround(ang * 1e6)) * 7919ULL ^ 7ULL;
  std::mt19937_64 rng(seed);

  const double h = 0.5 * width;
  const double t = 0.5 * thickness;
  // strip frame: x along the width axis, y the outward normal. Sensor on the -x, +y side.
  const Eigen::Vector2d sensor(-distance * std::sin(ang), distance * std::cos(ang));
  const double step = lidar.azimuthStepRad();
  if (step <= 0.0) {
    return out;
  }
  const double centre_bearing = std::atan2(-sensor.y(), -sensor.x());
  const double half_span = std::atan2(std::hypot(h, t), distance) + step;

  std::vector<double> el;
  if (lidar.n_elevation > 1) {
    const double fov = lidar.fov_elevation_deg * M_PI / 180.0;
    for (int i = 0; i < lidar.n_elevation; ++i) {
      el.push_back((static_cast<double>(i) / (lidar.n_elevation - 1) - 0.5) * fov);
    }
  } else {
    el.push_back(0.0);
  }

  std::uniform_real_distribution<double> phase_dist(0.0, step);
  std::normal_distribution<double> range_noise(0.0, lidar.range_sigma);

  std::vector<Eigen::Vector2d> errs;
  std::vector<Eigen::Vector2d> every;
  const int n_beams = static_cast<int>(std::ceil(2.0 * half_span / step)) + 1;

  for (int sample = 0; sample < kBiasSamples; ++sample) {
    const double phase = phase_dist(rng);
    const double k0 = std::ceil((centre_bearing - half_span - phase) / step);
    Points pts;
    for (int k = 0; k < n_beams; ++k) {
      const double az = phase + step * (k0 + k);
      const Eigen::Vector2d d(std::cos(az), std::sin(az));
      double best = std::numeric_limits<double>::infinity();
      if (d.y() < -1e-12) {  // the front face, y = +t
        const double r = (t - sensor.y()) / d.y();
        if (r > 0.0 && std::abs(sensor.x() + r * d.x()) <= h) {
          best = r;
        }
      }
      if (d.x() > 1e-12) {  // the end toward the sensor, x = -h
        const double r = (-h - sensor.x()) / d.x();
        if (r > 0.0 && std::abs(sensor.y() + r * d.y()) <= t) {
          best = std::min(best, r);
        }
      }
      if (!std::isfinite(best)) {
        continue;
      }
      for (double e : el) {
        const double z = best * std::tan(e);
        if (z < dz - 0.5 * height || z > dz + 0.5 * height) {
          continue;
        }
        double slant = best / std::cos(e);
        if (slant < lidar.min_range || slant > lidar.max_range) {
          continue;
        }
        slant += range_noise(rng);
        pts.emplace_back(
          slant * std::cos(e) * d.x(), slant * std::cos(e) * d.y(), slant * std::sin(e));
      }
    }
    if (pts.empty()) {
      continue;
    }
    const auto fit = fitStripCentre(pts, lidar, width, thickness);
    // the fit is relative to the sensor; the true centre is at -sensor from it
    const Eigen::Vector2d err = fit.first.head<2>() + sensor;
    every.push_back(err);
    if (n_columns.has_value() && std::min(beamColumns(pts, lidar), 4) != *n_columns) {
      continue;
    }
    errs.push_back(err);
  }

  // this geometry hardly ever gives that many columns, so fall back on the unconditioned mean
  if (static_cast<int>(errs.size()) < kBiasMinMatches) {
    errs = every;
  }
  if (errs.empty()) {
    return out;
  }

  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for (const auto & e : errs) {
    mean += e;
  }
  mean /= static_cast<double>(errs.size());

  // the spread, resolved the way the measurement model wants it: along the line of sight to the
  // strip's centre and across it
  const Eigen::Vector2d e_r = -sensor / sensor.norm();
  const Eigen::Vector2d e_h(-e_r.y(), e_r.x());
  double sum_r = 0.0;
  double sum_h = 0.0;
  double sum_r2 = 0.0;
  double sum_h2 = 0.0;
  for (const auto & e : errs) {
    const double pr = e.dot(e_r);
    const double ph = e.dot(e_h);
    sum_r += pr;
    sum_h += ph;
    sum_r2 += pr * pr;
    sum_h2 += ph * ph;
  }
  const double n = static_cast<double>(errs.size());
  out.mean_along = mean.x();
  out.mean_out = mean.y();
  out.sd_radial = std::sqrt(std::max(sum_r2 / n - (sum_r / n) * (sum_r / n), 0.0));
  out.sd_tangential = std::sqrt(std::max(sum_h2 / n - (sum_h / n) * (sum_h / n), 0.0));
  return out;
}

/// Cached moments on a grid of distance and incidence angle.
FitMoments stripFitMoments(
  double ang, double distance, const LidarSpec & lidar, double marker_width,
  double marker_thickness, double marker_height, double dz, std::optional<int> n_columns)
{
  const int columns = n_columns.has_value() ? std::min(*n_columns, 4) : -1;
  const int d_cell = static_cast<int>(std::lround(distance / kBiasGridM));
  const int a_cell = static_cast<int>(std::lround(std::abs(ang) / kBiasGridRad));
  const MomentKey key{
    d_cell, a_cell, lidar.n_azimuth, lidar.n_elevation, lidar.fov_azimuth_deg,
    lidar.fov_elevation_deg, roundTo(marker_width, 4), roundTo(marker_thickness, 4),
    roundTo(marker_height, 3), roundTo(dz, 1), columns};

  std::lock_guard<std::mutex> lock(g_moment_mutex);
  auto it = g_moment_cache.find(key);
  if (it != g_moment_cache.end()) {
    return it->second;
  }
  const FitMoments moments = simulateFit(
    d_cell * kBiasGridM, a_cell * kBiasGridRad, lidar, marker_width, marker_thickness,
    marker_height, roundTo(dz, 1), n_columns);
  g_moment_cache.emplace(key, moments);
  return moments;
}

}  // namespace

int predictedBeamCount(
  const Eigen::Vector3d & point_sensor, const LidarSpec & lidar, double marker_height,
  double marker_width)
{
  const double r = point_sensor.norm();
  if (r < 1e-6) {
    return 1;
  }
  const double across = marker_width / std::max(r * lidar.azimuthStepRad(), 1e-9);
  const double down = marker_height / std::max(r * lidar.elevationStepRad(), 1e-9);
  return std::max(static_cast<int>(across * down), 1);
}

Eigen::Matrix3d measurementInformation(
  const Eigen::Vector3d & point_sensor, int n_beams, const LidarSpec & lidar,
  const LandmarkSpec & spec, double marker_height, double marker_width,
  std::optional<double> seen_width_m, std::optional<int> n_columns,
  std::optional<std::pair<double, double>> fit_sigma)
{
  const double r = point_sensor.norm();
  const double floor = spec.min_sigma_m * spec.min_sigma_m;
  if (r < 1e-6) {
    return Eigen::Matrix3d::Identity() / floor;
  }
  const double n = std::max(n_beams, 1);

  const Eigen::Vector3d horizontal(point_sensor.x(), point_sensor.y(), 0.0);
  const double nh = horizontal.norm();
  if (nh < 1e-6) {  // straight up or down: the strip says nothing about position
    return Eigen::Matrix3d::Zero();
  }
  const Eigen::Vector3d e_r = horizontal / nh;
  const Eigen::Vector3d e_h = Eigen::Vector3d::UnitZ().cross(e_r);
  const Eigen::Vector3d e_v = Eigen::Vector3d::UnitZ();

  double var_r = lidar.range_sigma * lidar.range_sigma / n;
  double var_h = 0.0;
  double var_v = 0.0;

  if (fit_sigma.has_value() && std::min(fit_sigma->first, fit_sigma->second) > 0.0) {
    var_r = fit_sigma->first * fit_sigma->first;
    var_h = fit_sigma->second * fit_sigma->second;
    Eigen::Matrix3d info = e_r * e_r.transpose() / std::max(var_r, floor) +
      e_h * e_h.transpose() / std::max(var_h, floor);
    if (spec.use_vertical) {
      var_v = std::pow(r * lidar.elevationStepRad(), 2) / 12.0 + marker_height * marker_height / 12.0;
      info += e_v * e_v.transpose() / std::max(var_v, floor);
    }
    return info;
  }

  // How many independent samples each axis actually has. The returns arrive in columns: every
  // ring in a column shares one azimuth, so it is the columns that say where the strip sits
  // across the line of sight and the rings that say where it sits up and down. Averaging the
  // horizontal over all n returns calls a strip crossed by one column of six rings six
  // measurements along the tunnel when it is one (docs/failures.md number 31). Range is
  // different: the rings do give n independent looks at it.
  double columns = n_columns.has_value() && *n_columns > 0 ?
    static_cast<double>(*n_columns) :
    marker_width / std::max(r * lidar.azimuthStepRad(), 1e-9);
  columns = std::min(std::max(columns, 1.0), n);
  const double rings = std::max(n / columns, 1.0);

  // Foreshortening. A strip seen end on hides its own far half, so what comes back is its near
  // edge and the range is short by up to half a width. The geometric fit removes most of it and
  // the rest is paid for here, or the filter treats a systematic offset as independent evidence.
  if (seen_width_m.has_value() && marker_width > 0.0) {
    const double hidden = std::clamp(1.0 - *seen_width_m / marker_width, 0.0, 1.0);
    var_r += std::pow(0.5 * marker_width * hidden, 2);
  }
  if (spec.use_beam_quantisation) {
    var_h += std::pow(r * lidar.azimuthStepRad(), 2) / (12.0 * columns);
    var_v += std::pow(r * lidar.elevationStepRad(), 2) / (12.0 * rings);
  }
  if (spec.use_target_extent) {
    var_h += marker_width * marker_width / (12.0 * columns);
    var_v += marker_height * marker_height / (12.0 * rings);
  }

  Eigen::Matrix3d info = e_r * e_r.transpose() / std::max(var_r, floor) +
    e_h * e_h.transpose() / std::max(var_h, floor);
  if (spec.use_vertical) {
    info += e_v * e_v.transpose() / std::max(var_v, floor);
  }
  return info;
}

int beamColumns(const Points & points_sensor, const LidarSpec & lidar)
{
  if (points_sensor.empty()) {
    return 0;
  }
  const double step = lidar.azimuthStepRad();
  if (step <= 0.0) {
    return 1;
  }
  std::vector<double> az;
  az.reserve(points_sensor.size());
  for (const auto & p : points_sensor) {
    az.push_back(std::atan2(p.y(), p.x()));
  }
  std::sort(az.begin(), az.end());
  // unwrap, so a strip straddling +/- pi is not counted as two clusters
  double offset = 0.0;
  std::vector<long long> cells;
  cells.reserve(az.size());
  for (std::size_t i = 0; i < az.size(); ++i) {
    if (i > 0) {
      const double d = az[i] - az[i - 1];
      if (d > M_PI) {
        offset -= 2.0 * M_PI;
      } else if (d < -M_PI) {
        offset += 2.0 * M_PI;
      }
    }
    cells.push_back(std::llround((az[i] + offset) / step));
  }
  std::sort(cells.begin(), cells.end());
  return static_cast<int>(std::unique(cells.begin(), cells.end()) - cells.begin());
}

std::pair<Eigen::Vector3d, double> fitStripCentre(
  const Points & points_sensor, const LidarSpec & lidar, double marker_width,
  double marker_thickness)
{
  if (points_sensor.empty()) {
    return {Eigen::Vector3d::Zero(), 0.0};
  }
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto & p : points_sensor) {
    centroid += p;
  }
  centroid /= static_cast<double>(points_sensor.size());

  const Eigen::Vector2d c2 = centroid.head<2>();
  const double r = c2.norm();
  if (r < 1e-6 || points_sensor.size() < 2) {
    return {centroid, 0.0};
  }
  const Eigen::Vector2d u = c2 / r;

  // the strip's width axis: the principal direction of the horizontal spread, falling back to
  // the perpendicular of the line of sight when the returns are too few or too collinear with it
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  std::vector<Eigen::Vector2d> spread;
  spread.reserve(points_sensor.size());
  for (const auto & p : points_sensor) {
    const Eigen::Vector2d s = p.head<2>() - c2;
    spread.push_back(s);
    cov += s * s.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
  Eigen::Vector2d axis = es.eigenvectors().col(1);
  const Eigen::Vector2d perp(-u.y(), u.x());
  if (es.eigenvalues()(1) < 1e-8) {
    axis = perp;
  }
  if (axis.dot(perp) < 0.0) {
    axis = -axis;
  }

  std::vector<double> s_along;
  s_along.reserve(spread.size());
  for (const auto & s : spread) {
    s_along.push_back(s.dot(axis));
  }
  const double s_min = *std::min_element(s_along.begin(), s_along.end());
  const double s_max = *std::max_element(s_along.begin(), s_along.end());
  const double beam = std::max(r * lidar.azimuthStepRad(), 1e-9);
  const double extent = (s_max - s_min) + beam;

  double offset = 0.0;
  if (extent >= 0.8 * marker_width) {
    // the whole width is in view, so the midpoint of the extremes is the centre and their
    // quantisation errors cancel
    offset = 0.5 * (s_max + s_min);
  } else {
    // the far edge is not being seen: anchor to the near edge and step half a width across
    std::size_t near = 0;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < points_sensor.size(); ++i) {
      const double d = points_sensor[i].head<2>().norm();
      if (d < best) {
        best = d;
        near = i;
      }
    }
    std::vector<double> sorted = s_along;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t mid = sorted.size() / 2;
    const double median = sorted.size() % 2 == 0 ?
      0.5 * (sorted[mid - 1] + sorted[mid]) : sorted[mid];
    const double s_near = s_along[near];
    const double toward_far = s_near <= median ? 1.0 : -1.0;
    // the near edge sits half a beam spacing outside the nearest return
    offset = s_near - toward_far * 0.5 * beam + toward_far * 0.5 * marker_width;
  }

  const Eigen::Vector2d centre2 = c2 + axis * offset + u * (0.5 * marker_thickness);
  const double seen = std::min(extent, marker_width);
  return {Eigen::Vector3d(centre2.x(), centre2.y(), centroid.z()), seen};
}

Eigen::Vector2d stripFitBias(
  const Eigen::Vector2d & los_in, const Eigen::Vector2d & normal_in, double distance,
  const LidarSpec & lidar, double marker_width, double marker_thickness, double marker_height,
  double dz, std::optional<int> n_columns)
{
  const Eigen::Vector2d los = los_in / std::max(los_in.norm(), 1e-12);
  const Eigen::Vector2d normal = normal_in / std::max(normal_in.norm(), 1e-12);
  const Eigen::Vector2d axis(-normal.y(), normal.x());
  // incidence angle, signed: 0 is face on, +90 degrees is looking along +axis
  const double ang = std::atan2(los.dot(axis), -los.dot(normal));
  if (marker_thickness <= 0.0 || std::abs(ang) >= 0.5 * M_PI) {
    return Eigen::Vector2d::Zero();
  }
  const FitMoments m = stripFitMoments(
    ang, distance, lidar, marker_width, marker_thickness, marker_height, dz, n_columns);
  const double along = ang >= 0.0 ? m.mean_along : -m.mean_along;
  return along * axis + m.mean_out * normal;
}

std::pair<double, double> stripFitSigma(
  const Eigen::Vector2d & los, const Eigen::Vector2d & normal_in, double distance,
  const LidarSpec & lidar, double marker_width, double marker_thickness, double marker_height,
  double dz, std::optional<int> n_columns)
{
  const double n = normal_in.norm();
  if (n < 1e-9 || distance < 1e-6 || marker_width <= 0.0) {
    return {0.0, 0.0};
  }
  const Eigen::Vector2d normal = normal_in / n;
  const Eigen::Vector2d axis(-normal.y(), normal.x());
  const double ang = std::atan2(los.dot(axis), -los.dot(normal));
  const FitMoments m = stripFitMoments(
    ang, distance, lidar, marker_width, marker_thickness, marker_height, dz, n_columns);
  return {m.sd_radial, m.sd_tangential};
}

}  // namespace locrec
