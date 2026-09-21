// Sensor geometry. The estimator needs the beam spacing and the range, nothing else: the scan
// itself arrives from the driver, so no ray casting lives on this side.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace locrec
{

struct LidarSpec
{
  int n_azimuth = 180;
  int n_elevation = 16;
  double fov_azimuth_deg = 360.0;
  double fov_elevation_deg = 30.0;  ///< total vertical opening, beams symmetric about the xy plane
  double max_range = 20.0;
  double min_range = 0.4;
  double range_sigma = 0.02;  ///< zero-mean Gaussian range noise, metres, 1 sigma

  /// Beam spacing along the dense (scanning) axis, radians.
  double azimuthStepRad() const
  {
    return (fov_azimuth_deg * M_PI / 180.0) / std::max(n_azimuth, 1);
  }

  double elevationStepRad() const
  {
    if (n_elevation <= 1) {
      return 0.0;
    }
    return (fov_elevation_deg * M_PI / 180.0) / std::max(n_elevation - 1, 1);
  }

  double beamSpacingAt(double distance) const {return distance * azimuthStepRad();}

  /// Distance beyond which the scan under-samples a map of this voxel pitch.
  ///
  /// A voxel map of pitch v can only represent structure the scan samples at v/2 or finer. Past
  /// that range the nearest map point to a far scan point is up to a beam spacing away in the
  /// wrong direction, and registration inherits a systematic pull (docs/failures.md number 2).
  double nyquistRange(double voxel) const
  {
    const double step = azimuthStepRad();
    return step > 0.0 ? 0.5 * voxel / step : std::numeric_limits<double>::infinity();
  }

  bool operator==(const LidarSpec & o) const
  {
    return n_azimuth == o.n_azimuth && n_elevation == o.n_elevation &&
           fov_azimuth_deg == o.fov_azimuth_deg && fov_elevation_deg == o.fov_elevation_deg &&
           max_range == o.max_range && min_range == o.min_range && range_sigma == o.range_sigma;
  }
};

/// VLP-16 class: 16 rings over 30 deg, 1 deg azimuth, short range. The UGV sensor.
inline LidarSpec spinning360()
{
  LidarSpec s;
  s.n_azimuth = 360;
  s.n_elevation = 16;
  s.fov_azimuth_deg = 360.0;
  s.fov_elevation_deg = 30.0;
  s.max_range = 10.0;
  return s;
}

/// Livox-class solid state: 90 x 60 deg at about 0.5 deg, long range. The drone sensor.
inline LidarSpec limitedFov()
{
  LidarSpec s;
  s.n_azimuth = 180;
  s.n_elevation = 112;
  s.fov_azimuth_deg = 90.0;
  s.fov_elevation_deg = 60.0;
  s.max_range = 30.0;
  return s;
}

}  // namespace locrec
