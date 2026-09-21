// Retroreflective markers as landmarks.
//
// A marker is a vertical strip bolted to the tunnel wall. The robot mounts it while it still
// knows where it is, registers its position in the estimator's own frame at that moment, and
// then, for as long as the strip stays in view, every scan that picks it up ties the pose back to
// that registered position. A marker exports the localizability the robot had at drop time into
// the blind stretch ahead.
//
// Nothing here removes accumulated drift: a marker anchors the pose to wherever the robot
// thought it was when the marker went down, so drift is bounded by the error at drop time plus
// whatever accrues after the marker leaves view. It is not a loop closure.
#pragma once

#include <map>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "locrec_core/lidar.hpp"
#include "locrec_core/se3.hpp"

namespace locrec
{

/// Noise model for one marker observation. The three error sources are measured, not guessed:
/// range noise from the LiDAR spec, beam quantisation (the centroid can sit anywhere inside the
/// beam spacing), and target extent (the centroid of a handful of returns on a strip is not the
/// strip's centre).
struct LandmarkSpec
{
  bool use_beam_quantisation = true;
  bool use_target_extent = true;
  /// Whether the measurement constrains the vertical at all. It should not: the returns land
  /// wherever the elevation channels happen to cross the strip, so the vertical component of
  /// their centroid slides with the geometry. That is a bias, not zero-mean noise, and believing
  /// it took drift from about 1 m to about 40 m. The strip is a horizontal-plane landmark.
  bool use_vertical = false;
  /// Floor on any axis, so a lucky geometry cannot claim more than a centimetre.
  double min_sigma_m = 0.01;
};

/// One marker seen in one scan, as the detector reports it.
struct MarkerDetection
{
  int slot = 0;
  Eigen::Vector3d point_sensor = Eigen::Vector3d::Zero();
  int n_beams = 1;
  double range_m = 0.0;
  /// How much of the strip's width the returns actually covered, when the detector fitted it.
  std::optional<double> seen_width_m;
  /// Distinct beam azimuths among the returns. One column means the strip's position between two
  /// beams was never observed (docs/failures.md number 31).
  std::optional<int> n_columns;
};

struct Landmark
{
  int slot = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  int registered_at_step = 0;
  /// Absolute pose covariance when the marker was registered. Bookkeeping only: it says how far
  /// from the start of the run the whole chain might be, and weighting a fix with it makes every
  /// marker deep in a tunnel look worthless.
  std::optional<Matrix6d> covariance;
  int n_observations = 0;

  // ---- the relative record, which is what a fix is actually weighted by ----
  std::optional<Eigen::Isometry3d> T_drop;      ///< estimated sensor pose when the strip went on
  std::optional<Eigen::Vector3d> offset_drop;   ///< marker position in that drop frame
  std::optional<Eigen::Matrix2d> R_drop;        ///< covariance of the drop-time observation
  std::optional<Matrix6d> q_ref;                ///< accumulated prior covariance at the last reset
  std::optional<Eigen::Matrix3d> rel;           ///< pose covariance relative to this anchor, (yaw,x,y)
  /// Horizontal outward normal of the strip's face. Implied by the drop pose for a marker this
  /// robot mounted; carried explicitly for a teammate's, because the fit's bias is mirrored
  /// between a strip approached from ahead and one left behind.
  std::optional<Eigen::Vector2d> normal;
  /// True when another robot mounted this one and this robot uses it on trust.
  bool foreign = false;
};

class LandmarkBook
{
public:
  Landmark & registerLandmark(const Landmark & lm)
  {
    by_slot_[lm.slot] = lm;
    return by_slot_[lm.slot];
  }
  Landmark * get(int slot)
  {
    auto it = by_slot_.find(slot);
    return it == by_slot_.end() ? nullptr : &it->second;
  }
  const Landmark * get(int slot) const
  {
    auto it = by_slot_.find(slot);
    return it == by_slot_.end() ? nullptr : &it->second;
  }
  std::size_t size() const {return by_slot_.size();}
  std::vector<int> slots() const
  {
    std::vector<int> out;
    out.reserve(by_slot_.size());
    for (const auto & kv : by_slot_) {
      out.push_back(kv.first);
    }
    return out;
  }

private:
  std::map<int, Landmark> by_slot_;
};

/// How many returns a strip at this offset should produce. Needed the moment a marker is bolted
/// on, where there is no observation to count beams from but the drop-time covariance still has
/// to be recorded.
int predictedBeamCount(
  const Eigen::Vector3d & point_sensor, const LidarSpec & lidar, double marker_height,
  double marker_width);

/// 3x3 information of one marker observation, in the sensor frame.
///
/// The covariance is diagonal in a horizontal ray frame (radial, horizontal tangential, vertical)
/// and then rotated into the sensor frame. The horizontal tangential axis is the one that
/// matters: a strip on the side wall sits roughly abeam, so that direction is the tunnel axis,
/// which is exactly what the odometry cannot observe on its own.
///
/// The basis is built in the horizontal plane, not in the ray frame, so world z is an exact null
/// direction: with a basis tilted by the ray's elevation, a 0.4 m height difference between the
/// registered point and the beam centroid leaks 7 cm of fictitious along-track error.
///
/// n_columns is how many distinct azimuths the returns came from, which is what the horizontal
/// axis averages over. fit_sigma is the measured spread of the strip fit at this geometry
/// (stripFitSigma): it is the whole error of the fit, so it replaces the modelled terms.
Eigen::Matrix3d measurementInformation(
  const Eigen::Vector3d & point_sensor, int n_beams, const LidarSpec & lidar,
  const LandmarkSpec & spec, double marker_height, double marker_width,
  std::optional<double> seen_width_m = std::nullopt,
  std::optional<int> n_columns = std::nullopt,
  std::optional<std::pair<double, double>> fit_sigma = std::nullopt);

/// Distinct beam azimuths among a strip's returns.
int beamColumns(const Points & points_sensor, const LidarSpec & lidar);

/// Estimate the mounting point of a strip from the returns that landed on it, and how much of
/// the strip's width was seen. At grazing incidence a strip occludes its own far half, so the
/// centroid sits up to half a width short; and the returns land on the face, which stands proud
/// of the wall by half the thickness.
std::pair<Eigen::Vector3d, double> fitStripCentre(
  const Points & points_sensor, const LidarSpec & lidar, double marker_width,
  double marker_thickness = 0.0);

/// Expected horizontal error of fitStripCentre (fit minus true centre) for a strip whose facing
/// is known: a 2-vector in the frame los and normal are given in. Zero unless the strip has a
/// thickness. See docs/failures.md number 31, and note the default is off in OdometryConfig
/// because the bias is sensor-specific (number 34).
Eigen::Vector2d stripFitBias(
  const Eigen::Vector2d & los, const Eigen::Vector2d & normal, double distance,
  const LidarSpec & lidar, double marker_width, double marker_thickness, double marker_height,
  double dz, std::optional<int> n_columns = std::nullopt);

/// Spread of the strip fit for this view, along the line of sight and across it, metres. Same
/// Monte Carlo as stripFitBias: it already contains the range noise, the beam comb and the
/// strip's geometry, so it replaces the modelled terms rather than adding to them.
std::pair<double, double> stripFitSigma(
  const Eigen::Vector2d & los, const Eigen::Vector2d & normal, double distance,
  const LidarSpec & lidar, double marker_width, double marker_thickness, double marker_height,
  double dz, std::optional<int> n_columns = std::nullopt);

}  // namespace locrec
