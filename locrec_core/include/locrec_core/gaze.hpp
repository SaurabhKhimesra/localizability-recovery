// Where the drone points its sensor.
//
// The drone's sensor yaw is free of its direction of travel, so it can look for the structure
// that would constrain the direction the tunnel does not. Two policies are kept, because the
// comparison between them is the result: on Gazebo the forward baseline and the glance policy are
// level (1.91 m against 1.89 m final along-track error over 8 seeds), so turning the sensor to
// look for localizability does not pay. See docs/RESEARCH.md.
#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "locrec_core/lidar.hpp"
#include "locrec_core/localizability.hpp"
#include "locrec_core/se3.hpp"

namespace locrec
{

/// Unit normals for the confirmed local map, by local PCA. small_gicp computes covariances on the
/// downsampled source cloud during registration, not on the map, so they are estimated here.
Points mapNormals(const Points & points, int num_neighbors = 20, int num_threads = 1);

/// 3x3 translational information the map would give from this yaw.
///
/// Only the geometry is predicted, not the returns: a map point counts if it is inside the range
/// and inside the horizontal field of view. Grazing incidence is charged for by weighting each
/// patch with the cosine between its normal and the line of sight, because a wall seen edge on
/// returns few beams and constrains little.
Eigen::Matrix3d predictedInformation(
  const Points & points, const Points & normals, const Eigen::Vector3d & sensor_position,
  double yaw, const LidarSpec & spec);

/// E-optimal score of a candidate yaw: the weakest eigenvalue of what the robot would know after
/// looking there, with what it already has normalised alongside it.
double scoreYaw(
  const Points & points, const Points & normals, const Eigen::Vector3d & origin, double yaw,
  const LidarSpec & spec, const std::optional<Eigen::Matrix3d> & have, double scale);

/// Yaw locked to the direction of travel. The do-nothing baseline.
class ForwardGaze
{
public:
  double operator()(double track_yaw) const {return track_yaw;}
  static constexpr bool needs_map = false;
};

/// Look back at the best structure behind, briefly, then return to forward.
///
/// A forward-restricted greedy policy showed that inside a forward cone there is usually nothing
/// better to look at than forward, and an unrestricted one showed that a sensor free to stare
/// backwards will do exactly that and stop mapping the tunnel it is flying into. This is the
/// bounded version: when localizability collapses and there is structure behind worth looking at,
/// turn to it, hold for at most hold_s, then return to forward and do not look again for
/// cooldown_s. Scans keep going into the map throughout.
///
/// Pre-registered: the three durations and the cone were fixed before the grid ran.
class GlanceGaze
{
public:
  explicit GlanceGaze(
    double ratio_threshold, double cone_deg = 60.0, double step_deg = 15.0, double hold_s = 3.0,
    double cooldown_s = 5.0)
  : ratio_threshold_(ratio_threshold), cone_deg_(cone_deg), step_deg_(step_deg),
    hold_s_(hold_s), cooldown_s_(cooldown_s), cooling_(cooldown_s) {}

  /// The yaw to command, given the track heading, the estimate, the live localizability and the
  /// confirmed map with its normals.
  double operator()(
    double track_yaw, const Eigen::Isometry3d & T_est, const Localizability & loc,
    const Points & map_points, const Points & map_normals, const LidarSpec & spec, double dt);

  double glanceFraction() const
  {
    return static_cast<double>(glancing_steps_) / std::max(steps_, 1);
  }
  static constexpr bool needs_map = true;

private:
  std::optional<Eigen::Vector3d> structureBehind(
    const Points & points, const Points & normals, const Eigen::Vector3d & origin,
    double forward, const LidarSpec & spec, const Localizability & loc) const;

  double ratio_threshold_;
  double cone_deg_;
  double step_deg_;
  double hold_s_;
  double cooldown_s_;
  double holding_ = 0.0;
  double cooling_;  ///< starts ready to glance
  std::optional<Eigen::Vector3d> target_point_;
  int steps_ = 0;
  int glancing_steps_ = 0;
};

}  // namespace locrec
