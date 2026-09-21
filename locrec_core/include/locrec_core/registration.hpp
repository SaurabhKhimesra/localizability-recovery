// Scan-to-map registration, and the 6x6 Hessian that is the localizability signal.
//
// The work is small_gicp's, the same library the offline study registers with; this is the thin
// wrapper that keeps its settings in one place. num_threads is not only a speed setting: over a
// 300 m run the result depends on it, deterministically (docs/failures.md number 30).
#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "locrec_core/se3.hpp"

namespace locrec
{

struct RegistrationConfig
{
  double downsampling_resolution = 0.2;
  double max_correspondence_distance = 1.0;
  int max_iterations = 30;
  int num_threads = 4;
  std::string type = "GICP";  ///< GICP, PLANE_ICP or ICP
};

struct RegistrationOutput
{
  Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();
  Matrix6d H = Matrix6d::Zero();  ///< [rotation(0:3), translation(3:6)], in the target frame
  bool converged = false;
  int num_inliers = 0;
  double error = 0.0;
};

RegistrationOutput align(
  const Points & target, const Points & source, const RegistrationConfig & cfg);

}  // namespace locrec
