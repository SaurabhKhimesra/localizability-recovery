// Localizability from the registration Hessian.
//
// small_gicp returns the 6x6 Gauss-Newton information matrix H of the final linearisation,
// ordered [rotation(0:3), translation(3:6)] and expressed in the target frame (for us: the local
// map frame, the estimator's world). That ordering is pinned by a unit test that builds a
// corridor cloud and asserts the weak direction lands in H.block<3,3>(3,3) along the corridor.
//
// The eigenvalues of the translational block scale with the number of correspondences and with
// the noise weighting, so two quantities are reported:
//
//   ratio                 lambda_min / lambda_max, scale free, the primary signal
//   lambda_min_per_point  absolute, a guard against the degenerate-but-tiny-scan case where the
//                         ratio looks fine only because every direction is badly observed
//
// No threshold is baked in. Thresholds are a declared study parameter, calibrated on seeds
// disjoint from the evaluation seeds (locrec/experiments/calibrate_thresholds_gazebo.py).
#pragma once

#include <Eigen/Core>

#include "locrec_core/se3.hpp"

namespace locrec
{

struct LocalizabilityConfig
{
  /// Declare degenerate when lambda_min/lambda_max falls below this. No default: a default here
  /// would be a second source for a calibrated number (docs/failures.md number 20).
  double ratio_threshold = 0.0;
  /// Declare degenerate when the absolute weakest direction is below this even if the ratio is fine.
  double min_lambda_per_point = 0.0;
};

struct Localizability
{
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();   ///< translational, ascending
  Eigen::Matrix3d eigenvectors = Eigen::Matrix3d::Identity();  ///< columns match eigenvalues
  double ratio = 0.0;
  double lambda_min_per_point = 0.0;
  Eigen::Vector3d weak_direction = Eigen::Vector3d::UnitX();  ///< least observable direction
  Eigen::Vector3d rot_eigenvalues = Eigen::Vector3d::Zero();
  double rot_ratio = 0.0;
  int n_points = 0;

  bool isDegenerate(const LocalizabilityConfig & cfg) const
  {
    return ratio < cfg.ratio_threshold || lambda_min_per_point < cfg.min_lambda_per_point;
  }

  /// Re-express the directions in another frame. R_target_source maps a vector in the new frame
  /// into the frame H was given in, so directions transform by its transpose.
  Localizability inFrame(const Eigen::Matrix3d & R_target_source) const
  {
    Localizability out = *this;
    out.eigenvectors = R_target_source.transpose() * eigenvectors;
    out.weak_direction = R_target_source.transpose() * weak_direction;
    return out;
  }
};

/// Eigen-analysis of the translational (and, secondarily, rotational) block.
Localizability analyseHessian(const Matrix6d & H, int n_points);

}  // namespace locrec
