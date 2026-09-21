// Minimal SE(3) helpers. Right-handed, column-vector convention.
//
// A pose T maps sensor coordinates into the world (map) frame: p_world = T * p_sensor.
#pragma once

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace locrec
{

using Points = std::vector<Eigen::Vector3d>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

inline Eigen::Matrix3d rotz(double yaw)
{
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return S;
}

inline Eigen::Isometry3d makeT(const Eigen::Matrix3d & R, const Eigen::Vector3d & t)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = R;
  T.translation() = t;
  return T;
}

/// Rotation vector of R, numerically safe near 0 and pi.
inline Eigen::Vector3d so3Log(const Eigen::Matrix3d & R)
{
  const double cos_theta = std::clamp((R.trace() - 1.0) * 0.5, -1.0, 1.0);
  const double theta = std::acos(cos_theta);
  const Eigen::Vector3d asym(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  if (theta < 1e-8) {
    return asym * 0.5;
  }
  if (M_PI - theta < 1e-6) {
    // near pi the antisymmetric part vanishes, so the axis comes from the symmetric part
    const Eigen::Matrix3d A = (R + Eigen::Matrix3d::Identity()) * 0.5;
    Eigen::Vector3d axis = A.diagonal().cwiseMax(0.0).cwiseSqrt();
    int k = 0;
    axis.maxCoeff(&k);
    if (axis(k) > 0.0) {
      axis = A.col(k) / axis(k);
    }
    axis /= (axis.norm() + 1e-12);
    return axis * theta;
  }
  return (theta / (2.0 * std::sin(theta))) * asym;
}

inline Eigen::Matrix3d so3Exp(const Eigen::Vector3d & w)
{
  const double theta = w.norm();
  if (theta < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  const Eigen::Matrix3d K = skew(w / theta);
  return Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * (K * K);
}

inline Points transformPoints(const Eigen::Isometry3d & T, const Points & pts)
{
  Points out;
  out.reserve(pts.size());
  for (const auto & p : pts) {
    out.push_back(T * p);
  }
  return out;
}

/// (translation error in m, rotation error in rad) of an estimate against the truth.
inline std::pair<double, double> poseError(
  const Eigen::Isometry3d & T_est, const Eigen::Isometry3d & T_true)
{
  const Eigen::Isometry3d dT = T_true.inverse() * T_est;
  return {dT.translation().norm(), so3Log(dT.linear()).norm()};
}

/// Symmetrise and clip negative eigenvalues to zero.
///
/// Differences of covariances appear all over the relative formulation, and a difference of two
/// covariances is only positive semidefinite in exact arithmetic. Clipping is the honest repair:
/// it never removes uncertainty the arithmetic actually supports.
template<typename Derived>
inline typename Derived::PlainObject psd(const Eigen::MatrixBase<Derived> & M_in)
{
  typename Derived::PlainObject M = 0.5 * (M_in + M_in.transpose());
  Eigen::SelfAdjointEigenSolver<typename Derived::PlainObject> es(M);
  if ((es.eigenvalues().array() >= 0.0).all()) {
    return M;
  }
  const auto w = es.eigenvalues().cwiseMax(0.0);
  return es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
}

}  // namespace locrec
