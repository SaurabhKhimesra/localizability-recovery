#include "locrec_core/localizability.hpp"

#include <Eigen/Eigenvalues>

namespace locrec
{

Localizability analyseHessian(const Matrix6d & H, int n_points)
{
  const Eigen::Matrix3d Ht = 0.5 * (H.block<3, 3>(3, 3) + H.block<3, 3>(3, 3).transpose());
  const Eigen::Matrix3d Hr = 0.5 * (H.block<3, 3>(0, 0) + H.block<3, 3>(0, 0).transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> trans(Ht);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> rot(Hr);

  Localizability out;
  out.eigenvalues = trans.eigenvalues().cwiseMax(0.0);
  out.eigenvectors = trans.eigenvectors();
  out.rot_eigenvalues = rot.eigenvalues().cwiseMax(0.0);

  const double lam_max = out.eigenvalues(2);
  out.ratio = lam_max > 0.0 ? out.eigenvalues(0) / lam_max : 0.0;
  const double rot_max = out.rot_eigenvalues(2);
  out.rot_ratio = rot_max > 0.0 ? out.rot_eigenvalues(0) / rot_max : 0.0;
  out.lambda_min_per_point = n_points > 0 ? out.eigenvalues(0) / n_points : 0.0;
  out.n_points = n_points;

  Eigen::Vector3d weak = out.eigenvectors.col(0);
  // the sign is arbitrary; fix it so traces are readable
  int k = 0;
  weak.cwiseAbs().maxCoeff(&k);
  if (weak(k) < 0.0) {
    weak = -weak;
  }
  out.weak_direction = weak;
  return out;
}

}  // namespace locrec
