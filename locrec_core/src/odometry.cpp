#include "locrec_core/odometry.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <Eigen/Eigenvalues>

namespace locrec
{

namespace
{

/// Indices of (yaw, x, y) in the [rot, trans] twist ordering. The whole absolute stage lives in
/// this subspace: horizontal, and rotation only about the vertical.
constexpr int kState[3] = {2, 3, 4};

Eigen::Matrix3d subState(const Matrix6d & M)
{
  Eigen::Matrix3d out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = M(kState[i], kState[j]);
    }
  }
  return out;
}

void setSubState(Matrix6d & M, const Eigen::Matrix3d & block)
{
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      M(kState[i], kState[j]) = block(i, j);
    }
  }
}

/// Rows of the 2x3 Jacobian of the predicted marker position in (yaw, x, y).
///
/// Linearised about the current sensor position, a yaw increment moves a world point by
/// dpsi * z_hat x (q - t), so the yaw column is the in-plane perpendicular of the lever from
/// sensor to anchor. Anchors close to the sensor contribute almost nothing to yaw, which is
/// correct: heading needs a baseline. With use_yaw false the column is zero, which is how a
/// single anchor is kept away from attitude.
Eigen::Matrix<double, 2, 3> jacobian(
  const Eigen::Vector3d & q, const Eigen::Vector3d & sensor_t, bool use_yaw)
{
  Eigen::Matrix<double, 2, 3> J = Eigen::Matrix<double, 2, 3>::Zero();
  if (use_yaw) {
    const Eigen::Vector2d lever = q.head<2>() - sensor_t.head<2>();
    J(0, 0) = -lever.y();
    J(1, 0) = lever.x();
  }
  J(0, 1) = 1.0;
  J(1, 2) = 1.0;
  return J;
}

/// 6x6 map from a world-origin twist to one about the point t.
///
/// small_gicp linearises with a left perturbation, so its Hessian is written in coordinates where
/// a rotation increment turns the world about the origin. A hundred and sixty metres down a
/// tunnel that makes the rotation block enormous and couples it hard into translation, which is
/// fine for GICP alone but wrong to combine with a prior written about the sensor.
Matrix6d shiftTwistFrame(const Eigen::Vector3d & t)
{
  Matrix6d M = Matrix6d::Identity();
  M.block<3, 3>(3, 0) = -skew(t);
  return M;
}

/// Lift P until it dominates floor in the Loewner order, projecting both onto the floor's
/// eigenbasis. Only deficient directions are lifted, so P is never made larger than it has to be.
Eigen::Matrix2d loewnerFloor(const Eigen::Matrix2d & P_in, const Eigen::Matrix2d & floor_in)
{
  const Eigen::Matrix2d P = 0.5 * (P_in + P_in.transpose());
  const Eigen::Matrix2d F = 0.5 * (floor_in + floor_in.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(F);
  const Eigen::Matrix2d V = es.eigenvectors();
  Eigen::Vector2d deficit;
  for (int i = 0; i < 2; ++i) {
    const double d_post = V.col(i).transpose() * P * V.col(i);
    deficit(i) = std::max(0.0, es.eigenvalues()(i) - d_post);
  }
  if ((deficit.array() <= 0.0).all()) {
    return P;
  }
  return P + V * deficit.asDiagonal() * V.transpose();
}

Vector6d solveOrLstsq(const Matrix6d & A, const Vector6d & rhs)
{
  const Vector6d x = A.ldlt().solve(rhs);
  if (x.allFinite()) {
    return x;
  }
  return A.completeOrthogonalDecomposition().solve(rhs);
}

Eigen::Vector3d solveOrLstsq3(const Eigen::Matrix3d & A, const Eigen::Vector3d & rhs)
{
  const Eigen::Vector3d x = A.ldlt().solve(rhs);
  if (x.allFinite()) {
    return x;
  }
  return A.completeOrthogonalDecomposition().solve(rhs);
}

/// Combine the registration and the prior into one correction.
///
/// The prior's mean correction is zero by construction (the scan was already placed at the
/// predicted pose), so the MAP correction solves (H_cal + H_prior) x = H_cal x_gicp, which passes
/// well-observed directions through untouched and collapses unobserved ones onto the prior.
///
/// This is a blend applied after an unconstrained GICP has converged, not a factor inside its
/// Gauss-Newton loop: small_gicp does not expose its iteration. The consequence is one-sided,
/// since the unconstrained solve can wander further before the blend reins it in, so the baseline
/// is if anything understated.
///
/// Returns the corrected transform and the calibrated, guarded registration information in the
/// sensor-centred frame.
std::pair<Eigen::Isometry3d, Matrix6d> fuse(
  const Eigen::Isometry3d & T_corr, const Matrix6d & H_in, const Matrix6d & prior_info,
  const Eigen::Vector3d & sensor_position, int n_inliers, double residual, bool calibrate,
  std::optional<double> gicp_ratio_floor)
{
  Vector6d x_world;
  x_world.head<3>() = so3Log(T_corr.linear());
  x_world.tail<3>() = T_corr.translation();

  Matrix6d H = 0.5 * (H_in + H_in.transpose());
  if (calibrate) {
    const double dof = std::max(n_inliers - 6, 1);
    const double scale = (std::isfinite(residual) && residual > 1e-12) ? dof / residual : 1.0;
    H *= scale;
  }

  if (gicp_ratio_floor.has_value()) {
    const Eigen::Matrix3d Ht = 0.5 * (H.block<3, 3>(3, 3) + H.block<3, 3>(3, 3).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Ht);
    Eigen::Vector3d ev = es.eigenvalues().cwiseMax(0.0);
    const double cut = *gicp_ratio_floor * std::max(ev(2), 1e-30);
    for (int i = 0; i < 3; ++i) {
      if (ev(i) < cut) {
        ev(i) = 0.0;
      }
    }
    H.block<3, 3>(3, 3) = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    H.block<3, 3>(0, 3).setZero();
    H.block<3, 3>(3, 0).setZero();
  }

  const Matrix6d M = shiftTwistFrame(sensor_position);
  Matrix6d Minv = Matrix6d::Identity();
  Minv.block<3, 3>(3, 0) = -M.block<3, 3>(3, 0);

  const Matrix6d H_s = Minv.transpose() * H * Minv;
  const Vector6d x_s = M * x_world;

  const Matrix6d A = H_s + prior_info;
  const Vector6d rhs = H_s * x_s;
  const Vector6d x_fused = Minv * solveOrLstsq(A, rhs);

  return {makeT(so3Exp(x_fused.head<3>()), x_fused.tail<3>()), H_s};
}

/// Zero the translational correction along directions the Hessian says are weak: the classical
/// solution-remapping guard (Zhang, Kaess and Singh 2016). Off by default; the soft fusion
/// supersedes it.
std::pair<Eigen::Isometry3d, int> remapCorrection(
  const Eigen::Isometry3d & T_corr, const Localizability & loc, double eigenvalue_floor)
{
  Eigen::Matrix3d keep = Eigen::Matrix3d::Zero();
  int weak = 0;
  for (int i = 0; i < 3; ++i) {
    if (loc.eigenvalues(i) < eigenvalue_floor) {
      ++weak;
    } else {
      keep += loc.eigenvectors.col(i) * loc.eigenvectors.col(i).transpose();
    }
  }
  if (weak == 0) {
    return {T_corr, 0};
  }
  Eigen::Isometry3d out = T_corr;
  out.translation() = keep * T_corr.translation();
  return {out, weak};
}

/// Running sum of the points that landed in one voxel. Eigen does not zero a vector it default
/// constructs, so a std::pair<Eigen::Vector3d, int64_t> made by operator[] starts from whatever
/// was in memory; this one starts from zero.
struct CentroidSum
{
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  int64_t count = 0;

  void add(const Eigen::Vector3d & p)
  {
    sum += p;
    ++count;
  }
  Eigen::Vector3d centroid() const {return sum / static_cast<double>(count);}
};

constexpr int kVoxelBits = 21;
constexpr int64_t kVoxelOffset = int64_t{1} << (kVoxelBits - 1);

}  // namespace

Matrix6d priorInformation(
  const MotionPriorSpec & spec, double dt, const Eigen::Isometry3d & delta_T_prior,
  const std::optional<Eigen::Matrix3d> & R_world_body)
{
  const double sigma_rot = std::hypot(
    spec.gyro_bias_rad_s * dt, spec.gyro_noise_rad_s * std::sqrt(std::max(dt, 1e-12)));

  const Eigen::Vector3d d = delta_T_prior.translation();
  const double dist = d.norm();
  const double var_rw = spec.odom_noise_m_per_sqrt_m * spec.odom_noise_m_per_sqrt_m * dist +
    spec.odom_min_sigma_m * spec.odom_min_sigma_m;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * var_rw;
  if (dist > 1e-9) {
    const Eigen::Vector3d u = d / dist;
    cov += std::pow(spec.odom_scale_sigma * dist, 2) * (u * u.transpose());
  }
  if (R_world_body.has_value()) {
    cov = *R_world_body * cov * R_world_body->transpose();
  }

  Matrix6d info = Matrix6d::Zero();
  info.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() / std::pow(std::max(sigma_rot, 1e-12), 2);
  info.block<3, 3>(3, 3) = cov.inverse();
  return info;
}

int64_t voxelKey(const Eigen::Vector3d & point, double resolution)
{
  int64_t idx[3];
  for (int i = 0; i < 3; ++i) {
    const int64_t v = static_cast<int64_t>(std::floor(point(i) / resolution));
    if (std::abs(v) >= kVoxelOffset) {
      throw std::out_of_range("point outside the packed voxel range; raise the resolution");
    }
    idx[i] = v + kVoxelOffset;
  }
  return (idx[0] << (2 * kVoxelBits)) | (idx[1] << kVoxelBits) | idx[2];
}

Points voxelDownsample(const Points & points, double resolution)
{
  std::unordered_map<int64_t, CentroidSum> cells;
  for (const auto & p : points) {
    cells[voxelKey(p, resolution)].add(p);
  }
  Points out;
  out.reserve(cells.size());
  for (const auto & kv : cells) {
    out.push_back(kv.second.centroid());
  }
  return out;
}

LocalMap::LocalMap(double radius, double resolution, int max_points, int min_observations)
: radius_(radius), resolution_(resolution), max_points_(max_points),
  min_observations_(min_observations)
{
}

void LocalMap::add(const Points & points_world, const std::optional<Eigen::Vector3d> & centre_world)
{
  if (points_world.empty()) {
    return;
  }
  std::optional<Eigen::Vector3d> centre;
  if (centre_world.has_value()) {
    centre = toMap(*centre_world);
  }

  // one observation per voxel per scan, so counts mean "how many scans saw it"
  std::unordered_map<int64_t, CentroidSum> scan;
  for (const auto & p : points_world) {
    const Eigen::Vector3d q = toMap(p);
    scan[voxelKey(q, resolution_)].add(q);
  }
  for (const auto & kv : scan) {
    auto & voxel = voxels_[kv.first];
    voxel.sum += kv.second.centroid();
    voxel.count += 1;
  }

  if (centre.has_value()) {
    for (auto it = voxels_.begin(); it != voxels_.end(); ) {
      const Eigen::Vector3d centroid = it->second.sum / static_cast<double>(it->second.count);
      it = (centroid - *centre).norm() > radius_ ? voxels_.erase(it) : std::next(it);
    }
  }

  if (static_cast<int>(voxels_.size()) > max_points_) {
    // drop the furthest voxels, not every Nth: a stride thins the near field the registration
    // depends on just as hard as the far field it should shed
    const Eigen::Vector3d ref = centre.value_or(Eigen::Vector3d::Zero());
    std::vector<std::pair<double, int64_t>> by_distance;
    by_distance.reserve(voxels_.size());
    for (const auto & kv : voxels_) {
      const Eigen::Vector3d centroid = kv.second.sum / static_cast<double>(kv.second.count);
      by_distance.emplace_back((centroid - ref).norm(), kv.first);
    }
    std::nth_element(
      by_distance.begin(), by_distance.begin() + max_points_, by_distance.end());
    for (std::size_t i = max_points_; i < by_distance.size(); ++i) {
      voxels_.erase(by_distance[i].second);
    }
  }
}

void LocalMap::applyTransform(const Eigen::Matrix3d & R, const Eigen::Vector3d & t)
{
  R_ = R * R_;
  t_ = R * t_ + t;
}

Points LocalMap::points() const
{
  Points out;
  out.reserve(voxels_.size());
  for (const auto & kv : voxels_) {
    if (kv.second.count < min_observations_) {
      continue;
    }
    const Eigen::Vector3d centroid = kv.second.sum / static_cast<double>(kv.second.count);
    out.push_back(R_ * centroid + t_);
  }
  return out;
}

std::size_t LocalMap::size() const
{
  std::size_t n = 0;
  for (const auto & kv : voxels_) {
    if (kv.second.count >= min_observations_) {
      ++n;
    }
  }
  return n;
}

Odometry::Odometry(
  const Eigen::Isometry3d & T0, const OdometryConfig & cfg, const MotionPriorSpec & prior_spec,
  double dt, std::optional<LidarSpec> lidar_spec, const LandmarkSpec & landmark_spec,
  double marker_height, double marker_width, double marker_thickness)
: cfg_(cfg), prior_spec_(prior_spec), dt_(dt), lidar_spec_(lidar_spec),
  landmark_spec_(landmark_spec), marker_height_(marker_height), marker_width_(marker_width),
  marker_thickness_(marker_thickness), T_(T0),
  map_(cfg.map_radius, cfg.map_resolution, 80000, cfg.map_min_observations),
  scale_var_(prior_spec.odom_scale_sigma * prior_spec.odom_scale_sigma)
{
  Vector6d p0;
  p0 << 1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6;
  P_ = p0.asDiagonal();
}

void Odometry::resetMap()
{
  map_ = LocalMap(cfg_.map_radius, cfg_.map_resolution, 80000, cfg_.map_min_observations);
}

void Odometry::registerLandmark(int slot, const Eigen::Vector3d & offset_sensor)
{
  Landmark lm;
  lm.slot = slot;
  lm.position = T_ * offset_sensor;
  lm.registered_at_step = n_steps_;
  lm.covariance = P_;
  lm.T_drop = T_;
  lm.offset_drop = offset_sensor;
  lm.R_drop = dropCovariance(offset_sensor);
  lm.q_ref = Q_cum_;
  lm.rel = Eigen::Matrix3d::Zero();
  landmarks_.registerLandmark(lm);
  travelled_ref_[slot] = travelled_;
  // it inherits the error the estimate has now, so it moves with the next fix
  since_fix_.push_back(slot);
}

void Odometry::registerForeignLandmark(
  int slot, const Eigen::Vector3d & position_world, const Eigen::Matrix2d & R_drop,
  const Eigen::Vector2d & normal)
{
  const double n = normal.norm();
  if (n < 1e-9) {
    throw std::invalid_argument("normal must be a non-zero horizontal direction");
  }
  Landmark lm;
  lm.slot = slot;
  lm.position = position_world;
  lm.registered_at_step = n_steps_;
  lm.R_drop = R_drop;
  // the relation this anchor constrains is this robot's pose against the shared frame, so the
  // prior on it is this robot's own odometry since it was last fixed in that frame
  lm.q_ref = frame_q_ref_;
  lm.rel = Eigen::Matrix3d::Zero();
  lm.normal = normal / n;
  lm.foreign = true;
  landmarks_.registerLandmark(lm);
  travelled_ref_[slot] = travelled_;
}

Eigen::Matrix2d Odometry::dropCovariance(const Eigen::Vector3d & offset_sensor) const
{
  if (!lidar_spec_.has_value()) {
    return Eigen::Matrix2d::Identity() * landmark_spec_.min_sigma_m * landmark_spec_.min_sigma_m;
  }
  const int n = predictedBeamCount(offset_sensor, *lidar_spec_, marker_height_, marker_width_);
  const Eigen::Matrix3d omega = measurementInformation(
    offset_sensor, n, *lidar_spec_, landmark_spec_, marker_height_, marker_width_);
  const Eigen::Matrix3d R = T_.linear();
  const Eigen::Matrix3d world = R * omega * R.transpose();
  return world.block<2, 2>(0, 0).completeOrthogonalDecomposition().pseudoInverse();
}

Eigen::Matrix3d Odometry::relativeCovariance(const Landmark & lm) const
{
  if (lm.foreign) {
    // A teammate's marker is not relative to a drop of this robot's, so what separates the pose
    // from it is the odometry since this robot was last fixed in the shared frame, wherever in
    // that frame the fix came from.
    const Eigen::Matrix3d grown = subState(Q_cum_ - frame_q_ref_);
    return psd(Eigen::Matrix3d(frame_rel_ + grown));
  }
  if (!lm.q_ref.has_value()) {
    return Eigen::Matrix3d::Zero();
  }
  const Eigen::Matrix3d grown = subState(Q_cum_ - *lm.q_ref);
  const Eigen::Matrix3d base = lm.rel.value_or(Eigen::Matrix3d::Zero());
  return psd(Eigen::Matrix3d(base + grown));
}

std::pair<Eigen::Vector3d, std::optional<std::pair<double, double>>> Odometry::stripFit(
  const Landmark & lm, const Eigen::Vector3d & sensor_position, const MarkerDetection * obs) const
{
  const Eigen::Vector3d no_bias = Eigen::Vector3d::Zero();
  if (marker_thickness_ <= 0.0 || !lidar_spec_.has_value()) {
    return {no_bias, std::nullopt};
  }
  // the strip faces the way the robot stood when it mounted it: its normal is the horizontal
  // direction from the mounting point back to the drop pose
  Eigen::Vector2d normal;
  if (lm.normal.has_value()) {
    normal = *lm.normal;
  } else if (lm.T_drop.has_value() && lm.offset_drop.has_value()) {
    normal = -(lm.T_drop->linear() * *lm.offset_drop).head<2>();
  } else {
    return {no_bias, std::nullopt};
  }
  const Eigen::Vector2d los = lm.position.head<2>() - sensor_position.head<2>();
  const double distance = los.norm();
  if (normal.norm() < 1e-9 || distance < 1e-6) {
    return {no_bias, std::nullopt};
  }
  const double dz = lm.position.z() - sensor_position.z();
  std::optional<int> columns;
  if (obs != nullptr) {
    columns = obs->n_columns;
  }

  Eigen::Vector3d bias = Eigen::Vector3d::Zero();
  if (cfg_.correct_strip_bias) {
    const Eigen::Vector2d b = stripFitBias(
      los, normal, distance, *lidar_spec_, marker_width_, marker_thickness_, marker_height_, dz,
      columns);
    bias << b.x(), b.y(), 0.0;
  }
  std::optional<std::pair<double, double>> sigma;
  if (cfg_.measure_strip_spread) {
    sigma = stripFitSigma(
      los, normal, distance, *lidar_spec_, marker_width_, marker_thickness_, marker_height_, dz,
      columns);
  }
  return {bias, sigma};
}

std::vector<Odometry::AnchorTerm> Odometry::landmarkTerms(
  const Eigen::Isometry3d & T, const std::vector<MarkerDetection> & observations)
{
  std::vector<AnchorTerm> terms;
  if (observations.empty() || !lidar_spec_.has_value()) {
    return terms;
  }
  const Eigen::Matrix3d R = T.linear();
  const Eigen::Vector3d t = T.translation();
  for (const auto & obs : observations) {
    Landmark * lm = landmarks_.get(obs.slot);
    if (lm == nullptr) {
      continue;
    }
    Eigen::Vector3d q = T * obs.point_sensor;
    const auto fit = stripFit(*lm, t, &obs);
    q -= fit.first;
    const Eigen::Matrix3d omega_sensor = measurementInformation(
      obs.point_sensor, obs.n_beams, *lidar_spec_, landmark_spec_, marker_height_, marker_width_,
      obs.seen_width_m, obs.n_columns, fit.second);
    const Eigen::Matrix3d world = R * omega_sensor * R.transpose();
    const Eigen::Matrix2d R_meas =
      world.block<2, 2>(0, 0).completeOrthogonalDecomposition().pseudoInverse();

    AnchorTerm term;
    term.slot = obs.slot;
    term.q = q;
    term.anchor = lm->position;
    if (cfg_.relative_anchors) {
      term.R_obs = R_meas + lm->R_drop.value_or(Eigen::Matrix2d::Zero());
      term.Q_since = relativeCovariance(*lm);
    } else {
      // the old formulation: the anchor's absolute covariance is folded into the measurement, and
      // the pose prior is the absolute one
      term.R_obs = R_meas;
      if (lm->covariance.has_value()) {
        term.R_obs += lm->covariance->block<2, 2>(3, 3);
      }
      term.Q_since = psd(subState(P_));
    }
    lm->n_observations += 1;
    terms.push_back(term);
  }
  return terms;
}

void Odometry::propagateCovariance(
  const Matrix6d & prior_info, const Eigen::Vector3d & translation,
  const Matrix6d * registration_info)
{
  Matrix6d F = Matrix6d::Identity();
  F.block<3, 3>(3, 0) = -skew(translation);
  Matrix6d Q_step = prior_info.inverse();
  if (cfg_.credit_registration && registration_info != nullptr) {
    const Matrix6d H = 0.5 * (*registration_info + registration_info->transpose());
    const Matrix6d fused = (Q_step.inverse() + H).inverse();
    if (fused.allFinite()) {
      Q_step = 0.5 * (fused + fused.transpose());
    }
  }
  // The relative accumulator takes the step covariance as it stands, without the change of twist
  // centre. Over one step the lever arm is the step length, so the coupling it drops is second
  // order in the step, and the translation block, which is the only part a horizontal fix reads,
  // is unaffected by it.
  Q_cum_ += 0.5 * (Q_step + Q_step.transpose());
  Matrix6d P = F * P_ * F.transpose() + Q_step;
  if (registration_info != nullptr) {
    const Matrix6d H = 0.5 * (*registration_info + registration_info->transpose());
    const Matrix6d updated = (P.inverse() + H).inverse();
    if (updated.allFinite()) {
      P = updated;
    }
    P = 0.5 * (P + P.transpose());
  }
  P_ = P;
}

int Odometry::resurvey(const std::vector<MarkerDetection> & observations)
{
  int updated = 0;
  const double P_trace = P_.block<2, 2>(3, 3).trace();
  for (const auto & term : landmarkTerms(T_, observations)) {
    Landmark * lm = landmarks_.get(term.slot);
    // A marker a teammate mounted is never re-surveyed. Its coordinates are the shared frame's
    // definition; overwriting them with this robot's estimate would quietly redefine the frame
    // the teammate is still using.
    if (lm == nullptr || lm->foreign || !lm->covariance.has_value()) {
      continue;
    }
    if (P_trace < lm->covariance->block<2, 2>(3, 3).trace()) {
      lm->position = term.q;
      lm->covariance = P_;
      // the anchor has been re-registered from the current pose, so its relative record starts
      // again from here
      lm->T_drop = T_;
      lm->offset_drop = T_.linear().transpose() * (term.q - T_.translation());
      lm->R_drop = term.R_obs;
      lm->q_ref = Q_cum_;
      lm->rel = Eigen::Matrix3d::Zero();
      ++updated;
    }
  }
  return updated;
}

std::pair<int, int> Odometry::absoluteFix(const std::vector<MarkerDetection> & observations)
{
  std::vector<AnchorTerm> terms = landmarkTerms(T_, observations);
  if (terms.empty()) {
    return {0, 0};
  }

  const Eigen::Vector3d t_before = T_.translation();
  std::vector<int> distinct;
  for (const auto & term : terms) {
    if (std::find(distinct.begin(), distinct.end(), term.slot) == distinct.end()) {
      distinct.push_back(term.slot);
    }
  }
  std::sort(distinct.begin(), distinct.end());
  const bool use_yaw = cfg_.yaw_fix && distinct.size() >= 2;

  // The prior is the pose relative to the best known anchor in view. Every other anchor is then
  // uncertain relative to that one by the difference between their accumulated odometry, and that
  // difference goes into their measurement noise where it belongs. With one anchor the extra term
  // is zero and this reduces to the textbook update.
  const AnchorTerm * best = &terms.front();
  for (const auto & term : terms) {
    if (term.Q_since.block<2, 2>(1, 1).trace() < best->Q_since.block<2, 2>(1, 1).trace()) {
      best = &term;
    }
  }
  const Eigen::Matrix3d P_rel = psd(best->Q_since);
  // the leg this fix is measured over, captured before the anchor records are reset below,
  // because that reset is what the leg is measured from
  const auto ref_it = travelled_ref_.find(best->slot);
  const double leg = travelled_ - (ref_it == travelled_ref_.end() ? travelled_ : ref_it->second);

  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  std::map<int, Eigen::Matrix2d> extras;
  for (const auto & term : terms) {
    const Eigen::Matrix2d extra =
      psd(Eigen::Matrix2d(term.Q_since.block<2, 2>(1, 1) - P_rel.block<2, 2>(1, 1)));
    extras[term.slot] = extra;
    const Eigen::Matrix2d W = (term.R_obs + extra).inverse();
    const Eigen::Matrix<double, 2, 3> J = jacobian(term.q, T_.translation(), use_yaw);
    A += J.transpose() * W * J;
    b += -J.transpose() * W * (term.q - term.anchor).head<2>();
  }

  // A ridge rather than a pseudo-inverse: right at the drop the relative covariance is zero,
  // which means the pose is already exactly where the anchor says, and the correction must be
  // zero rather than unconstrained.
  const Eigen::Matrix3d S =
    (P_rel + 1e-10 * Eigen::Matrix3d::Identity()).inverse() + A;
  const Eigen::Vector3d x = solveOrLstsq3(S, b);

  const double dpsi = use_yaw ? x(0) : 0.0;
  const double dx = x(1);
  const double dy = x(2);
  const Eigen::Matrix3d R_corr = rotz(dpsi);
  const Eigen::Vector3d t_corr(dx, dy, 0.0);
  // the yaw acts about the current sensor position, not about the world origin
  const Eigen::Vector3d offset = t_before - R_corr * t_before + t_corr;

  T_ = makeT(R_corr * T_.linear(), R_corr * T_.translation() + offset);
  map_.applyTransform(R_corr, offset);

  // Landmarks registered since the last fix drifted with the estimate, so they move with the
  // correction. The ones this fix was computed from do not: they are the anchor, and moving them
  // with the pose would make the whole thing circular.
  std::vector<int> still_since;
  for (int slot : since_fix_) {
    if (std::find(distinct.begin(), distinct.end(), slot) != distinct.end()) {
      still_since.push_back(slot);
      continue;
    }
    if (Landmark * lm = landmarks_.get(slot)) {
      lm->position = R_corr * lm->position + offset;
    }
  }
  since_fix_ = still_since;

  const Eigen::Matrix3d posterior = psd(Eigen::Matrix3d(S.inverse()));
  Landmark * best_lm = landmarks_.get(best->slot);
  const Eigen::Vector2d heading = T_.linear().col(0).head<2>();
  const double q_along = heading.transpose() * P_rel.block<2, 2>(1, 1) * heading;
  const double r_along = heading.transpose() * best->R_obs * heading;

  FixRecord record;
  record.step = n_steps_;
  record.n_terms = static_cast<int>(terms.size());
  record.slots = distinct;
  record.dx = dx;
  record.dy = dy;
  record.dpsi = dpsi;
  record.along = heading.dot(Eigen::Vector2d(dx, dy));
  record.residual_along = heading.dot((best->q - best->anchor).head<2>());
  record.range_m = (best->anchor - t_before).head<2>().norm();
  record.q_since_along = q_along;
  record.r_along = r_along;
  record.gain_along = q_along / std::max(q_along + r_along, 1e-18);
  record.used_yaw = use_yaw;
  const double residual_along = record.residual_along;
  fix_log_.push_back(record);

  if (cfg_.relative_anchors) {
    // This robot has just been fixed in the shared frame, so its pose against that frame is the
    // posterior of this fix, and grows from here by its own odometry alone. Markers a teammate
    // mounted are held against this rather than against a mark of their own: they were all put
    // down in one frame, so being fixed against any of them is being fixed against all of them.
    frame_rel_ = posterior;
    frame_q_ref_ = Q_cum_;

    // Each anchor's relative record restarts here. There is no floor and no reset to an absolute
    // number: the relative covariance cannot fall below R_drop + R_meas because those terms are
    // inside S, which is the correct bound and the reason repeated observations of one strip can
    // be treated as independent without compounding into a confidence nothing supports.
    for (const auto & term : terms) {
      Landmark * lm = landmarks_.get(term.slot);
      if (lm == nullptr) {
        continue;
      }
      Eigen::Matrix3d rel = posterior;
      rel.block<2, 2>(1, 1) += extras[term.slot];
      lm->rel = rel;
      lm->q_ref = Q_cum_;
      travelled_ref_[term.slot] = travelled_;
    }

    // Absolute covariance is bookkeeping: where the whole chain might be with respect to the
    // start of the run. Reported, never used in a gain.
    if (best_lm != nullptr && best_lm->covariance.has_value()) {
      Matrix6d P_abs = P_;
      setSubState(P_abs, Eigen::Matrix3d(subState(*best_lm->covariance) + posterior));
      P_ = psd(P_abs);
    }
  } else {
    // The old formulation drove the gain from the absolute covariance, and needed a floor at the
    // anchor's own covariance to stop repeated views of one strip compounding into a confidence
    // the measurement could not support. Kept so the ablation can measure what the change was
    // worth.
    Eigen::Matrix3d P_meas = posterior;
    if (best_lm != nullptr && best_lm->covariance.has_value()) {
      P_meas.block<2, 2>(1, 1) =
        loewnerFloor(P_meas.block<2, 2>(1, 1), best_lm->covariance->block<2, 2>(3, 3));
    }
    setSubState(P_, P_meas);
  }

  if (cfg_.estimate_scale) {
    updateScale(leg, residual_along, q_along, r_along);
  }
  ++n_fixes_;
  return {static_cast<int>(terms.size()), use_yaw ? 1 : 0};
}

void Odometry::updateScale(double leg, double residual_along, double q_along, double r_along)
{
  // The residual along the direction of travel, accumulated over a leg of length L, is an
  // observation of (s - 1) * L. Short legs carry almost no information about a multiplicative
  // error, which the lever arm handles on its own: with L near zero the gain is near zero.
  //
  // The residual is used twice, once for the pose and once here. A jointly augmented state would
  // handle the correlation properly; sequentially it makes the scale estimate slightly
  // over-confident, which the random walk works against. The filter recovers 63 per cent of the
  // scale error and leaves 37 per cent, systematically (docs/failures.md number 35).
  const double L = leg;
  if (L <= 1e-6) {
    return;
  }
  const double S = L * L * scale_var_ + std::max(q_along + r_along, 1e-12);
  const double K = L * scale_var_ / S;
  scale_ += K * (residual_along - L * (scale_ - 1.0));
  scale_var_ = std::max((1.0 - K * L) * scale_var_, 1e-10);
  // a sanity bound far outside anything the prior spec allows, so a pathological fix cannot run
  // the prior away
  scale_ = std::clamp(scale_, 0.8, 1.2);
}

OdomStep Odometry::step(
  const Points & scan_points_sensor, const Eigen::Isometry3d & delta_T_prior_in,
  const std::vector<MarkerDetection> & marker_observations)
{
  const Eigen::Isometry3d T_before = T_;
  Eigen::Isometry3d delta_T_prior = delta_T_prior_in;
  if (cfg_.estimate_scale && std::abs(scale_ - 1.0) > 1e-12) {
    delta_T_prior.translation() /= scale_;
  }
  const double step_distance = delta_T_prior.translation().norm();
  travelled_ += step_distance;
  scale_var_ += cfg_.scale_random_walk_per_m * cfg_.scale_random_walk_per_m * step_distance;
  const Eigen::Isometry3d T_pred = T_ * delta_T_prior;

  Points reg_pts;
  if (cfg_.registration_range.has_value()) {
    for (const auto & p : scan_points_sensor) {
      if (p.norm() <= *cfg_.registration_range) {
        reg_pts.push_back(p);
      }
    }
  } else {
    reg_pts = scan_points_sensor;
  }

  const Matrix6d prior_info =
    priorInformation(prior_spec_, dt_, delta_T_prior, T_.linear());

  if (static_cast<int>(reg_pts.size()) < cfg_.min_scan_points ||
    static_cast<int>(map_.size()) < cfg_.min_scan_points)
  {
    T_ = T_pred;
    propagateCovariance(prior_info, T_pred.translation() - T_before.translation(), nullptr);
    if (!scan_points_sensor.empty()) {
      map_.add(transformPoints(T_, scan_points_sensor), T_.translation());
    }
    std::pair<int, int> fix{0, 0};
    if (cfg_.absolute_fix) {
      fix = absoluteFix(marker_observations);
    }
    ++n_steps_;

    OdomStep out;
    out.T = T_;
    out.T_prior = T_pred;
    out.n_scan_points = static_cast<int>(scan_points_sensor.size());
    out.registered = false;
    out.landmarks_used = fix.first;
    out.yaw_fixed = fix.second;
    return out;
  }

  RegistrationConfig reg_cfg;
  reg_cfg.downsampling_resolution = cfg_.downsampling_resolution;
  reg_cfg.max_correspondence_distance = cfg_.max_correspondence_distance;
  reg_cfg.max_iterations = cfg_.max_iterations;
  reg_cfg.num_threads = cfg_.num_threads;
  reg_cfg.type = cfg_.registration_type;

  const Points src_world = transformPoints(T_pred, reg_pts);
  const RegistrationOutput res = align(map_.points(), src_world, reg_cfg);
  const Localizability loc = analyseHessian(res.H, res.num_inliers);
  Eigen::Isometry3d T_corr = res.T_target_source;

  // ---- stage A: relative -------------------------------------------
  Matrix6d H_guarded = Matrix6d::Zero();
  bool have_H = false;
  if (cfg_.fuse_prior) {
    const auto fused = fuse(
      T_corr, res.H, prior_info, T_pred.translation(), res.num_inliers, res.error,
      cfg_.calibrate_information, cfg_.gicp_ratio_floor);
    T_corr = fused.first;
    H_guarded = fused.second;
    have_H = true;
  }

  int remapped = 0;
  if (cfg_.remap_degenerate && cfg_.remap_eigenvalue_floor > 0.0) {
    const auto out = remapCorrection(T_corr, loc, cfg_.remap_eigenvalue_floor);
    T_corr = out.first;
    remapped = out.second;
  }

  T_ = T_corr * T_pred;
  propagateCovariance(
    prior_info, T_.translation() - T_before.translation(), have_H ? &H_guarded : nullptr);
  map_.add(transformPoints(T_, scan_points_sensor), T_.translation());

  // ---- stage B: absolute -------------------------------------------
  std::pair<int, int> fix{0, 0};
  if (cfg_.absolute_fix) {
    if (cfg_.resurvey_anchors) {
      resurvey(marker_observations);
    }
    fix = absoluteFix(marker_observations);
  }
  ++n_steps_;

  OdomStep out;
  out.T = T_;
  out.T_prior = T_pred;
  out.localizability = loc;
  out.converged = res.converged;
  out.num_inliers = res.num_inliers;
  out.error = res.error;
  out.n_scan_points = static_cast<int>(reg_pts.size());
  out.registered = true;
  out.remapped_axes = remapped;
  out.landmarks_used = fix.first;
  out.yaw_fixed = fix.second;
  return out;
}

}  // namespace locrec
