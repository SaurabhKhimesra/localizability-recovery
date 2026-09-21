#include "locrec_core/gaze.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

namespace locrec
{

namespace
{

double wrapToPi(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

}  // namespace

Points mapNormals(const Points & points, int num_neighbors, int num_threads)
{
  Points normals(points.size(), Eigen::Vector3d::Zero());
  if (points.size() < 8) {
    return normals;
  }
  std::vector<Eigen::Vector4d> raw;
  raw.reserve(points.size());
  for (const auto & p : points) {
    raw.emplace_back(p.x(), p.y(), p.z(), 1.0);
  }
  auto cloud = std::make_shared<small_gicp::PointCloud>(raw);
  small_gicp::KdTree<small_gicp::PointCloud> tree(cloud);
  small_gicp::estimate_normals_omp(*cloud, tree, num_neighbors, num_threads);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d n = cloud->normal(i).head<3>();
    const double norm = n.norm();
    normals[i] = norm > 1e-9 ? Eigen::Vector3d(n / norm) : Eigen::Vector3d::Zero();
  }
  return normals;
}

Eigen::Matrix3d predictedInformation(
  const Points & points, const Points & normals, const Eigen::Vector3d & sensor_position,
  double yaw, const LidarSpec & spec)
{
  Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
  const double half_fov = spec.fov_azimuth_deg * M_PI / 180.0 * 0.5;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d rel = points[i] - sensor_position;
    const double dist = rel.norm();
    if (dist <= spec.min_range || dist > spec.max_range) {
      continue;
    }
    const double delta = wrapToPi(std::atan2(rel.y(), rel.x()) - yaw);
    if (std::abs(delta) > half_fov) {
      continue;
    }
    const Eigen::Vector3d & n = normals[i];
    const Eigen::Vector3d los = rel / dist;
    const double weight = std::abs(n.dot(los));
    info += weight * (n * n.transpose());
  }
  return info;
}

double scoreYaw(
  const Points & points, const Points & normals, const Eigen::Vector3d & origin, double yaw,
  const LidarSpec & spec, const std::optional<Eigen::Matrix3d> & have, double scale)
{
  Eigen::Matrix3d base = Eigen::Matrix3d::Zero();
  if (have.has_value()) {
    const double tr = have->trace();
    if (tr > 0.0) {
      base = *have / tr;
    }
  }
  const Eigen::Matrix3d predicted = predictedInformation(points, normals, origin, yaw, spec);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(base + predicted / (scale > 0.0 ? scale : 1.0));
  return es.eigenvalues()(0);
}

std::optional<Eigen::Vector3d> GlanceGaze::structureBehind(
  const Points & points, const Points & normals, const Eigen::Vector3d & origin, double forward,
  const LidarSpec & spec, const Localizability & loc) const
{
  // what the robot already knows, reconstructed from the detector's own output
  const Eigen::Matrix3d have =
    loc.eigenvectors * loc.eigenvalues.asDiagonal() * loc.eigenvectors.transpose();

  const double step = step_deg_ * M_PI / 180.0;
  const double cone = cone_deg_ * M_PI / 180.0;
  std::vector<double> inside;
  std::vector<double> outside;
  for (double offset = -M_PI; offset < M_PI; offset += step) {
    (std::abs(offset) > cone ? outside : inside).push_back(offset);
  }
  if (outside.empty()) {
    return std::nullopt;
  }

  // one common scale for every candidate, so the scores are comparable
  double scale = 0.0;
  for (double offset : inside) {
    scale = std::max(scale, predictedInformation(points, normals, origin, forward + offset, spec).trace());
  }
  for (double offset : outside) {
    scale = std::max(scale, predictedInformation(points, normals, origin, forward + offset, spec).trace());
  }
  if (scale <= 0.0) {
    scale = 1.0;
  }

  // inside the cone the sensor is already looking, so a glance has to beat the best of it
  double best_inside = 0.0;
  for (double offset : inside) {
    best_inside = std::max(
      best_inside, scoreYaw(points, normals, origin, forward + offset, spec, have, scale));
  }
  double best_outside = -1.0;
  double best_offset = 0.0;
  for (double offset : outside) {
    const double score = scoreYaw(points, normals, origin, forward + offset, spec, have, scale);
    if (score > best_outside) {
      best_outside = score;
      best_offset = offset;
    }
  }
  if (best_outside <= best_inside) {
    // nothing out there beats what forward already offers, so spend no slew on it
    return std::nullopt;
  }

  // the cluster itself, so that the sensor tracks it as the robot moves on
  const double yaw = forward + best_offset;
  const double half_fov = spec.fov_azimuth_deg * M_PI / 180.0 * 0.5;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  int n = 0;
  for (const auto & p : points) {
    const Eigen::Vector3d rel = p - origin;
    const double dist = rel.norm();
    if (dist <= spec.min_range || dist > spec.max_range) {
      continue;
    }
    if (std::abs(wrapToPi(std::atan2(rel.y(), rel.x()) - yaw)) > half_fov) {
      continue;
    }
    centroid += p;
    ++n;
  }
  if (n == 0) {
    return std::nullopt;
  }
  return centroid / static_cast<double>(n);
}

double GlanceGaze::operator()(
  double track_yaw, const Eigen::Isometry3d & T_est, const Localizability & loc,
  const Points & map_points, const Points & map_normals_in, const LidarSpec & spec, double dt)
{
  ++steps_;
  const Eigen::Vector3d origin = T_est.translation();

  if (target_point_.has_value()) {
    holding_ += dt;
    if (holding_ >= hold_s_) {
      // the glance is over: back to forward, and the cooldown starts now
      target_point_.reset();
      holding_ = 0.0;
      cooling_ = 0.0;
      return track_yaw;
    }
    ++glancing_steps_;
    const Eigen::Vector3d rel = *target_point_ - origin;
    return std::atan2(rel.y(), rel.x());
  }

  if (cooling_ < cooldown_s_) {
    cooling_ += dt;
    return track_yaw;
  }

  if (loc.ratio >= ratio_threshold_ || map_points.size() < 8 ||
    map_normals_in.size() != map_points.size())
  {
    return track_yaw;
  }

  const auto target = structureBehind(map_points, map_normals_in, origin, track_yaw, spec, loc);
  if (!target.has_value()) {
    return track_yaw;
  }
  target_point_ = target;
  holding_ = 0.0;
  ++glancing_steps_;
  const Eigen::Vector3d rel = *target - origin;
  return std::atan2(rel.y(), rel.x());
}

}  // namespace locrec
