#include "locrec_core/registration.hpp"

#include <stdexcept>
#include <vector>

#include <small_gicp/registration/registration_helper.hpp>

namespace locrec
{

namespace
{

std::vector<Eigen::Vector4d> toEigenPoints(const Points & pts)
{
  std::vector<Eigen::Vector4d> out;
  out.reserve(pts.size());
  for (const auto & p : pts) {
    out.emplace_back(p.x(), p.y(), p.z(), 1.0);
  }
  return out;
}

small_gicp::RegistrationSetting::RegistrationType parseType(const std::string & name)
{
  if (name == "GICP") {
    return small_gicp::RegistrationSetting::GICP;
  }
  if (name == "PLANE_ICP") {
    return small_gicp::RegistrationSetting::PLANE_ICP;
  }
  if (name == "ICP") {
    return small_gicp::RegistrationSetting::ICP;
  }
  if (name == "VGICP") {
    return small_gicp::RegistrationSetting::VGICP;
  }
  throw std::invalid_argument("unknown registration type: " + name);
}

}  // namespace

RegistrationOutput align(
  const Points & target, const Points & source, const RegistrationConfig & cfg)
{
  small_gicp::RegistrationSetting setting;
  setting.type = parseType(cfg.type);
  setting.downsampling_resolution = cfg.downsampling_resolution;
  setting.max_correspondence_distance = cfg.max_correspondence_distance;
  setting.max_iterations = cfg.max_iterations;
  setting.num_threads = cfg.num_threads;

  const auto result = small_gicp::align(
    toEigenPoints(target), toEigenPoints(source), Eigen::Isometry3d::Identity(), setting);

  RegistrationOutput out;
  out.T_target_source = result.T_target_source;
  out.H = result.H;
  out.converged = result.converged;
  out.num_inliers = static_cast<int>(result.num_inliers);
  out.error = result.error;
  return out;
}

}  // namespace locrec
