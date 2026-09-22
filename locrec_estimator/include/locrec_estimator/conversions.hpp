// PointCloud2 and Odometry into plain Eigen types.
//
// Field layout is read from the message rather than assumed. A Velodyne driver, an Ouster driver
// and a rosbag replay all describe x, y and z with the same names but at different offsets and
// sometimes different datatypes, and a hard-coded stride is the classic way to get a point cloud
// that looks plausible and is wrong.
#pragma once

#include <optional>

#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "locrec_core/se3.hpp"

namespace locrec_estimator
{

/// Finite xyz points from a PointCloud2.
///
/// Rows are unpacked with row_step rather than width * point_step, because organised clouds pad
/// their rows. Non-finite points are dropped: most drivers mark an invalid return with NaN, and a
/// NaN reaching the registration turns the whole Hessian into NaN, which surfaces as a detector
/// that silently never fires.
locrec::Points pointCloud2ToXyz(
  const sensor_msgs::msg::PointCloud2 & msg, std::optional<int> max_points = std::nullopt);

/// A builtin_interfaces/Time as integer nanoseconds, so stamps compare exactly.
int64_t stampToNs(const builtin_interfaces::msg::Time & stamp);

/// The pose in a nav_msgs/Odometry: child frame in odometry frame. The quaternion is normalised;
/// a zero or non-finite one is an error rather than an identity, because an identity rotation is
/// a plausible prior that is wrong.
Eigen::Isometry3d odometryToPose(const nav_msgs::msg::Odometry & msg);

}  // namespace locrec_estimator
