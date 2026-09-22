#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "locrec_estimator/conversions.hpp"

using locrec_estimator::odometryToPose;
using locrec_estimator::pointCloud2ToXyz;
using locrec_estimator::stampToNs;

namespace
{

/// A cloud in the layout a Gazebo or Velodyne driver produces: float32 xyz, with an intensity
/// field after them, so the stride is wider than the three points.
sensor_msgs::msg::PointCloud2 cloudOf(const std::vector<std::array<float, 3>> & points)
{
  sensor_msgs::msg::PointCloud2 msg;
  msg.height = 1;
  msg.width = points.size();
  msg.point_step = 16;
  msg.row_step = msg.point_step * msg.width;
  msg.is_bigendian = false;
  msg.is_dense = false;
  const char * names[3] = {"x", "y", "z"};
  for (int i = 0; i < 3; ++i) {
    sensor_msgs::msg::PointField field;
    field.name = names[i];
    field.offset = 4 * i;
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
    msg.fields.push_back(field);
  }
  sensor_msgs::msg::PointField intensity;
  intensity.name = "intensity";
  intensity.offset = 12;
  intensity.datatype = sensor_msgs::msg::PointField::FLOAT32;
  intensity.count = 1;
  msg.fields.push_back(intensity);

  msg.data.resize(msg.row_step);
  for (std::size_t i = 0; i < points.size(); ++i) {
    std::memcpy(msg.data.data() + i * msg.point_step, points[i].data(), 12);
    const float value = 1.0F;
    std::memcpy(msg.data.data() + i * msg.point_step + 12, &value, 4);
  }
  return msg;
}

}  // namespace

TEST(Conversions, reads_xyz_at_the_offsets_the_message_declares) {
  const auto msg = cloudOf({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});
  const locrec::Points pts = pointCloud2ToXyz(msg);
  ASSERT_EQ(pts.size(), 2u);
  EXPECT_NEAR(pts[0].x(), 1.0, 1e-6);
  EXPECT_NEAR(pts[0].z(), 3.0, 1e-6);
  EXPECT_NEAR(pts[1].y(), 5.0, 1e-6);
}

TEST(Conversions, drops_non_finite_returns) {
  // A NaN reaching the registration turns the whole Hessian into NaN, which surfaces as a
  // detector that silently never fires.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto msg = cloudOf({{1.0F, 2.0F, 3.0F}, {nan, nan, nan}, {7.0F, 8.0F, 9.0F}});
  const locrec::Points pts = pointCloud2ToXyz(msg);
  ASSERT_EQ(pts.size(), 2u);
  EXPECT_NEAR(pts[1].x(), 7.0, 1e-6);
}

TEST(Conversions, thins_a_cloud_that_is_over_the_cap) {
  std::vector<std::array<float, 3>> points;
  for (int i = 0; i < 100; ++i) {
    points.push_back({static_cast<float>(i), 0.0F, 0.0F});
  }
  const locrec::Points pts = pointCloud2ToXyz(cloudOf(points), 10);
  EXPECT_LE(pts.size(), 10u);
  EXPECT_GT(pts.size(), 5u);
}

TEST(Conversions, a_missing_field_is_an_error) {
  auto msg = cloudOf({{1.0F, 2.0F, 3.0F}});
  msg.fields.erase(msg.fields.begin() + 2);  // no z
  EXPECT_THROW(pointCloud2ToXyz(msg), std::invalid_argument);
}

TEST(Conversions, short_data_is_an_error_rather_than_a_plausible_cloud) {
  auto msg = cloudOf({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});
  msg.data.resize(msg.data.size() - 8);
  EXPECT_THROW(pointCloud2ToXyz(msg), std::invalid_argument);
}

TEST(Conversions, unpacks_an_organised_cloud_by_row_step) {
  // organised clouds pad their rows, so width * point_step is not the stride
  auto msg = cloudOf({{1.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}});
  msg.height = 2;
  msg.width = 1;
  msg.row_step = 24;  // 16 bytes of point plus 8 of padding
  std::vector<uint8_t> data(48, 0);
  std::memcpy(data.data(), msg.data.data(), 16);
  std::memcpy(data.data() + 24, msg.data.data() + 16, 16);
  msg.data = data;

  const locrec::Points pts = pointCloud2ToXyz(msg);
  ASSERT_EQ(pts.size(), 2u);
  EXPECT_NEAR(pts[0].x(), 1.0, 1e-6);
  EXPECT_NEAR(pts[1].x(), 2.0, 1e-6);
}

TEST(Conversions, stamps_compare_exactly_as_nanoseconds) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 3;
  stamp.nanosec = 250000000;
  EXPECT_EQ(stampToNs(stamp), 3250000000LL);
}

TEST(Conversions, reads_the_odometry_pose_and_normalises_its_quaternion) {
  nav_msgs::msg::Odometry msg;
  msg.pose.pose.position.x = 1.0;
  msg.pose.pose.position.y = 2.0;
  msg.pose.pose.position.z = 3.0;
  // a quarter turn about z, deliberately not normalised
  msg.pose.pose.orientation.z = 2.0 * std::sin(M_PI / 4.0);
  msg.pose.pose.orientation.w = 2.0 * std::cos(M_PI / 4.0);

  const Eigen::Isometry3d T = odometryToPose(msg);
  EXPECT_NEAR(T.translation().y(), 2.0, 1e-12);
  EXPECT_NEAR(locrec::so3Log(T.linear()).z(), M_PI / 2.0, 1e-9);
  EXPECT_NEAR(T.linear().determinant(), 1.0, 1e-12);
}

TEST(Conversions, a_zero_quaternion_is_an_error_rather_than_an_identity) {
  // an identity rotation is a plausible prior that is wrong, so a driver that leaves the
  // orientation unset is refused rather than believed
  nav_msgs::msg::Odometry msg;
  msg.pose.pose.orientation.w = 0.0;  // the message's own default is the identity quaternion
  EXPECT_THROW(odometryToPose(msg), std::invalid_argument);
}

TEST(Conversions, a_non_finite_position_is_an_error) {
  nav_msgs::msg::Odometry msg;
  msg.pose.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(odometryToPose(msg), std::invalid_argument);
}
