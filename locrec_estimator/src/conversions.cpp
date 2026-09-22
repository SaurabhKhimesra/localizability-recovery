#include "locrec_estimator/conversions.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace locrec_estimator
{

namespace
{

/// Read one field of one point, honouring its declared datatype and the cloud's endianness.
double readField(const uint8_t * point, uint32_t offset, uint8_t datatype, bool big_endian)
{
  const auto load = [&](std::size_t size) {
      uint8_t bytes[8] = {0};
      std::memcpy(bytes, point + offset, size);
      if (big_endian) {
        for (std::size_t i = 0; i < size / 2; ++i) {
          std::swap(bytes[i], bytes[size - 1 - i]);
        }
      }
      return *reinterpret_cast<uint64_t *>(bytes);
    };

  switch (datatype) {
    case sensor_msgs::msg::PointField::INT8: {
        int8_t v = 0;
        std::memcpy(&v, point + offset, 1);
        return v;
      }
    case sensor_msgs::msg::PointField::UINT8: {
        uint8_t v = 0;
        std::memcpy(&v, point + offset, 1);
        return v;
      }
    case sensor_msgs::msg::PointField::INT16: {
        const uint64_t raw = load(2);
        int16_t v = 0;
        std::memcpy(&v, &raw, 2);
        return v;
      }
    case sensor_msgs::msg::PointField::UINT16: {
        const uint64_t raw = load(2);
        uint16_t v = 0;
        std::memcpy(&v, &raw, 2);
        return v;
      }
    case sensor_msgs::msg::PointField::INT32: {
        const uint64_t raw = load(4);
        int32_t v = 0;
        std::memcpy(&v, &raw, 4);
        return v;
      }
    case sensor_msgs::msg::PointField::UINT32: {
        const uint64_t raw = load(4);
        uint32_t v = 0;
        std::memcpy(&v, &raw, 4);
        return v;
      }
    case sensor_msgs::msg::PointField::FLOAT32: {
        const uint64_t raw = load(4);
        float v = 0.0F;
        std::memcpy(&v, &raw, 4);
        return v;
      }
    case sensor_msgs::msg::PointField::FLOAT64: {
        const uint64_t raw = load(8);
        double v = 0.0;
        std::memcpy(&v, &raw, 8);
        return v;
      }
    default:
      throw std::invalid_argument(
              "unsupported PointField datatype " + std::to_string(static_cast<int>(datatype)));
  }
}

}  // namespace

locrec::Points pointCloud2ToXyz(
  const sensor_msgs::msg::PointCloud2 & msg, std::optional<int> max_points)
{
  const sensor_msgs::msg::PointField * fields[3] = {nullptr, nullptr, nullptr};
  const char * names[3] = {"x", "y", "z"};
  for (const auto & field : msg.fields) {
    for (int i = 0; i < 3; ++i) {
      if (field.name == names[i]) {
        fields[i] = &field;
      }
    }
  }
  for (int i = 0; i < 3; ++i) {
    if (fields[i] == nullptr) {
      throw std::invalid_argument(std::string("point cloud has no ") + names[i] + " field");
    }
  }

  const std::size_t n_rows = std::max<std::size_t>(msg.height, 1);
  const std::size_t point_step = msg.point_step;
  const std::size_t row_step = msg.row_step != 0 ? msg.row_step : msg.width * point_step;
  if (point_step == 0) {
    throw std::invalid_argument("point cloud has a zero point_step");
  }
  const std::size_t expected = n_rows * row_step;
  if (msg.data.size() < expected) {
    throw std::invalid_argument(
            "point cloud data is " + std::to_string(msg.data.size()) + " bytes, expected " +
            std::to_string(expected));
  }

  const std::size_t n_points = static_cast<std::size_t>(msg.width) * n_rows;
  const std::size_t per_row = row_step / point_step;

  locrec::Points out;
  out.reserve(std::min<std::size_t>(n_points, 1u << 20));
  for (std::size_t p = 0; p < n_points; ++p) {
    const std::size_t row = p / per_row;
    const std::size_t column = p % per_row;
    const uint8_t * point = msg.data.data() + row * row_step + column * point_step;
    const Eigen::Vector3d xyz(
      readField(point, fields[0]->offset, fields[0]->datatype, msg.is_bigendian),
      readField(point, fields[1]->offset, fields[1]->datatype, msg.is_bigendian),
      readField(point, fields[2]->offset, fields[2]->datatype, msg.is_bigendian));
    if (xyz.allFinite()) {
      out.push_back(xyz);
    }
  }

  if (max_points.has_value() && static_cast<int>(out.size()) > *max_points) {
    const std::size_t stride = static_cast<std::size_t>(
      std::ceil(static_cast<double>(out.size()) / *max_points));
    locrec::Points thinned;
    thinned.reserve(out.size() / stride + 1);
    for (std::size_t i = 0; i < out.size(); i += stride) {
      thinned.push_back(out[i]);
    }
    out = std::move(thinned);
  }
  return out;
}

int64_t stampToNs(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

Eigen::Isometry3d odometryToPose(const nav_msgs::msg::Odometry & msg)
{
  const auto & p = msg.pose.pose.position;
  const auto & q = msg.pose.pose.orientation;
  const Eigen::Vector3d t(p.x, p.y, p.z);
  Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
  const double norm = quat.norm();
  if (!t.allFinite() || !std::isfinite(norm) || norm < 1e-9) {
    throw std::invalid_argument("odometry pose has a non-finite position or a zero quaternion");
  }
  quat.normalize();
  return locrec::makeT(quat.toRotationMatrix(), t);
}

}  // namespace locrec_estimator
