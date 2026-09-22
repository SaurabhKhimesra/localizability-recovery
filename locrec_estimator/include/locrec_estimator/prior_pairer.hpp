// Pair each scan with the odometry pose at its own stamp, with no ROS include.
//
// A scan's prior is the odometry at that scan's stamp, and "the latest odometry message" is not
// that. The simulator publishes a scan before the odometry that shares its stamp, so the latest
// is either one scan stale or racing the scan depending on which callback runs first, and a real
// driver promises no order at all. So a scan waits here until the odometry history reaches its
// stamp. An exact stamp match is used as it is; otherwise the pose is interpolated between the
// two samples either side of it, linearly in translation and along the shortest rotation. Nothing
// is extrapolated, and a scan that cannot be paired is dropped and counted rather than passed on
// without a prior.
//
// A scan can also be made to wait for a side message that carries exactly its stamp, which is how
// the marker report reaches the estimator. That join is exact rather than interpolated, because a
// report is about one scan and no other.
#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "locrec_core/se3.hpp"

namespace locrec_estimator
{

/// The pose a fraction alpha of the way from T0 to T1.
inline Eigen::Isometry3d interpolatePose(
  const Eigen::Isometry3d & T0, const Eigen::Isometry3d & T1, double alpha)
{
  const Eigen::Vector3d rotvec = locrec::so3Log(T0.linear().transpose() * T1.linear());
  return locrec::makeT(
    T0.linear() * locrec::so3Exp(alpha * rotvec),
    (1.0 - alpha) * T0.translation() + alpha * T1.translation());
}

template<typename Payload, typename Side>
class PriorPairer
{
public:
  struct Paired
  {
    int64_t stamp = 0;
    Payload payload;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    std::optional<Side> side;
  };

  explicit PriorPairer(
    std::size_t max_pending = 10, std::size_t max_history = 4096, bool require_side = false)
  : max_pending_(max_pending), max_history_(max_history), require_side_(require_side) {}

  std::vector<Paired> addSide(int64_t stamp, Side payload)
  {
    side_[stamp] = std::move(payload);
    if (!newest_side_.has_value() || stamp > *newest_side_) {
      newest_side_ = stamp;
    }
    return release();
  }

  std::vector<Paired> addOdometry(int64_t stamp, const Eigen::Isometry3d & pose)
  {
    ++n_odometry_;
    const auto it = std::lower_bound(odom_stamps_.begin(), odom_stamps_.end(), stamp);
    const auto index = it - odom_stamps_.begin();
    if (it != odom_stamps_.end() && *it == stamp) {
      odom_poses_[index] = pose;
    } else {
      odom_stamps_.insert(it, stamp);
      odom_poses_.insert(odom_poses_.begin() + index, pose);
    }
    return release();
  }

  std::vector<Paired> addScan(int64_t stamp, Payload payload)
  {
    if (last_released_.has_value() && stamp <= *last_released_) {
      ++dropped_out_of_order_;
      return {};
    }
    const auto it = std::upper_bound(
      pending_.begin(), pending_.end(), stamp,
      [](int64_t value, const std::pair<int64_t, Payload> & entry) {return value < entry.first;});
    pending_.insert(it, {stamp, std::move(payload)});
    while (pending_.size() > max_pending_) {
      pending_.erase(pending_.begin());
      ++dropped_overflow_;
    }
    return release();
  }

  std::size_t waiting() const {return pending_.size();}
  int nOdometry() const {return n_odometry_;}
  int droppedStale() const {return dropped_stale_;}
  int droppedOverflow() const {return dropped_overflow_;}
  int droppedOutOfOrder() const {return dropped_out_of_order_;}
  int droppedNoSide() const {return dropped_no_side_;}
  int dropped() const
  {
    return dropped_stale_ + dropped_overflow_ + dropped_out_of_order_ + dropped_no_side_;
  }

private:
  std::vector<Paired> release()
  {
    std::vector<Paired> out;
    while (!pending_.empty()) {
      const int64_t stamp = pending_.front().first;
      if (odom_stamps_.empty() || odom_stamps_.back() < stamp) {
        break;  // the odometry has not reached this scan yet
      }
      std::optional<Side> side;
      if (require_side_) {
        const auto it = side_.find(stamp);
        if (it == side_.end()) {
          if (!newest_side_.has_value() || *newest_side_ < stamp) {
            break;  // its side message may still be on the way
          }
          // a later one has arrived, so this scan's never will
          pending_.erase(pending_.begin());
          ++dropped_no_side_;
          continue;
        }
        side = it->second;
      }
      Paired paired;
      paired.stamp = stamp;
      paired.payload = std::move(pending_.front().second);
      pending_.erase(pending_.begin());
      if (odom_stamps_.front() > stamp) {
        ++dropped_stale_;
        continue;
      }
      paired.pose = poseAt(stamp);
      paired.side = std::move(side);
      last_released_ = stamp;
      out.push_back(std::move(paired));
    }
    trim();
    return out;
  }

  Eigen::Isometry3d poseAt(int64_t stamp) const
  {
    const auto it = std::lower_bound(odom_stamps_.begin(), odom_stamps_.end(), stamp);
    const auto i = it - odom_stamps_.begin();
    if (it != odom_stamps_.end() && *it == stamp) {
      return odom_poses_[i];
    }
    const double t0 = static_cast<double>(odom_stamps_[i - 1]);
    const double t1 = static_cast<double>(odom_stamps_[i]);
    return interpolatePose(
      odom_poses_[i - 1], odom_poses_[i], (static_cast<double>(stamp) - t0) / (t1 - t0));
  }

  void trim()
  {
    // every scan still to come is stamped after this, so only the last sample at or before it is
    // ever needed again
    std::optional<int64_t> floor =
      pending_.empty() ? last_released_ : std::optional<int64_t>(pending_.front().first);
    if (floor.has_value()) {
      const auto it = std::upper_bound(odom_stamps_.begin(), odom_stamps_.end(), *floor);
      const auto j = (it - odom_stamps_.begin()) - 1;
      if (j > 0) {
        odom_stamps_.erase(odom_stamps_.begin(), odom_stamps_.begin() + j);
        odom_poses_.erase(odom_poses_.begin(), odom_poses_.begin() + j);
      }
    }
    if (odom_stamps_.size() > max_history_) {
      const auto excess = odom_stamps_.size() - max_history_;
      odom_stamps_.erase(odom_stamps_.begin(), odom_stamps_.begin() + excess);
      odom_poses_.erase(odom_poses_.begin(), odom_poses_.begin() + excess);
    }
    // side messages at or before the floor belong to scans that are already gone
    if (floor.has_value()) {
      const int64_t oldest_needed = pending_.empty() ? *floor + 1 : pending_.front().first;
      for (auto it = side_.begin(); it != side_.end(); ) {
        it = it->first < oldest_needed ? side_.erase(it) : std::next(it);
      }
    }
    while (side_.size() > max_history_) {
      side_.erase(side_.begin());
    }
  }

  std::size_t max_pending_;
  std::size_t max_history_;
  bool require_side_;
  std::vector<int64_t> odom_stamps_;
  std::vector<Eigen::Isometry3d> odom_poses_;
  std::vector<std::pair<int64_t, Payload>> pending_;
  std::map<int64_t, Side> side_;
  std::optional<int64_t> newest_side_;
  std::optional<int64_t> last_released_;
  int n_odometry_ = 0;
  int dropped_stale_ = 0;      ///< scans older than every odometry sample still held
  int dropped_overflow_ = 0;   ///< scans pushed out by newer ones while waiting for odometry
  int dropped_out_of_order_ = 0;  ///< scans stamped at or before a scan already released
  int dropped_no_side_ = 0;    ///< scans whose side message never came although a later one did
};

}  // namespace locrec_estimator
