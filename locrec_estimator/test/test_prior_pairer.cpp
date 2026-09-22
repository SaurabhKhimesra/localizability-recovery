#include <gtest/gtest.h>

#include <string>

#include "locrec_estimator/prior_pairer.hpp"

using locrec_estimator::PriorPairer;
using locrec_estimator::interpolatePose;

namespace
{

Eigen::Isometry3d at(double x, double yaw = 0.0)
{
  return locrec::makeT(locrec::rotz(yaw), Eigen::Vector3d(x, 0.0, 0.0));
}

using Pairer = PriorPairer<std::string, int>;

}  // namespace

TEST(InterpolatePose, walks_between_the_two_samples) {
  const Eigen::Isometry3d T = interpolatePose(at(0.0, 0.0), at(2.0, 1.0), 0.25);
  EXPECT_NEAR(T.translation().x(), 0.5, 1e-12);
  EXPECT_NEAR(locrec::so3Log(T.linear()).z(), 0.25, 1e-12);
}

TEST(PriorPairerTest, holds_a_scan_until_the_odometry_reaches_its_stamp) {
  Pairer pairer;
  EXPECT_TRUE(pairer.addScan(100, "scan").empty());
  EXPECT_EQ(pairer.waiting(), 1u);
  // odometry before the scan is not enough: nothing is extrapolated
  EXPECT_TRUE(pairer.addOdometry(50, at(0.5)).empty());
  const auto released = pairer.addOdometry(150, at(1.5));
  ASSERT_EQ(released.size(), 1u);
  EXPECT_EQ(released[0].payload, "scan");
  EXPECT_NEAR(released[0].pose.translation().x(), 1.0, 1e-12) << "interpolated to the scan stamp";
  EXPECT_EQ(pairer.dropped(), 0);
}

TEST(PriorPairerTest, an_exact_stamp_is_used_as_it_is) {
  Pairer pairer;
  pairer.addOdometry(100, at(7.0));
  pairer.addOdometry(200, at(9.0));
  const auto released = pairer.addScan(100, "scan");
  ASSERT_EQ(released.size(), 1u);
  EXPECT_NEAR(released[0].pose.translation().x(), 7.0, 1e-12);
}

TEST(PriorPairerTest, releases_in_stamp_order_whichever_message_arrived_first) {
  Pairer pairer;
  pairer.addScan(300, "third");
  pairer.addScan(100, "first");
  pairer.addScan(200, "second");
  pairer.addOdometry(0, at(0.0));
  const auto released = pairer.addOdometry(400, at(4.0));
  ASSERT_EQ(released.size(), 3u);
  EXPECT_EQ(released[0].payload, "first");
  EXPECT_EQ(released[1].payload, "second");
  EXPECT_EQ(released[2].payload, "third");
}

TEST(PriorPairerTest, a_scan_older_than_the_history_is_dropped_and_counted) {
  Pairer pairer;
  pairer.addOdometry(1000, at(1.0));
  pairer.addOdometry(2000, at(2.0));
  const auto released = pairer.addScan(500, "stale");
  EXPECT_TRUE(released.empty());
  EXPECT_EQ(pairer.droppedStale(), 1);
  EXPECT_EQ(pairer.dropped(), 1);
}

TEST(PriorPairerTest, a_scan_at_or_before_one_already_released_is_dropped) {
  Pairer pairer;
  pairer.addOdometry(100, at(1.0));
  ASSERT_EQ(pairer.addScan(100, "first").size(), 1u);
  EXPECT_TRUE(pairer.addScan(100, "again").empty());
  EXPECT_EQ(pairer.droppedOutOfOrder(), 1);
}

TEST(PriorPairerTest, the_queue_is_bounded) {
  Pairer pairer(3);
  for (int i = 0; i < 10; ++i) {
    pairer.addScan(100 + i, "scan");
  }
  EXPECT_EQ(pairer.waiting(), 3u);
  EXPECT_EQ(pairer.droppedOverflow(), 7);
}

TEST(PriorPairerTest, with_require_side_a_scan_also_waits_for_its_own_report) {
  Pairer pairer(10, 4096, true);
  pairer.addOdometry(100, at(1.0));
  EXPECT_TRUE(pairer.addScan(100, "scan").empty()) << "the report has not come yet";

  const auto released = pairer.addSide(100, 42);
  ASSERT_EQ(released.size(), 1u);
  ASSERT_TRUE(released[0].side.has_value());
  EXPECT_EQ(*released[0].side, 42) << "the join is exact: a report is about one scan and no other";
}

TEST(PriorPairerTest, a_scan_whose_report_can_no_longer_come_is_dropped) {
  Pairer pairer(10, 4096, true);
  pairer.addOdometry(50, at(0.5));
  pairer.addOdometry(300, at(3.0));
  pairer.addScan(100, "orphan");
  pairer.addScan(200, "fine");
  const auto released = pairer.addSide(200, 7);
  ASSERT_EQ(released.size(), 1u);
  EXPECT_EQ(released[0].payload, "fine");
  EXPECT_EQ(pairer.droppedNoSide(), 1);
}

TEST(PriorPairerTest, counts_the_odometry_it_has_seen) {
  Pairer pairer;
  EXPECT_EQ(pairer.nOdometry(), 0);
  pairer.addOdometry(1, at(0.0));
  pairer.addOdometry(2, at(0.1));
  EXPECT_EQ(pairer.nOdometry(), 2);
}
