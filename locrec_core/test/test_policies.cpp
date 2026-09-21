#include <gtest/gtest.h>

#include "locrec_core/policies.hpp"

using namespace locrec;  // NOLINT(build/namespaces)

namespace
{
constexpr double kThreshold = 0.01;
constexpr double kRange = 7.0;
constexpr double kMargin = 1.5;
// the duty cycle: twice the detection range, less the margin
constexpr double kReach = 2.0 * kRange - kMargin;
}  // namespace

TEST(Scheduler, spends_nothing_while_the_geometry_carries_the_estimate) {
  LocalizabilityScheduler policy(kThreshold, kRange, kMargin);
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(policy(kThreshold * 10.0, i * 0.5));
  }
}

TEST(Scheduler, drops_on_the_falling_edge_one_step_into_the_blind_stretch) {
  LocalizabilityScheduler policy(kThreshold, kRange, kMargin);
  EXPECT_FALSE(policy(kThreshold * 10.0, 0.0));
  EXPECT_FALSE(policy(kThreshold * 10.0, 0.5));
  // the first degenerate scan, and the marker goes on the wall there
  EXPECT_TRUE(policy(kThreshold * 0.1, 1.0));
}

TEST(Scheduler, then_spaces_by_twice_the_detection_range) {
  LocalizabilityScheduler policy(kThreshold, kRange, kMargin);
  ASSERT_TRUE(policy(kThreshold * 0.1, 0.0));
  int drops = 0;
  for (int i = 1; i <= 100; ++i) {
    if (policy(kThreshold * 0.1, i * 0.5)) {
      ++drops;
    }
  }
  // 50 m of blind tunnel at a 12.5 m duty cycle
  EXPECT_EQ(drops, static_cast<int>(50.0 / kReach));
}

TEST(Scheduler, does_not_pay_for_chatter_at_the_threshold) {
  // On a 300 m blind run the ratio crosses the threshold 77 times. A rule that fires on every
  // crossing spends markers on chatter, so the chain is not forgotten when a stretch turns
  // localizable again.
  LocalizabilityScheduler policy(kThreshold, kRange, kMargin);
  int drops = 0;
  for (int i = 0; i < 20; ++i) {
    const double distance = i * 0.5;
    const double ratio = (i % 2 == 0) ? kThreshold * 0.1 : kThreshold * 10.0;
    if (policy(ratio, distance)) {
      ++drops;
    }
  }
  EXPECT_EQ(drops, 1);
}
