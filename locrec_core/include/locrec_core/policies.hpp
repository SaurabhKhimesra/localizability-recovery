// Marker-drop policy for the ground robot.
//
// The policy sees the same thing the estimator sees and nothing else: the live localizability
// ratio and how far the estimate says the robot has come. Markers are a budgeted resource, so a
// policy that drops one every step would bound drift beautifully and be useless underground.
#pragma once

#include <algorithm>
#include <optional>

namespace locrec
{

/// Drop on the falling edge, then space by range. Three causal rules.
///
///  1. This step degenerate and the previous step not: drop immediately, one step into the blind
///     stretch. The valuable place to leave a marker is the last place you still know where you
///     are, not fifty metres in, because an anchor inherits the error the estimate had when it
///     was registered.
///  2. Still inside a blind stretch: drop when the robot has travelled twice the detection range
///     since the last anchor, less a margin. Never on the ratio alone. The ratio says whether a
///     stretch needs anchors; the detection range says how far apart they go inside one.
///  3. Above threshold: never drop. The geometry is carrying the estimate, so spend nothing.
///
/// The spacing is twice the detection range rather than once it: an anchor is useful over the
/// whole window in which it can be seen, and the fix that matters is the one that ties the new
/// anchor to the old chain. Spacing at one range instead spent 64 markers on a 300 m blind run,
/// three times uniform 15 m spacing for no measured gain.
///
/// The chain is deliberately not forgotten when a stretch turns localizable again. On a 300 m
/// blind run the ratio crosses the threshold 77 times, so a rule that fires on every crossing
/// pays for chatter in hardware.
class LocalizabilityScheduler
{
public:
  LocalizabilityScheduler(double ratio_threshold, double reliable_range_m, double margin_m = 1.5)
  : ratio_threshold_(ratio_threshold), reliable_range_m_(reliable_range_m), margin_m_(margin_m)
  {
  }

  /// True when a marker should be mounted now.
  bool operator()(double ratio, double estimated_distance)
  {
    const bool degenerate = ratio < ratio_threshold_;
    was_degenerate_ = degenerate;
    if (!degenerate) {
      return false;
    }
    const double reach = std::max(2.0 * reliable_range_m_ - margin_m_, margin_m_);
    if (!last_drop_.has_value() || estimated_distance - *last_drop_ >= reach) {
      last_drop_ = estimated_distance;
      return true;
    }
    return false;
  }

  double ratioThreshold() const {return ratio_threshold_;}
  double reliableRange() const {return reliable_range_m_;}
  double margin() const {return margin_m_;}
  bool wasDegenerate() const {return was_degenerate_;}

private:
  double ratio_threshold_;
  double reliable_range_m_;
  double margin_m_;
  std::optional<double> last_drop_;
  bool was_degenerate_ = false;
};

}  // namespace locrec
