// Two-stage estimation: a relative frontend and an absolute correction.
//
// Stage A, relative. The motion prior predicts the pose, the new scan is placed there, small_gicp
// registers it against the local map, and the correction is blended with the prior by their
// information matrices. The 6x6 Hessian of that registration is the localizability signal. This
// stage is self-consistent and it drifts: the map is built from the estimate, so if the estimate
// slides along a tunnel the map slides with it and the registration stays perfectly satisfied.
//
// Stage B, absolute. When a marker is observed, the correction comes from the predicted minus the
// measured landmark position, with the gain taken from a covariance integrated since the last
// absolute fix and reset on each one. The correction is applied to the pose and rigidly to the
// local map and to every landmark registered since the last fix. That last part is not a detail:
// the registration constraint is relative, so shifting pose and map together leaves it satisfied,
// while shifting the pose alone means the next registration drags it straight back.
//
// Nothing here knows about tunnels or recovery motions. The scheduling lives in policies.hpp.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "locrec_core/landmarks.hpp"
#include "locrec_core/lidar.hpp"
#include "locrec_core/localizability.hpp"
#include "locrec_core/registration.hpp"
#include "locrec_core/se3.hpp"

namespace locrec
{

/// A dead-reckoning prior with the error structure of IMU + wheel odometry. Units are physical,
/// not per-step, so the prior does not silently change when the scan spacing changes. The
/// defaults are chosen from hardware, not from results.
struct MotionPriorSpec
{
  /// 1 sigma relative error on translated distance, constant per run. This is the error the
  /// tunnel axis cannot observe away, and the parameter the whole study is about.
  double odom_scale_sigma = 0.02;
  /// 1 sigma random-walk translation noise, metres per sqrt(metre) travelled.
  double odom_noise_m_per_sqrt_m = 0.01;
  /// 1 sigma residual gyro bias, rad/s: what is left after a LiDAR-inertial system has estimated
  /// the in-run bias of a MEMS gyro (0.05 deg/s). Assuming raw uncorrected bias would make the
  /// prior a straw man.
  double gyro_bias_rad_s = 8.7e-4;
  /// 1 sigma white gyro noise as a rate, rad/s, about 0.3 deg/sqrt(hr) at the scan rate.
  double gyro_noise_rad_s = 3e-4;
  /// Floor on the per-step translation sigma, metres. A random walk per sqrt(metre) goes to zero
  /// information as the step goes to zero, which would let a stationary platform claim a perfect
  /// prior and veto the registration outright.
  double odom_min_sigma_m = 2e-3;
};

/// 6x6 information of the one-step prior error, ordered [rot, trans], in the world frame.
///
/// Computed from the increment the prior actually produced, not from a nominal step length. The
/// translation block is anisotropic: the scale error acts only along the direction of travel, the
/// random walk on all three axes, so in the body frame the covariance is
/// sigma_rw^2 I + (scale_sigma * |d|)^2 u u^T. Treating it as isotropic would either over-trust
/// the along-track axis or under-trust the lateral ones, and the whole study turns on telling
/// those apart.
Matrix6d priorInformation(
  const MotionPriorSpec & spec, double dt, const Eigen::Isometry3d & delta_T_prior,
  const std::optional<Eigen::Matrix3d> & R_world_body = std::nullopt);

/// Pack a 3-D voxel index into one integer. The range is plus or minus 2^20 voxels, which at a
/// 0.2 m pitch is plus or minus 200 km.
int64_t voxelKey(const Eigen::Vector3d & point, double resolution);

/// Deterministic voxel-grid downsample: one centroid per occupied voxel.
Points voxelDownsample(const Points & points, double resolution);

/// Voxel-hashed local map in the estimator's world frame, cropped to a radius.
///
/// The crop radius must exceed the sensor range, otherwise the live scan sticks out past the ends
/// of the map and the only axial correspondences left are at the map's boundary: registration
/// then pulls every new scan back toward the map centroid and the estimate stops advancing while
/// the robot keeps moving. That failure is silent and looks like excellent convergence
/// (docs/failures.md number 1).
///
/// A voxel is offered for registration only once min_observations separate scans have landed in
/// it. At the far end of a long-range narrow-FOV scan a single sweep leaves beams further apart
/// than the voxel pitch, so a map built from one viewpoint is full of one-hit voxels whose
/// nearest neighbour is in the wrong place.
class LocalMap
{
public:
  LocalMap(
    double radius = 30.0, double resolution = 0.2, int max_points = 80000,
    int min_observations = 2);

  void add(const Points & points_world, const std::optional<Eigen::Vector3d> & centre = std::nullopt);

  /// Move the whole map rigidly, by moving its frame.
  ///
  /// An absolute fix moves the pose, and the registration constraint is relative, so moving the
  /// map by the same rigid transform leaves it exactly as satisfied as it was. The shift is
  /// applied to the grid's frame rather than to the points, so no rebinning happens: rebinning
  /// looked harmless and was not, because every pass snapped the points to fresh centroids.
  void applyTransform(const Eigen::Matrix3d & R, const Eigen::Vector3d & t);

  /// Confirmed voxel centroids: those seen by at least min_observations scans.
  Points points() const;
  std::size_t size() const;
  std::size_t nVoxels() const {return voxels_.size();}

private:
  struct Voxel
  {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int64_t count = 0;
  };

  Eigen::Vector3d toMap(const Eigen::Vector3d & p) const {return R_.transpose() * (p - t_);}

  double radius_;
  double resolution_;
  int max_points_;
  int min_observations_;
  std::unordered_map<int64_t, Voxel> voxels_;
  // the voxel grid lives in its own frame, and an absolute fix moves that frame rather than the
  // points, so the grid is never rebinned
  Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_ = Eigen::Vector3d::Zero();
};

struct OdometryConfig
{
  double downsampling_resolution = 0.2;
  double max_correspondence_distance = 1.0;
  int max_iterations = 30;
  int num_threads = 4;
  std::string registration_type = "GICP";
  /// Crop radius of the local map, metres. Must exceed the LiDAR max range.
  double map_radius = 30.0;
  double map_resolution = 0.2;
  /// Scans that must have hit a voxel before it is used for registration.
  int map_min_observations = 2;
  /// Only scan points within this range of the sensor are used for registration; the full scan
  /// still goes into the map.
  ///
  /// This is not an optimisation. A scan reaches the sensor's full range, but the map's forward
  /// boundary sits one step behind that, so the leading shell of every scan has no map support.
  /// Those points find their nearest correspondence behind themselves and drag the solution back
  /// by a few centimetres every step, which in a straight tunnel compounds into a systematic lag
  /// of ten percent of distance travelled while the registration reports healthy convergence.
  std::optional<double> registration_range = std::nullopt;
  /// Below this many returns the registration is skipped and the prior is trusted.
  int min_scan_points = 60;
  /// Fuse the registration correction with the zero-mean one-step prior using both information
  /// matrices. This is the baseline behaviour, not a recovery policy: accepting the raw GICP
  /// correction along a direction the Hessian says is unobserved is a known way to make odometry
  /// worse than dead reckoning, and a baseline that does that would be a straw man.
  bool fuse_prior = true;
  /// Rescale the GICP Hessian by (inliers - 6) / residual before fusing, the usual Gauss-Newton
  /// covariance estimate with an unknown noise scale.
  bool calibrate_information = true;
  /// Absolute guard: zero the registration's translational information along directions whose
  /// eigenvalue ratio falls below this. The registration's along-track Hessian claims a 4 to 5 mm
  /// standard deviation in a tunnel with no along-track structure at all, which is an artefact of
  /// treating a thousand correlated correspondences as independent evidence, and it must not be
  /// allowed to outvote an absolute measurement. Calibrated, not tuned.
  std::optional<double> gicp_ratio_floor = 1.9e-3;
  /// Weight a marker fix by the odometry accumulated since that marker was dropped, rather than
  /// by the marker's absolute covariance. A dropped marker does not know where it is in the
  /// world; what it knows is where the robot was when it went on the wall.
  bool relative_anchors = true;
  /// Let two or more anchors in view correct heading as well as position.
  bool yaw_fix = true;
  /// Estimate the odometry's distance scale error as a state. The prior's scale error is one
  /// number per run, not a new draw per step, and modelling it as white noise is what stops a
  /// marker chain from behaving like a survey traverse.
  bool estimate_scale = true;
  /// Standard deviation the scale estimate may wander by, per metre.
  double scale_random_walk_per_m = 1e-4;
  /// Subtract the strip fit's expected bias from every marker observation. Off by default on the
  /// evidence: it is sensor-specific and only one sensor was measured (docs/failures.md 34).
  bool correct_strip_bias = false;
  /// Take the spread of a marker observation from the same Monte Carlo as its bias, rather than
  /// from a hand-written beam-quantisation term (docs/failures.md 31).
  bool measure_strip_spread = true;
  /// Subtract what the registration observed from the odometry a marker fix is weighted against.
  bool credit_registration = false;
  /// Re-register an anchor whenever the pose is better known than the anchor is. Off: in practice
  /// it re-registers an anchor from an estimate that is itself drifting.
  bool resurvey_anchors = false;
  /// Stage B: apply marker observations as an absolute correction after the relative stage.
  bool absolute_fix = true;
  /// Hard version of the guard: project the correction out of weak directions. Off by default;
  /// the soft fusion supersedes it.
  bool remap_degenerate = false;
  double remap_eigenvalue_floor = 0.0;
};

struct OdomStep
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  ///< estimated sensor pose
  Eigen::Isometry3d T_prior = Eigen::Isometry3d::Identity();
  std::optional<Localizability> localizability;
  bool converged = false;
  int num_inliers = 0;
  double error = std::numeric_limits<double>::quiet_NaN();
  int n_scan_points = 0;
  bool registered = false;
  int remapped_axes = 0;
  int landmarks_used = 0;
  int yaw_fixed = 0;
};

/// One record per absolute fix: what it moved, and what weighted it. Instrumentation rather than
/// state: a fix that is silently small, or in the wrong direction, looks exactly like a fix that
/// worked from outside the estimator.
struct FixRecord
{
  int step = 0;
  int n_terms = 0;
  std::vector<int> slots;
  double dx = 0.0;
  double dy = 0.0;
  double dpsi = 0.0;
  double along = 0.0;
  double residual_along = 0.0;
  double range_m = 0.0;
  double q_since_along = 0.0;
  double r_along = 0.0;
  double gain_along = 0.0;
  bool used_yaw = false;
};

class Odometry
{
public:
  Odometry(
    const Eigen::Isometry3d & T0, const OdometryConfig & cfg = OdometryConfig(),
    const MotionPriorSpec & prior_spec = MotionPriorSpec(), double dt = 0.5,
    std::optional<LidarSpec> lidar_spec = std::nullopt,
    const LandmarkSpec & landmark_spec = LandmarkSpec(), double marker_height = 1.0,
    double marker_width = 0.15, double marker_thickness = 0.0);

  /// Record where the robot believes it just bolted a marker. The offset is in the sensor frame
  /// at the moment of the drop, so the landmark inherits exactly the pose error the estimate had
  /// then, and no more.
  void registerLandmark(int slot, const Eigen::Vector3d & offset_sensor);

  /// Record a marker another robot mounted, in that robot's frame. This is the whole of what one
  /// robot has to tell another: where the strip is, how well the first robot knew that when it
  /// bolted it on, and which way the face points. The position is taken as exact, because this
  /// robot is localizing in the other robot's frame and there the strip's coordinates are the
  /// definition of it.
  void registerForeignLandmark(
    int slot, const Eigen::Vector3d & position_world, const Eigen::Matrix2d & R_drop,
    const Eigen::Vector2d & normal);

  OdomStep step(
    const Points & scan_points_sensor, const Eigen::Isometry3d & delta_T_prior,
    const std::vector<MarkerDetection> & marker_observations = {});

  const OdometryConfig & config() const {return cfg_;}
  Eigen::Isometry3d & pose() {return T_;}
  const Eigen::Isometry3d & pose() const {return T_;}
  LocalMap & map() {return map_;}
  const LocalMap & map() const {return map_;}
  LandmarkBook & landmarks() {return landmarks_;}
  const LandmarkBook & landmarks() const {return landmarks_;}
  const LandmarkSpec & landmarkSpec() const {return landmark_spec_;}
  const Matrix6d & covariance() const {return P_;}
  double scale() const {return scale_;}
  double scaleVariance() const {return scale_var_;}
  double travelled() const {return travelled_;}
  int nFixes() const {return n_fixes_;}
  int nSteps() const {return n_steps_;}
  const std::vector<FixRecord> & fixLog() const {return fix_log_;}
  void resetMap();

private:
  struct AnchorTerm
  {
    int slot = 0;
    Eigen::Vector3d q = Eigen::Vector3d::Zero();       ///< where the observed marker is now
    Eigen::Vector3d anchor = Eigen::Vector3d::Zero();  ///< where the anchor's record says it is
    Eigen::Matrix2d R_obs = Eigen::Matrix2d::Identity();  ///< drop-time plus current observation
    Eigen::Matrix3d Q_since = Eigen::Matrix3d::Zero();    ///< pose covariance against this anchor
  };

  Eigen::Matrix2d dropCovariance(const Eigen::Vector3d & offset_sensor) const;
  Eigen::Matrix3d relativeCovariance(const Landmark & lm) const;
  std::pair<Eigen::Vector3d, std::optional<std::pair<double, double>>> stripFit(
    const Landmark & lm, const Eigen::Vector3d & sensor_position,
    const MarkerDetection * obs) const;
  std::vector<AnchorTerm> landmarkTerms(
    const Eigen::Isometry3d & T, const std::vector<MarkerDetection> & observations);
  void propagateCovariance(
    const Matrix6d & prior_info, const Eigen::Vector3d & translation,
    const Matrix6d * registration_info);
  int resurvey(const std::vector<MarkerDetection> & observations);
  std::pair<int, int> absoluteFix(const std::vector<MarkerDetection> & observations);
  void updateScale(double leg, double residual_along, double q_along, double r_along);

  OdometryConfig cfg_;
  MotionPriorSpec prior_spec_;
  double dt_;
  std::optional<LidarSpec> lidar_spec_;
  LandmarkSpec landmark_spec_;
  double marker_height_;
  double marker_width_;
  double marker_thickness_;

  Eigen::Isometry3d T_;
  LocalMap map_;
  LandmarkBook landmarks_;
  /// Covariance of the pose error accumulated since the last absolute fix, in the twist frame
  /// centred on the current sensor position.
  Matrix6d P_;
  /// Running sum of the per-step prior covariance, in the world frame. A difference of two marks
  /// on this is the odometry accumulated between them, which is what weights every marker fix. It
  /// is the prior only: a scan-to-map registration is relative to a map the estimate built and
  /// cannot certify how far the robot has come since a marker went down.
  Matrix6d Q_cum_ = Matrix6d::Zero();
  /// Covariance of the pose against the shared frame at this robot's last fix in it.
  Eigen::Matrix3d frame_rel_ = Eigen::Matrix3d::Zero();
  /// Accumulated odometry covariance at this robot's last fix in the shared frame. A marker this
  /// robot mounted marks Q_cum itself at the drop, because the drop is a fix; a marker another
  /// robot mounted cannot, because this robot was somewhere else when it went on.
  Matrix6d frame_q_ref_ = Matrix6d::Zero();
  double scale_ = 1.0;
  double scale_var_;
  double travelled_ = 0.0;
  std::map<int, double> travelled_ref_;
  std::vector<int> since_fix_;
  int n_fixes_ = 0;
  int n_steps_ = 0;
  std::vector<FixRecord> fix_log_;
};

}  // namespace locrec
