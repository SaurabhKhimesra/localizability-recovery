# locrec_core

The algorithm, in C++17 with Eigen and [small_gicp](https://github.com/koide3/small_gicp). No ROS
dependency, so it builds, runs and is unit tested on its own.

| header | what it holds |
|---|---|
| `se3.hpp` | poses, rotation logs and exponentials, and the positive-semidefinite repair used all over the relative formulation |
| `lidar.hpp` | sensor geometry: beam spacing, range, and the sampling limit that caps the registration range |
| `registration.hpp` | scan-to-map registration through small_gicp, and the 6x6 Hessian it returns |
| `localizability.hpp` | the eigen-analysis of that Hessian: the ratio, the weak direction, and whether the scan is degenerate |
| `landmarks.hpp` | what one observation of a retroreflective strip is worth, the geometric fit of the strip's mounting point, and the Monte Carlo that measures the fit's bias and spread |
| `odometry.hpp` | the estimator: the local map, the motion prior's information, the two-stage step, the marker fix and the distance-scale state |
| `policies.hpp` | the marker scheduler: drop on the falling edge, then space by range |
| `gaze.hpp` | where a drone points its sensor, forward or glancing back at structure |

## Build

```bash
colcon build --packages-select locrec_core
```

small_gicp has no rosdep key. The build fetches the pinned release (v1.0.1) at configure time and
compiles the two translation units it needs straight into this library, so nothing extra has to be
installed at runtime. For an offline build, point it at a checkout instead:

```bash
colcon build --packages-select locrec_core \
  --cmake-args -DSMALL_GICP_SOURCE_DIR=/path/to/small_gicp
```

## Tests

```bash
colcon test --packages-select locrec_core && colcon test-result --all
```

One of them is load bearing beyond its own package:
`test_localizability.cpp::a_corridor_is_degenerate_along_its_own_axis` registers a featureless
corridor against itself and asserts the weak direction lands in the translational block along the
corridor. small_gicp does not document the layout of its Hessian, and everything downstream reads
that block.
