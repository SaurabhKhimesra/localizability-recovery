# Localizability of the latest scan, read off the registration Hessian.
std_msgs/Header header

# Eigenvalues of the translational Hessian block, ascending.
float64[3] eigenvalues

# Unit vector of the least observable translation direction, in the frame named
# by header.frame_id. This is the direction the estimate is free to slide along.
float64[3] weak_direction

# eigenvalues[0] / eigenvalues[2]. Scale free, so it is the primary signal.
float64 ratio

# eigenvalues[0] divided by the number of correspondences. Absolute guard against
# the degenerate-but-tiny-scan case where the ratio looks fine only because every
# direction is equally badly observed.
float64 lambda_min_per_point

bool is_degenerate
int32 n_points

# Known markers that corrected the estimate on this scan. Zero unless the node was
# started with use_markers and a marker it registered is in view.
int32 landmarks_used
