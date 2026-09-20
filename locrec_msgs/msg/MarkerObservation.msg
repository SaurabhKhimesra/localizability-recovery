# One marker seen in one scan, as the simulator's retroreflector detector reports it.
int32 slot

# Fitted mounting point of the strip, in the sensor frame of the scan.
geometry_msgs/Point point_sensor

# Returns that landed on the strip. One is enough: a beam is a range and a bearing.
int32 n_beams
float64 range_m

# Distinct azimuths among those returns. The rings in one column share an azimuth, so
# this, not n_beams, is how many independent looks the strip got across the line of
# sight, which is the direction along the tunnel that the odometry cannot see.
int32 n_columns

# How much of the strip's width the returns covered, metres. NaN when the strip was
# not fitted, in which case the measurement model cannot charge for foreshortening.
float64 seen_width_m
