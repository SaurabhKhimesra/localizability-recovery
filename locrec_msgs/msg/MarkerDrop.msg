# A marker the robot has just mounted on the wall beside it.
int32 slot

# Stamp of the scan at whose pose the marker was mounted. The estimator registers
# the landmark from the estimate it holds for exactly that scan.
builtin_interfaces/Time pose_stamp

# Mounting point in the sensor frame at that pose. The robot measures the wall with
# one sideways ray, so the offset it believes it used is the offset it gets.
geometry_msgs/Point offset_sensor

# Ground truth, for display only. The estimator never reads it.
geometry_msgs/Point position_world
