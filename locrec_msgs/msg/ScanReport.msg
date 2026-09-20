# What the simulator knows about one scan besides the cloud and the odometry. The stamp is
# that scan's stamp, and one report is published for every scan, before the scan, empty or
# not, so a consumer can join the two exactly and never has to guess whether one is coming.
std_msgs/Header header

# Markers mounted since the previous scan. Apply these before this scan is registered:
# that is the order the estimator needs them in.
MarkerDrop[] drops

# Markers seen in this scan.
MarkerObservation[] detections

# Heading of the path the vehicle is flying at this scan, radians, in the odometry frame.
# A drone's sensor yaw is free of its direction of travel, and a gaze policy steers the
# sensor relative to this heading, as the study's run_pass gives it Drone.track_yaw().
# NaN for a vehicle whose sensor is locked to its heading.
float64 track_yaw
