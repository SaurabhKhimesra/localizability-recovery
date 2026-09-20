# One strip a vehicle mounted, told to the rest of the team.
#
# This is the whole of it. Not a map, not a trajectory, not the sender's covariance over its own
# run: a receiver localizes IN THE SENDER'S FRAME, and in that frame the strip's coordinates are
# the definition of the frame rather than something to be uncertain about. Weighting a fix by the
# sender's absolute covariance would make a strip 250 m into a tunnel look worthless exactly where
# it is the only thing on offer.
std_msgs/Header header

# Which slot, so the receiver's detector can associate what it sees with what it was told.
int32 slot

# Where the sender believes the strip is, in the sender's frame.
geometry_msgs/Point position

# Row-major 2x2 horizontal covariance of the observation that placed it: the floor on how well
# any later fix can place a vehicle relative to this strip.
float64[4] drop_covariance

# Horizontal outward normal of the strip's face, in the sender's frame.
#
# Carried rather than inferred. The end face a strip shows is the one toward the sensor, so the
# fit reads it short on the approach and long once it is passed, and the correction along the wall
# flips sign with it. A vehicle that guessed the facing from its own pose, the way it may for a
# strip it mounted itself, would apply that correction backwards.
geometry_msgs/Vector3 normal
