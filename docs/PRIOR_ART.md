# Prior art: what exists, and what this actually claims

Searched 2026-09-18, before any claim of novelty is written anywhere. Written because the
robot-team result was commissioned as "something which hasn't been done yet", and the honest
answer is that most of it has, separately. What follows is what was found and what it leaves.

## The closest work found

**Leap-SLAM**, Information Fusion 2026,
[ScienceDirect S1566253526004173](https://www.sciencedirect.com/science/article/abs/pii/S1566253526004173).
Two robots in a GNSS-denied tunnel. Degeneracy indices built from the information matrix drive a
"leapfrogging" collaboration in which a Beacon Robot carrying reflective markers gives a Master
Robot external constraints where the geometry is feature-starved. This is the same problem, the
same class of degeneracy metric, reflective markers, and a two-robot tunnel team. Anyone claiming
the locrec team result is new has to say how it differs from this paper first. Two differences
are real: their beacon robot **carries** its markers and drives with them, so nothing is left
behind and the constraint exists only while both robots are in the same place; and their beacon
robot carried single-antenna RTK GNSS in the reported experiments, which is external positioning
this study does not have.

**Deployable, Data-Driven Unmanned Vehicle Navigation System in GPS-Denied, Feature-Deficient
Environments**, [arXiv 2101.09750](https://arxiv.org/pdf/2101.09750). A vehicle carries ranging
beacons and drops them as it goes, with a learned policy choosing where, then localizes against
the ranges. This is landmark deployment triggered by the state of localization, which is the UGV
half of locrec, with UWB ranging in place of passive LiDAR strips and one vehicle rather than two.

**Integrated Air-Ground Robotic System for Autonomous Post-Blast Operations in GNSS-Denied
Tunnels**, [Remote Sensing 18(8) 1133](https://www.mdpi.com/2072-4292/18/8/1133). A UAV and a
ground machine share a coordinate system in a tunnel, and the UAV's localization is initialized
from reflective markers. The markers are placed, not deployed by the ground vehicle mid-mission.

## The rest of the field, by element

* **Degeneracy and localizability metrics for LiDAR odometry** are a crowded area, and every one
  found adapts the *estimator* rather than the *environment*: LP-ICP
  ([arXiv 2501.02580](https://arxiv.org/html/2501.02580)), LF-GICP
  ([arXiv 2608.19522](https://arxiv.org/html/2608.19522)), DARE-SLAM
  ([arXiv 2102.05117](https://arxiv.org/pdf/2102.05117)), and the equivariance condition of
  [arXiv 2608.15532](https://arxiv.org/html/2608.15532), which is worth reading against this
  study's own ratio.
* **Fiducial and retroreflective landmarks in SLAM** are standard, and so is optimising where to
  put them ([arXiv 2211.01513](https://arxiv.org/pdf/2211.01513)). In all of it the markers are
  placed beforehand, by people or by a planner with a map.
* **Cooperative localization in a shared or inherited frame** is standard: LAMP 2.0
  ([arXiv 2205.13135](https://arxiv.org/pdf/2205.13135)), Swarm-SLAM
  ([arXiv 2301.06230](https://arxiv.org/pdf/2301.06230)), and marsupial UGV-UAV teams from
  DARPA SubT ([arXiv 2111.06482](https://arxiv.org/pdf/2111.06482)).

## What is left, stated narrowly

Not found in the search: a system where a **localizability metric decides the moment to mount a
passive landmark that is then left behind permanently**, and a **second vehicle that carries no
markers of its own** localizes against those left-behind landmarks in the first vehicle's frame.
The combination is what is unclaimed. Every ingredient has prior art, and the nearest neighbour,
Leap-SLAM, differs mainly in that its markers travel with a robot instead of staying on the wall.

That is the most that may be written. "First", "never been done" and "unprecedented" are not
supported by this search and are not to be used. A search of ICRA, IROS and RA-L for 2025 and
2026 by hand would be the next step before any submission, since the web search above reaches
abstracts and not proceedings indexes.
