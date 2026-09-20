# Real data: the next validation step

Every result in this repository is from simulation. The next step is a public sequence, and the
tooling for it is written. The selection rule was the smallest public sequence with LiDAR plus
IMU, ground-truth poses, and a straight low-texture section of at least 30 m.

## The choice

**Hilti SLAM Challenge 2022, Exp14 Basement 2.** 6 GB, the smallest sequence in
that dataset carrying 6DoF ground truth, indoors, a basement. Hesai PandarXT-32
LiDAR plus IMU, distributed as rosbags, no registration form, mirrored on Hugging
Face. Licence CC BY-NC-SA 3.0, non-commercial with attribution, which is
compatible with a portfolio repository as long as the data is not redistributed.

Second choice if the basement has no 30 m straight run: **Exp18 Corridor Gallery
2**, 8 GB, also 6DoF ground truth, and a corridor by name.

Rejected: Newer College is behind an access form and is an outdoor college quad,
so it is both gated and textured. Exp07 Long Corridor is 11 GB but carries only
3DoF ground truth, which cannot measure along-track error in three dimensions.
SubT has the right environment and is far too large for one night.

Both choices are under 20 GB. The analysis has not been run yet: it needs the bag downloaded to a
machine with the disk for it.

## What is already written

`locrec/experiments/real_sequence.py` is complete and does not need ROS: it reads ROS 1
and ROS 2 bags with `rosbags`, which is pure Python, and reuses the same
PointCloud2 conversion the ROS node uses, so the format handling is the code the
unit tests already cover.

It produces the ratio over distance with drift-growth segments shaded, the ROC of
"detector fires" against "along-track error grows over the next five seconds", and
the lead-time distribution.

## Run it

```bash
pip install rosbags
# download Exp14 Basement 2 from https://hilti-challenge.com/dataset-2022
python locrec/experiments/real_sequence.py ~/data/exp14_basement_2.bag \
    --lidar-topic /hesai/pandar \
    --gt-topic /gt_pose \
    --range 40 \
    --out docs/
```

Run it once with `--max-scans 200` first. It prints the available topics if
neither name matches, which is the usual first failure with someone else's bag.

## What would make this a negative result

The detector firing where drift does not grow, or drift growing where it does not
fire. Both are worth reporting. The number to watch is the AUC: at 0.5 the ratio
carries no information about where this sequence goes wrong, and the honest
conclusion would be that the simulation's degeneracy is cleaner than a real
basement's.
