#!/usr/bin/env bash
# Screen-record the real windows of a live run, tiled the way an engineer's desktop is: Gazebo and
# rviz across the top, two live rqt_plot windows across the bottom. No titles, no captions,
# nothing drawn over them.
#
#   ros2 run locrec_ros record_windows.sh team 1 300 ~/videos/locrec_team_seed1.mp4 6
#   ros2 run locrec_ros record_windows.sh ugv  1 300 ~/videos/locrec_ugv_seed1.mp4  5
#
# arguments: vehicle seed length output [speedup, default 5]
#
# The plots are rqt_plot on real topics. The localizability ratio is what each estimator publishes.
# The along-track error and, in the team, the drone's distance from the robot's frame are published
# by demo_viewer, which is the only node with the ground truth to compute them from; no estimator
# sees them.
#
# How it works, and why this way. The desktop is GNOME on Wayland, where one app cannot read
# another's pixels. These windows run through XWayland, and ffmpeg's x11grab reads a single X
# window with plain GetImage, which XWayland allows (GStreamer's ximagesrc uses shared memory,
# which XWayland refuses). Each window is captured on its own, at its own size, so none has to be
# placed or kept on top, and they are tiled afterwards. The run is recorded in real time and sped
# up once at the end. locrec_ros/README.md has the five things that each failed once.
set -euo pipefail
vehicle="${1:?vehicle: ugv or team}"
seed="${2:-1}"
length="${3:-300}"
out="${4:?output mp4}"
speed="${5:-5}"
fps=15
mkdir -p "$(dirname "$out")"
here="$(cd "$(dirname "$0")" && pwd)"

case "$vehicle" in
    ugv)
        plot_a=(/demo/error/baseline/data /demo/error/markers/data)
        plot_b=(/baseline/localizability/ratio /markers/localizability/ratio) ;;
    team)
        # rqt_plot colours curves in the order they are added, so solo and team come first in both
        # plots and keep one colour each across them
        plot_a=(/demo/error/solo/data /demo/error/team/data /demo/error/markers/data)
        plot_b=(/demo/frame_gap/solo/data /demo/frame_gap/team/data) ;;
    *) echo "vehicle must be ugv or team"; exit 2 ;;
esac

FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
work=$(mktemp -d /tmp/locrec_windows_XXXX)
log="$work/launch.log"
# A launch of its own on the ROS graph and in Gazebo: a leftover launch from an earlier run would
# otherwise feed this one its /tf and scans (docs/failures.md number 37).
export ROS_DOMAIN_ID=$(( 60 + RANDOM % 30 ))
export GZ_PARTITION="locrec_windows_$$"

descendants() {  # every process under a PID, deepest first
    local kids; kids=$(ps -eo pid=,ppid= | awk -v p="$1" '$2 == p {print $1}')
    for k in $kids; do descendants "$k"; echo "$k"; done
}
stop_tree() {
    # By PID, never by pattern: `pkill -f` matches this script's own command line. And TERM rather
    # than INT, because a background job in a non-interactive shell starts with SIGINT ignored.
    local tree; tree="$(descendants "$1") $1"
    kill -TERM $tree 2>/dev/null || true
    sleep 4
    kill -KILL $tree 2>/dev/null || true
}
cleanup() {
    for p in ${grab_pids:-}; do kill -TERM "$p" 2>/dev/null || true; done
    for p in ${plot_pids:-} ${launch_pid:-}; do stop_tree "$p"; done
}
trap cleanup EXIT

ros2 launch locrec_ros gazebo.launch.py vehicle:="$vehicle" seed:="$seed" length:="$length" \
    gazebo_gui:=true rviz:=true render:=false gazebo_gui_config:=gazebo_gui_half.config \
    rviz_config:="gazebo_${vehicle}_recording.rviz" > "$log" 2>&1 &
launch_pid=$!

window_id() {  # pattern [ids to skip] -> the largest X window whose name matches, or nothing yet
    # The largest, not the newest: Qt opens small helper windows that carry the application's name,
    # and taking the newest match grabbed a 3x3 one. The skip list tells two rqt_plot windows
    # apart, since both have the same name. `|| true`: under pipefail a grep that finds nothing yet
    # is a failure, and set -e would end the script on the first poll.
    xwininfo -root -tree 2>/dev/null | grep -iE "$1" | grep -v "has no name" | awk -v skip=" ${2:-} " '
        { if (index(skip, " " $1 " ")) next
          for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+x[0-9]+[+-]/) {
              split($i, g, /[x+-]/); a = g[1] * g[2]
              if (g[1] >= 300 && g[2] >= 300 && a > best) { best = a; id = $1 } } }
        END { if (id) print id }' || true
}
# Gazebo opens a window, loads its plugins, and replaces it: the first test grabbed the first one,
# which was black and gone 1.5 s later. So a window counts only once the same one has been the
# largest match for STABLE seconds in a row.
STABLE=8
settle() {  # pattern [ids to skip]
    local last="" seen=0 id
    for _ in $(seq 180); do
        id=$(window_id "$1" "${2:-}")
        if [ -n "$id" ] && [ "$id" = "$last" ]; then seen=$((seen + 1)); else seen=0; fi
        last="$id"
        [ "$seen" -ge "$STABLE" ] && { echo "$id"; return 0; }
        sleep 1
    done
    return 0
}
GZ_PAT='"Gazebo Sim'
RVIZ_PAT='RViz'
PLOT_PAT='rqt'

echo "waiting for the Gazebo and rviz windows to settle..."
declare -A win pat pid seg dropped
win[gazebo]=$(settle "$GZ_PAT");  pat[gazebo]="$GZ_PAT"
win[rviz]=$(settle "$RVIZ_PAT");  pat[rviz]="$RVIZ_PAT"

# The plots start once the nodes are up, so the topics they are given already exist and rqt_plot
# can read their types. -e: do not restore whatever an earlier rqt session was plotting.
# QT_QPA_PLATFORM=xcb, as the launch file sets for rviz and Gazebo: left to itself Qt opens a
# native Wayland window, which X cannot see and so nothing can capture.
export QT_QPA_PLATFORM=xcb
# rqt_plot opens at 321x169 and has no geometry option (it rejects Qt's -qwindowgeometry), so each
# plot window is resized once it appears, through locrec_ros/scripts/xresize.py. The two plots share a title, so
# each is found by its client window's class, the second skipping the first, and captured through
# the window manager's frame around it, title bar included, as Gazebo and rviz are.
rqt_client() {  # [client ids to skip] -> an rqt_plot client window, at any size
    xwininfo -root -tree 2>/dev/null | grep '("rqt_plot" "rqt_plot")' \
        | awk -v skip=" ${1:-} " '{ if (!index(skip, " " $1 " ")) { print $1; exit } }' || true
}
frame_of() {  # client window -> the frame the window manager put around it
    xwininfo -tree -id "$1" 2>/dev/null | awk '/Parent window id:/ { print $4; exit }' || true
}
open_plot() {  # skip topics... -> starts rqt_plot, prints "pid client frame"
    local skip="$1"; shift
    # rqt_plot itself, through a launcher that waits for discovery: started plainly it adds its
    # topics before its node has discovered them, and silently plots nothing
    python "$here/rqt_plot_waiting.py" -e "$@" > "$work/plot_$RANDOM.log" 2>&1 &
    local rp=$! client=""
    for _ in $(seq 60); do client=$(rqt_client "$skip"); [ -n "$client" ] && break; sleep 1; done
    [ -n "$client" ] || { echo "$rp"; return 0; }
    sleep 2
    python "$here/xresize.py" "$client" 960 480
    sleep 3
    echo "$rp $client $(frame_of "$client")"
}
read -r pa ca fa <<< "$(open_plot "" "${plot_a[@]}")"
read -r pb cb fb <<< "$(open_plot "${ca:-}" "${plot_b[@]}")"
plot_pids="${pa:-} ${pb:-}"
win[plot_a]="${fa:-}"; pat[plot_a]=""
win[plot_b]="${fb:-}"; pat[plot_b]=""

names=(gazebo rviz plot_a plot_b)
for n in "${names[@]}"; do
    if [ -z "${win[$n]:-}" ]; then
        echo "the $n window never settled; what X had:"
        xwininfo -root -tree 2>/dev/null | grep -iE "gazebo|rviz|rqt" | head -20 || true
        echo "launch log: $log"
        exit 1
    fi
done

# Crop to even dimensions: the compositor sizes the windows as it likes, and a window 1087 px tall
# made x264 refuse to open its encoder, so the Gazebo capture came out empty.
even="crop=trunc(iw/2)*2:trunc(ih/2)*2"
grab() {  # name window segment -> starts ffmpeg in the background, prints its pid
    "$FF" -loglevel error -y -f x11grab -framerate $fps -window_id "$2" -i "$DISPLAY" -vf "$even" \
        -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p "$work/$1_$3.mkv" > "$work/$1_$3.log" 2>&1 &
    echo $!
}
for n in "${names[@]}"; do seg[$n]=0; pid[$n]=$(grab "$n" "${win[$n]}" 0); done
grab_pids="${pid[*]}"
echo "recording gazebo ${win[gazebo]}, rviz ${win[rviz]}, plots ${win[plot_a]} ${win[plot_b]}"

# If a window is replaced mid-run its capture dies; find the window again and carry on in a new
# segment rather than lose the rest of the run. Every restart is logged.
until grep -q "end of tunnel" "$log"; do
    kill -0 "$launch_pid" 2>/dev/null || { echo "the launch ended early; log in $log"; exit 1; }
    for n in "${names[@]}"; do
        kill -0 "${pid[$n]}" 2>/dev/null && continue
        if [ -z "${pat[$n]}" ]; then
            [ -z "${dropped[$n]:-}" ] && { echo "  $n capture ended and is not restarted"; dropped[$n]=1; }
            continue
        fi
        others=""; for m in "${names[@]}"; do [ "$m" != "$n" ] && others="$others ${win[$m]}"; done
        w=$(window_id "${pat[$n]}" "$others")
        [ -n "$w" ] || continue
        seg[$n]=$(( seg[$n] + 1 ))
        echo "  $n capture ended; restarting on window $w as segment ${seg[$n]}"
        win[$n]="$w"; pid[$n]=$(grab "$n" "$w" "${seg[$n]}")
    done
    grab_pids="${pid[*]}"
    sleep 2
done
sleep 4  # the last few frames of the vehicle coming to rest
kill -TERM ${pid[*]} 2>/dev/null || true
# The captures were started inside $(...), so they are not this shell's children and `wait`
# returns at once; poll instead, or the tiling below reads files ffmpeg is still finalising.
for _ in $(seq 60); do
    alive=0; for p in ${pid[*]}; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 0.5
done
grab_pids=""

# Tile: Gazebo and rviz 960x600 across the top, the two plots 960x480 across the bottom. Each
# window's segments are joined, dropping any that recorded nothing, and fitted into its tile
# without stretching.
declare -A tile_w=( [gazebo]=960 [rviz]=960 [plot_a]=960 [plot_b]=960 )
declare -A tile_h=( [gazebo]=600 [rviz]=600 [plot_a]=480 [plot_b]=480 )
inputs=(); chains=""; idx=0
for n in "${names[@]}"; do
    parts=""; k=0; W=${tile_w[$n]}; H=${tile_h[$n]}
    for f in "$work/${n}"_*.mkv; do
        [ -s "$f" ] && "$FF" -v error -i "$f" -f null - 2>/dev/null || continue
        inputs+=(-i "$f")
        chains+="[$idx:v]scale=$W:$H:force_original_aspect_ratio=decrease,pad=$W:$H:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[${n}$idx];"
        parts+="[${n}$idx]"; idx=$((idx + 1)); k=$((k + 1))
    done
    [ "$k" -gt 0 ] || { echo "no usable $n capture; raw files in $work"; exit 1; }
    chains+="${parts}concat=n=$k:v=1:a=0[$n];"
done
"$FF" -loglevel error -y "${inputs[@]}" -filter_complex "${chains}\
[gazebo][rviz]hstack=inputs=2[top];[plot_a][plot_b]hstack=inputs=2[bottom];\
[top][bottom]vstack=inputs=2,setpts=PTS/$speed,fps=30[v]" \
    -map "[v]" -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart "$out"
"$FF" -loglevel error -y -sseof -3 -i "$out" -frames:v 1 "${out%.mp4}_last.png" || true
echo "wrote $out ($(du -h "$out" | cut -f1)); raw captures kept in $work"
