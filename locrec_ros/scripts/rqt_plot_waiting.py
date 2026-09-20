"""rqt_plot, unchanged, except that it waits for its topics to be discovered before adding them.

    python locrec_ros/scripts/rqt_plot_waiting.py -e /demo/error/markers/data /markers/localizability/ratio

rqt_plot adds the topics it is given while its widget is being built, by asking the ROS graph for
each topic's type (rqt_plot.plot_widget.get_plot_fields). Its node was created a moment earlier and
DDS discovery has not reached the other nodes yet, so every topic "does not exist", nothing is
added, and nothing is logged either: add_topic loops over an empty list. Both plot windows in the
first recording came up empty that way. Discovery runs on DDS's own threads and fills the graph
cache without the node being spun, so waiting in that one call is enough.
"""
import sys
import time

import rqt_plot.plot_widget as plot_widget

_get_plot_fields = plot_widget.get_plot_fields
WAIT_S = 15.0
NOT_YET = ("does not exist", "no topic types")


def get_plot_fields_once_discovered(node, topic_name):
    t0 = time.monotonic()
    while True:
        fields, message = _get_plot_fields(node, topic_name)
        waited = time.monotonic() - t0
        if fields or waited > WAIT_S or not any(s in message for s in NOT_YET):
            if fields and waited > 0.05:
                print(f"rqt_plot: {topic_name} discovered after {waited:.1f} s", file=sys.stderr)
            elif not fields:
                print(f"rqt_plot: {topic_name}: {message}", file=sys.stderr)
            return fields, message
        time.sleep(0.2)


plot_widget.get_plot_fields = get_plot_fields_once_discovered

# The second thing an engineer does with rqt_plot before recording: show the whole run. The Plot
# plugin switches x autoscaling off (rqt_plot/plot.py) and autoscrolls with a window as wide as the
# axis was at startup, which is the 0 to 1 s fallback, so it shows the last second forever and every
# curve looks flat. Autoscaling x instead keeps the axis growing from the start of the run to now,
# which is what the toolbar's "home" button gives. The y axis is left as rqt_plot sets it: it only
# ever extends to fit the data, so an error that grows is seen to grow.
import rqt_plot.plot as plot_plugin  # noqa: E402

_plot_init = plot_plugin.Plot.__init__


def _plot_init_whole_run(self, context):
    _plot_init(self, context)
    self._data_plot.set_autoscale(x=True)


plot_plugin.Plot.__init__ = _plot_init_whole_run

if __name__ == "__main__":
    # the name Qt takes the window's class from, so the window is still recognisably rqt_plot
    sys.argv[0] = "rqt_plot"
    from rqt_plot.main import main

    main()
