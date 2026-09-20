"""Ask the window manager to resize an X window: xresize.py <window id> <width> <height>.

What wmctrl and xdotool do, without either, through libX11. rqt_plot opens at 321x169 and takes no
geometry option of its own (it parses its arguments with argparse and rejects Qt's
-qwindowgeometry), so locrec_ros/scripts/record_windows.sh resizes it after it appears. The request goes to
mutter as an ordinary ConfigureRequest, which it honours for XWayland clients.
"""
import ctypes
import sys

x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_uint, ctypes.c_uint]
x11.XResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint, ctypes.c_uint]
x11.XFlush.argtypes = [ctypes.c_void_p]
x11.XCloseDisplay.argtypes = [ctypes.c_void_p]

win, width, height = int(sys.argv[1], 16), int(sys.argv[2]), int(sys.argv[3])
display = x11.XOpenDisplay(None)
if not display:
    sys.exit("cannot open the X display")
x11.XResizeWindow(display, win, width, height)
x11.XFlush(display)
x11.XCloseDisplay(display)
