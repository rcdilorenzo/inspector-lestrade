from funcy import partial, compose, identity, memoize, curry
import os, contextlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time

YOLO_DIR = os.path.expanduser('~/workspaces/data/yolov3')
FOLDER_DIR = os.path.expanduser('~/Documents/data/gazecapture')
OUT_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), './eye-gaze-capture.csv')
)

@curry
def prop(key, value):
    return value[key]

def log(x):
    print(x)
    return x

def time_this(f, pre_f = identity, post_f = identity, value = None):
    value_initial = pre_f(value)
    start = time()
    value_main = f(value_initial)
    duration = time() - start
    return (duration, post_f(value_main))

def time_this_f(f, pre_f = identity, post_f = identity):
    return lambda x: time_this(f, pre_f, post_f, value = x)

# https://stackoverflow.com/a/28321717/2740693
def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                result = func(*a, **ka)
        return result
    return wrapper

def load_df():
    return pd.read_csv(OUT_CSV)

@memoize
def capture_df():
    return load_df()

frame = compose(plt.imread, partial(os.path.join, FOLDER_DIR),
                lambda row: row['Frame'])

# Visualizations

def hist3d_data(x_values, y_values):
    min_value = min(np.min(x_values), np.min(y_values))
    max_value = max(np.max(x_values), np.max(y_values))
    print('min', min_value)
    print('max', max_value)

    # Mostly from
    hist, xedges, yedges = np.histogram2d(x_values, y_values, bins=20,
                                          range=[[min_value, max_value], [min_value, max_value]])

    # Construct arrays for the anchor positions of the bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
    # with indexing='ij'.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing='ij')
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 10 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    return ((xpos, ypos, zpos), (dx, dy, dz))
