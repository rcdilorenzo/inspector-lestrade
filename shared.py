from funcy import partial, compose, identity
import os, contextlib
import matplotlib.pyplot as plt
from time import time

YOLO_DIR = os.path.expanduser('~/workspaces/data/yolov3')
FOLDER_DIR = os.path.expanduser('~/Documents/data/gazecapture')
OUT_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), './eye-gaze-capture.csv')
)

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

frame = compose(plt.imread, partial(os.path.join, FOLDER_DIR),
                lambda row: row['Frame'])