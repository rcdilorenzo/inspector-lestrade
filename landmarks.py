from funcy import memoize
from glob import glob
from funcy import rpartial
from toolz import pipe
from toolz.curried import *
import numpy as np
import re
import os
from shared import frame
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ALL_LANDMARKS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), './data/facial-landmarks.npy')
)

BATCHES_PATTERN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), './data/facial-landmarks-batch*.npz')
)

def regex_numeric(value):
    return int(re.search('(\d+).npz', value).group(1))

def from_npz(values, npz_file):
    next_values = npz_file['arr_0']
    npz_file.close()
    return np.append(values, next_values)

@memoize
def landmarks(mmap=None):
    if os.path.exists(ALL_LANDMARKS_PATH):
        return np.load(ALL_LANDMARKS_PATH, mmap_mode=mmap)
    else:
        return pipe(
            BATCHES_PATTERN,
            glob,
            sorted(key = regex_numeric),
            list,
            map(np.load),
            lambda npzs: reduce(from_npz, npzs, np.empty(0)),
            __save_and_return_landmarks
        )

def __save_and_return_landmarks(data):
    np.save(ALL_LANDMARKS_PATH, data)
    return data

def plot_landmarks(row):
    preds = row.at['Landmarks']
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(frame(row))
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.axis('off')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()

def bounding_box(box):
    print('box', box)
    return mpl.patches.Rectangle(
        (box[0], box[2]), box[1]-box[0], box[3]-box[2],
        linewidth=1,
        edgecolor='r',
        facecolor='none')

def show_eyes(row, frame1, frame2, axis=None):
    axis.imshow(frame(row))
    axis.set_axis_off()
    axis.add_patch(bounding_box(frame1))
    axis.add_patch(bounding_box(frame2))
    plt.show()
