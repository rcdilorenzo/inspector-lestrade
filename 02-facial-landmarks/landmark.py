from toolz import curry
from funcy import compose, partial
from os.path import join, expanduser
import matplotlib as mpl
import matplotlib.pyplot as plt
import face_alignment
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import pandas as pd
import sys
import pickle
import numpy as np
from time import time
import progressbar

FOLDER_DIR = '~/Documents/data/gazecapture'
OUT_CSV = '../eye-gaze-capture.csv'
SIZE = 100000

capture_df = pd.read_csv(OUT_CSV)

@curry
def col(key, row):
    return row[key]

from_dir = partial(join, expanduser(FOLDER_DIR))

frame = compose(plt.imread, from_dir, col('Frame'))

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                  enable_cuda=True, flip_input=False)

landmarks = []
batch = int(sys.argv[1])

start_index = (batch - 1) * SIZE
end_index = min(batch * SIZE, capture_df.shape[0])

print('start', start_index)
print('end', end_index)

with progressbar.ProgressBar(max_value=(end_index - start_index), redirect_stdout=True) as bar:

    for i, row in capture_df[start_index:end_index].iterrows():
        raw_features = fa.get_landmarks(frame(row))
        features = np.empty(0) if raw_features is None else raw_features[-1]
        landmarks.append(features)

        if (i + 1) % 1000 == 0:
            print('Saving {} features'.format(len(landmarks)))
            np.savez('./facial-landmarks-batch-{}.npz'.format(batch), landmarks)

        bar.update(i - start_index)

print('Saving {} features'.format(len(landmarks)))
np.savez('./facial-landmarks-batch-{}.npz'.format(batch), landmarks)
