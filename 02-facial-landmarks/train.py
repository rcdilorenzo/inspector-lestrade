import os
import sys

# Setup visible GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import keras.backend as K

# Configure tensorflow options
K.clear_session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Library imports
from funcy import compose
from toolz.curried import *

# Import shared from directory above
sys.path.insert(0, '../')
from shared import capture_df, frame

# Import local files
from landmarks import landmarks
from generator import InspectNNGenerator, SET_TYPE_TEST, SET_TYPE_TRAIN, SET_TYPE_VALIDATION


# ========================================
# Setup data and scaling transformations
# ========================================

# Filter for rows with landmarks
has_landmarks = lambda x: x.shape[0] > 0

# Create data frame and add landmarks as column
df = capture_df()
df['Landmarks'] = landmarks()

# Filter to only rows that have landmarks
df = df[df.Landmarks.apply(has_landmarks)]

generator = InspectNNGenerator(session, df, 8, set_type=SET_TYPE_TRAIN)

shapes = compose(list, map(lambda x: x.shape))

batch1 = generator.__getitem__(1)
