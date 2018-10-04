import os
import sys

# Setup visible GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import keras.backend as K

# Configure tensorflow options
K.clear_session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

from shared import sample_df, frame
from landmarks import landmarks
from generator import InspectNNGenerator
from generator import SET_TYPE_TEST, SET_TYPE_TRAIN, SET_TYPE_VALIDATION

import numpy as np

# =============================================================================
# Data and Scaling Transformations
#   (See face-landmarks-sample.ipynb for details on how this was created)
# =============================================================================

BOTH_EYES_IDX = 0
ONE_EYE_IDX   = 1

# Load sample data frame and add landmarks as column
df = sample_df()
df['Landmarks'] = np.load('02-facial-landmarks/sample_landmarks.npy')

generator = InspectNNGenerator(session, df, 2, set_type=SET_TYPE_TRAIN)

# Remove randomization
generator.data_frame = df

inputs, outputs = generator.__getitem__(0)

# =============================================================================
# Generator Tests
# =============================================================================

def test_input_size():
    assert len(inputs) == 3

def test_input_row_counts_match():
    assert list(map(len, inputs)) == [2, 2, 2]

def test_eye_inputs():
    left_eyes = inputs[0]
    right_eyes = inputs[1]

    assert left_eyes[BOTH_EYES_IDX].shape == (128, 128, 3)
    assert left_eyes[ONE_EYE_IDX].shape == (128, 128, 3)

    assert right_eyes[BOTH_EYES_IDX].shape == (128, 128, 3)
    assert right_eyes[ONE_EYE_IDX].shape == (128, 128, 3)

def test_landmark_inputs():
    landmarks = inputs[2]

    assert landmarks.shape == (2, 68, 3)

def test_output_shape():
    assert outputs.shape == (2, 3)

def test_gaze_likelihood_output():
    # 1 = both eyes visible
    # 0 = one or both eyes missing
    assert outputs[BOTH_EYES_IDX, 2] == 1.0
    assert outputs[ONE_EYE_IDX, 2] == 0.0

