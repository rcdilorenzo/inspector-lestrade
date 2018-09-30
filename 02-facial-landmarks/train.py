import os
import sys

# Setup visible GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import keras.backend as K

# Configure tensorflow options
K.clear_session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Library imports
from funcy import compose
from toolz.curried import *
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint

# Import shared from directory above
sys.path.insert(0, '../')
from shared import capture_df, frame

# Import local files
from landmarks import landmarks
from generator import InspectNNGenerator, SET_TYPE_TEST, SET_TYPE_TRAIN, SET_TYPE_VALIDATION


# ========================================
# Data and Scaling Transformations
# ========================================

# Filter for rows with landmarks
has_landmarks = lambda x: x.shape[0] > 0

# Create data frame and add landmarks as column
df = capture_df()
df['Landmarks'] = landmarks()

# Filter to only rows that have landmarks
df = df[df.Landmarks.apply(has_landmarks)]

# Data Format
#   Inputs:  [left_eyes, right_eyes, landmarks]
#   Outputs: [XCam, YCam] (centimeters relative position to lens)
generator = InspectNNGenerator(session, df, 8, set_type=SET_TYPE_TRAIN)
val_generator = InspectNNGenerator(session, df, 8, set_type=SET_TYPE_VALIDATION)

print('item', generator.__getitem__(1))

# ========================================
# Loss Function
# ========================================

def euclidean_distance_mse(actual, pred):
    x = actual[:, 0]
    y = actual[:, 1]
    x_hat = pred[:, 0]
    y_hat = pred[:, 1]

    distance = K.sqrt(K.square(x_hat - x) - K.square(y_hat - y))
    return K.mean(K.square(distance), axis=-1)

# ========================================
# Callbacks
# ========================================

board = TensorBoard(log_dir='./logs')

checkpoint = ModelCheckpoint('./initial.weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss')

# ========================================
# Model Layers
# ========================================

left_eye_input = Input(shape=(128,128,3))
right_eye_input = Input(shape=(128,128,3))
landmark_input = Input(shape=(68,3))

left_path = pipe(
    left_eye_input,
    Conv2D(16, (3, 3), activation='relu'),
    Conv2D(8, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dense(8, activation='relu'),
    Flatten()
)

right_path = pipe(
    right_eye_input,
    Conv2D(16, (3, 3), activation='relu'),
    Conv2D(8, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dense(8, activation='relu'),
    Flatten()
)

landmarks = pipe(
    landmark_input,
    Dense(64, activation='relu'),
    Dense(8, activation='relu'),
    Flatten()
)

coordinate = pipe(
    concatenate([left_path, right_path, landmarks]),
    Dense(16, activation='relu'),
    Dense(2, activation='sigmoid', name='coord_output')
)

model = Model(inputs=[left_eye_input, right_eye_input, landmark_input],
              outputs=[coordinate])

print('model', model.summary())

model.compile(optimizer='Adam', loss=euclidean_distance_mse)

model.fit_generator(generator, validation_data=val_generator,
                    callbacks=[board, checkpoint], epochs=100)

