import os
import sys

# Setup visible GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import keras.backend as K

# Configure tensorflow options
K.clear_session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Library imports
from funcy import compose, partial
from toolz.curried import *
from keras.optimizers import Adam
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint

# Import local files
from shared import capture_df, frame
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
BATCH_SIZE = 128
generator = InspectNNGenerator(session, df, BATCH_SIZE, set_type=SET_TYPE_TRAIN)
val_generator = InspectNNGenerator(session, df, BATCH_SIZE, set_type=SET_TYPE_VALIDATION)

# ========================================
# Loss Function
# ========================================

# (1/b) ∑^b_(i=1)([(^G-G)^2 + 1]^2 * [(^y - y)^2 + (^x - x)^2])
def loss_func(actual, pred):
    x_diff = tf.square(pred[:, 0] - actual[:, 0])
    y_diff = tf.square(pred[:, 1] - actual[:, 1])
    g_diff = tf.square(pred[:, 2] - actual[:, 2])

    return K.mean(tf.square(g_diff + 1) * (y_diff + x_diff))

# ========================================
# Callbacks
# ========================================

NAME = 'v11-fixed+modified-activation'

board = TensorBoard(log_dir='./logs/' + NAME)

checkpoint = ModelCheckpoint('./models/' + NAME + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_loss')

# ========================================
# Model Layers
# ========================================

def coordinate_activation(x):
    return x / (K.abs(x / 30) + 1)

left_eye_input = Input(shape=(128,128,3))
right_eye_input = Input(shape=(128,128,3))
landmark_input = Input(shape=(68,3))

def applicative(input_layer, layers):
    return list(map(lambda f: f(input_layer), layers))

@curry
def inception_module(prefix, count, input_layer):
    return concatenate(applicative(input_layer, [
        Conv2D(count, (1, 1), activation='relu',
               padding='same', name=(prefix + '_conv1x1')),
        compose(
            Conv2D(count, (3, 3), activation='relu',
                   padding='same', name=(prefix + '_conv3x3')),
            Conv2D(count, (1, 1), activation='relu',
                   padding='same', name=(prefix + '_conv1x1pre3x3'))
        ),
        compose(
            Conv2D(count, (3, 3), activation='relu',
                   padding='same', name=(prefix + '_conv5x5')),
            Conv2D(count, (1, 1), activation='relu',
                   padding='same', name=(prefix + '_conv1x1pre5x5'))
        ),
        compose(
            Conv2D(count, (1, 1), activation='relu',
                   padding='same', name=(prefix + '_conv1x1_postpool')),
            MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                         padding='same', name=(prefix + '_max')),
        )
    ]), axis=3)

def eye_path(input_layer, prefix='na'):
    return pipe(
        input_layer,
        inception_module(prefix + '_mni1', 6),
        inception_module(prefix + '_mni2', 6),
        Flatten(name=(prefix + '_flttn'))
    )


left_path = eye_path(left_eye_input, prefix='left')
right_path = eye_path(right_eye_input, prefix='right')

landmarks = pipe(
    landmark_input,
    Dense(16, activation='linear'),
    BatchNormalization(),
    Dense(16, activation='linear'),
    Dense(8, activation='linear'),
    Flatten()
)

grouped = concatenate([left_path, right_path, landmarks])

coordinate = pipe(
    grouped,
    Dense(64, activation='linear'),
    Dense(32, activation='linear'),
    Dense(2, activation=coordinate_activation, name='coord_output')
)

gaze_likelihood = pipe(
    grouped,
    Dense(8, activation='relu'),
    BatchNormalization(),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid', name='gaze_likelihood')
)

output = pipe(
    concatenate([coordinate, gaze_likelihood]),
)

model = Model(inputs=[left_eye_input, right_eye_input, landmark_input],
              outputs=[output])

print('model', model.summary())

model.compile(optimizer=Adam(lr=1e1), loss=loss_func)

model.fit_generator(generator, validation_data=val_generator,
                    callbacks=[board, checkpoint], epochs=100)

