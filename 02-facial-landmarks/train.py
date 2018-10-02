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
from funcy import compose, partial
from toolz.curried import *
from keras.optimizers import Adam
from keras.layers import *
from keras.layers.normalization import BatchNormalization
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
val_generator = InspectNNGenerator(session, df, 8, max_size=2000, set_type=SET_TYPE_VALIDATION)

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

NAME = 'inception6nodes_few_dense'

board = TensorBoard(log_dir='./logs/' + NAME)

checkpoint = ModelCheckpoint('./models/' + NAME + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_loss')

# ========================================
# Model Layers
# ========================================

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
        BatchNormalization(),
        Flatten(name=(prefix + '_flttn'))
    )


left_path = eye_path(left_eye_input, prefix='left')
right_path = eye_path(right_eye_input, prefix='right')

landmarks = pipe(
    landmark_input,
    Dense(16, activation='relu'),
    Dropout(0.25),
    Dense(16, activation='relu'),
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

model.compile(optimizer=Adam(lr=1e3), loss=euclidean_distance_mse)

model.fit_generator(generator, validation_data=val_generator, steps_per_epoch=5000,
                    callbacks=[board, checkpoint], epochs=100)

