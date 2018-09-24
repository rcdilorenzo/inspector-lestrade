import os
import sys
from funcy import first, second, curry
import matplotlib as mpl
from random import sample as rand_subset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Add path to keras-yolo implementation
sys.path.append(os.path.abspath('./keras-yolo3'))
import train
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

# Add path to shared file
sys.path.insert(0, '../')
from shared import *
import openimages as oi

# Setup GPU Options
K.clear_session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

# Setup constants

## Primary adjustables
MAX_LINES = 1000000
VERSION_NAME = 'v5-full-adam1e3-train10'
BATCH_SIZE = 32
BATCH_SIZE_FINE_TUNE = 8
FINE_TUNE_TRAIN = True
PHASE_ONE_EPOCHS = 100
PHASE_TWO_EPOCHS = 75

## Other constants
YOLO_ANCHORS_FILE = './annotations-yolo-format.txt'
YOLO_FILE = os.path.join(YOLO_DIR, 'yolov3-320.h5')
CLASSES_FILE = './classes-yolo-format.txt'
ANCHORS_FILE = './keras-yolo3/model_data/yolo_anchors.txt'
TEMP_MODEL_FORMAT = 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
LOSS_F = {'yolo_loss': lambda y_true, y_pred: y_pred}
INPUT_SHAPE = (416, 416) # multiple of 32, hw

## Derived constants
MODEL_DIR = './models/' + VERSION_NAME
LOG_DIR = './logs/' + VERSION_NAME + '/'
TRAINED_BASE_FILE = os.path.join(MODEL_DIR, 'trained_weights_base.h5')
TRAINED_FINAL_FILE = os.path.join(MODEL_DIR, 'trained_weights_final.h5')

# Create directories
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


# Setup Model
class_names = train.get_classes(CLASSES_FILE)
num_classes = len(class_names)
anchors = train.get_anchors(ANCHORS_FILE)

logging = TensorBoard(log_dir=LOG_DIR)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, TEMP_MODEL_FORMAT),
    monitor='val_loss', save_weights_only=True,
    save_best_only=True, period=3)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=10, verbose=1)

val_split = 0.1
with open(YOLO_ANCHORS_FILE) as f:
    lines = f.readlines()

## Shuffle
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)

## Cap at maximum
lines = lines[0:MAX_LINES]
print('Total records: ', len(lines))

num_val = int(len(lines) * val_split)
num_train = len(lines) - num_val
train_steps = lambda size: max(1, num_train // size)
val_steps = lambda size: max(1, num_val // size)

def data_generator(subset, size):
    return train.data_generator_wrapper(subset, size, INPUT_SHAPE,
                                        anchors, num_classes)


# First-phase training (freeze most of layers)

model = train.create_model(INPUT_SHAPE, anchors, num_classes,
                           freeze_body=2, weights_path=YOLO_FILE)

## Unfreeze top ten layers
for i in range(len(model.layers) - 10, len(model.layers)):
    model.layers[i].trainable = True

model.compile(optimizer=Adam(lr=1e-3), loss=LOSS_F)

print('Train on {} samples, val on {} samples, with batch size {}.'.format(
    num_train, num_val, BATCH_SIZE))

model.fit_generator(
    data_generator(lines[:num_train], BATCH_SIZE),
    steps_per_epoch=train_steps(BATCH_SIZE),
    validation_data=data_generator(lines[num_train:], BATCH_SIZE),
    validation_steps=val_steps(BATCH_SIZE),
    epochs=PHASE_ONE_EPOCHS,
    initial_epoch=0,
    callbacks=[logging, checkpoint]
)

model.save_weights(TRAINED_BASE_FILE)

# Second-phase training
if FINE_TUNE_TRAIN:
    model = train.create_model(INPUT_SHAPE, anchors, num_classes,
                               weights_path=TRAINED_BASE_FILE)

    # Unfreeze
    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    model.compile(optimizer=Adam(lr=1e-4), loss=LOSS_F)
    print('Unfreeze all of the layers.')

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(
        num_train, num_val, BATCH_SIZE_FINE_TUNE))

    model.fit_generator(
        data_generator(lines[:num_train], BATCH_SIZE_FINE_TUNE),
        steps_per_epoch=train_steps(BATCH_SIZE_FINE_TUNE),
        validation_data=data_generator(lines[num_train:], BATCH_SIZE_FINE_TUNE),
        validation_steps=val_steps(BATCH_SIZE_FINE_TUNE),
        epochs=PHASE_ONE_EPOCHS + PHASE_TWO_EPOCHS,
        initial_epoch=PHASE_ONE_EPOCHS,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping]
    )

    model.save_weights(TRAINED_FINAL_FILE)
