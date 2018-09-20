from funcy import partial, compose, identity, memoize, flatten
from toolz.curried import *
from toolz import first, second, curry
import os
import pandas as pd

OPEN_IMAGES_DIR = os.path.expanduser('/media/rcdilorenzo/datahouse/openimagesv4')

TRAIN_DIRS = ['train_00', 'train_01', 'train_02', 'train_03',
              'train_04', 'train_05', 'train_06', 'train_07',
              'train_08']

ALL_LABELS = []

oi_data_frame = compose(pd.read_csv, partial(os.path.join, OPEN_IMAGES_DIR))
sans_ext = lambda f: f[:-4]
ls = compose(os.listdir, partial(os.path.join, OPEN_IMAGES_DIR))
image_ids_from = compose(list, map(sans_ext), flatten, map(ls))

IMAGE_PATHS_BY_FOLDER = {k: ls(k) for k in TRAIN_DIRS + ['test', 'validation']}.items()

@memoize
def class_desc():
    return pd.read_csv(os.path.join(OPEN_IMAGES_DIR, 'class-descriptions-boxable.csv'),
                       header=0, names=['LabelName', 'Description'])

EYE_LABEL = class_desc()[class_desc().Description == 'Human eye'].iloc[0].at['LabelName']

def oi_filtered_df(name, ids, labels = ALL_LABELS):
    df = oi_data_frame(name)
    if labels == ALL_LABELS:
        return df[df.ImageID.isin(ids)]
    else:
        return df[(df.ImageID.isin(ids)) & (df.LabelName.isin(labels))]

# Annotation bounding boxes (train set)
@memoize
def train_ann_bbox(labels = [EYE_LABEL]):
    return oi_filtered_df('train-annotations-bbox.csv',
                          train_image_ids(), labels)

# Label w/ confidence by image id (train set)
@memoize
def train_ann_label_box(labels = [EYE_LABEL]):
    return oi_filtered_df('train-annotations-human-imagelabels-boxable.csv',
                          train_image_ids(), labels)

# Image name with URL (train set)
@memoize
def train_img_box():
    return oi_data_frame('train-images-boxable.csv')

# Image metadata (e.g. copyright, origin, author, rotation amount) (train set)
@memoize
def train_img_box_rot():
    return oi_data_frame('train-images-boxable-with-rotation.csv')

# Annotation bounding boxes (validation set)
@memoize
def val_ann_bbox(labels = [EYE_LABEL]):
    return oi_filtered_df('validation-annotations-bbox.csv',
                          val_image_ids(), labels)

# Label w/ confidence by image id (validation set)
@memoize
def val_ann_label_box(labels = [EYE_LABEL]):
    return oi_filtered_df('validation-annotations-human-imagelabels-boxable.csv',
                          val_image_ids(), labels)

# Image metadata (e.g. copyright, origin, author, rotation amount) (validation set)
@memoize
def val_img_rot():
    return oi_data_frame('validation-images-with-rotation.csv')

# Annotation bounding boxes (test set)
@memoize
def test_ann_bbox(labels = [EYE_LABEL]):
    return oi_filtered_df('test-annotations-bbox.csv', test_image_ids(), labels)

# Label w/ confidence by image id (test set)
@memoize
def test_ann_label_box(labels = [EYE_LABEL]):
    return oi_filtered_df('test-annotations-human-imagelabels-boxable.csv',
                          test_image_ids(), labels)

# Image metadata (e.g. copyright, origin, author, rotation amount) (test set)
@memoize
def test_img_rot():
    return oi_data_frame('test-images-with-rotation.csv')

@memoize
def train_image_ids():
    return image_ids_from(TRAIN_DIRS)

@memoize
def test_image_ids():
    return image_ids_from(['test'])

@memoize
def val_image_ids():
    return image_ids_from(['validation'])

def image_path(row):
    filename = row.at['ImageID'] + '.jpg'
    
    folder = pipe(IMAGE_PATHS_BY_FOLDER,
                  filter(value_includes(filename)),
                  first, first)
    return os.path.join(OPEN_IMAGES_DIR, folder, filename)

# General Helpers

@curry
def value_includes(value, kv_pair):
    return value in second(kv_pair)
