import subprocess
import os
import sys
sys.path.insert(0, '02-facial-landmarks/')

# Library imports
import tensorflow as tf
from funcy import compose, partial
from toolz.curried import *
from keras.optimizers import Adam
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint
import textwrap

# Import local files
from weights import weights_for, weights_count
from shared import capture_df, frame
from landmarks import landmarks
from generator import InspectNNGenerator, SET_TYPE_TEST, SET_TYPE_TRAIN, SET_TYPE_VALIDATION

# ====================
# Helper Functions
# ====================

def print_banner(message):
    print('')
    print('==================================================')
    print(textwrap.fill(message, 50))
    print('==================================================')

def read_file(path):
    data = None
    with open(path, 'r') as f:
        data = f.read()
    return data

def find_or_save_architecture_version(version):
    PATH = './architectures/lestrade-v{0}.json'.format(version)
    if os.path.isfile(PATH):
        print_banner('Found previously saved v{}'.format(version))
        return model_from_json(read_file(PATH))
    else:
        save_architecture_version(version)
        return model_from_json(read_file(PATH))

def save_architecture_version(version):
    print_banner('Searching commit matching "Model v{}-..."'.format(version))

    FIND_SHA = """git log --grep='Model v{0}-' |
    head -n 1 |
    awk '{{ print $2 }}'
    """.format(version)

    sha = subprocess.check_output(FIND_SHA, shell=True).strip().decode('ascii')

    if len(sha) == 0:
        raise ValueError('No model version found for {}.'.format(version))

    print_banner('Found commit {}'.format(sha[0:6]))

    print_banner('Loading architecture for v{}'.format(version))

    SHOW_FILE = """git show {0}:02-facial-landmarks/train.py |
    sed -e '/fit_generator/,+1d'
    """.format(sha)

    code = subprocess.check_output(SHOW_FILE, shell=True).decode('utf8')
    print(len(code))

    SAVE_VERSION = """with open('./architectures/lestrade-v{0}.json', 'w') as f:
        f.write(model.to_json())
    """.format(version)

    exec(code + SAVE_VERSION)

    print_banner('Saved v{} to "architectures" folder'.format(version))


# ====================
# Model Class
# ====================

class InspectorLestrade:
    def __init__(self, version, load_weights=True, weights_rank=1):
        self.version = version
        self.model = find_or_save_architecture_version(version)

        print_banner('Using InspectorLestrade v{}'.format(version))

        if load_weights:
            self.load_weights(weights_rank)

    def load_weights(self, rank=1):
        if weights_count(self.version) == 0:
            raise ValueError('No weights exist for v{}.'.format(self.version))

        weights_path = '02-facial-landmarks/' + weights_for(self.version, rank)
        self.model.load_weights(weights_path)

        print_banner('Loaded ' + weights_path)

