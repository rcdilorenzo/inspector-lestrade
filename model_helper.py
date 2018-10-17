import os
import subprocess
import textwrap
from toolz import memoize
from shared import log
# Architecture imports (for exec operations)
import tensorflow as tf
from funcy import compose, partial
from toolz.curried import *
from keras.optimizers import Adam
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint

def print_banner(message):
    print('')
    print('============================================================')
    print(textwrap.fill(message, 60, replace_whitespace=False))
    print('============================================================')

def read_file(path):
    data = None
    with open(path, 'r') as f:
        data = f.read()
    return data

def find_or_save_architecture_version(version):
    PATH = './architectures/lestrade-v{0}.json'.format(version)
    if os.path.isfile(PATH):
        print_banner('Found previously saved v{}'.format(version))
    else:
        save_architecture_version(version)

    return model_from_json(
        read_file(PATH),
        custom_objects=custom_objects(version))

@memoize
def sha_for_version(version, progress=False):
    if progress:
        print_banner('Searching commit matching "Model v{}-..."'.format(version))

    FIND_SHA = """git log --grep='Model v{0}[- ]' |
    head -n 1 |
    awk '{{ print $2 }}'
    """.format(version)

    sha = subprocess.check_output(FIND_SHA, shell=True).strip().decode('ascii')

    if len(sha) == 0:
        raise ValueError('No model version found for {}.'.format(version))

    if progress:
        print_banner('Found commit {}'.format(sha[0:6]))
    return sha


def train_file_for_version(version, custom_pipe='', progress=False):
    sha = sha_for_version(version, progress)

    if progress:
        print_banner('Loading architecture for v{}'.format(version))

    suffix = '| {}'.format(custom_pipe)

    SHOW_FILE = """git show {0}:02-facial-landmarks/train.py |
    sed -e '/fit_generator/,+1d' |
    sed -e '/compile/,+1d' {1}
    """.format(sha, suffix if len(custom_pipe) > 0 else '')

    return subprocess.check_output(SHOW_FILE, shell=True).decode('utf8')

@memoize
def custom_objects(version):
    exec(train_file_for_version(version, 'sed -n "/_activation(/,/^$/p"'))
    return {k: v for (k, v) in locals().items() if not '__' in k}

def save_architecture_version(version):
    SAVE_VERSION = """with open('./architectures/lestrade-v{0}.json', 'w') as f:
        f.write(model.to_json())
    """.format(version)

    raw_code = train_file_for_version(version, progress=True) + SAVE_VERSION
    code = compile(raw_code, 'train-temp.py', 'exec')
    exec(code, globals(), globals())

    print_banner('Saved v{} to "architectures" folder'.format(version))
