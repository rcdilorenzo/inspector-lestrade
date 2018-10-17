import pandas as pd
import os
import re
from toolz import memoize

# This weights script assumes the following file format:
#
#   v[<int>]-[name-separated-by-dashes].[step<int>]-[val_loss<float>].hdf5
#
# Example:
#
#   v6-1e1-linear-dense-lm.34-27.78.hdf5

def extract_loss(filename):
    m = re.search(r'(\d+)\.(\d+).hdf5', filename)
    return float(m.group(1) + '.' + m.group(2))

def extract_step(filename):
    return int(re.search(r'\.(\d+)-', filename).group(1))

def extract_version(filename):
    return int(re.search(r'^v(\d+)', filename).group(1))

def extract_model_val_loss(filename):
    return float(str(extract_version(filename)) + str(extract_loss(filename)))

def weight_stats(filename):
    return {
        'val_loss': extract_loss(filename),
        'step': extract_step(filename),
        'model': extract_version(filename),
    }

@memoize
def weights():
    files = os.listdir(os.path.abspath(
        os.path.join(os.path.dirname(__file__), './models')
    ))

    return pd.DataFrame({
        'filename': files,
        'model': list(map(extract_version, files)),
        'step': list(map(extract_step, files)),
        'val_loss': list(map(extract_loss, files)),
        'unique_loss': list(map(extract_model_val_loss, files))
    }).sort_values(['model', 'val_loss', 'step'])

def weights_count(model_version):
    df = weights()
    return df[df.model == model_version].shape[0]

def weights_for(model_version, rank=1):
    df = weights()
    available_weights = df[df.model == model_version].sort_values('val_loss')
    index = min(max(1, rank), available_weights.shape[0]) - 1

    return available_weights.iloc[index].at['filename']
