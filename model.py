import os
import sys
sys.path.insert(0, '02-facial-landmarks/')

# Import local files
from model_helper import *
from weights import weights_for, weights_count, weight_stats
from shared import capture_df, frame
from landmarks import landmarks
from generator import InspectNNGenerator, SET_TYPE_TEST, SET_TYPE_TRAIN, SET_TYPE_VALIDATION

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

        filename = weights_for(self.version, rank)
        self.model.load_weights('02-facial-landmarks/models/' + filename)

        print_banner('Loaded ' + filename + '\n  ' +
                     str(weight_stats(filename)))

