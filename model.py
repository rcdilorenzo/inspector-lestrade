import os
import sys
sys.path.insert(0, '02-facial-landmarks/')

import tensorflow as tf
import face_alignment
import keras.backend as K

# Import local files
from model_helper import *
import facial_model as fm
from weights import weights_for, weights_count, weight_stats
from shared import capture_df, frame
from landmarks import landmarks
from generator import InspectorLestradeGenerator, SET_TYPE_TEST
from generator import find_or_create_scaler, scale_landmarks

def default_session():
    K.clear_session()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(session)
    return session

class InspectorLestrade:
    def __init__(self, version, session=default_session(),
                 load_weights=True, weights_rank=1):

        self.session = session
        self.version = version
        self.model = find_or_save_architecture_version(version)

        print_banner('Using InspectorLestrade v{}'.format(version))

        if load_weights:
            self.load_weights(weights_rank)

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                  enable_cuda=False, flip_input=False)

        # Landmark scaler
        self.landmark_scaler = find_or_create_scaler(None)

        # Image preprocessing
        (predictions, image, factor, left_eye, right_eye,
         _l, _r, gaze_likelihood) = fm.eyes_tensor()

        self.predictions_input = predictions
        self.image_input = image
        self.factor_input = factor
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.gaze_likelihood = gaze_likelihood

    def load_weights(self, rank=1):
        if weights_count(self.version) == 0:
            raise ValueError('No weights exist for v{}.'.format(self.version))

        filename = weights_for(self.version, rank)
        self.model.load_weights('02-facial-landmarks/models/' + filename)

        print_banner('Loaded ' + filename + '\n  ' +
                     str(weight_stats(filename)))

    def predict(self, image, print_errors=True):
        facial_features = self.fa.get_landmarks(image)

        if facial_features is None:
            if print_errors:
                print('Error: No facial features detected.')
            return None

        facial_features = np.array(facial_features)

        eye_ops = (self.left_eye, self.right_eye)
        data = {
            self.predictions_input: facial_features[0, :, 0:2],
            self.image_input: image,
            self.factor_input: 1.5
        }
        (left_eye, right_eye) = self.session.run(eye_ops, feed_dict=data)

        (x, y, likelihood) = self.model.predict_on_batch([
            np.array([left_eye]),
            np.array([right_eye]),
            scale_landmarks(self.landmark_scaler, facial_features)
        ])[0]

        print('Prediction - point: ({}, {}), likelihood: {}%'.format(
              x, y, likelihood * 100))

        return (x, y, likelihood)




