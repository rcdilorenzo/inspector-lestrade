import os
import sys
sys.path.insert(0, '02-facial-landmarks/')

import tensorflow as tf
import face_alignment
import keras.backend as K
from keras.optimizers import Adam

# Import local files
from model_helper import *
import facial_model as fm
from weights import weights_for, weights_count, weight_stats
from shared import capture_df, frame
from landmarks import landmarks
from generator import InspectorLestradeGenerator, SET_TYPE_TEST
from generator import find_or_create_scaler, scale_landmarks


# ========================================
# Loss Function (from train.py)
# ========================================

# (1/b) ∑^b_(i=1)([(^G-G)^2 + 1]^2 * [(^y - y)^2 + (^x - x)^2])
def loss_func(actual, pred):
    x_diff = tf.square(pred[:, 0] - actual[:, 0])
    y_diff = tf.square(pred[:, 1] - actual[:, 1])
    g_diff = tf.square(pred[:, 2] - actual[:, 2])

    return K.mean(tf.square(g_diff + 1) * (y_diff + x_diff))

# Mean Euclidean distance for all values where both eyes visible
def scaled_distance_loss(actual, pred):
    combined = tf.concat([actual, pred], axis=1)
    masked = tf.boolean_mask(combined, tf.math.equal(combined[:, 2], 1.0))
    x_diff = tf.square(masked[:, 0] - masked[:, 3])
    y_diff = tf.square(masked[:, 1] - masked[:, 4])
    return K.mean(tf.sqrt(x_diff + y_diff))


# ========================================
# Constants / Helper functions
# ========================================

def default_session():
    K.clear_session()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(session)
    return session

# Note: To auto-download a custom weight, simply replace/add to the data in this constant.
URLS = {
    10: {
        'url': 'https://github.com/rcdilorenzo/msds-686-eye-tracking/releases/download/v0.0.1/v10-2incep-dense-custom-act.01-63.85.hdf5',
        'path': '02-facial-landmarks/models/v10-2incep-dense-custom-act.01-63.85.hdf5',
    },
    11: {
        'url': 'https://github.com/rcdilorenzo/msds-686-eye-tracking/releases/download/v0.0.1/v11-fixed+modified-activation.01-1803.19.hdf5',
        'path': '02-facial-landmarks/models/v11-fixed+modified-activation.01-1803.19.hdf5'
    },
    12: {
        'url': 'https://github.com/rcdilorenzo/msds-686-eye-tracking/releases/download/v0.0.1/v12-1e2-shortened-activation.01-1803.19.hdf5',
        'path': '02-facial-landmarks/models/v12-1e2-shortened-activation.01-1803.19.hdf5'
    }
}

class InspectorLestrade:
    def __init__(self, version, session=default_session(),
                 load_weights=True, weights_rank=1, download_allowed=False):

        self.session = session
        self.version = version
        self.__download_model(download_allowed)
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

    def __download_model(self, allowed_by_option):
        model_data = URLS.get(self.version)

        if model_data is None or os.path.exists(model_data['path']):
            return

        if not allowed_by_option:
            print('Model available for download but must be specifically requested.')
            print('Use download_allowed=True to enable.')
            return

        import wget
        from tqdm.auto import tqdm
        with tqdm() as t:
            wget.download(model_data['url'], out=model_data['path'], bar=(lambda progress, total, _: t.update(progress / total)))

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

    def evaluate(self, workers=1, use_scaled_distance=False):
        loss_f = scaled_distance_loss if use_scaled_distance else loss_func
        self.model.compile(optimizer=Adam(lr=1e2), loss=loss_f)

        has_landmarks = lambda x: x.shape[0] > 0
        df = capture_df()
        df['Landmarks'] = landmarks()
        df = df[df.Landmarks.apply(has_landmarks)]

        generator = InspectorLestradeGenerator(
            self.session, df, 32,
            set_type=SET_TYPE_TEST
        )
        return self.model.evaluate_generator(generator, workers=workers, verbose=1)




