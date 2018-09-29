from funcy import memoize
import tensorflow as tf

LEFT = 42
RIGHT = 36

def eye_tensor(image, predictions, factor, index = LEFT):
    eye_points = predictions[index:(index + 6), :]
    x = eye_points[:, 0]
    y = eye_points[:, 1]

    # Find bounding box
    min_x_raw, max_x_raw = tf.reduce_min(x), tf.reduce_max(x)
    min_y_raw, max_y_raw = tf.reduce_min(y), tf.reduce_max(y)

    # Expand by factor
    width = tf.to_float(max_x_raw - min_x_raw)
    width_delta = tf.to_int32(tf.round((width * factor - width) / 2))
    height = tf.to_float(max_y_raw - min_y_raw)
    height_delta = tf.to_int32(tf.round((height * factor - height) / 2))
    min_x = min_x_raw - width_delta
    max_x = max_x_raw + width_delta
    min_y = min_y_raw - height_delta
    max_y = max_y_raw + height_delta

    eye = image[min_y:max_y, min_x:max_x]
    return (eye, (min_x, max_x, min_y, max_y))

@memoize
def eyes_tensor():
    predictions = tf.placeholder(tf.int32, shape=(68, 2))
    image = tf.placeholder(tf.int32)
    factor = tf.placeholder_with_default(tf.constant(1.5), shape=())

    (left_eye, left_box) = eye_tensor(
        image, predictions, factor, LEFT)
    (right_eye, right_box) = eye_tensor(
        image, predictions, factor, RIGHT)

    return (predictions, image, factor, left_eye, right_eye, left_box, right_box)
