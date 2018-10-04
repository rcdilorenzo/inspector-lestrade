from funcy import memoize
import tensorflow as tf

IMAGE_SIZE = (128, 128)
LEFT = 42
RIGHT = 36

def eye_tensor(image, predictions, factor, index = LEFT):
    eye_points = predictions[index:(index + 6), :]
    x = eye_points[:, 0]
    y = eye_points[:, 1]

    # Image dimensions
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    # Find bounding box
    min_x_raw, max_x_raw = tf.reduce_min(x), tf.reduce_max(x)
    min_y_raw, max_y_raw = tf.reduce_min(y), tf.reduce_max(y)

    # Expand by factor and reform as square
    width = tf.to_float(max_x_raw - min_x_raw)
    height = tf.to_float(max_y_raw - min_y_raw)
    sq_size = tf.to_int32(tf.round(tf.reduce_max([width * factor, height * factor])))

    # Compute deltas
    width_delta = tf.to_int32(tf.round((tf.to_float(sq_size) - width) / 2))
    height_delta = tf.to_int32(tf.round((tf.to_float(sq_size) - height) / 2))

    # Pre-compute max_x and max_y
    max_x = max_x_raw + width_delta
    max_y = max_y_raw + height_delta

    # Calculate whether eye visible within image
    both_eyes_visible = tf.logical_and(max_x < image_width, max_y < image_height)

    # Update frame based on delta (but with min/max boundaries)
    max_x = tf.reduce_min([tf.reduce_max([max_x, 1]), tf.shape(image)[1]])
    min_x = tf.reduce_max([tf.reduce_min([min_x_raw - width_delta, max_x - 1]), 0])
    max_y = tf.reduce_min([tf.reduce_max([max_y, 1]), tf.shape(image)[0]])
    min_y = tf.reduce_max([tf.reduce_min([min_y_raw - height_delta, max_y - 1]), 0])

    # Create image and scale to (128, 128)
    unscaled_shape = tf.stack([max_y - min_y, max_x - min_x, tf.constant(3)])
    eye = tf.reshape(image[min_y:max_y, min_x:max_x] / 255, unscaled_shape)
    scaled_eye = tf.image.resize_images(eye, IMAGE_SIZE)

    # Return original bounding box and resized image
    return (scaled_eye, (min_x, max_x, min_y, max_y), both_eyes_visible)

@memoize
def eyes_tensor():
    predictions = tf.placeholder(tf.int32, shape=(68, 2))
    image = tf.placeholder(tf.int32)
    factor = tf.placeholder_with_default(tf.constant(1.5), shape=())

    (left_eye, left_box, left_eye_visible) = eye_tensor(
        image, predictions, factor, LEFT)
    (right_eye, right_box, right_eye_visible) = eye_tensor(
        image, predictions, factor, RIGHT)

    gaze_likelihood = tf.to_float(tf.logical_and(left_eye_visible, right_eye_visible))

    return (predictions, image, factor, left_eye, right_eye, left_box, right_box, gaze_likelihood)
