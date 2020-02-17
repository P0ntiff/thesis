import os
import tensorflow as tf
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from ..util.image_util import get_preprocess_for_model

# high level wrapper for DeepLIFT
# TODO: replace with direct implementation


# suppress output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def attribute(model_name, model, img_path, output_img_path):
    preprocess = get_preprocess_for_model(model_name)

    img_size = (224, 224)
    if model_name == 'inception' or model_name == 'xception':
        img_size = (299, 299)

    input_img = load_img(img_path, target_size=img_size)
    input_img = img_to_array(input_img)
    expanded_img = np.expand_dims(input_img, axis=0)
    preprocessed_img = preprocess(expanded_img)

    print(model.summary())
    # TODO: implement
