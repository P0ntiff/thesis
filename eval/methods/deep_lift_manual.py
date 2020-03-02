import os
import tensorflow as tf
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from ..util.image_util import get_preprocess_for_model, ImageHandler

# high level wrapper for DeepLIFT
# TODO: replace with direct implementation


# suppress output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def attribute(model, ih: ImageHandler):

    print(model.summary())
    # TODO: implement
