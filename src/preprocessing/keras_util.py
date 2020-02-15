from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils


def get_preprocess_for_model(model_name):
    if model_name == 'inception' or model_name == 'xception':
        return preprocess_input
    return imagenet_utils.preprocess_input
