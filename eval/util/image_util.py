import numpy as np

from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras import backend as K

from ..util.constants import RESULTS_BASE_PATH, IMG_BASE_PATH
from ..util.imagenet_annotator import get_image_file_name


def get_preprocess_for_model(model_name):
    """ Gets a different preprocessing function based on the requirement from the model"""
    if model_name == 'inception' or model_name == 'xception':
        return preprocess_input
    return imagenet_utils.preprocess_input


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.common.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


class ImageHandler:
    # For feeding into model architectures
    STD_IMG_SIZE = (224, 224)
    NONSTD_IMG_SIZE = (299, 299)

    def __init__(self, img_no: int, model_name: str):
        self.img_no = img_no
        self.model_name = model_name
        if self.model_name == 'inception' or self.model_name == 'xception':
            self.size = self.NONSTD_IMG_SIZE
        else:
            self.size = self.STD_IMG_SIZE
        self.input_img_path = get_image_file_name(IMG_BASE_PATH, img_no) + '.JPEG'

        self.original_img = load_img(self.input_img_path, target_size=self.size)
        self.raw_img = img_to_array(self.original_img)
        self.expanded_img = self.expand()
        self.processed_img = self.process()

    def get_preprocess_for_model(self):
        return get_preprocess_for_model(self.model_name)

    def expand(self):
        # reshape to 4D tensor (batchsize, height, width, channels)
        return np.expand_dims(self.raw_img, axis=0)

    def process(self):
        # normalise / preprocess based on each model's needs
        preprocessor = get_preprocess_for_model(self.model_name)
        return preprocessor(self.expanded_img)

    def get_output_path(self, method: str):
        return RESULTS_BASE_PATH + method + '/' + method + '_' + str(self.img_no) + '.png'

    def get_original_img(self):
        return self.original_img

    def get_raw_img(self):
        return self.raw_img

    def get_expanded_img(self):
        return self.expanded_img

    def get_processed_img(self):
        return self.processed_img
