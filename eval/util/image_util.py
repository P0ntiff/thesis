import numpy as np
import os
import requests
import logging
import json

from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K

from ..util.constants import IMG_BASE_PATH
from ..util.constants import IMAGENET_CLASSES_OUTPUT
from ..util.constants import IMAGENET_OBJ_DET_CLASSES_INPUT
from ..util.constants import IMAGENET_OBJ_DET_CLASSES_OUTPUT

# results folder
RESULTS_BASE_PATH = 'results/adapted'


def get_classification_classes():
    """ Gets the list of 1000 classification classes for ILSVRC (2012)"""
    if not os.path.exists('data'):
        logging.error("Error, not executing from top level directory")
        return -1
    # get from cache
    if os.path.exists(IMAGENET_CLASSES_OUTPUT):
        with open(IMAGENET_CLASSES_OUTPUT, 'r') as f:
            return json.load(f)
    classes_json = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')

    # cache
    with open(IMAGENET_CLASSES_OUTPUT, 'w') as f:
        f.write(classes_json.text)
    return classes_json.json()


def get_object_detection_classes():
    """ Gets the 200 imagenet labels for the object detection task"""
    if not os.path.exists('data'):
        logging.error("Error, not executing from top level directory")
        return -1
    if os.path.exists(IMAGENET_OBJ_DET_CLASSES_OUTPUT):
        with open(IMAGENET_OBJ_DET_CLASSES_OUTPUT, 'r') as f:
            return json.load(f)
    # build the json from the raw text
    output = {}
    with open(IMAGENET_OBJ_DET_CLASSES_INPUT, 'r') as f:
        counter = 0
        for line in f.read().splitlines():
            divider = line.find(' ')
            output[counter] = [line[0:divider], line[divider + 1:]]
            counter += 1
    with open(IMAGENET_OBJ_DET_CLASSES_OUTPUT, 'w') as f:
        json.dump(output, f)


def get_classification_mappings():
    # 1000 classes for image localisation / image classification tasks for ImageNet dataset
    class_labels = get_classification_classes()
    # return map of WNID : label
    return {v[0]: v[1] for k, v in class_labels.items()}


def get_detection_mappings():
    # 200 classes for object detection data
    obj_detect_labels = get_object_detection_classes()
    return {v[0]: v[1] for k, v in obj_detect_labels.items()}


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


def get_image_file_name(base_path: str, img_no: int):
    zeros = ''.join(['0' for _ in range(0, 8 - len(str(img_no)))])

    return base_path + zeros + str(img_no)


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

        # output base path for the model (vgg/inception) folder
        self.output_base_path = RESULTS_BASE_PATH + "_" + self.model_name + "/"

    def expand(self):
        # reshape to 4D tensor (batchsize, height, width, channels)
        return np.expand_dims(self.raw_img, axis=0)

    def process(self):
        # normalise / preprocess based on each model's needs
        preprocessor = get_preprocess_for_model(self.model_name)
        return preprocessor(self.expanded_img)

    def get_output_path(self, method: str):
        method_path = method + "/" + \
                      method + '_' + str(self.img_no) + '_' + self.model_name + '.png'
        return self.output_base_path + method_path

    def get_original_img(self):
        return self.original_img

    def get_raw_img(self):
        return self.raw_img

    def get_expanded_img(self):
        return self.expanded_img

    def get_processed_img(self):
        return self.processed_img

    def get_size(self):
        return self.size