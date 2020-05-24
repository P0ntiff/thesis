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
import numpy as np

from ..util.constants import IMG_BASE_PATH
from ..util.constants import IMAGENET_CLASSES_OUTPUT
from ..util.constants import IMAGENET_OBJ_DET_CLASSES_INPUT
from ..util.constants import IMAGENET_OBJ_DET_CLASSES_OUTPUT

# results folder
RESULTS_BASE_PATH = 'results/adapted'

# For feeding into model architectures
STD_IMG_SIZE = (224, 224)
NONSTD_IMG_SIZE = (299, 299)


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


# def normalize(self, x):
#     """Utility function to normalize a tensor by its L2 norm"""
#     return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

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


def deprocess_gradcam(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x



def get_image_file_name(base_path: str, img_no: int):
    zeros = ''.join(['0' for _ in range(0, 8 - len(str(img_no)))])

    return base_path + zeros + str(img_no)


class ImageHelper:
    def __init__(self):
        pass

    @classmethod
    def expand(cls, raw_img):
        # raw_img is of shape (X, Y, RGB)
        # reshapes to 4D array (batchsize, height, width, channels)
        return np.expand_dims(raw_img, axis=0)

    @classmethod
    def process(cls, model_name: str, expanded_img):
        # normalise / preprocess based on each model's needs
        preprocessor = get_preprocess_for_model(model_name)
        return preprocessor(expanded_img)

    @classmethod
    def getSize(cls, model_name: str):
        if model_name == 'inception' or model_name == 'xception':
            return NONSTD_IMG_SIZE
        else:
            return STD_IMG_SIZE


class ImageHandler:
    def __init__(self, img_no: int, model_name: str):
        self.img_no = img_no
        self.model_name = model_name
        self.size = ImageHelper.getSize(model_name)
        self.input_img_path = get_image_file_name(IMG_BASE_PATH, img_no) + '.JPEG'

        self.original_img = load_img(self.input_img_path, target_size=self.size)
        self.raw_img = img_to_array(self.original_img)
        self.expanded_img = ImageHelper.expand(self.raw_img)
        self.processed_img = ImageHelper.process(self.model_name, self.expanded_img)

        # output base path for the model (vgg/inception) folder
        self.output_base_path = RESULTS_BASE_PATH + "_" + self.model_name + "/"

    def get_output_path(self, method: str):
        method_path = method + "/" + \
                      method + '_' + str(self.img_no) + '_' + self.model_name + '_test.png'
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


class BatchImageHelper:
    def __init__(self, img_nos: list, model_name: str):
        self.img_nos = img_nos
        self.model_name = model_name
        self.size = ImageHelper.getSize(model_name)
        self.input_paths = [get_image_file_name(IMG_BASE_PATH, img_no) + '.JPEG' for img_no in img_nos]

        self.original_images = [load_img(path, target_size=self.size) for path in self.input_paths]
        self.raw_images = [img_to_array(img) for img in self.original_images]
        self.expanded_images = np.concatenate([ImageHelper.expand(img) for img in self.raw_images], axis=0)
        #self.processed_images = concat([ImageHelper.process(model_name, img) for img in self.expanded_images], axis=0)

        # output base paths for the model
        self.output_base_path = RESULTS_BASE_PATH + "_" + self.model_name + "/"

    def get_output_path(self, img_no: int, method: str):
        method_path = method + "/" + \
                      method + '_' + str(img_no) + '_' + self.model_name + '_batch.png'
        return self.output_base_path + method_path

    def get_original_images(self):
        return self.original_images

    def get_raw_images(self):
        return self.raw_images

    def get_expanded_images(self):
        return self.expanded_images

    # def get_processed_images(self):
    #     return self.processed_images

    def get_size(self):
        return self.size
