import json
import requests
import os
import logging

from ..util.constants import IMAGENET_CLASSES_OUTPUT
from ..util.constants import IMAGENET_OBJ_DET_CLASSES_INPUT
from ..util.constants import IMAGENET_OBJ_DET_CLASSES_OUTPUT


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

