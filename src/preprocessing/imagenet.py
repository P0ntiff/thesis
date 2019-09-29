import json
import requests
import os
import logging

logging.basicConfig(level=logging.ERROR)

IMAGENET_CLASSES_OUTPUT = 'data/imagenet_classes.json'
IMAGENET_OBJ_DET_CLASSES_OUTPUT = 'data/imagenet_obj_det_classes.json'
IMAGENET_OBJ_DET_CLASSES_INPUT = 'data/imagenet_obj_det_classes_raw.txt'

# Gets the list of 1000 imagenet classes
def getClassificationClasses():
    if not os.path.exists('data'):
        logging.error("Error, not executing from top level directory")
        return -1
    # get from cache
    if os.path.exists(IMAGENET_CLASSES_OUTPUT):
        with open(IMAGENET_CLASSES_OUTPUT, 'r') as f:
            return json.load(f)
    classesJSON = requests.get(
        'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
    # cache
    with open(IMAGENET_CLASSES_OUTPUT, 'w') as f:
        f.write(classesJSON.text)
    return classesJSON.json()

# Gets the 200 imagenet labels for the object detection task
def getObjectDetectionClasses():
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


def getClassificationMappings():
    # 1000 classes for image localisation / image classification tasks for ImageNet dataset
    classLabels = getClassificationClasses()
    # return map of WNID : label
    return {v[0]: v[1] for k, v in classLabels.items()}


def getDetectionMappings():
    # 200 classes for object detection data
    objDetectLabels = getObjectDetectionClasses()
    return {v[0]: v[1] for k, v in objDetectLabels.items()}



labelMap = getClassificationMappings()
objMap = getDetectionMappings()


# check on what classes are only in the object detection data
for k, v in objMap.items():
    if k not in labelMap:
        print('{} {}'.format(k , v))