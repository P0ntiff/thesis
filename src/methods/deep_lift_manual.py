
import os
import sys
sys.path.append('src/')

from keras.applications import VGG16
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from preprocessing.keras_util import getPreprocessForModel

import matplotlib.pyplot as plt
import numpy as np

# high level wrapper for DeepLIFT
# TODO: replace with direct implementation
import tensorflow as tf

from shap.plots.colors import red_transparent_blue

# suppress output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3
}

modelName = 'vgg16'
model = MODELS[modelName](weights='imagenet')

IMG_BASE_PATH = 'data/imagenet_val_subset/ILSVRC2012_val_000000'

def attribute(modelName, model, imgPath, outputImgPath):
    preprocess = getPreprocessForModel(modelName)

    imgSize = (224, 224)
    if modelName == 'inception' or modelName == 'xception':
        imgSize = (299, 299)

    inputImg = load_img(imgPath, target_size=imgSize)
    inputImg = img_to_array(inputImg)
    expandedImg = np.expand_dims(inputImg, axis=0)
    preprocessedImg = preprocess(expandedImg)

    print(model.summary())
    # TODO: implement


for i in range(1, 4):
    image = str(i)
    if i < 10:
        image = '0' + image
    attribute(modelName, model, IMG_BASE_PATH + image + '.JPEG', 'results/deeplift/deeplift_' + image + '.png')