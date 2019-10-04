
import sys
sys.path.append('src/')
import os
import logging
# pretrained models
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import ResNet50
from keras.applications import VGG16
import tensorflow as tf
# helper functions
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import argparse
import numpy as np

#import matplotlib.pyplot as plt
#from nets import inception
from preprocessing.keras_util import getPreprocessForModel

logging.basicConfig(level=logging.INFO)
# suppress output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

IMG_BASE_PATH = 'data/imagenet_val_subset/ILSVRC2012_val_000000'

def predict(modelName, imagePath):
    # branched image size needs for two diff model classes
    imageSize = (224, 224)
    if modelName == 'inception' or modelName == 'xception':
        imageSize = (299, 299)
    # network weights are cached on subsequent runs
    logging.info('Loading {} model architecture and weights...'.format(modelName))
    Network = MODELS[modelName]
    model = Network(weights='imagenet')

    # load image
    logging.info('Preprocessing image "..{}"...'.format(imagePath[-13:]))
    img = load_img(imagePath, target_size=imageSize)
    img = img_to_array(img)

    # reshape to 4D tensor (batchsize, height, width, channels)
    img = np.expand_dims(img, axis=0)

    # normalise / preprocess based on each model's needs
    preprocess = getPreprocessForModel(modelName)
    img = preprocess(img)

    # classify
    logging.info('Classifying...')
    predictions = model.predict(img)
    decodedPredictions = decode_predictions(predictions)

    # print the top 5 predictions, labels and probabilities
    for (i, (imgnetID, label, p)) in enumerate(decodedPredictions[0]):
        print('{}: {}m {}, probability={:.2f}'.format(i + 1, imgnetID, label, p))
    print('')
    return decodedPredictions[0]


predict('vgg16', IMG_BASE_PATH + '08.JPEG')
# predict('vgg16', IMG_BASE_PATH + '09.JPEG')

# predict('resnet', IMG_BASE_PATH + '08.JPEG')
# predict('resnet', IMG_BASE_PATH + '09.JPEG')
