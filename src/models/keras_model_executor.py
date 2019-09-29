# pretrained models
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import ResNet50
from keras.applications import VGG16

# helper functions
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import argparse
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

#import tensorflow as tf
#import matplotlib.pyplot as plt
#from nets import inception

MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}


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
    logging.info('Preprocessing image...')
    img = load_img(imagePath, target_size=imageSize)
    img = img_to_array(img)
    # reshape to 4D tensor (batchsize, height, width, channels)
    img = np.expand_dims(img, axis=0)
    # normalise / preprocess based on each model's needs
    preprocess = imagenet_utils.preprocess_input
    if modelName == 'inception' or modelName == 'xception':
        preprocess = preprocess_input
    img = preprocess(img)

    # classify
    logging.info('Classifying...')
    predictions = model.predict(img)
    decodedPredictions = imagenet_utils.decode_predictions(predictions)

    # print the top 5 predictions, labels and probabilities
    for (i, (imgnetID, label, p)) in enumerate(decodedPredictions[0]):
        print('{}: {}m {}, probability={:.2f}'.format(i + 1, imgnetID, label, p))

    print(decodedPredictions[0])
    return decodedPredictions[0]



#predict('vgg16', 'data/ILSVRC2012_val_00000022.JPEG')

