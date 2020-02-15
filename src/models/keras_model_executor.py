import os
import logging

import tensorflow as tf
import numpy as np

# pretrained models
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import ResNet50
from keras.applications import VGG16

# helper functions
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

#import matplotlib.pyplot as plt
from ..preprocessing.keras_util import get_preprocess_for_model

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


def predict(model_name, image_path):
    # branched image size needs for two diff model classes
    image_size = (224, 224)
    if model_name == 'inception' or model_name == 'xception':
        image_size = (299, 299)
    # network weights are cached on subsequent runs
    logging.info('Loading {} model architecture and weights...'.format(model_name))
    Network = MODELS[model_name]
    model = Network(weights='imagenet')

    # load image
    logging.info('Preprocessing image "..{}"...'.format(image_path[-13:]))
    img = load_img(image_path, target_size=image_size)
    img = img_to_array(img)

    # reshape to 4D tensor (batchsize, height, width, channels)
    img = np.expand_dims(img, axis=0)

    # normalise / preprocess based on each model's needs
    preprocess = get_preprocess_for_model(model_name)
    img = preprocess(img)

    # classify
    logging.info('Classifying...')
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions)

    # print the top 5 predictions, labels and probabilities
    for (i, (imgnetID, label, p)) in enumerate(decoded_predictions[0]):
        print('{}: {}m {}, probability={:.2f}'.format(i + 1, imgnetID, label, p))
    print('')
    return decoded_predictions[0]


predict('vgg16', IMG_BASE_PATH + '04.JPEG')
predict('vgg16', IMG_BASE_PATH + '06.JPEG')
predict('vgg16', IMG_BASE_PATH + '08.JPEG')

# predict('resnet', IMG_BASE_PATH + '08.JPEG')
# predict('resnet', IMG_BASE_PATH + '09.JPEG')
