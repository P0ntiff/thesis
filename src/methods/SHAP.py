import sys
sys.path.append('src/')
import os

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import json

# helper functions
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from preprocessing.keras_util import getPreprocessForModel
from preprocessing.imagenet import getClassificationClasses

import shap


RAW_CLASSES = getClassificationClasses()


IMG_BASE_PATH = 'data/imagenet_val_subset/ILSVRC2012_val_000000'

preprocess = getPreprocessForModel('vgg16')

backgroundData, Y = shap.datasets.imagenet50()

imgSize = (224, 224)
inputImage = 'data/imagenet_val_subset/ILSVRC2012_val_00000013.JPEG'
inputImage = load_img(inputImage, target_size=imgSize)
inputImage = img_to_array(inputImage)

inputImage = np.expand_dims(inputImage, axis=0)
img = preprocess(inputImage)


model = VGG16(weights='imagenet', include_top=True)


# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)

# combines expectation with sampling values from whole background data set
e = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(backgroundData, 7),
    local_smoothing=0 # std dev of smoothing noise
)

shap_values, indexes = e.shap_values(map2layer(inputImage, 7), ranked_outputs=2)

# get the names for the classes
#index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# using the top indexes (i.e 1:1000)
imgLabels = np.vectorize(lambda x: RAW_CLASSES[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, inputImage, imgLabels)