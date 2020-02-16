import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import innvestigate
import innvestigate.utils

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from ..util.keras_util import get_preprocess_for_model
from ..util.constants import IMG_BASE_PATH

# high level wrapper for DeepLIFT
# TODO: replace with direct implementation

# suppress output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def attribute(model_name: str, model, img_path: str, output_img_path: str):
    preprocess = get_preprocess_for_model(model_name)

    # strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    img_size = (224, 224)
    if model_name == 'inception' or model_name == 'xception':
        img_size = (299, 299)

    input_img = load_img(img_path, target_size=img_size)
    input_img = img_to_array(input_img)
    expanded_img = np.expand_dims(input_img, axis=0)
    preprocessed_img = preprocess(expanded_img)

    analyzer = innvestigate.analyzer.DeepLIFT(model)
    #analyzer = innvestigate.create_analyzer("deep_lift.wrapper", model)
    a = analyzer.analyze(preprocessed_img)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))

    # Plot
    # TODO : use extent to put input image down in greyscale

    #plt.imshow(inputImg, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, a[0].shape[0], a[0].shape[1], -1))
    plt.imshow(input_img, alpha=0.4, extent=(-1, a[0].shape[0], a[0].shape[1], -1))
    #print(np.abs(a[0]))
    maxVal = np.nanpercentile(np.abs(a[0]), 99.9)
    #plt.imshow(a[0], cmap=red_transparent_blue, vmin=-maxVal, vmax=maxVal)
    plt.imshow(a[0], cmap='seismic', clim=(-1, 1))
    plt.savefig(output_img_path)
    plt.show()
    plt.cla()
