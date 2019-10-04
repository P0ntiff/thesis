import os
import sys
sys.path.append('src/')
import innvestigate
import innvestigate.utils

from keras.applications import VGG16
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



model = VGG16(weights='imagenet')
preprocess = getPreprocessForModel('vgg16')

# strip softmax layer
model = innvestigate.utils.model_wo_softmax(model)

imgSize = (224, 224)
img = load_img('data/imagenet_val_subset/ILSVRC2012_val_00000003.JPEG', target_size=imgSize)
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess(img)

analyser = innvestigate.analyzer.DeepLIFT(model)

a = analyser.analyze(img)

# Aggregate along color channels and normalize to [-1, 1]
a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
a /= np.max(np.abs(a))
# Plot
plt.imshow(a[0], cmap="red_transparent_blue", clim=(-1, 1))

plt.show()