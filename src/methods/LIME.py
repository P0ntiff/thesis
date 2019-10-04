import os
import sys
sys.path.append('src/')

import os
import keras
from keras.applications import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt

import numpy as np

from preprocessing.keras_util import getPreprocessForModel


preprocess = getPreprocessForModel('inception')

imgSize = (299, 299)
img = load_img('data/imagenet_val_subset/ILSVRC2012_val_00000015.JPEG', target_size=imgSize)
img = img_to_array(img)
expandedImg = np.expand_dims(img, axis=0)

model = InceptionV3(weights='imagenet')
predictions = model.predict(preprocess(expandedImg))

for x in decode_predictions(predictions)[0]:
    print(x)


# explanation
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries


explainer = lime_image.LimeImageExplainer()

# the function has to take a normal, unprocessed image
explanation = explainer.explain_instance(img, model.predict, top_labels=5, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()