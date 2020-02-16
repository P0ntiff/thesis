from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

# LIME explanation
from lime import lime_image
from skimage.segmentation import mark_boundaries

from ..util.keras_util import get_preprocess_for_model


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.common.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def attribute(model_name: str, model, img_path: str, output_img_path: str):
    preprocess = get_preprocess_for_model(model_name)
    img_size = (224, 224)

    if model_name == 'inception' or model_name == 'xception':
        img_size = (299, 299)
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img)
    expanded_img = np.expand_dims(img, axis=0)
    processed_img = preprocess(expanded_img)

    explainer = lime_image.LimeImageExplainer()

    def prediction_wrapper(x):
        return model.predict(preprocess(x))

    # the function has to take a normal, unprocessed image
    explanation = explainer.explain_instance(img, prediction_wrapper, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    #plt.imshow(img, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, temp.shape[0], temp.shape[1], -1))
    #im = axes[row,i+1].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)

    # for inception, remember to / 2 and + 0.5
    plt.imshow(mark_boundaries(deprocess_image(temp), mask))
    plt.savefig(output_img_path)
    plt.show()

