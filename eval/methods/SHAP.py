import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2

import shap
from shap.plots.colors import red_transparent_blue

from eval.util.image_util import ImageHandler
from eval.util.imagenet import get_classification_classes

from shap import force_plot

RAW_CLASSES = get_classification_classes()


# Modified version of SHAP plotter function (see shap.plots.image)
def get_saliency(shap_values, ih: ImageHandler, save: bool = True, visualise: bool = True):
    """ Plots SHAP values for image inputs.
    """
    multi_output = True

    if type(shap_values) != list:
        multi_output = False
        shap_values = [shap_values]

    sh = shap_values[0]

    # aggregate along third axis and normalise
    sv = sh[0].sum(-1)
    sv /= np.max(np.abs(sv))
    print(sv.shape)

    if save:
        plt.imshow(sv, cmap='seismic', clim=(-1, 1))
        plt.savefig(ih.get_output_path('shap'))
        plt.cla()

    if visualise:
        plt.figure(figsize=(15, 10))
        plt.subplot(121)
        plt.title('SHAP')
        plt.axis('off')
        plt.imshow(ih.get_original_img())

        plt.subplot(122)
        plt.axis('off')
        plt.imshow(sv, cmap='seismic', clim=(-1, 1))

        # plt.subplot(133)
        # plt.axis('off')
        # im = cv2.resize(sv, (224, 224), cv2.INTER_LINEAR)
        # plt.imshow(im, cmap='seismic', clim=(-1, 1))

        plt.show()
    plt.cla()


def attribute(model, ih: ImageHandler):
    layer_n = 7
    background_data, Y = shap.datasets.imagenet50()

    # explain how the input to a layer of the model explains the top class
    def map2layer(img, layer):
        feed_dict = dict(zip([model.layers[0].input], [img]))
        return K.get_session().run(model.layers[layer].input, feed_dict)

    # combines expectation with sampling values from whole background data set
    e = shap.GradientExplainer(
        (model.layers[layer_n].input, model.layers[-1].output),
        map2layer(background_data, layer_n),
        local_smoothing=0  # std dev of smoothing noise
    )

    # get outputs for top prediction count "ranked_outputs"

    input_to_layer_n = map2layer(ih.get_processed_img(), layer_n)

    shap_values, indexes = e.shap_values(input_to_layer_n, ranked_outputs=1)

    # plot the explanations and save to file
    get_saliency(shap_values, ih=ih)

    # modified_SHAP_plot(shap_values, input_image, imgLabels)

