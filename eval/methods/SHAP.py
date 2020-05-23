import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

import shap

from eval.util.image_util import ImageHandler
from eval.util.image_util import get_classification_classes


class Shap:
    def __init__(self, model, layer_n : int = 7):
        self.model = model
        self.layer_n = layer_n
        self.background_data, _ = shap.datasets.imagenet50()
        self.raw_classes = get_classification_classes()

        # combines expectation with sampling values from whole background data set
        self.explainer = shap.GradientExplainer(
            (model.layers[layer_n].input, model.layers[-1].output),
            self.map2layer(self.background_data, self.layer_n),
            local_smoothing=0  # std dev of smoothing noise
        )

    # explain how the input to a layer of the model explains the top class
    def map2layer(self, img, layer):
        feed_dict = dict(zip([self.model.layers[0].input], [img]))
        return K.get_session().run(self.model.layers[layer].input, feed_dict)

    def attribute(self, ih: ImageHandler, visualise: bool = False, save: bool = True):
        # get outputs for top prediction count "ranked_outputs"
        input_to_layer_n = self.map2layer(ih.get_processed_img(), self.layer_n)

        shap_values, indexes = self.explainer.shap_values(input_to_layer_n, ranked_outputs=1)

        # plot the explanations (SHAP value matrices) and save to file
        # print(len(shap_values))
        # if type(shap_values) != list:
        #     shap_values = [shap_values]

        # sh = shap_values[0]
        sh = shap_values
        # print(sh.size)
        # aggregate along third axis and normalise
        sv = sh[0].sum(-1)
        sv /= np.max(np.abs(sv))
        # print(sv.shape)

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

        # modified_SHAP_plot(shap_values, input_image, imgLabels)
