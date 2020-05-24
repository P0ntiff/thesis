import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.applications.vgg16 import preprocess_input
import shap

from eval.util.image_util import ImageHandler, get_preprocess_for_model, BatchImageHelper
from eval.util.image_util import get_classification_classes


class Shap:
    def __init__(self, model, model_name: str):
        self.model = model
        # currently this has to be picked per-model. For VGG, taken as layer 7  (conv layer about halfway in)
        # block3_conv1 and block5_conv3
        self.layer_name = 'block5_conv3'
        print(model.layers[7].name)
        self.model.get_layer(self.layer_name)
        self.model_name = model_name
        print('Collecting background sample')
        bih = BatchImageHelper(list(range(50, 100)), model_name=self.model_name)
        print('Background sample collected.')

        self.background_data = bih.get_expanded_images()
        print(self.model.get_layer(self.layer_name).name)
        print(self.model.get_layer(self.layer_name).output_shape)
        #print(model.layers[layer_n])
        #print(self.background_data.shape)
        #self.raw_classes = get_classification_classes()
        self.preprocess = get_preprocess_for_model(model_name)

        # combines expectation with sampling values from whole background data set
        self.explainer = shap.GradientExplainer(
            (self.model.get_layer(self.layer_name).input, model.layers[-1].output),
            self.map2layer(self.background_data),
            local_smoothing=0  # std dev of smoothing noise
        )

    # explain how the input to a layer of the model explains the top class
    def map2layer(self, img):
        #print(img.shape)
        feed_dict = dict(zip([self.model.layers[0].input], [self.preprocess(img.copy())]))
        return K.get_session().run(self.model.get_layer(self.layer_name).input, feed_dict)

    def attribute(self, ih: ImageHandler, visualise: bool = False, save: bool = True):
        # get outputs for top prediction count "ranked_outputs"
        input_to_layer_n = self.map2layer(ih.get_expanded_img())

        shap_values, indexes = self.explainer.shap_values(input_to_layer_n, ranked_outputs=1)

        # plot the explanations (SHAP value matrices) and save to file
        # print(len(shap_values))
        if type(shap_values) != list:
            shap_values = [shap_values]

        sh = shap_values[0]
        #sh = shap_values
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
