import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.applications.vgg16 import preprocess_input
import shap

from eval.util.image_util import ImageHandler, get_preprocess_for_model, BatchImageHelper
from eval.util.image_util import get_classification_classes


class Shap:
    def __init__(self, model, model_name: str, layer_no: int):
        self.model = model
        self.model_name = model_name
        self.layer_no = layer_no
        print('Collecting SHAP background sample')
        bih = BatchImageHelper(list(range(50, 100)), model_name=self.model_name)

        self.background_data = bih.get_expanded_images()
        self.preprocess = get_preprocess_for_model(model_name)

        # combines expectation with sampling values from whole background data set
        self.explainer = self.generate_explainer(layer_no)

    def generate_explainer(self, layer_no: int):
        return shap.GradientExplainer(
            (self.model.layers[layer_no].input, self.model.layers[-1].output),
            self.map2layer(self.background_data),
            local_smoothing=0  # std dev of smoothing noise
        )

    def reset_explainer(self, layer_no: int):
        if layer_no is None:
            return
        if layer_no != self.layer_no:
            self.explainer = self.generate_explainer(layer_no)
            self.layer_no = layer_no

    def get_layer_no(self):
        return self.layer_no

    def map2layer(self, img):
        # explain how the input to a layer of the model explains the top class
        feed_dict = dict(zip([self.model.layers[0].input], [self.preprocess(img.copy())]))
        return K.get_session().run(self.model.layers[self.layer_no].input, feed_dict)

    def attribute(self, ih: ImageHandler):
        # get outputs for top prediction count "ranked_outputs"
        input_to_layer_n = self.map2layer(ih.get_expanded_img())
        shap_values, indexes = self.explainer.shap_values(input_to_layer_n, ranked_outputs=1)

        # plot the explanations (SHAP value matrices) and save to file
        # print(len(shap_values))
        if type(shap_values) != list:
            shap_values = [shap_values]

        sh = shap_values[0]
        # aggregate along third axis (the RGB axis) and normalise to (-1, 1)
        sv = sh[0].sum(-1)
        sv /= np.max(np.abs(sv))

        return sv
