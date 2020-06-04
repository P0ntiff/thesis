import logging

# keras models
from keras.applications import InceptionV3, VGG16
from keras.applications.imagenet_utils import decode_predictions

# util
from eval.util.constants import *
from eval.util.image_util import ImageHandler, show_figure, apply_threshold, get_classification_mappings

# methods
from eval.methods.LIME import Lime
from eval.methods.deep_lift import DeepLift
from eval.methods.SHAP import Shap
from eval.methods.grad_cam import GradCam


def check_invalid_attribution(attribution, ih):
    # attribution should be a 2D array returned by each method
    # i.e grayscale not RGB
    if attribution is None:
        return 1
    # check shape of attribution is what is expected
    if attribution.ndim != 2:
        print('Attribution returned has {} dimensions'.format(attribution.ndim))
        return 1
    if attribution.shape != ih.get_size():
        print('Attribution returned with shape {} is not the expected shape of {}'.format(
            attribution.shape, ih.get_size()))
        return 1
    return 0


class Attributer:
    def __init__(self, model_name: str):
        self.models = {
            VGG: VGG16,
            INCEPT: InceptionV3,
        }
        self.curr_model_name = model_name
        self.curr_model = self.load_model(model_name)
        # set up methods
        self.lime_method = None
        self.deep_lift_method = None
        self.shap_method = None
        self.gradcam_method = None
        # Classes for imagenet
        self.class_map = get_classification_mappings()

    def load_model(self, model_name: str):
        self.curr_model_name = model_name
        print('Loading {} model architecture and weights...'.format(model_name))
        return self.models[self.curr_model_name](weights='imagenet')

    def build_model(self):
        """Function returning a new keras model instance.
        """
        if self.curr_model_name == VGG:
            return VGG16(include_top=True, weights='imagenet')
        elif self.curr_model_name == INCEPT:
            return InceptionV3(include_top=True, weights='imagenet')

    def initialise_for_method(self, method_name: str, layer_no: int = None):
        if method_name == LIFT and self.deep_lift_method is None:
            self.deep_lift_method = DeepLift(self.curr_model, self.build_model)
        elif method_name == LIME and self.lime_method is None:
            self.lime_method = Lime(self.curr_model, self.curr_model_name)
        elif method_name == SHAP and self.shap_method is None:
            self.shap_method = Shap(self.curr_model, self.curr_model_name, layer_no)
        elif method_name == GRAD and self.gradcam_method is None:
            self.gradcam_method = GradCam(self.curr_model, self.build_model, layer_no)

    def predict_for_model(self, ih: ImageHandler, top_n: int = 5, print_to_stdout: bool = True) -> (str, float):
        # returns a tuple with the top prediction, and the probability of the top prediction (i.e confidence)
        # classify
        logging.info('Classifying...')
        predictions = self.curr_model.predict(ih.get_processed_img())
        decoded_predictions = decode_predictions(predictions, top=top_n)

        # print the top 5 predictions, labels and probabilities
        if print_to_stdout:
            print('Model predictions:')
        max_p = 0.00
        max_pred = ''
        for (i, (img_net_ID, label, p)) in enumerate(decoded_predictions[0]):
            if print_to_stdout:
                print('{}: {}, Probability={:.2f}, ImageNet ID={}'.format(i + 1, label, p, img_net_ID))
            if p > max_p:
                max_p = p
                max_pred = label
        if print_to_stdout:
            print('')
        return max_pred, max_p

    def attribute(self, ih: ImageHandler, method: str, layer_no: int = None,
                  take_absolute: bool = False, take_threshold: bool = False, sigma_multiple: int = 0,
                  visualise: bool = False, save: bool = True):
        if layer_no is None:
            layer_no = LAYER_TARGETS[method][self.curr_model_name]
        self.initialise_for_method(method_name=method, layer_no=layer_no)
        # get the 2D numpy array which represents the attribution
        attribution = self.collect_attribution(ih, method=method, layer_no=layer_no)
        # check if applying any thresholds / adjustments based on +ve / -ve evidence
        if take_threshold or take_absolute:
            attribution = apply_threshold(attribution, sigma_multiple, take_absolute)
        if check_invalid_attribution(attribution, ih):
            return
        if save:
            ih.save_figure(attribution, method)
        if visualise:
            show_figure(attribution)
        return attribution

    def attribute_panel(self, ih: ImageHandler, methods: list = METHODS,
                        take_threshold: bool = False, sigma_multiple: int = 0, take_absolute: bool = False,
                        visualise: bool = False, save: bool = True):
        output_attributions = {}
        for method in methods:
            layer_no = LAYER_TARGETS[method][self.curr_model_name]
            output_attributions[method] = self.attribute(ih=ih, method=method, layer_no=layer_no,
                                                         take_absolute=take_absolute, take_threshold=take_threshold,
                                                         sigma_multiple=sigma_multiple,
                                                         visualise=visualise, save=save)
        return output_attributions

    def collect_attribution(self, ih: ImageHandler, method: str, layer_no: int = None):
        """Top level wrapper for collecting attributions from each method. """
        print('Collecting attribution for `{}`'.format(method))
        if method == LIFT:
            return self.deep_lift_method.attribute(ih)
        elif method == LIME:
            return self.lime_method.attribute(ih)
        elif method == SHAP:
            self.shap_method.reset_explainer(layer_no=layer_no)
            return self.shap_method.attribute(ih)
        elif method == GRAD:
            self.gradcam_method.reset_layer_no(layer_no=layer_no)
            return self.gradcam_method.attribute(ih)
        else:
            print('Error: Invalid attribution method chosen')
            return None
