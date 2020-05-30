import sys
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import pandas as pd
import numpy as np

# keras models
from keras.applications import InceptionV3, VGG16

# methods
from eval.methods.LIME import Lime
from eval.methods.deep_lift import DeepLift
from eval.methods.SHAP import Shap
from eval.methods.grad_cam import GradCam

# util
from eval.util.constants import GOOD_EXAMPLES, LIFT, LIME, SHAP, GRAD, VGG, INCEPT, RESULTS_EVAL_PATH, INTERSECT
from eval.util.image_util import ImageHandler, show_figure, apply_threshold, ImageHelper
from eval.util.image_util import get_classification_mappings
from keras.applications.imagenet_utils import decode_predictions

# misc
from eval.util.imagenet_annotator import draw_annotations, get_masks_for_eval

# K.clear_session()
# tf.global_variables_initializer()
# tf.local_variables_initializer()
# K.set_session(tf.Session(graph=model.output.graph)) init = K.tf.global_variables_initializer() K.get_session().run(init)
# logging.basicConfig(level=logging.ERROR)
# #suppress output
# tf.logging.set_verbosity(tf.logging.ERROR)


METHODS = [LIFT, LIME, SHAP, GRAD]
MODELS = [VGG, INCEPT]
METRICS = [INTERSECT]

VGG_LAYER_MAP = {"block5_conv3": 17,
                 "block4_conv3": 13,
                 "block3_conv3": 9,
                 "block3_conv1": 7,      # shap original target
                 "block2_conv2": 5}
INCEPTION_LAYER_MAP = {"conv2d_94": 299,
                       "conv2d_188": 299,
                       "mixed9": 279,
                       "mixed10": 310}

LAYER_TARGETS = {
    SHAP:
        {INCEPT: INCEPTION_LAYER_MAP["conv2d_188"],
         VGG: VGG_LAYER_MAP["block3_conv3"]},   # block5_conv3, block3_conv1
    GRAD:
        {INCEPT: INCEPTION_LAYER_MAP["conv2d_188"],
         VGG: VGG_LAYER_MAP["block5_conv3"]},
    LIFT:
        {INCEPT: None,
         VGG: None},
    LIME:
        {INCEPT: None,
         VGG: None}
}


def attributer_wrapper(method: str, model: str):
    # draw_annotations([i for i in range(16, 300)])
    # run some attributions
    att = Attributer(model)
    for i in range(1, 8):
        att.attribute(img_no=GOOD_EXAMPLES[i],
                      method=method,
                      layer_no=LAYER_TARGETS[method][model])


def evaluator_wrapper(method: str, model: str):
    # current evaluation metric
    metric = INTERSECT
    evaluator = Evaluator(metric=metric, model_name=model)

    # test attributor
    for i in range(1, 8):
        evaluator.att.attribute(img_no=GOOD_EXAMPLES[i],
                                method=method,
                                layer_no=LAYER_TARGETS[method][model],
                                threshold=0.1, take_absolute=True,
                                visualise=False, save=True)


def annotator_wrapper():
    # just for testing this works independently so it can be embedded elsewhere
    get_masks_for_eval(GOOD_EXAMPLES[:8], ImageHelper.get_size(VGG))


def print_confident_predictions(model_name: str):
    att = Attributer(model_name)
    att.initialise_for_method("")
    for i in range(1, 300):
        label, max_p = att.predict_for_model(img_no=i)
        if max_p > 0.75:
            print('image_no: {}, label: {}, probability: {:.2f}'.format(i, label, max_p))


def check_invalid_attribution(attribution, ih):
    # attribution should be a (X,Y,RGB) array returned by each method
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


class Analyser:
    def __init__(self):
        # analytics class for comparing metrics across models and methods
        pass


class Evaluator:
    def __init__(self, metric: str, model_name: str):
        self.att = Attributer(model_name=model_name)
        self.metric = metric
        self.model_name = model_name
        self.experiment_length = 5
        self.file_headers = ["img_no"] + METHODS
        self.result_file = RESULTS_EVAL_PATH + '/' + metric + '_results.csv'
        self.results_df = self.read_file(self.result_file, wipe=True)
        # test
        self.write_file(self.results_df)

    def read_file(self, file_path: str, wipe=False):
        # gets a dataframe from the results file
        if wipe:
            f = open(file_path, "w+")
            f.close()
            return pd.DataFrame(columns=self.file_headers)
        df = pd.read_csv(file_path)
        return df

    def write_file(self, df):
        df.to_csv(self.result_file, index=False)

    def collect_result_batch(self, method: str):
        df = pd.DataFrame(columns=self.file_headers)
        for img_no in range(0, self.experiment_length):
            res = self.collect_result(GOOD_EXAMPLES[img_no], method)
            # if img_no > len(self.results_df.index):

    def collect_result(self, img_no: int, method: str):
        evaluation = 1
        if self.metric == INTERSECT:
            evaluation = self.evaluate_intersection(img_no, method)
        elif self.metric is None:
            print('Unimplemented evaluation metric')

    def evaluate_intersection(self, img_no: int, method: str):
        return 1


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
        logging.info('Loading {} model architecture and weights...'.format(model_name))
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
            self.deep_lift_method = DeepLift(self.curr_model)
        elif method_name == LIME and self.lime_method is None:
            self.lime_method = Lime(self.curr_model, self.curr_model_name)
        elif method_name == SHAP and self.shap_method is None:
            self.shap_method = Shap(self.curr_model, self.curr_model_name, layer_no)
        elif method_name == GRAD and self.gradcam_method is None:
            self.gradcam_method = GradCam(self.curr_model, self.build_model, layer_no)

    def predict_for_model(self, img_no: int, top_n: int = 5, print_to_stdout: bool = True) -> (str, float):
        # returns a tuple with the top prediction, and the probability of the top prediction (i.e confidence)
        img = ImageHandler(img_no=img_no, model_name=self.curr_model_name)
        # classify
        logging.info('Classifying...')
        predictions = self.curr_model.predict(img.get_processed_img())
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

    def attribute(self, img_no: int, method: str, layer_no: int = None,
                  threshold: float = None, take_absolute: bool = None,
                  visualise: bool = False, save: bool = True):
        self.initialise_for_method(method_name=method, layer_no=layer_no)
        ih = ImageHandler(img_no=img_no, model_name=self.curr_model_name)
        # get the 2D numpy array which represents the attribution
        attribution = self.collect_attribution(ih, method=method, layer_no=layer_no)
        # check if applying any thresholds / adjustments based on +ve / -ve evidence
        if threshold is not None:
            attribution = apply_threshold(attribution, threshold, take_absolute)
        if check_invalid_attribution(attribution, ih):
            return
        if save:
            ih.save_figure(attribution, method)
        if visualise:
            show_figure(attribution)

    def collect_attribution(self, ih: ImageHandler, method: str, layer_no: int = None):
        """Top level wrapper for collecting attributions from each method. """
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


if __name__ == "__main__":
    if sys.argv[1] == 'annotate':
        annotator_wrapper()
        sys.exit()
    if len(sys.argv) != 4:
        print('Usage: main <mode>[attribute|evaluate] <method>[shap|deeplift|gradcam|lime] <model>[vgg16|inception]')
        sys.exit()
    if sys.argv[1] not in ['attribute', 'evaluate']:
        print('Unrecognised mode: {}'.format(sys.argv[1]))
    if sys.argv[2] not in METHODS:
        print('Unrecognised method: {}'.format(sys.argv[1]))
        sys.exit()
    if sys.argv[3] not in MODELS:
        print('Unrecognised model: {}'.format(sys.argv[2]))
        sys.exit()

    # send commands to wrappers
    if sys.argv[1] == 'attribute':
        attributer_wrapper(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'evaluate':
        evaluator_wrapper(sys.argv[2], sys.argv[3])
