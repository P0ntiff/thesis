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
from eval.util.constants import *
from eval.util.image_util import ImageHandler, show_figure, apply_threshold, ImageHelper
from eval.util.image_util import get_classification_mappings
from keras.applications.imagenet_utils import decode_predictions

# image annotations and mask util
from eval.util.imagenet_annotator import draw_annotations, get_masks_for_eval, get_mask_for_eval

# K.clear_session()
# tf.global_variables_initializer()
# tf.local_variables_initializer()
# K.set_session(tf.Session(graph=model.output.graph)) init = K.tf.global_variables_initializer() K.get_session().run(init)
# logging.basicConfig(level=logging.ERROR)
# #suppress output
# tf.logging.set_verbosity(tf.logging.ERROR)


METHODS = [LIME, LIFT, SHAP, GRAD]
MODELS = [VGG, INCEPT]
METRICS = [INTERSECT, INTENSITY]

VGG_LAYER_MAP = {"block5_conv3": 17,
                 "block4_conv3": 13,
                 "block3_conv3": 9,
                 "block3_conv1": 7,  # shap original target
                 "block2_conv2": 5}
INCEPTION_LAYER_MAP = {"conv2d_94": 299,
                       "conv2d_188": 299,
                       "mixed9": 279,
                       "mixed10": 310}

LAYER_TARGETS = {
    SHAP:
        {INCEPT: INCEPTION_LAYER_MAP["conv2d_188"],
         VGG: VGG_LAYER_MAP["block3_conv3"]},  # block5_conv3, block3_conv1
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
        ih = ImageHandler(img_no=GOOD_EXAMPLES[i], model_name=model)
        att.attribute(ih=ih,
                      method=method,
                      layer_no=LAYER_TARGETS[method][model])


def evaluate_panel_wrapper(metric: str, model: str):
    evaluator = Evaluator(metric=metric, model_name=model)
    evaluator.collect_panel_result_batch(range(1, 5))


def evaluator_wrapper(method: str, model: str):
    # hardcoded evaluation metric
    metric = INTERSECT
    evaluator = Evaluator(metric=metric, model_name=model)
    evaluator.collect_result_batch(method, range(2, 3))


def annotator_wrapper():
    # just for testing this works independently so it can be embedded elsewhere
    get_masks_for_eval(GOOD_EXAMPLES[:8], ImageHelper.get_size(VGG))


def print_confident_predictions(model_name: str):
    att = Attributer(model_name)
    att.initialise_for_method("")
    for i in range(1, 300):
        ih = ImageHandler(img_no=i, model_name=model_name)
        label, max_p = att.predict_for_model(ih=ih)
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
        self.file_headers = METHODS  # [m + "+_" + self.model_name for m in METHODS]
        self.result_file = "{}/{}/{}_results.csv".format(
            RESULTS_EVAL_PATH, model_name, metric)
        self.results_df = self.read_file(self.result_file, wipe=False)

    def read_file(self, file_path: str, wipe=False):
        # gets a dataframe from the results file
        if wipe:
            f = open(file_path, "w+")
            f.close()
            df = pd.DataFrame(columns=['img_no'] + self.file_headers).set_index('img_no')
            return df
        if not os.path.exists(file_path):
            open(file_path, "w").close()
        df = pd.read_csv(file_path).set_index('img_no')
        return df

    def write_results_to_file(self):
        self.results_df.to_csv(self.result_file, index=True, index_label='img_no')

    def get_image_handler_and_mask(self, img_no):
        # this gets the image wrapped in the ImageHandler object, and the bounding box annotation mask for the image,
        # ImageHandler is used to calculate attributions by each method, and the the mask is used for evaluation
        ih = ImageHandler(img_no=img_no, model_name=self.model_name)
        # bounding box in the format of the model's input shape / attribution shape
        annotation_mask = get_mask_for_eval(img_no=img_no, target_size=ih.get_size(),
                                            save=False, visualise=False)
        return ih, annotation_mask

    def collect_panel_result_batch(self, experiment_range: range):
        new_rows = {}
        for img_no in experiment_range:
            new_row = {}
            ih, annotation_mask = self.get_image_handler_and_mask(img_no)
            for method in METHODS:
                result = self.collect_result(ih, annotation_mask, method)
                # TODO replace GOOD_EXAMPLES no's with actual img_nos later
                if img_no <= len(self.results_df.index):
                    self.results_df.at[img_no, method] = result
                else:
                    new_row[method] = result
            new_rows[img_no] = new_row
            if img_no % 10 == 0:
                self.append_to_results_df(new_rows)
                new_rows = {}
        self.append_to_results_df(new_rows)

    def append_to_results_df(self, new_rows_dict, write=True):
        new_data = pd.DataFrame.from_dict(new_rows_dict, columns=self.file_headers, orient='index')
        self.results_df = self.results_df.append(new_data)
        if write:
            self.write_results_to_file()

    def collect_result_batch(self, method: str, experiment_range: range):
        new_rows = {}
        for img_no in experiment_range:
            # TODO replace GOOD_EXAMPLES no's with actual img_nos later (and in above func)
            ih, annotation_mask = self.get_image_handler_and_mask(img_no)
            result = self.collect_result(ih, annotation_mask, method)
            if img_no <= len(self.results_df.index):
                self.results_df.at[img_no, method] = result
            else:
                new_rows[img_no] = {method: result}
            if img_no % 10 == 0:
                self.append_to_results_df(new_rows)
                new_rows = {}

        self.append_to_results_df(new_rows)

    def collect_result(self, ih: ImageHandler, mask, method: str):
        # threshold for each attribution's "explainability" is the number of std deviations above
        # the mean contribution score for a pixel
        # against the ground truth bounding box label
        # for normal intersection over union, this is one sigma. for "intensity" over union this is taken as two sigma
        if self.metric == INTERSECT:
            return self.evaluate_intersection(ih, mask, method, sigma=1)
        elif self.metric == INTENSITY:
            return self.evaluate_intensity(ih, mask, method, sigma=2)

    def evaluate_intersection(self, ih: ImageHandler, mask, method: str, sigma: int, print_debug: bool = True) -> float:
        # calculate an attribution and use a provided bounding box mask to calculate the IOU metric
        # attribution has threshold applied, and abs value set (positive and negative evidence treated the same)
        attribution = self.att.attribute(ih=ih,
                                         method=method,
                                         layer_no=LAYER_TARGETS[method][self.model_name],
                                         threshold=True, sigma_multiple=sigma, take_absolute=True,
                                         visualise=False, save=False)
        # calculate the intersection of the attribution and the bounding box mask
        intersect_array = np.zeros(attribution.shape)
        intersect_array[(attribution > 0.0) * (mask > 0.0)] = 1
        #show_figure(intersect_array)
        # get the union array for the IOU calculation
        union_array = np.zeros(attribution.shape)
        union_array[(attribution > 0.0) + (mask > 0.0)] = 1
        #show_figure(union_array)
        # calculate intersection and union areas for numerator and denominator respectively
        intersect_area = intersect_array.sum()
        union_area = union_array.sum()
        #mask_area = mask.sum()
        intersection_over_union = intersect_area / union_area
        if print_debug:
            print('Evaluating `{}` on example `{}` ({})'.format(method, ih.img_no, 'intersection'))
            #print('--Mask Area =\t {}'.format(mask_area))
            print('--Intersect Area =\t {}'.format(intersect_area))
            print('--Union Area =\t {}'.format(union_area))
            print('--Intersection / Union =\t{:.2f}%'.format(intersection_over_union * 100))
            print('')

        return intersection_over_union

    def evaluate_intensity(self, ih: ImageHandler, mask, method: str, sigma: int, print_debug: bool = True) -> float:
        # # calculate an attribution and use a provided bounding box mask to calculate the IOU metric
        # # attribution has threshold applied, and abs value set (positive and negative evidence treated the same)
        attribution = self.att.attribute(ih=ih,
                                         method=method,
                                         layer_no=LAYER_TARGETS[method][self.model_name],
                                         threshold=True, sigma_multiple=sigma, take_absolute=True,
                                         visualise=False, save=False)
        show_figure(attribution)
        # calculate the weight/confidence of the attribution intersected with the bounding box mask
        intensity_array = np.copy(attribution)
        intensity_array[(attribution > 0.0) * (mask < 0.1)] = 0
        show_figure(intensity_array)
        # get the union array for the IOU calculation
        union_array = np.zeros(attribution.shape)
        union_array[(attribution > 0.0) + (mask > 0.0)] = 1
        show_figure(union_array)
        # calculate intersection and union areas for numerator and denominator respectively
        intensity_area = intensity_array.sum()
        union_area = union_array.sum()
        intensity_over_union = intensity_area / union_area
        if print_debug:
            print('Evaluating `{}` on example `{}` ({})'.format(method, ih.img_no, 'intensity'))
            print('--Intersect Area =\t {}'.format(intensity_area))
            print('--Union Area =\t {}'.format(union_area))
            print('--Intensity / Union =\t{:.2f}%'.format(intensity_over_union * 100))
            print('')

        return intensity_over_union


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
                  threshold: bool = False, sigma_multiple: int = 0, take_absolute: bool = False,
                  visualise: bool = False, save: bool = True):

        self.initialise_for_method(method_name=method, layer_no=layer_no)
        # get the 2D numpy array which represents the attribution
        attribution = self.collect_attribution(ih, method=method, layer_no=layer_no)
        # check if applying any thresholds / adjustments based on +ve / -ve evidence
        if threshold or take_absolute:
            attribution = apply_threshold(attribution, sigma_multiple, take_absolute)
        if check_invalid_attribution(attribution, ih):
            return
        if save:
            ih.save_figure(attribution, method)
        if visualise:
            show_figure(attribution)
        return attribution

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
    # 'evaluate_panel' command line option
    if sys.argv[1] == 'evaluate_panel':
        if len(sys.argv) != 4:
            print('Usage: main evaluate_panel <metric>[intersect|intensity] <model>[vgg16|inception]')
            sys.exit()
        if sys.argv[2] not in METRICS:
            print('Unrecognised metric: {}'.format(sys.argv[2]))
            sys.exit()
        if sys.argv[3] not in MODELS:
            print('Unrecognised model: {}'.format(sys.argv[3]))
        evaluate_panel_wrapper(sys.argv[2], sys.argv[3])
        sys.exit()
    # individual method command line options (attributing and evaluating)
    if len(sys.argv) != 4:
        print('Usage: main <mode>[attribute|evaluate] <method>[shap|deeplift|gradcam|lime] <model>[vgg16|inception]')
        sys.exit()
    if sys.argv[1] not in ['attribute', 'evaluate', 'evaluate_panel']:
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
