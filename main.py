import sys
import logging
import os

# keras models
from keras.applications import InceptionV3, VGG16

# methods
from eval.methods.LIME import Lime
from eval.methods.deep_lift import DeepLift
from eval.methods.SHAP import Shap
from eval.methods.grad_cam import GradCam

# util
from eval.util.constants import GOOD_EXAMPLES, LIFT, LIME, SHAP, GRAD, VGG, INCEPT
from eval.util.image_util import ImageHandler, BatchImageHelper, show_figure
from eval.util.image_util import get_classification_mappings
from keras.applications.imagenet_utils import decode_predictions

# misc
from eval.util.imagenet_annotator import draw_annotations

# K.clear_session()
# tf.global_variables_initializer()
# tf.local_variables_initializer()
# K.set_session(tf.Session(graph=model.output.graph)) init = K.tf.global_variables_initializer() K.get_session().run(init)
# logging.basicConfig(level=logging.ERROR)
# #suppress output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)


METHODS = [LIFT, LIME, SHAP, GRAD]

MODELS = [VGG, INCEPT]

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


def main(method: str, model: str):
    # draw_annotations([i for i in range(16, 300)])
    instance_count = 4

    # run some attributions
    att = Attributer(model)
    for i in range(2, 3):
        att.attribute(img_no=GOOD_EXAMPLES[i],
                      method=method,
                      layer_no=LAYER_TARGETS[method][model])


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
                  visualise: bool = False, save: bool = True):
        self.initialise_for_method(method_name=method, layer_no=layer_no)
        ih = ImageHandler(img_no=img_no, model_name=self.curr_model_name)
        # get the 2D numpy array which represents the attribution
        attribution = self.collect_attribution(ih, method=method, layer_no=layer_no)
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
    if len(sys.argv) != 3:
        print('Usage: main [method] [model]')
        sys.exit()
    if sys.argv[1] not in METHODS:
        print('unrecognised method: ' + sys.argv[1])
        sys.exit()
    if sys.argv[2] not in MODELS:
        print('unrecognised model: ' + sys.argv[2])
        sys.exit()
    main(sys.argv[1], sys.argv[2])
