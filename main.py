import sys
import logging

# keras models
from keras.applications import InceptionV3, VGG16, ResNet50, Xception
from keras import backend as K

# methods
from eval.methods.LIME import Lime
from eval.methods.deep_lift import DeepLift
from eval.methods.SHAP import Shap
from eval.methods.grad_cam import GradCam

# util
from eval.util.constants import GOOD_EXAMPLES
from eval.util.image_util import ImageHandler, BatchImageHelper
from eval.util.image_util import get_classification_mappings
from keras.applications.imagenet_utils import decode_predictions


# misc
from eval.util.imagenet_annotator import draw_annotations
K.clear_session()
# tf.global_variables_initializer()
# tf.local_variables_initializer()
# K.set_session(tf.Session(graph=model.output.graph)) init = K.tf.global_variables_initializer() K.get_session().run(init)
# logging.basicConfig(level=logging.ERROR)
# suppress output
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)


METHODS = ['deeplift', 'lime', 'shap', 'gradcam']
MODELS = ['vgg16', 'inception']


def main(method: str, model: str):
    # draw_annotations([i for i in range(16, 300)])
    instance_count = 4

    # run some attributions
    att = Attributer(model)
    att.initialise_for_method(method)
    if 1:
        for i in range(1, instance_count):
            att.attribute(img_no=GOOD_EXAMPLES[i], method=method)
    else:
        att.attribute(img_no=GOOD_EXAMPLES[2], method=method)


def print_confident_predictions(model_name: str):
    att = Attributer(model_name)
    att.initialise_for_method("")
    for i in range(1, 300):
        label, max_p = att.predict_for_model(img_no=i)
        if max_p > 0.75:
            print('image_no: {}, label: {}, probability: {:.2f}'.format(i, label, max_p))


class Attributer:
    def __init__(self, model_name: str = 'vgg16'):
        self.models = {
            "vgg16": VGG16,
            "inception": InceptionV3,
            "xception": Xception,
            "resnet": ResNet50
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
        if self.curr_model_name == 'vgg16':
            return VGG16(include_top=True, weights='imagenet')
        elif self.curr_model_name == 'inception':
            return InceptionV3(include_top=True, weights='imagenet')

    def initialise_for_method(self, method_name):
        if method_name == "deeplift":
            self.deep_lift_method = DeepLift(self.curr_model)
        elif method_name == "lime":
            self.lime_method = Lime(self.curr_model, self.curr_model_name)
        elif method_name == "shap":
            self.shap_method = Shap(self.curr_model, self.curr_model_name)
        elif method_name == "gradcam":
            self.gradcam_method = GradCam(self.curr_model, self.curr_model_name, self.build_model)

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

    def attribute(self, img_no: int, method):
        """Top level wrapper for collecting attributions from each method. """
        ih = ImageHandler(img_no=img_no, model_name=self.curr_model_name)
        if method == 'deeplift':
            self.deep_lift_method.attribute(ih)
        elif method == 'lime':
            self.lime_method.attribute(ih)
        elif method == 'shap':
            # TODO: note requires layer specified, come up with better solution/justification
            self.shap_method.attribute(ih)
        elif method == 'gradcam':
            self.gradcam_method.attribute(ih)
        else:
            print('Error: Invalid attribution method chosen')


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
