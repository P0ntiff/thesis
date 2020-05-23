import sys
import logging

# keras models
from keras.applications import InceptionV3, VGG16, ResNet50, Xception
from keras import backend as K

# methods
from eval.methods import grad_cam
from eval.methods.LIME import Lime
from eval.methods.deep_lift import DeepLift
from eval.methods.SHAP import Shap
from eval.methods.grad_cam import GradCam

# util
from eval.util.image_util import ImageHandler
from eval.util.image_util import get_classification_mappings
from keras.applications.imagenet_utils import decode_predictions


# misc
K.clear_session()
# tf.global_variables_initializer()
# tf.local_variables_initializer()
# K.set_session(tf.Session(graph=model.output.graph)) init = K.tf.global_variables_initializer() K.get_session().run(init)


# logging.basicConfig(level=logging.ERROR)
# suppress output
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)


# strong confidence ImageNet examples (p > 0.75) for VGG-16
good_examples = [7, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 24, 25, 27, 31, 33, 35, 36, 42, 43, 45, 47, 52, 53, 54, 56,
           58, 63, 66, 67, 68, 69, 72, 73, 74, 75, 78, 81, 82, 86, 89, 90, 92, 93, 96, 97, 98, 99, 100, 103, 107,
           109, 113, 116, 120, 122, 123, 124, 125, 127, 129, 130, 131, 133, 135, 138, 139, 142, 143, 144, 145, 149,
           153, 154, 156, 157, 158, 160, 164, 165, 166, 167, 168, 169, 171, 172, 173, 177, 179, 186, 187, 188, 192,
           194, 196, 198, 199, 200, 201, 206, 208, 209, 211, 213, 215, 216, 217, 218, 222, 225, 226, 227, 230, 231,
           233, 234, 235, 236, 237, 238, 239, 240, 243, 246, 247, 248, 250, 251, 252, 256, 258, 259, 262, 264, 266,
           271, 272, 275, 276, 278, 280, 281, 282, 283, 284, 287, 288, 289, 290, 293, 294, 299]

# CURRENT ATTRIBUTION METHOD
METHODS = ['deeplift', 'lime', 'shap', 'gradcam']

# CURRENT MODEL
MODELS = ['vgg16', 'inception']


def main(method: str, model: str):
    # run some attributions
    att = Attributer(model)

    for i in range(1, 5):
        att.attribute(img_no=good_examples[i], method=method)


def print_confident_predictions(model_name: str):
    att = Attributer(model_name)
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
        self.lime_method = Lime(self.curr_model, model_name)
        self.deep_lift_method = DeepLift(self.curr_model)
        self.shap_method = Shap(self.curr_model)
        self.gradcam_method = GradCam(self.curr_model)
        # Classes for imagenet
        self.class_map = get_classification_mappings()

    def load_model(self, model_name: str):
        self.curr_model_name = model_name
        logging.info('Loading {} model architecture and weights...'.format(model_name))
        return self.models[self.curr_model_name](weights='imagenet')

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
