import os
import logging
import numpy as np
import tensorflow as tf

# keras models
from keras.applications import InceptionV3, VGG16, ResNet50, Xception

# methods
from eval.methods import deep_lift, LIME, SHAP, grad_cam

# util
from eval.util.image_util import ImageHandler
from eval.util.imagenet_annotator import draw_annotation
from eval.util.imagenet import get_classification_mappings
from keras.applications.imagenet_utils import decode_predictions

# constants
from eval.util.constants import IMG_BASE_PATH, XML_BASE_PATH


logging.basicConfig(level=logging.INFO)
# suppress output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

METHODS = ['deeplift', 'lime', 'shap', 'gradcam']

### ---- CURRENT ATTRIBUTION METHOD ----
CURR_METHOD = 'gradcam'


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

        # Classes for imagenet
        self.class_map = get_classification_mappings()

    def load_model(self, model_name: str):
        self.curr_model_name = model_name
        logging.info('Loading {} model architecture and weights...'.format(model_name))
        return self.models[self.curr_model_name](weights='imagenet')

    def draw_annotation(self, img_no: int, save=True):
        draw_annotation(IMG_BASE_PATH + str(img_no) + '.JPEG',
                        XML_BASE_PATH + str(img_no) + '.xml',
                        class_map=self.class_map, save_to_file=save)

    def predict_for_model(self, img_suffix: str):
        img = ImageHandler(img_suffix=img_suffix, model_name=self.curr_model_name)
        # classify
        logging.info('Classifying...')
        predictions = self.curr_model.predict(img)
        decoded_predictions = decode_predictions(predictions)

        # print the top 5 predictions, labels and probabilities
        for (i, (img_net_ID, label, p)) in enumerate(decoded_predictions[0]):
            print('{}: {}m {}, probability={:.2f}'.format(i + 1, img_net_ID, label, p))
        print('')
        return decoded_predictions[0]

    def attribute(self, img_suffix: str, method: str = CURR_METHOD):
        """Top level wrapper for collecting attributions from each method. """
        ih = ImageHandler(img_suffix=img_suffix, model_name=self.curr_model_name)
        if method == 'deeplift':
            deep_lift.attribute(self.curr_model, ih)
        elif method == 'lime':
            LIME.attribute(self.curr_model, ih)
        elif method == 'shap':
            # TODO: note requires layer specified, come up with better solution/justification
            SHAP.attribute(self.curr_model, 7, ih)
        elif method == 'gradcam':
            grad_cam.attribute(self.curr_model, ih)


def main():
    # run some attributions
    att = Attributer('vgg16')
    for i in range(3, 6):
        img_no = str(i)
        if i < 10:
            img_no = '0' + img_no
        att.attribute(img_suffix=img_no)


if __name__ == "__main__":
    main()
