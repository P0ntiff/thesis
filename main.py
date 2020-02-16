import logging


# methods
from eval.methods import deep_lift
from eval.methods import LIME
from eval.methods import SHAP
from eval.methods import grad_cam
from eval.util.imagenet_annotator import draw_annotation
from eval.util.imagenet import get_classification_mappings

#constants
from eval.util.constants import IMG_BASE_PATH
from eval.util.constants import XML_BASE_PATH

# model stuff
from eval.models.keras_model_executor import predict
from keras.applications import InceptionV3
from keras.applications import VGG16

logging.basicConfig(level=logging.ERROR)

RESULTS_BASE_PATH = 'results/initial/'

METHODS = ['deeplift', 'lime', 'shap', 'gradcam']
CURR_METHOD = 'gradcam'

MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3
}
CURR_MODEL = 'vgg16'

# Classes for imagenet
CLASS_MAP = get_classification_mappings()


def main():
    #draw_annotation(IMG_BASE_PATH + '14.JPEG',
    #                XML_BASE_PATH + '14.xml', class_map=class_map, save_to_file=True)

    #predict('vgg16', IMG_BASE_PATH + '04.JPEG')
    # TODO: OOPify the methods so that resources are cached effectively (not spun up every time)

    # run some attributions
    for i in range(1, 5):
        img_no = str(i)
        if i < 10:
            img_no = '0' + img_no
        attribute(CURR_METHOD, CURR_MODEL, img_no)


def attribute(method: str, model_name: str, img_no: str):
    """Top level wrapper for collecting attributions from each method. """
    input_img_path = IMG_BASE_PATH + img_no + '.JPEG'
    output_img_path = RESULTS_BASE_PATH + method + '/' + method + '_' + img_no + '.png'
    model = MODELS[model_name](weights='imagenet')
    if method == 'deeplift':
        deep_lift.attribute(model_name, model, input_img_path, output_img_path)
    elif method == 'lime':
        LIME.attribute(model_name, model, input_img_path, output_img_path)
    elif method == 'shap':
        # TODO: note requires layer specified, come up with better solution/justification
        SHAP.attribute(model_name, model, 7, input_img_path, output_img_path)
    elif method == 'gradcam':
        grad_cam.attribute(model_name, model, input_img_path, output_img_path)


if __name__ == "__main__":
    main()
