import sys

# util
from eval.util.constants import *
from eval.util.image_util import ImageHandler, ImageHelper
# image annotations and mask util
from eval.util.imagenet_annotator import get_masks_for_eval, draw_annotations, demo_annotator
# Attributer and Evaluator
from eval.Attributer import Attributer, print_confident_predictions
from eval.Evaluator import Evaluator


def attributer_wrapper(method: str, model: str):
    # draw_annotations([i for i in range(16, 300)])
    # run some attributions
    att = Attributer(model)
    for i in range(2, 3):
        ih = ImageHandler(img_no=GOOD_EXAMPLES[i], model_name=model)
        att.attribute(ih=ih,
                      method=method,
                      layer_no=LAYER_TARGETS[method][model],
                      save=True, visualise=False)


def attribute_panel_wrapper(model: str):
    att = Attributer(model)
    for i in range(2, 3):
        ih = ImageHandler(img_no=GOOD_EXAMPLES[i], model_name=model)
        for method in METHODS:
            att.attribute(ih=ih,
                          method=method,
                          layer_no=LAYER_TARGETS[method][model],
                          save=True, visualise=False)


def evaluate_panel_wrapper(metric: str, model: str):
    evaluator = Evaluator(metric=INTERSECT, model_name=model)
    evaluator.collect_panel_result_batch(range(160, 165))

    evaluator = Evaluator(metric=INTENSITY, model_name=model)
    evaluator.collect_panel_result_batch(range(1, 2))


def evaluator_wrapper(method: str, model: str):
    # hardcoded evaluation metric
    metric = INTERSECT
    evaluator = Evaluator(metric=metric, model_name=model)
    evaluator.collect_result_batch(method, range(2, 3))


def annotator_wrapper():
    # hardcoded test output function
    demo_annotator(img_no=11, target_size=STD_IMG_SIZE)

    #get_masks_for_eval(GOOD_EXAMPLES[2:3], ImageHelper.get_size(VGG), visualise=True, save=False)
    # output predictions for interest
    print_confident_predictions(VGG, experiment_range=range(2, 3))




class Analyser:
    def __init__(self):
        # analytics class for comparing metrics across models and methods
        pass



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
    # `attribute_panel` command line option
    if sys.argv[1] == 'attribute_panel':
        if len(sys.argv) != 3:
            print('Usage: main attribute_panel <model>[vgg16|inception]')
        if sys.argv[2] not in MODELS:
            print('Unrecognised model: {}'.format(sys.argv[2]))
        attribute_panel_wrapper(sys.argv[2])
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
