import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt

# util
from eval.util.constants import *
from eval.util.image_util import ImageHandler, get_classification_mappings, get_image_file_name

# image annotations and mask util
from eval.util.imagenet_annotator import demo_resizer, get_mask_for_eval, draw_annotations

# Attributer, Evaluator and Analyser classes
from eval.Attributer import Attributer
from eval.Evaluator import Evaluator
from eval.Analyser import Analyser


def attributer_wrapper(method: str, model: str):
    # draw_annotations([i for i in range(16, 300)])
    # run some attributions
    att = Attributer(model)
    for i in [11]:
        ih = ImageHandler(img_no=i, model_name=model)
        att.attribute(ih=ih,
                      method=method,
                      save=True, visualise=True,
                      take_threshold=True, take_absolute=True,
                      sigma_multiple=1)


def attribute_panel_wrapper(model_name: str):
    methods = [SHAP, GRAD]
    att = Attributer(model_name)
    for i in [11]:  # range(6, 7):
        ih = ImageHandler(img_no=i, model_name=model_name)
        att.attribute_panel(ih=ih, methods=methods,
                            save=True, visualise=True,
                            take_threshold=True, take_absolute=True,
                            sigma_multiple=1)


def evaluate_panel_wrapper(metric: str, model: str):
    #evaluator = Evaluator(metric=INTERSECT, model_name=model)
    #evaluator.collect_panel_result_batch(list(range(601, 1001)))

    evaluator = Evaluator(metric=INTENSITY, model_name=model)
    evaluator.collect_panel_result_batch(list(range(940, 1001)))


def evaluator_wrapper(method: str, model: str):
    # hardcoded evaluation metric
    metric = INTENSITY
    evaluator = Evaluator(metric=metric, model_name=model)
    evaluator.collect_result_batch(method, range(1, 600))


def annotator_wrapper():
    img_nos = list(range(1, 301))
    draw_annotations(img_nos)


def print_confident_predictions(att: Attributer, model_name: str, experiment_range: list):
    """ Prints confident predictions within a specified range of image numbers
        For collecting model predictions see Attributer.predict_for_model()
        Source of GOOD_EXAMPLES constant
    """
    for i in experiment_range:
        label, max_p = att.predict_for_model(ih=ih)
        if max_p > 0.75:
            print('image_no: {}, label: {}, probability: {:.2f}'.format(i, label, max_p))


def demo_attribute(img_nos: list = None, att: Attributer = None):
    if att is None:
        model_name = VGG
        att = Attributer(model_name=model_name)
    if img_nos is None:
        img_nos = [11, 13, 15]
        #img_nos = [6, 97, 278]
    for img_no in img_nos:
        # image handler for later (method attributions)
        ih = ImageHandler(img_no=img_no, model_name=VGG)
        # predictions
        max_pred, max_p = att.predict_for_model(ih)
        plt.figure(figsize=(15, 10))
        plt.suptitle('Attributions for example {}, prediction = `{}`, probability = {:.2f}'.format(
            img_no, max_pred, max_p))
        # original image
        plt.subplot(2, 4, 1)
        plt.axis('off')
        plt.title('ImageNet Example {}'.format(img_no))
        plt.imshow(plt.imread(get_image_file_name(IMG_BASE_PATH, img_no) + '.JPEG'))
        # annotated image
        plt.subplot(2, 4, 2)
        plt.title('Annotated Example {}'.format(img_no))
        plt.imshow(plt.imread(get_image_file_name(ANNOTATE_BASE_PATH, img_no) + '.JPEG'))
        # processed image
        plt.subplot(2, 4, 3)
        plt.title('Reshaped Example')
        plt.imshow(demo_resizer(img_no=img_no, target_size=ih.get_size()))
        # processed image
        plt.subplot(2, 4, 4)
        plt.title('Annotation Mask')
        plt.imshow(get_mask_for_eval(img_no=img_no, target_size=ih.get_size()), cmap='seismic', clim=(-1, 1))

        attributions = att.attribute_panel(ih=ih, methods=METHODS,
                                           save=False, visualise=False,
                                           take_threshold=False, take_absolute=False,
                                           sigma_multiple=1)
        # show attributions
        for i, a in enumerate(attributions.keys()):
            plt.subplot(2, 4, 5 + i)
            plt.title(a)
            plt.axis('off')
            plt.imshow(ih.get_original_img(), cmap='gray', alpha=0.75)
            plt.imshow(attributions[a], cmap='seismic', clim=(-1, 1), alpha=0.8)
        plt.show()
        plt.clf()
        plt.close()


def demo_evaluate(img_nos: list = None, att: Attributer = None, metric: str = None, model_name: str = VGG):
    if att is None:
        att = Attributer(model_name=model_name)
    if img_nos is None:
        img_nos = [6, 97, 278]
    evaluator = Evaluator(metric=metric, model_name=model_name, att=att)
    evaluator.collect_panel_result_batch(img_nos)


def analyser_wrapper(filter: str = ''):
    filter_high_confidence = False
    if filter == 'filter':
        filter_high_confidence = True
    analyser = Analyser(model_name=VGG, filter_high_confidence=filter_high_confidence)
    #analyser = Analyser(model_name=INCEPT, methods=[LIME, GRAD], metrics=[INTERSECT])
    analyser.view_panel_results()


def repl_wrapper():
    model_name = VGG
    att = Attributer(model_name)
    # REPL loop
    input_line = input(' >> ')
    while input_line != 'exit':
        split_input = input_line.split(' ')
        # demo attribute (visualiser)
        if split_input[0] == 'demo_attribute':
            img_nos = None
            if len(split_input) >= 2:
                img_nos = [int(i) for i in split_input[1:]]
            demo_attribute(img_nos, att)
        if split_input[0] == 'demo_evaluate':
            if split_input[1] not in METRICS:
                print('Invalid evaluation metric')
            else:
                img_nos = None
                if len(split_input) >= 3:
                    img_nos = [int(i) for i in split_input[2:]]
                demo_evaluate(img_nos, att, split_input[1], model_name=model_name)
        if split_input[0] == 'demo_analyse':
            if len(split_input) == 2:
                if split_input[1] == 'filter':
                    analyser_wrapper(split_input[1])
            else:
                analyser_wrapper()
        input_line = input(' >> ')


if __name__ == "__main__":
    if sys.argv[1] == 'repl':
        repl_wrapper()
        sys.exit()
    if sys.argv[1] == 'analyse':
        analyser_wrapper()
        sys.exit()
    if sys.argv[1] == 'demo_attribute':
        demo_attribute()
        sys.exit()
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
        print('Unrecognised method: {}'.format(sys.argv[2]))
        sys.exit()
    if sys.argv[3] not in MODELS:
        print('Unrecognised model: {}'.format(sys.argv[3]))
        sys.exit()
    # send commands to wrappers
    if sys.argv[1] == 'attribute':
        attributer_wrapper(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'evaluate':
        evaluator_wrapper(sys.argv[2], sys.argv[3])
