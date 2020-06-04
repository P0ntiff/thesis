import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# util
from eval.util.constants import *
from eval.util.image_util import ImageHandler, get_classification_mappings, get_image_file_name

# image annotations and mask util
from eval.util.imagenet_annotator import draw_annotation, demo_resizer, get_mask_for_eval

# Attributer and Evaluator
from eval.Attributer import Attributer, print_confident_predictions
from eval.Evaluator import Evaluator
from eval.Analyser import Analyser


def attributer_wrapper(method: str, model: str):
    # draw_annotations([i for i in range(16, 300)])
    # run some attributions
    att = Attributer(model)
    for i in [6, 11]:
        ih = ImageHandler(img_no=i, model_name=model)
        att.attribute(ih=ih,
                      method=method,
                      save=True, visualise=True,
                      take_threshold=True, take_absolute=True,
                      sigma_multiple=1)


def attribute_panel_wrapper(model: str):
    att = Attributer(model)
    for i in [283, 284]:  # range(6, 7):
        ih = ImageHandler(img_no=i, model_name=model)
        att.attribute_panel(ih=ih, methods=METHODS,
                            save=True, visualise=True,
                            take_threshold=True, take_absolute=True,
                            sigma_multiple=1)


def evaluate_panel_wrapper(metric: str, model: str):
    evaluator = Evaluator(metric=INTERSECT, model_name=model)
    evaluator.collect_panel_result_batch(range(1, 301))

    evaluator = Evaluator(metric=INTENSITY, model_name=model)
    evaluator.collect_panel_result_batch(range(220, 301))


def evaluator_wrapper(method: str, model: str):
    # hardcoded evaluation metric
    metric = INTERSECT
    evaluator = Evaluator(metric=metric, model_name=model)
    evaluator.collect_result_batch(method, range(2, 3))


def annotator_wrapper():
    img_nos = [1, 301]
    class_map = get_classification_mappings()
    for img_no in img_nos:
        # demo_annotator(img_no=283, target_size=STD_IMG_SIZE)
        draw_annotation(img_no=img_no, class_map=class_map)
        # get_masks_for_eval(GOOD_EXAMPLES[2:3], ImageHelper.get_size(VGG), visualise=True, save=False)
        # output predictions for interest
        print_confident_predictions(VGG, experiment_range=[img_no])


def demo_attribute(img_nos: list = None):
    if img_nos is None:
        img_nos = [1]
    img_nos = [13] # 6, 283, 56
    model_name = VGG
    class_map = get_classification_mappings()
    att = Attributer(model_name)
    plt.figure(figsize=(15, 10))
    for img_no in img_nos:
        # image handler for later (method attributions)
        ih = ImageHandler(img_no=img_no, model_name=VGG)
        # predictions
        print_confident_predictions(VGG, experiment_range=[img_no])
        # original image
        plt.subplot(2, 4, 1)
        plt.axis('off')
        plt.title('ImageNet Example {}'.format(img_no))
        plt.imshow(plt.imread(get_image_file_name(IMG_BASE_PATH, img_no) + '.JPEG'))
        # annotated image
        plt.subplot(2, 4, 2)
        plt.title('Annotated Example')
        plt.imshow(plt.imread(get_image_file_name(ANNOTATE_BASE_PATH, img_no) + '.JPEG'))
        # processed image
        plt.subplot(2, 4, 3)
        plt.title('Reshaped for Model Input')
        plt.imshow(demo_resizer(img_no=img_no, target_size=ih.get_size()))
        # processed image
        plt.subplot(2, 4, 4)
        plt.title('Annotation Mask')
        plt.imshow(get_mask_for_eval(img_no=img_no, target_size=ih.get_size()), cmap='seismic', clim=(-1, 1))

        attributions = att.attribute_panel(ih=ih, methods=METHODS,
                                           save=False, visualise=False,
                                           take_threshold=True, take_absolute=True,
                                           sigma_multiple=1)
        # show attributions
        for i, a in enumerate(attributions.keys()):
            plt.subplot(2, 4, 5 + i)
            plt.title(a)
            plt.axis('off')
            plt.imshow(ih.get_original_img(), cmap='gray', alpha=0.75)
            plt.imshow(attributions[a], cmap='seismic', clim=(-1, 1), alpha=0.8)
        plt.show()
        plt.cla()


def analyser_wrapper():
    analyser = Analyser(model_name=VGG)
    analyser.view_panel_results()


def repl_wrapper():
    input_line = input('  >> ')
    while input_line != 'exit':
        split_input = input_line.split(' ')

        if split_input[0] == 'demo_attribute':
            img_nos = [int(i) for i in split_input[1:]]
            demo_attribute(img_nos)


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
