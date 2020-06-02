import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def analyser_wrapper():
    analyser = Analyser(model_name=VGG)
    analyser.view_panel_results()


class Analyser:
    def __init__(self, model_name: str, methods = METHODS, metrics = METRICS, filter_high_confidence = False):
        self.methods = methods
        self.metrics = [INTERSECT, INTENSITY]
        self.filter_high_confidence = True
        # analytics class for comparing metrics across models and methods
        self.data_map = {m: {} for m in self.metrics}
        self.method_means = {m: {} for m in self.metrics}
        for metric in self.metrics:
            metric_result_file = "{}/{}/{}_results.csv".format(
                RESULTS_EVAL_PATH, model_name, metric)
            # read data from dataframe into a Python dictionary
            self.data_map[metric] = self.ingest_metric_data(metric_result_file)
            # get statistics / aggregations
            self.method_means[metric] = self.get_method_means(metric)

    def ingest_metric_data(self, file_path: str):
        df = pd.read_csv(file_path)
        if self.filter_high_confidence:
            df = df[df['img_no'].isin(GOOD_EXAMPLES)]
        metric_data_map = {'img_nos': df['img_no'].to_numpy()}
        for method in self.methods:
            metric_data_map[method] = df[method].to_numpy()
        return metric_data_map

    def get_method_means(self, metric: str):
        output_means = {}
        for method in self.methods:
            output_means[method] = np.nanmean(self.data_map[metric][method])
        return output_means

    def get_method_variances(self, metric: str):
        pass

    def view_panel_results(self):
        ind = np.arange(4)
        width = 0.4
        for i, metric in enumerate(self.metrics):
            method_means = tuple(mean for method, mean in self.method_means[metric].items())
            print(method_means)
            plt.bar(ind + width * (i - 1), method_means, width, label=metric)
        plt.ylabel('Eval Metric')
        plt.title(','.join(self.metrics) + ' results for ' + ','.join(self.methods))
        plt.xticks(ind + width / 2, tuple(self.methods))
        plt.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    if sys.argv[1] == 'analyse':
        analyser_wrapper()
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
