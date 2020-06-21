import os
import pandas as pd
import numpy as np

# Attributer class
from eval.Attributer import Attributer

# util
from eval.util.constants import *
from eval.util.image_util import ImageHandler, show_figure, show_intersect_union_subfigures
from eval.util.imagenet_annotator import get_mask_for_eval


class Evaluator:
    def __init__(self, metric: str, model_name: str, att: Attributer = None):
        if att is None:
            self.att = Attributer(model_name=model_name)
        else:
            self.att = att
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
            df = pd.DataFrame(columns=['img_no'] + self.file_headers).set_index(
                'img_no')
            return df
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(','.join(['img_no'] + self.file_headers))
        df = pd.read_csv(file_path).set_index('img_no')
        return df

    def write_results_to_file(self):
        self.results_df.to_csv(self.result_file, index=True, index_label='img_no')

    def get_image_handler_and_mask(self, img_no):
        # this gets the image wrapped in the ImageHandler object, and the
        # bounding box annotation mask for the image,
        # ImageHandler is used to calculate attributions by each method, and the
        # mask is used for evaluation
        ih = ImageHandler(img_no=img_no, model_name=self.model_name)
        # bounding box in the format of the model's input shape / attribution shape
        annotation_mask = get_mask_for_eval(img_no=img_no, target_size=ih.get_size(),
                                            save=False, visualise=False)
        return ih, annotation_mask

    def collect_panel_result_batch(self, experiment_range: list):
        new_rows = {}
        for img_no in experiment_range:
            new_row = {}
            ih, annotation_mask = self.get_image_handler_and_mask(img_no)
            for method in METHODS:
                result = self.collect_result(ih, annotation_mask, method)
                if img_no <= len(self.results_df.index):
                    self.results_df.at[img_no, method] = result
                else:
                    new_row[method] = result
            new_rows[img_no] = new_row
            if (img_no % 10) == 0:
                self.append_to_results_df(new_rows)
                new_rows = {}
        self.append_to_results_df(new_rows)

    def collect_result_batch(self, method: str, experiment_range: range):
        new_rows = {}
        for img_no in experiment_range:
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

    def append_to_results_df(self, new_rows_dict, write=True):
        new_data = pd.DataFrame.from_dict(new_rows_dict,
                                          columns=self.file_headers,
                                          orient='index')
        self.results_df = self.results_df.append(new_data, sort=True)
        if write:
            self.write_results_to_file()

    def collect_result(self, ih: ImageHandler, mask, method: str):
        # threshold for each attribution's "explainability" is the number of std
        # deviations above the mean contribution score for a pixel
        if self.metric == INTERSECT:
            return self.evaluate_intersection(ih, mask, method,
                                              sigma=INTERSECT_THRESHOLD)
        elif self.metric == INTENSITY:
            return self.evaluate_intensity(ih, mask, method,
                                           sigma=INTENSITY_THRESHOLD)

    def evaluate_intersection(self, ih: ImageHandler, mask, method: str,
                              sigma: int, print_debug: bool = False) -> float:
        # calculate an attribution and use a provided bounding box mask to
        # calculate the IOU metric. attribution has threshold applied, and abs
        # value set (positive and negative evidence treated the same)
        attribution = self.att.attribute(ih=ih,
                                         method=method,
                                         layer_no=LAYER_TARGETS[method][self.model_name],
                                         take_threshold=True, sigma_multiple=sigma,
                                         take_absolute=True,
                                         visualise=False, save=False)
        # calculate the intersection of the attribution and the bounding box mask
        intersect_array = np.zeros(attribution.shape)
        intersect_array[(attribution > 0.0) * (mask > 0.0)] = 1
        # get the union array for the IOU calculation
        union_array = np.zeros(attribution.shape)
        union_array[(attribution > 0.0) + (mask > 0.0)] = 1
        # calculate intersection and union areas for numerator and
        # denominator respectively
        intersect_area = intersect_array.sum()
        union_area = union_array.sum()
        intersection_over_union = intersect_area / union_area
        print('Evaluating `{}` on example `{}` ({})'.format(
            method, ih.img_no, 'intersection'))
        if print_debug:
            #print('--Mask Area =\t {}'.format(mask_area))
            print('--Intersect Area =\t {}'.format(intersect_area))
            print('--Union Area =\t {}'.format(union_area))
            print('--Intersection / Union =\t{:.2f}%'.format
                  (intersection_over_union * 100))
            print('')

        return intersection_over_union

    def evaluate_intensity(self, ih: ImageHandler, mask, method: str, sigma: int,
                           print_debug: bool = False) -> float:
        # # calculate an attribution and use a provided bounding box mask to
        # calculate the IOU metric attribution has threshold applied, and abs
        # value set (positive and negative evidence treated the same)
        attribution = self.att.attribute(ih=ih,
                                         method=method,
                                         layer_no=LAYER_TARGETS[method][self.model_name],
                                         take_threshold=True, sigma_multiple=sigma,
                                         take_absolute=True,
                                         visualise=False, save=True)
        # calculate the weight/confidence of the attribution intersected with
        # the bounding box mask
        intensity_array = np.copy(attribution)
        intensity_array[(attribution > 0.0) * (mask < 0.1)] = 0
        # get the union array for the IOU* calculation
        union_array = np.zeros(attribution.shape)
        union_array[(attribution > 0.0) + (mask > 0.0)] = 1
        intensity_area = intensity_array.sum()
        union_area = union_array.sum()
        intensity_over_union = intensity_area / union_area
        print('Evaluating `{}` on example `{}` ({})'.format(
            method, ih.img_no, 'intensity'))
        if print_debug:
            print('--Intersect Area =\t {}'.format(intensity_area))
            print('--Union Area =\t {}'.format(union_area))
            print('--Intensity / Union =\t{:.2f}%'.format(intensity_over_union * 100))
            print('')

        return intensity_over_union
