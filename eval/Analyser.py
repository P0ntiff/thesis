import pandas as pd
import numpy as np
from eval.util.constants import *
import matplotlib.pyplot as plt


class Analyser:
    def __init__(self, model_name: str, methods: list = METHODS, metrics: list = METRICS,
                 filter_high_confidence: bool = False):
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
        print(self.method_means)
        for i, metric in enumerate(self.metrics):
            method_means = tuple(mean for method, mean in self.method_means[metric].items())
            print(method_means)
            plt.bar(ind + width * (i - 1), method_means, width, label=metric)
        plt.ylabel('Eval Metric')
        plt.title(','.join(self.metrics) + ' results for ' + ','.join(self.methods))
        plt.xticks(ind + width / 2, tuple(self.methods))
        plt.legend(loc='best')
        plt.show()