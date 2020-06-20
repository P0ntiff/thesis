import pandas as pd
import numpy as np
from eval.util.constants import *
import matplotlib.pyplot as plt
from math import sqrt


class Analyser:
    def __init__(self, model_name: str, methods: list = METHODS, metrics: list = METRICS,
                 filter_high_confidence: bool = False):
        self.methods = methods
        self.metrics = metrics
        self.model_name = model_name
        self.filter_high_confidence = filter_high_confidence
        # analytics class for comparing metrics across models and methods
        self.data_map = {m: {} for m in self.metrics}
        self.method_means = {m: {} for m in self.metrics}
        self.method_std_deviations = {m: {} for m in self.metrics}
        for metric in self.metrics:
            metric_result_file = "{}/{}/{}_results.csv".format(
                RESULTS_EVAL_PATH, self.model_name, metric)
            # read data from dataframe into a Python dictionary
            self.data_map[metric] = self.ingest_metric_data(metric_result_file)
            # get statistics / aggregations
            self.method_means[metric] = self.get_method_means(metric)
            self.method_std_deviations[metric] = self.get_method_standard_deviations(metric)

    def ingest_metric_data(self, file_path: str):
        df = pd.read_csv(file_path)
        if self.filter_high_confidence:
            df = df[df['img_no'].isin(GOOD_EXAMPLES[self.model_name])]
        else:
            df = df[df['img_no'].isin(GOOD_EXAMPLES[self.model_name]) == False]
        print(df)
        metric_data_map = {'img_nos': df['img_no'].to_numpy()}
        for method in self.methods:
            metric_data_map[method] = df[method].to_numpy()
        return metric_data_map

    def get_method_means(self, metric: str):
        output_means = {}
        for method in self.methods:
            output_means[method] = np.nanmean(self.data_map[metric][method])
        return output_means

    def get_method_standard_deviations(self, metric: str):
        output_vars = {}
        for method in self.methods:
            output_vars[method] = sqrt(np.nanvar(self.data_map[metric][method]))
        return output_vars

    def view_panel_results(self):
        if len(self.metrics) != 2:
            metrics = ['IOU']
        else:
            metrics = ['IOU', 'IOU*']
        width = 0.5
        ind = np.arange(len(self.methods)) + width
        plt.figure(figsize=(6, 5))
        for i, metric in enumerate(self.metrics):
            method_means = tuple(mean for method, mean in self.method_means[metric].items())
            method_vars = tuple(var for method, var in self.method_std_deviations[metric].items())
            #plt.bar(ind + width * (i - 1), method_means, width, label=metric)
            plt.bar(x=ind + width * (i - 1),
                    height=method_means, width=width,
                    yerr=method_vars, label=metrics[i], align='center', color='orange')
        plt.ylabel('Mean IOU')
        title = ', '.join(metrics) + ' results for ' + ', '.join(self.methods)
        plt.title(title)
        plt.xticks(ticks=ind - width, labels=tuple(self.methods))
        plt.legend(loc='best')
        plt.show()

    def view_panel_performance_results(self):
        # quick function for report figure
        width = 0.5
        ind = np.arange(len(self.methods)) + width
        # timed from wrapper function
        performances = [1.5, 2.1, 3.2, 4.1, 5.1]
        plt.figure(figsize=(12, 10))
        plt.bar(x=ind,
                height=performances, width=width,
                align='center')
        plt.ylabel('Seconds')
        plt.title('Performance on 100 attributions')
        plt.xticks(ticks=ind - width, labels=tuple(self.methods))
        plt.show()
