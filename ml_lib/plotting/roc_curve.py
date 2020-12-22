import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

from RadarQC.ml_lib.utils.evaluation_methods import ContingencyTable
from sklearn.metrics import auc


class ROC(object):

    def __init__(self):

        # set up plotting domain
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # configure axes, etc.
        self.ax.plot([0, 1], [0, 1], "black", lw=2, linestyle='--')
        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.01])
        self.ax.set_xlabel('POFD')
        self.ax.set_ylabel('POD')

    def make_curve(self, true_labels, probabilities, nprob_bins=10, boostrap_indices_list=None):

        self.true_labels = true_labels
        self.probabilities = probabilities
        self.nprob_bins = nprob_bins
        self.bootstrap_indices_list = boostrap_indices_list

        if self.bootstrap_indices_list is None:
            self.nboots = 1
            self.bootstrap_indices_list = [np.arange(true_labels.shape[0])]
        else:
            self.nboots = len(self.bootstrap_indices_list)

        probability_thresholds = np.linspace(0., 1., self.nprob_bins+1)
        self.probs_as_string = np.char.mod('%d', probability_thresholds*100.0)
        num_thresholds = probability_thresholds.shape[0]

        # arrays to store the pofd and pod data
        self.pofd = np.full((self.nboots, num_thresholds), np.nan)
        self.pod = np.full((self.nboots, num_thresholds), np.nan)

        for iboot in range(self.nboots):
            sample_labels = self.true_labels[self.bootstrap_indices_list[iboot]]
            sample_predictions = self.probabilities[self.bootstrap_indices_list[iboot]]
            for ithresh in range(num_thresholds):
                table = ContingencyTable(sample_labels,
                                         sample_predictions,
                                         threshold=probability_thresholds[ithresh]
                                         )

                self.pofd[iboot, ithresh] = table.getPOFD()
                self.pod[iboot, ithresh] = table.getPOD()

        # lastly, compute the area under the curve for each bootstrap sample
        self.compute_auc()

    def compute_auc(self):

        # lists of AUCs and their optimal thresholds
        aucs = []
        optimal_threshold = []

        # define probability thresholds
        probability_thresholds = np.linspace(0., 1., self.pod.shape[1])

        for iboot in range(self.nboots):
            sort_indices = np.argsort(-self.pofd[iboot, :])
            pofd_by_threshold = self.pofd[iboot, sort_indices]
            pod_by_threshold = self.pod[iboot, sort_indices]

            nan_flags = np.logical_or(
                np.isnan(pofd_by_threshold),
                np.isnan(pod_by_threshold)
            )
            if np.all(nan_flags):
                return np.nan

            # find all finite indices
            real_indices = np.where(np.invert(nan_flags))[0]

            # determine the optimal probability threshold
            optimal_idx = np.argmax(pod_by_threshold - pofd_by_threshold)

            # store optimal probability threshold as well as AUC value
            optimal_threshold.append(probability_thresholds[optimal_idx])
            aucs.append(
                auc(pofd_by_threshold[real_indices], pod_by_threshold[real_indices]))

        # get confidence interval for auc
        self.auc_ci_bounds = np.nanpercentile(
            aucs, [2.5, 97.5], axis=0, interpolation='midpoint')
        self.auc_mean = np.mean(aucs)
        self.optimal_threshold = np.mean(optimal_threshold)

    def plot_curve(self, color="xkcd:water blue", curve_label='model1', line_width=2):

        # get mean pod and pofd
        mean_pod = np.nanmean(self.pod, axis=0)
        mean_pofd = np.nanmean(self.pofd, axis=0)

        # get confidence interval for pod and pofd
        pod_ci_bounds = np.nanpercentile(self.pod, [2.5, 97.5], axis=0,
                                         interpolation='midpoint')
        pofd_ci_bounds = np.nanpercentile(self.pofd, [2.5, 97.5], axis=0,
                                          interpolation='midpoint')

        self.ax.plot(mean_pofd, mean_pod, color=color, lw=line_width, marker='o',
                     mec='black', ms=5, label=f"{curve_label} AUC = {'%0.2f' % (self.auc_mean)}")

        self.ax.fill_between(mean_pofd, pod_ci_bounds[0, :], pod_ci_bounds[1, :],
                             color=color, alpha=0.25)
        self.ax.fill_betweenx(mean_pod, pofd_ci_bounds[0, :], pofd_ci_bounds[1, :],
                              color=color, alpha=0.25)

        # add probability to bubble
        # for i in range(self.nprob_bins):
        #     self.ax.text(mean_pofd[i], mean_pod[i]-0.01, self.probs_as_string[i],
        #                  ha='center', fontsize=8,  color='white')

        self.ax.legend(loc="lower right")

        return self.fig

    def get_params(self):

        out_dict = {}
        out_dict['AUC'] = (round(self.auc_ci_bounds[0], 3),
                           round(self.auc_mean, 3),
                           round(self.auc_ci_bounds[1], 3))
        out_dict['best_threshold'] = round(self.optimal_threshold, 3)

        return out_dict
