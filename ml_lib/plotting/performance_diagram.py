#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:06:48 2020

@author: shawn.handler
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

from RadarQC.ml_lib.utils.evaluation_methods import ContingencyTable
from sklearn.metrics import average_precision_score


class PerformanceDiagram(object):

    def __init__(self):

        # create the background fiels for plot
        pod_out = np.linspace(0.01, 1, 100)
        sr_out = np.linspace(0.01, 1, 100)

        xx, yy = np.meshgrid(sr_out, pod_out)

        FB = yy/xx
        CSI = (xx ** -1 + yy ** -1 - 1.) ** -1

        # set up plotting domain
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # configure axes, etc.
        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.0])
        self.ax.set_xlabel('SR', fontsize=10)
        self.ax.set_ylabel('POD', fontsize=10)

        # Plot the contoured CSI first on the background
        cf = self.ax.contourf(xx, yy, CSI, levels=np.linspace(0.0, 1.0, 11),
                              cmap=plt.cm.get_cmap('Greys'))
        self.fig.colorbar(cf, label='Critical Success Index (CSI)')

        # Next, plot the frequency bias contour (dashed) lines
        cs = self.ax.contour(xx, yy, FB, levels=[0.5, 1.0, 1.5, 2.0, 4.0],
                             linestyles='dashed', colors='black', alpha=0.75)
        self.ax.clabel(cs, inline=1, fmt='%1.1f')

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
        self.sr = np.full((self.nboots, num_thresholds), np.nan)
        self.pod = np.full((self.nboots, num_thresholds), np.nan)
        aucs = []

        for iboot in range(self.nboots):
            sample_labels = self.true_labels[self.bootstrap_indices_list[iboot]]
            sample_predictions = self.probabilities[self.bootstrap_indices_list[iboot]]
            for ithresh in range(num_thresholds):
                table = ContingencyTable(sample_labels,
                                         sample_predictions,
                                         threshold=probability_thresholds[ithresh]
                                         )

                self.sr[iboot, ithresh] = table.getSR()
                self.pod[iboot, ithresh] = table.getPOD()

            # lastly, compute the area under the curve for each bootstrap sample
            aucs.append(average_precision_score(
                sample_labels, sample_predictions))

        # get confidence interval for auc
        self.auc_ci_bounds = np.nanpercentile(
            aucs, [2.5, 97.5], axis=0, interpolation='midpoint')
        self.auc_mean = np.mean(aucs)

        # find threshold of FB~1
        table = ContingencyTable(self.true_labels, self.probabilities)
        self.optimal_threshold = table.find_optimal_threshold(
            num_forecast_bins=self.nprob_bins)

    def plot_curve(self, color="xkcd:water blue", curve_label='model1', line_width=2):

        # get mean pod and pofd
        mean_pod = np.nanmean(self.pod, axis=0)
        mean_sr = np.nanmean(self.sr, axis=0)

        # get confidence interval for pod and pofd
        pod_ci_bounds = np.nanpercentile(self.pod, [2.5, 97.5], axis=0,
                                         interpolation='midpoint')
        sr_ci_bounds = np.nanpercentile(self.sr, [2.5, 97.5], axis=0,
                                        interpolation='midpoint')

        self.ax.plot(mean_sr, mean_pod, color=color, lw=line_width, marker='o',
                     mec='black', ms=5, label=f"{curve_label} AUPD = {'%0.2f' % (self.auc_mean)}")

        self.ax.fill_between(mean_sr, pod_ci_bounds[0, :], pod_ci_bounds[1, :],
                             color=color, alpha=0.25)
        self.ax.fill_betweenx(mean_pod, sr_ci_bounds[0, :], sr_ci_bounds[1, :],
                              color=color, alpha=0.25)

        # add probability to bubble
        # for i in range(self.nprob_bins):
        #     self.ax.text(mean_sr[i], mean_pod[i]-0.01, self.probs_as_string[i],
        #                  ha='center', fontsize=8,  color='white')

        self.ax.legend(loc="lower right")

        return self.fig

    def get_params(self):

        out_dict = {}
        out_dict['AUPD'] = (round(self.auc_ci_bounds[0], 3),
                            round(self.auc_mean, 3),
                            round(self.auc_ci_bounds[1], 3))
        out_dict['best_threshold'] = round(self.optimal_threshold, 2)

        return out_dict
