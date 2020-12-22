#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:36:02 2020

@author: shawn.handler
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

from RadarQC.ml_lib.utils.evaluation_methods import BrierSkillScore_Decomposed, BrierSkillScore


class AttributesDiagram(object):

    def __init__(self):

        # plot background
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.0])

        self.ax.plot([0, 1], [0, 1], "k:", linestyle='--')

        self.ax.set_xlabel('Forecast probabilities')
        self.ax.set_ylabel('Conditional event frequency')

        # insert for probabilities
        left, bottom, width, height = [0.2, 0.6, 0.2, 0.2]
        self.ax_hist = self.fig.add_axes([left, bottom, width, height])
        self.ax_hist.set_yscale('log')

    def make_curve(self, true_labels, probabilities, nprob_bins=10, boostrap_indices_list=None):

        # store inputs to current object
        self.true_labels = true_labels
        self.probabilities = probabilities
        self.nprob_bins = nprob_bins
        self.bootstrap_indices_list = boostrap_indices_list

        if self.bootstrap_indices_list is None:
            self.nboots = 1
            self.bootstrap_indices_list = [np.arange(true_labels.shape[0])]
        else:
            self.nboots = len(self.bootstrap_indices_list)

        # arrays to store the pofd and pod data
        self.mean_forecast_prob_by_bin = np.full(
            (self.nboots, self.nprob_bins), np.nan)
        self.mean_observed_label_by_bin = np.full(
            (self.nboots, self.nprob_bins), np.nan)

        reliability = []
        resolution = []
        uncertainty = []
        bss = []

        for iboot in range(self.nboots):
            sample_labels = self.true_labels[self.bootstrap_indices_list[iboot]]
            sample_predictions = self.probabilities[self.bootstrap_indices_list[iboot]]

            # split forecast probs into bins (basically a histogram)
            bin_cutoffs = np.linspace(0., 1., num=self.nprob_bins + 1)
            input_to_bin_indices = np.digitize(
                sample_predictions, bin_cutoffs, right=False) - 1

            input_to_bin_indices[input_to_bin_indices < 0] = 0
            input_to_bin_indices[input_to_bin_indices >
                                 self.nprob_bins - 1] = self.nprob_bins - 1

            # get the number of examples for each bin
            self.num_examples_by_bin = np.full(self.nprob_bins, -1, dtype=int)

            for ibin in range(self.nprob_bins):
                self.num_examples_by_bin[ibin] = np.sum(
                    input_to_bin_indices == ibin)

            for ibin in range(self.nprob_bins):
                indices = np.where(input_to_bin_indices == ibin)[0]

                self.mean_forecast_prob_by_bin[iboot, ibin] = np.nanmean(
                    sample_predictions[indices])
                self.mean_observed_label_by_bin[iboot, ibin] = np.nanmean(
                    sample_labels[indices].astype(float))

            # compute brier skill score
            bss_dict = BrierSkillScore_Decomposed(sample_labels, sample_predictions,
                                                  pos_label=1, n_bins=self.nprob_bins)
            bss.append(BrierSkillScore(sample_labels, sample_predictions,
                                       pos_label=1))

            resolution.append(bss_dict['res'])
            reliability.append(bss_dict['rel'])
            uncertainty.append(bss_dict['unc'])

        self.mean_bss = round(np.mean(bss), 4)
        self.mean_rel = round(np.mean(reliability), 4)
        self.mean_res = round(np.mean(resolution), 4)
        self.mean_unc = round(np.mean(uncertainty), 4)

    def compute_uncertainty_bars(self):
        pass

    def plot_curve(self, color="xkcd:water blue", curve_label='model1', line_width=2):

        climo_x = np.nanmean(self.true_labels)
        climo_x = np.full(10, climo_x)
        no_res = np.nanmean(self.true_labels)
        y1 = 0.5*np.nanmean(self.true_labels)
        y2 = 0.5*(1+np.nanmean(self.true_labels))

        x = [0, 1]
        yy = [y1, y2]

        coefficients = np.polyfit(x, yy, 1)

        polynomial = np.poly1d(coefficients)
        x1 = np.linspace(0.0, 1.0, 10)
        y1 = polynomial(x1)

        x2 = climo_x
        y2 = np.linspace(0.0, 1.0, 10)

        yi = np.sort(np.c_[y1, y2].flatten())
        x1i = np.interp(yi, y1, x1)
        x2i = np.interp(yi, y2, x2)

        self.ax.plot([climo_x, climo_x], [0, 1], "k:", linestyle='--')
        self.ax.plot([0, 1], [no_res, no_res], "k:", linestyle='--')
        self.ax.plot(x1, y1, "k:", linestyle='--')

        # filled region denoting BSS > 0
        self.ax.fill_betweenx(yi, x1i, x2i, alpha=0.5, color='lightgray')

        # get mean fraction of positives and predicted
        mean_frac = np.nanmean(self.mean_forecast_prob_by_bin, axis=0)
        mean_pred = np.nanmean(self.mean_observed_label_by_bin, axis=0)

        # get confidence interval bounds
        mean_frac_bounds = np.nanpercentile(self.mean_forecast_prob_by_bin, [2.5, 97.5], axis=0,
                                            interpolation='midpoint')
        mean_pred_bounds = np.nanpercentile(self.mean_observed_label_by_bin, [2.5, 97.5], axis=0,
                                            interpolation='midpoint')

        # actual plot of mean data
        self.ax.plot(mean_pred, mean_frac, color=color, lw=line_width, marker='o',
                     mec='black', label=f"{curve_label} BSS = {'%0.3f' % (self.mean_bss)}")

        # filling between CI bounds
        self.ax.fill_between(mean_pred, mean_frac_bounds[0, :], mean_frac_bounds[1, :],
                             color=color, alpha=0.25)
        self.ax.fill_betweenx(mean_frac, mean_pred_bounds[0, :], mean_pred_bounds[1, :],
                              color=color, alpha=0.25)

        # add the probability distribution
        self.add_prob_dist(color=color)

        # plot the reliability uncertainty lines
        # if (lower_unc is not None):
        #     for i,xval in enumerate(mean_pred):
        #         self.ax.axvline(x=xval, ymin=lower_unc[i], ymax=upper_unc[i],
        #                         color=color, alpha=0.75, linewidth=self.line_width)

        self.ax.legend(loc="lower right")

        return self.fig

    def add_prob_dist(self, color='red'):

        bins = np.linspace(0.0, 1.0, self.nprob_bins+1)

        # compute histogram
        hist, edges = np.histogram(self.probabilities, bins=bins)

        xplot = (edges[1:]+edges[:-1])/2

        self.ax_hist.plot(xplot, hist, alpha=1.0,
                          linewidth=1.5, color=color)

    def get_params(self):

        out_dict = {}
        out_dict['BSS'] = self.mean_bss
        out_dict['REL'] = self.mean_rel
        out_dict['RES'] = self.mean_res
        out_dict['UNC'] = self.mean_unc

        return out_dict
