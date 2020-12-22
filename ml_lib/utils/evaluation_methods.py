import numpy as np
import pandas as pd
import os

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.calibration import calibration_curve

#-----------------------------------------------------------------------------------------#


def make_roc_curve(true_labels, your_probs, nbins=10):

    probabilities = np.linspace(0., 1., nbins+1)

    num_thresholds = probabilities.shape[0]

    pofd = np.full(num_thresholds, np.nan)
    pod = np.full(num_thresholds, np.nan)

    for i in range(num_thresholds):
        table = ContingencyTable(
            true_labels, your_probs, threshold=probabilities[i])

        pofd[i] = table.getPOFD()
        pod[i] = table.getPOD()

    return pofd, pod
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#


def my_roc_auc(pofd_array, pod_array):

    sort_indices = np.argsort(-pofd_array)
    pofd_by_threshold = pofd_array[sort_indices]
    pod_by_threshold = pod_array[sort_indices]

    nan_flags = np.logical_or(
        np.isnan(pofd_by_threshold),
        np.isnan(pod_by_threshold)
    )
    if np.all(nan_flags):
        return np.nan

    real_indices = np.where(np.invert(nan_flags))[0]

    # determine the optimal probability threshold
    probabilities = np.linspace(0., 1., pod_array.shape[0])
    optimal_idx = np.argmax(pod_array - pofd_array)
    optimal_threshold = probabilities[optimal_idx]

    return round(auc(pofd_array[real_indices], pod_array[real_indices]), 2), optimal_threshold
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#


def my_au_pd(true_labels, probabilities):

    return average_precision_score(true_labels, probabilities)
#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
def get_reliability_points(true_labels, your_probs, num_forecast_bins=10):

    # split forecast probs into bins (basically a histogram)

    bin_cutoffs = np.linspace(0., 1., num=num_forecast_bins + 1)
    input_to_bin_indices = np.digitize(
        your_probs, bin_cutoffs, right=False) - 1

    input_to_bin_indices[input_to_bin_indices < 0] = 0
    input_to_bin_indices[input_to_bin_indices >
                         num_forecast_bins - 1] = num_forecast_bins - 1

    num_examples_by_bin = np.full(num_forecast_bins, -1, dtype=int)

    for j in range(num_forecast_bins):
        num_examples_by_bin[j] = np.sum(input_to_bin_indices == j)

    mean_forecast_prob_by_bin = np.full(num_forecast_bins, np.nan)
    mean_observed_label_by_bin = np.full(num_forecast_bins, np.nan)

    for i in range(num_forecast_bins):
        indices = np.where(input_to_bin_indices == i)[0]

        mean_forecast_prob_by_bin[i] = np.nanmean(your_probs[indices])
        mean_observed_label_by_bin[i] = np.nanmean(
            true_labels[indices].astype(float))

    return mean_forecast_prob_by_bin, mean_observed_label_by_bin, num_examples_by_bin
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#


def get_performance_diag_points(true_labels, your_probs, num_forecast_bins=10):

    probabilities = np.linspace(0., 1., num=num_forecast_bins + 1)

    num_thresholds = probabilities.shape[0]

    SR = np.full(num_thresholds, np.nan)
    POD = np.full(num_thresholds, np.nan)

    for i in range(num_thresholds):

        table = ContingencyTable(
            true_labels, your_probs, threshold=probabilities[i])

        SR[i] = table.getSR()
        POD[i] = table.getPOD()

    return SR, POD
#-----------------------------------------------------------------------------------------#


def reliability_uncertainty(X, Y, nboots=100, nbins=10):
    '''
    Calculates the uncertainty of the event frequency based on Brocker and Smith (WAF, 2007)
    '''
    event_freq_set = []
    n_obs = X.shape[0]

    for _ in range(nboots):
        Z = np.random.uniform(size=n_obs)
        X_hat = np.random.choice(X, size=n_obs, replace=True)
        Y_hat = np.where(Z < X_hat, 1, 0)
        _, event_freq = get_reliability_points(
            Y_hat, X_hat, num_forecast_bins=nbins)
        event_freq_set.append(event_freq)

    return np.array(event_freq_set)


def BrierSkillScore(y_test, your_probs, pos_label=1, custom_mean=None):
    """Returns the Brier Skill Score (a float) in general form (not broken down via
    reliability, resolution, and uncertainty)."""

    N = your_probs.shape[0]

    bs_forecast = brier_score_loss(y_test, your_probs, pos_label=pos_label)
    bs_climatology = (1.0/N)*np.sum(((np.nanmean(y_test)-y_test)**2))

    bss = 1 - (bs_forecast/bs_climatology)

    return bss


def BrierSkillScore_Decomposed(y_test, your_probs, pos_label=1, n_bins=10):
    """Returns the Brier Skill Score (a float) decomposed into
    reliability, resolution, and uncertainty compenents as a dictionary."""

    forecast, observed, nexamples = get_reliability_points(
        y_test, your_probs, num_forecast_bins=n_bins)

    inner_rel = inner_res = 0

    for j in range(forecast.shape[0]):
        inner_rel += (nexamples[j])*((forecast[j]-observed[j])**2)
        inner_res += (nexamples[j])*((observed[j]-np.nanmean(y_test))**2)

    rel = inner_rel/np.sum(nexamples)
    res = inner_res/np.sum(nexamples)
    unc = np.nanmean(y_test)*(1-np.nanmean(y_test))

    bss = (res-rel)/unc

    return{'bss': bss,
           'res': res,
           'rel': rel,
           'unc': unc}


class ContingencyTable():

    def __init__(self, observed, predictions, threshold=None):

        self.y_observed = observed
        self.y_pred = predictions
        self.threshold = threshold

        self.TP = None
        self.FP = None
        self.FN = None
        self.TN = None

        self.createTable()

    def createTable(self):

        # set a default threshold if one isn't provided/defined
        if (self.threshold is None):
            self.threshold = 0.5

        # turn probability into binary 0 or 1 based on threshold value
        binary_prediction = np.where(self.y_pred >= self.threshold, 1, 0)

        # use scikit-learn confusion matrix here...
        conf_mat = confusion_matrix(self.y_observed, binary_prediction)

        # set the true positive, false positive, etc...
        self.TP = float(conf_mat[1, 1])
        self.FP = float(conf_mat[0, 1])
        self.FN = float(conf_mat[1, 0])
        self.TN = float(conf_mat[0, 0])

    def getPOD(self):
        """compute and return the prob of detection """

        if (self.TP + self.FN == 0):
            return np.nan
        else:
            return round(self.TP / (self.TP + self.FN), 2)

    def getPOFD(self):
        """compute and return the prob of false detection """

        if (self.FP + self.TN == 0):
            return np.nan
        else:
            return round(self.FP / (self.FP + self.TN), 2)

    def getSR(self):
        """compute and return the success ratio """

        if (self.TP + self.FP == 0):
            return np.nan
        else:
            return round(self.TP / (self.TP + self.FP), 2)

    def getFAR(self):
        """compute and return the false alarm ratio """

        if (self.TP + self.FP == 0):
            return np.nan
        else:
            return round(1 - self.getSR(), 2)

    def getFB(self):
        """compute and return the frequency bias """

        if (self.TP + self.FN == 0):
            return np.nan
        else:
            return round((self.TP+self.FP) / (self.TP+self.FN), 2)

    def getCSI(self):
        """compute and return the critical success index (threat score) """

        if (self.TP+self.FP+self.FN == 0):
            return np.nan
        else:
            return round(self.TP / (self.TP+self.FP+self.FN), 2)

    def getPSS(self):
        """compute and return the pierce skill score """

        if ((self.TP + self.FN == 0) | (self.FP + self.TN == 0)):
            return np.nan
        else:

            pss = ((self.TP*self.TN) -
                   (self.FP*self.FN)) / ((self.TP + self.FN) *
                                         (self.FP + self.TN))

            return round(pss, 2)

    def find_optimal_threshold(self, num_forecast_bins=10):

        probabilities = np.linspace(0., 1., num_forecast_bins+1)
        FB = []
        CSI = []
        for prob in probabilities:

            threshold = prob

            # turn probability into binary 0 or 1 based on threshold value
            binary_prediction = np.where(self.y_pred >= threshold, 1, 0)

            # use scikit-learn confusion matrix here...
            conf_mat = confusion_matrix(self.y_observed, binary_prediction)

            # set the true positive, false positive, etc...
            self.TP = float(conf_mat[1, 1])
            self.FP = float(conf_mat[0, 1])
            self.FN = float(conf_mat[1, 0])
            self.TN = float(conf_mat[0, 0])

            FB.append(self.getFB())
            CSI.append(self.getCSI())

        # find FB ~1
        fb_near_one = np.abs(1-np.array(FB))
        best_index = np.argmin(fb_near_one)

        print(probabilities[best_index], FB[best_index], CSI[best_index])

        # reset threshold and recompute table with best threshold
        self.threshold = probabilities[best_index]
        self.createTable()

        return probabilities[best_index]
