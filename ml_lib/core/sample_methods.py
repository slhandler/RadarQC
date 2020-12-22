import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from dateutil.relativedelta import *

from RadarQC.ml_lib.utils.utils_funcs import splitTargetFromLabels


class SampleTechniques():

    def __init__(self, data, target_column):
        #        super().__init__(data, target_column)

        self.data = data
        self.target_column = target_column

        self.data_x = None
        self.data_y = None

        self.splitTargetFromLabels()

    def splitTargetFromLabels(self):
        """--------------------------------------------------------------

        This routine will split the pandas DataFrame into training labels and
        the target array based on the string "target_column".

        -----------------------------------------------------------------"""

        self.data_x = self.data.loc[:, self.data.columns != self.target_column]
        self.data_y = np.ravel(
            self.data.loc[:, self.data.columns == self.target_column])

    def simpleRandomSample(self, n=None, frac=None, random_state=42):

        if n is not None:
            sampled_df = self.data.sample(
                n=n, replace=False, random_state=random_state)
        if frac is not None:
            sampled_df = self.data.sample(
                frac=frac, replace=False, random_state=random_state)

        sampled_df.reset_index(drop=True, inplace=True)

        return sampled_df

    def randomOverSample(self, n_ones=50000, n_zeros=50000):
        """--------------------------------------------------------------

        This routine will over sample the minority class to mitigate
        class imbalance problems.

        -----------------------------------------------------------------"""

        dict_params = {0: n_zeros, 1: n_ones}

#        self.splitTargetFromLabels()

        X_resampled, y_resampled = RandomOverSampler(ratio=dict_params,
                                                     random_state=30).fit_sample(self.data_x, self.data_y)

        # convert back to a dataframe
        X_resampled = pd.DataFrame(X_resampled, columns=self.data_x.columns)

        return X_resampled, y_resampled

    def randomUnderSample(self, n_ones=50000, n_zeros=50000):
        """--------------------------------------------------------------

        This routine will under sample the majority class to mitigate
        class imbalance problems.

        -----------------------------------------------------------------"""

        dict_params = {0: n_zeros, 1: n_ones}

#        self.splitTargetFromLabels()

        X_resampled, y_resampled = RandomUnderSampler(ratio=dict_params,
                                                      replacement='false', random_state=30).fit_sample(self.data_x, self.data_y)

        X_resampled = pd.DataFrame(X_resampled, columns=self.data_x.columns)

        return X_resampled, y_resampled

    def balance_classes(self, data_to_balance=None):
        """--------------------------------------------------------------

        This method will even out class imbalanced such that number of
        majority class samples equals those of the minority class samples.

        -----------------------------------------------------------------"""

        if data_to_balance is None:
            data_to_balance = self.data.copy()

        # determine the minority class
        vals = data_to_balance[self.target_column].value_counts().to_numpy()
        if (vals[0] < vals[1]):
            minority_class = 0
        else:
            minority_class = 1

        # get number minority class examples
        num_min = data_to_balance[self.target_column].value_counts()[
            minority_class]

        # get all majority class samples
        data_major = data_to_balance[data_to_balance[self.target_column]
                                     != minority_class].copy()

        # get a sample of the majority class data equal to number of minority samples
        data_major_sample = data_major.sample(n=num_min, replace=False)

        # get all the minority class samples
        data_minor = data_to_balance[data_to_balance[self.target_column]
                                     == minority_class].copy()

        # concatenate the even sets and reset the index
        balanced_data = pd.concat([data_major_sample, data_minor])
        balanced_data = balanced_data.reset_index(drop=True)

        del data_major
        del data_minor

        return balanced_data

    def balanceClassesByFeature(self, feature=None):
        """--------------------------------------------------------------

        This method will even out class imbalanced such that number of
        majority class samples equals those of the minority class samples
        based on a certain feature.

        -----------------------------------------------------------------"""

        # determine the minority class
        vals = self.data[self.target_column].value_counts().to_numpy()
        if (vals[0] < vals[1]):
            minority_class = 0
        else:
            minority_class = 1

        # get all majority class samples
        data_major = self.data[self.data[self.target_column] != minority_class]

        # get all the minority class samples
        data_minor = self.data[self.data[self.target_column] == minority_class]

        # get number of ones (minority class)
        num_minor = data_minor.shape[0]

        # create groups based on feature
        feature_groups = self.data.groupby(feature)

        # create an empty DataFrame
        data_major_to_use = pd.DataFrame()

        for feat in feature_groups.groups:

            print(f"Processing {feat}")

            data_tmp = self.data.loc[feature_groups.groups[feat], :].copy()
            data_tmp = data_tmp.reset_index(drop=True)

            # get the number of ones for this month
            num_minor = data_tmp.cat_rt.value_counts()[self.minority_class]

            # get a sample of the majority class data
            data_major_sample = data_zeros.sample(
                n=num_minor, replace=False, random_state=24)

            # add samples to dataFrame
            data_major_to_use = data_major_to_use.append(data_zero_sample)

        # concatenate the even sets and resort by date
        new_data = pd.concat([data_major_to_use, data_minor])
        new_data = new_data.sort_values(by=feature)
        new_data = new_data.reset_index(drop=True)

        # update data in object for future functions
        self.data = new_data.copy()

        del new_data
        return self.data

    def makeSMOTE_Samples(self, n_ones=50000, n_zeros=50000):
        """--------------------------------------------------------------

        This method imparts Synthetic Minority Oversampling Technique
        which synthetically creates new data for the minority class.
        This will only be invoked if a certain number of observations
        are present (user defined).

        -------------------------------------------------------------"""

        # create a mapping for the number of 0,1s to have
        dict_params = {0: n_zeros, 1: n_ones}

        X_resampled, y_resampled = SMOTE(
            ratio=dict_params, kind='regular', random_state=30).fit_sample(self.data_x, self.data_y)

        print(np.mean(self.data_y), np.mean(y_resampled),
              self.data_y[0].shape, y_resampled[0].shape)

        X_resampled = pd.DataFrame(X_resampled, columns=self.data_x.columns)

        return X_resampled, y_resampled

    def makeSMOTE_ENN_Samples(self, n_zeros=50000):
        """--------------------------------------------------------------

        This method imparts Synthetic Minority Oversampling Technique
        which synthetically creates new data for the minority class.
        This will only be invoked if a certain number of observations
        are present (user defined).

        -------------------------------------------------------------"""

        n_ones = int(0.5*n_zeros)
        dict_params = {0: n_zeros, 1: n_ones}

        print(self.data_y.shape)

        X_resampled, y_resampled = SMOTEENN(
            ratio=dict_params, random_state=30).fit_sample(self.data_x, self.data_y)

        print(np.mean(self.data_y), np.mean(y_resampled),
              self.data_y.shape, y_resampled.shape)

        X_resampled = pd.DataFrame(X_resampled, columns=self.data_x.columns)

        return X_resampled, y_resampled

    def make_SMOTE_TOMEK_Samples(self, n_zeros=50000):
        """--------------------------------------------------------------

        This method imparts Synthetic Minority Oversampling Technique
        which synthetically creates new data for the minority class.
        This will only be invoked if a certain number of observations
        are present (user defined).

        -------------------------------------------------------------"""

        n_ones = int(0.5*n_zeros)
        dict_params = {0: n_zeros, 1: n_ones}

        X_resampled, y_resampled = SMOTETomek(
            ratio=dict_params, random_state=30).fit_sample(self.data_x, self.data_y)

        print(np.mean(self.data_y), np.mean(y_resampled),
              self.data_y.shape, y_resampled.shape)

        X_resampled = pd.DataFrame(X_resampled, columns=self.data_x.columns)

        return X_resampled, y_resampled

    def make_near_miss_examples(self):
        pass

    def make_uniform_distribution_samples(self, feature_to_sample=None, number_of_obs=1000,
                                          num_bins=10):
        """--------------------------------------------------------------

        This method will return a sub-sample of your DataFrame but each bin is
        uniform.

        num_bins: int representing how many different ranges to split: Default is 10
        feature_to_sample: string of feature you want to sample about
        number_of_obs: int representing how many obs to have in each bin

        -------------------------------------------------------------"""

        # making dataset with equal number of samples for training

        print("Size before sampling %d" % (self.data.shape[0]))

        # get min and max value to create bin cutoffs
        min_value = np.floor(self.data[feature_to_sample])
        max_value = np.ceil(self.data[feature_to_sample])
        bin_cutoffs = np.linspace(min_value, max_value, num=num_bins + 1)

        # create new DataFrame
        df_to_return = pd.DataFrame(columns=self.data.columns)

        for i in range(1, num_bins):

            temp_df = self.data[(self.data[feature_to_sample] >= bin_cutoffs[i-1]) &
                                (self.data[feature_to_sample] < bin_cutoffs[i])].copy()

            if (temp_df.shape[0] < number_of_obs):
                sampled_obs = temp_df
            else:
                sampled_obs = temp_df.sample(
                    n=number_of_obs, replace=False, random_state=24)

            df_to_return = df_to_return.append(sampled_obs)

        df_to_return.reset_index(drop=True, inplace=True)

        # reassign to newdata
        self.data = df_to_return.copy()

        del df_to_return

        print("Size after sampling %d" % (self.data.shape[0]))

        return self.data

    def make_subsampled_distribution(self, feature_to_sample=None, num_bins=10,
                                     fraction_of_obs=0.5):
        """--------------------------------------------------------------

        This method will return a reduced sample of the dataFrame but will
        preserve the observations true distribution.

        num_bins: int representing how many different ranges to split: Default is 10
        feature_to_sample: string of feature you want to sample about
        fraction_of_obs: float representing how much of your dataset you want to sample.
                            Default is 50% (0.5)

        -------------------------------------------------------------"""

        print("Size before sampling %d" % (self.data.shape[0]))

        # get min and max value to create bin cutoffs
        min_value = np.floor(self.data[feature_to_sample])
        max_value = np.ceil(self.data[feature_to_sample])
        bin_cutoffs = np.linspace(min_value, max_value, num=num_bins + 1)

        # create new DataFrame
        df_to_return = pd.DataFrame(columns=self.data.columns)

        for i in range(1, num_bins):

            temp_df = self.data[(self.data[feature_to_sample] >= bin_cutoffs[i-1]) &
                                (self.data[feature_to_sample] < bin_cutoffs[i])].copy()

            sampled_obs = temp_df.sample(
                frac=fraction_of_obs, replace=False, random_state=24)

            df_to_return = df_to_return.append(sampled_obs)

        df_to_return = df_to_return.reset_index(drop=True)

        # reassign to newdata
        self.data = df_to_return

        print("Size after sampling %d" % (self.data.shape[0]))

        del df_to_return

        return self.data

    def make_specific_range_samples(self, data_range, feature_to_sample=None):
        """--------------------------------------------------------------

        This method will return samples only in a specific range based on
        a feature provided.

        data_range: tuple of type (min,max)
        feature_to_sample: string of feature you want to sample about

        -------------------------------------------------------------"""

        print("Size before sampling %d" % (self.data.shape[0]))

        tmp_data = self.data[(self.data[feature_to_sample] >= data_range[0]) &
                             (self.data[feature_to_sample] <= data_range[1])].copy()

        self.data = tmp_data.reset_index(drop=True, inplace=False)

        del tmp_data

        print("Size after sampling %d" % (self.data.shape[0]))

        return self.data

    def smart_sample_by_feature(self, feature=None):
        """--------------------------------------------------------------

        This method will resample the DataFrame based on whether or not it
        was precipitating. By default, a 50/50 split of precip vs no precip
        is done.

        -------------------------------------------------------------"""

        # make date have datetime and make a calendar date variable
        self.data['date'] = pd.to_datetime(
            self.data['date'], format="%Y-%m-%d %H:%M:%S")

        if ('date_calendar' not in self.data.columns):
            self.data['date_calendar'] = pd.to_datetime(
                self.data['date'].dt.strftime('%Y-%m-%d'))
        else:
            self.data['date_calendar'] = pd.to_datetime(
                self.data['date_calendar'])

        # get precip obs
        print(self.data.shape)
        precip_data = self.data[self.data[feature] == 1]
        print(precip_data.shape)

        # create new DataFrame
        df_to_return = pd.DataFrame(columns=self.data.columns)

        # get unqiue calendar dates when it was precipitating
        times = pd.to_datetime(precip_data.date_calendar.unique())

        # loop over each calendar date
        for time in times:
            #print(f"Processing {time}")
            tmp_precip = precip_data[precip_data['date_calendar'] == time]

            if (tmp_precip.shape[0] <= 0):
                print("Time "+str(time)+" contains no data, skipping...")
                continue
            else:
                # get min/max values for latitude and longitude box
                min_lat, max_lat = tmp_precip.lat.min(), tmp_precip.lat.max()
                min_lon, max_lon = tmp_precip.lon.min(), tmp_precip.lon.max()

                # find nonprecip obs within lat/lon box
                tmp_full = self.data[(self.data['date_calendar'] == time) &
                                     (self.data[feature] == 0) &
                                     ((self.data['lon'] > min_lon) & (self.data['lon'] < max_lon)) &
                                     ((self.data['lat'] > min_lat) & (self.data['lat'] < max_lat))]

                tmp_full.reset_index(drop=True, inplace=True)

                # sample nonprecip obs if larger than number of precip obs (should be)
                if (tmp_full.shape[0] > tmp_precip.shape[0]):
                    sampled_non_precip = tmp_full.sample(n=tmp_precip.shape[0],
                                                         replace=False)
                else:
                    sampled_non_precip = tmp_full

                # concat individual dataframes into one
                combined_day = pd.concat([tmp_precip, sampled_non_precip])

            # free memory
            del tmp_full
            del tmp_precip

            # append concatenated daily dataframe to overall dataframe
            df_to_return = df_to_return.append(combined_day)
            # print(df_to_return.shape)

        # reset_index to be monotonically increasing
        df_to_return = df_to_return.reset_index(drop=True)

        # reindex based on calendar date
        df_to_return = df_to_return.sort_values(by='date_calendar')

        # reset_index to be monotonically increasing
        df_to_return = df_to_return.reset_index(drop=True)

        # reassign to data in object
        self.data = df_to_return

        print("Size after sampling %d" % (self.data.shape[0]))

        return self.data

    def get_daily_random_samples(self, fraction_of_obs=0.1, balance=False):
        """--------------------------------------------------------------

        This method will return random samples in the domain for each day.

        fraction_of_obs: a float [0,1] representing the fraction of
            observations that will be sampled for each day.

        -------------------------------------------------------------"""

        # make date have datetime and make a calendar date variable
        self.data['date'] = pd.to_datetime(
            self.data['date'], format="%Y-%m-%d %H:%M:%S")

        if ('date_calendar' not in self.data.columns):
            self.data['date_calendar'] = pd.to_datetime(
                self.data['date'].dt.strftime('%Y-%m-%d'))
        else:
            self.data['date_calendar'] = pd.to_datetime(
                self.data['date_calendar'])

        # create new DataFrame
        df_to_return = pd.DataFrame(columns=self.data.columns)

        # get unqiue calendar dates
        dates = pd.to_datetime(self.data.date_calendar.unique())

        for date in dates:

            # get all data for current day
            tmp_data = self.data[self.data['date_calendar'] == date].copy()

            if (tmp_data.shape[0] <= 10):
                print(
                    f"Time {date} contains no or very little data, skipping...")
                continue

            # sample a subset of obs for the day
            sampled_obs = tmp_data.sample(
                frac=fraction_of_obs, random_state=31)

            # if balance is True, then balance the class to be 50-50
            if (balance is True):
                sampled_obs = self.balanceClasses(sampled_obs)

            # append data to new DataFrame
            df_to_return = df_to_return.append(sampled_obs)

            del tmp_data
            # print(df_to_return.shape)

        # reshuffle based on calendar date
        df_to_return = df_to_return.sort_values(by='date_calendar')

        # reset_index to be monotonically increasing
        df_to_return = df_to_return.reset_index(drop=True)

        # reassign to newdata
        self.data = df_to_return

        print("Size after sampling %d" % (self.data.shape[0]))
        print("Fraction of subfreezing obs: %f" %
              self.data[self.target_column].mean())

        del df_to_return

        return self.data
