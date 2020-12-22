import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from RadarQC.ml_lib.utils.utils_funcs import perdelta

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut


class DataSplitter():

    def __init__(self, predictors, targets, target_name):

        self.data_x = predictors
        self.data_y = targets
        self.taregt_name = target_name

    def trainTestSplit(self, test_size=0.25, random_state=3):
        """--------------------------------------------------------------

        Will perform a simple train test split based on the size of the
        test_size variable. Default is 0.25 for test set size and 0.75 for
        training set size.

        -----------------------------------------------------------------"""

        X_train, X_test, y_train, y_test = train_test_split(self.data_x, self.data_y,
                                                            test_size=test_size, random_state=random_state)

        print("Number of training samples: %d") % X_train.shape[0]
        print("Number of calibration samples: %d") % X_test.shape[0]

        return X_train, X_test, y_train, y_test

    def splitRandomly(self, test_size=0.25, random_state=3):
        """--------------------------------------------------------------

        Call the trainTestSplit routine.

        -----------------------------------------------------------------"""

        self.trainTestSplit()

    def splitByMonth(self):
        """--------------------------------------------------------------

        Will split the DataFrame by month. Returns a custom cross-validate
        iterator to be used for future tuning, training, etc.

        -----------------------------------------------------------------"""

        logo = LeaveOneGroupOut()

        test_indices = []
        train_indices = []
        inner_train_indices = []
        val_indices = []

        if ('month' not in self.data_x.columns):
            self.data_x['date'] = pd.to_datetime(
                self.data_x['date'], format="%Y-%m-%d %H:%M:%S")
            self.data_x['month'] = self.data_x['date'].dt.month

        split_crit = self.data_x['month']

        for train_index, test_index in logo.split(self.data_x, self.data_y, groups=split_crit):

            train_indices.append(train_index)
            test_indices.append(test_index)

            temp_data = self.data_x.iloc[train_index]
            temp_targets = self.data_y[train_index]

            logo_inner = LeaveOneGroupOut()
            inner_split_crit = temp_data['month']

            for inner_train_index, val_index in logo_inner.split(temp_data, temp_targets, groups=inner_split_crit):
                inner_train_indices.append(inner_train_index)
                val_indices.append(val_index)

        cv_dict = {}
        cv_dict['train_indices'] = train_indices
        cv_dict['test_indices'] = test_indices
        cv_dict['inner_train_indices'] = inner_train_indices
        cv_dict['validation_indices'] = val_indices

        return cv_dict

    def splitByWeek(self, nweeks=1, offset=0, skip_factor=0):
        """--------------------------------------------------------------

        Will split the DataFrame into a weekly list.

        nweeks: how many weeks to pull at a time (default is 1)
        offset: used to offset the start or end of the date (hours).
        skip_factor: how many hours to skip after a one-week period.

        -----------------------------------------------------------------"""

        print("Spliting DataFrame into weekly chunks...")

        self.data_x['date'] = pd.to_datetime(
            self.data_x['date'], format="%Y-%m-%d %H:%M:%S")

        first_date = self.data_x.date.iloc[0]
        end_date = self.data_x.date.iloc[-1]

        hours_in_week = 24*7
        weeks_df = pd.DataFrame(columns=self.data_x.columns)

        for i, result in enumerate(perdelta(first_date+timedelta(hours=offset), end_date,
                                            timedelta(hours=(hours_in_week*nweeks)+skip_factor))):
            print(result, result+timedelta(hours=hours_in_week*nweeks))

            mask = (self.data_x['date'] > result) & (
                self.data_x['date'] < (result+timedelta(hours=hours_in_week*nweeks)))
            temp_df = self.data_x.loc[mask]
            temp_df['week_number'] = 0
            temp_df['week_number'] = i
            weeks_df = weeks_df.append(temp_df)

        weeks_df.reset_index(drop=True, inplace=True)

        return weeks_df

    def splitByDay(self, offset=0, skip_factor=0):
        """--------------------------------------------------------------

        Will split the DataFrame into a daily list.
        offset: used to offset the start or end of the date (hours).
        skip_factor: how many hours to skip after a one-week period.
        -----------------------------------------------------------------"""

        print("Spliting DataFrame into daily chunks...")

        self.data_x['date'] = pd.to_datetime(
            self.data_x['date'], format="%Y-%m-%d %H:%M:%S")

        first_date = self.data_x.date.iloc[0]
        end_date = self.data_x.date.iloc[-1]

        list_of_day_dfs = []

        for result in perdelta(first_date+timedelta(hours=offset), end_date-timedelta(hours=offset), timedelta(hours=24+skip_factor)):
            print(result)
            mask = (self.data_x['date'] > result) & (
                self.data_x['date'] < (result+timedelta(hours=24)))
            list_of_day_dfs.append(self.data_x.loc[mask])

        print(len(list_of_day_dfs))
        return list_of_day_dfs

    def splitByDate(self, date_to_split=None, is_greater=False):
        """--------------------------------------------------------------

        Will split the DataFrame based on a specific date.

        date_to_split: the date to split on.
        is_greater: keyword for splitting. If false, get values for dates less
            than date_to_split. If true, get values for dates greater than
            date_to_split
        -----------------------------------------------------------------"""

        print("Spliting DataFrame by date, which is %s" % date_to_split)
        print(self.data_x.shape)

        if (is_greater is False):
            filter = self.data_x['date'] <= date_to_split

        else:
            filter = self.data_x['date'] >= date_to_split

        df_to_return = self.data_x[filter]

        # reset the index
        df_to_return.reset_index(drop=True, inplace=True)
        print(df_to_return.shape)

        return df_to_return
