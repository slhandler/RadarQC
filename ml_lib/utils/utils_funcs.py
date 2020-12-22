import numpy as np
import pandas as pd
import sys, os
from datetime import datetime, timedelta, time

#----------------------------------------------------------------------------------------#
def splitTargetFromLabels(data, target_column=None):

    """--------------------------------------------------------------

    This routine will split a pandas DataFrame into a predictor matrix and target 
    label array where the target array is based on the string "target_column".

    -----------------------------------------------------------------"""

    prediction_matrix  = data.drop(target_column, axis=1)
    target_array = data[target_column].to_numpy()

    return prediction_matrix, target_array
#----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
def perdelta(start, end, delta_t):
    curr = start
    while curr <= end:
        yield curr
        curr += delta_t 
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
def removePredictor(data0, pred, axis):

    """Here, data0 is a pandas DataFrame."""

    data0 = data0.drop(pred, axis=axis)
    cols  = data0.columns

    return data0, cols
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
def addPredictor(data0, name, new_data):

    """Here, data0 is a pandas DataFrame."""

    data0[name] = new_data
    cols  = data0.columns

    return data0, cols
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
def removeDuplicates(seq):
    seen = set()
    seen_add = seen.add

    return [x for x in seq if not (x in seen or seen_add(x))]
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
def checkFileTime(file, start_date, end_date):

    """Given a file, check if the timestamp falls in between the dates we care about.
       Returns a boolean."""

    splits = file.split('.')
    datetime_file = datetime.strptime(splits[0], '%Y%m%d-%H%M%S')

    if ((datetime_file >= start_date) & (datetime_file <= end_date)):return True
    else: return False

#-----------------------------------------------------------------------------------------#