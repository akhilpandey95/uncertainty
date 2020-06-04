# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

# function for processing the dataset
def data_processing(file_path, target):
    """
    Process the dataset and prepare it for using it
    against the graph neural network model

    Parameters
    ----------
    arg1 | file_path: str
        The file path indicating the location of the dataset
    arg2 | target: str
        The type of target variable going to be used for the experiment

    Returns
    -------
    Dataframe
        pandas.DataFrame

    """
    try:
        # read the dataset
        data = pd.read_csv(file_path, low_memory=False)

        # get a sample of the data
        data = data.sample(frac=0.25, random_state=2019)

        # reset the index for the dataframe
        data = data.reset_index(drop=True)

        # return the dataframe
        return data
    except:
        return pd.DataFrame()

