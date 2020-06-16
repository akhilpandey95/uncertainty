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

        # return the dataframe
        return data
    except:
        return pd.DataFrame()

# function for preparing embeddings
def prepare_embeddings(data, X):
    """
    Prepare the one hot encoded vector for the smile strings of every compound

    Parameters
    ----------
    arg1 | data: pandas.DataFrame
        A dataframe consisting of necessary columns for extracting Smiles
    arg2 | X: numpy.ndarray
        An array consisting of texts from the dataframe that would be converted
        into word embeddings

    Returns
    -------
    Array
        numpy.ndarray

    """
    try:
        # find the maximum words and maximum length for the given dataset
        max_words = max(list(map(lambda x: len(x.split()), tqdm(data[X]))))

        # find max length of the text for the given dataset
        max_len = max(list(map(len, tqdm(data[X]))))

        # init the tokenizer class object
        tok = Tokenizer(char_level=True)

        # fit the tokenizer on the text data
        tok.fit_on_texts(data[X])

        # generate the sequences
        sequences = tok.texts_to_sequences(data[X])

        # obtain the sequence matrix
        X = sequence.pad_sequences(sequences, maxlen=max_len)

        # return the dataframe
        return X
    except:
        return np.zeros(20).reshape(2, 10)
