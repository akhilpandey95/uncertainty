# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import os
import pickle
import logging
import deepchem as dc

# set the logger
logger = logging.getLogger(__name__)

# set the default dir
DEFAULT_DIR = '/tmp'

def load_qm7(featurizer='CoulombMatrix',
             split='random',
             reload=True,
             move_mean=True,
             data_dir=None,
             save_dir=None,
             **kwargs):
    # Featurize qm7 dataset
    logger.info("About to featurize qm7 dataset.")
    if data_dir is None:
        data_dir = DEFAULT_DIR
    if save_dir is None:
        save_dir = DEFAULT_DIR
    dataset_file = os.path.join(data_dir, "gdb7.sdf")

    if not os.path.exists(dataset_file):
        dc.utils.download_url(url=GDB7_URL, dest_dir=data_dir)
        dc.utils.untargz_file(os.path.join(data_dir, 'gdb7.tar.gz'), data_dir)

    qm7_tasks = ["u0_atom"]
    if featurizer == 'CoulombMatrix':
        featurizer = dc.feat.CoulombMatrixEig(23)
    loader = dc.data.SDFLoader(
      tasks=qm7_tasks,
      featurizer=featurizer)
    dataset = loader.featurize(dataset_file)

    if split == None:
        raise ValueError()

    splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
    }
    splitter = splitters[split]
    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)

    transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset, move_mean=move_mean)
    ]

    for transformer in transformers:
        train_dataset = transformer.transform(train_dataset)
        valid_dataset = transformer.transform(valid_dataset)
        test_dataset = transformer.transform(test_dataset)

    return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers

# grab the data from the mat file
tasks, (train_data, val_data, test_data), transformers = load_qm7()

# separate the train, val, and test datasets
X_train, y_train = train_data.X, train_data.y
X_val, y_val = val_data.X, val_data.y
X_test, y_test = test_data.X, test_data.y

# save the input training data
with open('X_train.pkl','wb') as f:
    # dumpy the training tensors
    pickle.dump(X_train, f)

# save the input validation data
with open('X_val.pkl','wb') as f:
    # dumpy the validation tensors
    pickle.dump(X_val, f)

# save the input test data
with open('X_test.pkl','wb') as f:
    # dumpy the test tensors
    pickle.dump(X_test, f)

# save the outputs from training data
with open('y_train.pkl','wb') as f:
    # dumpy the training targets
    pickle.dump(y_train, f)

# save the outputs from validation data
with open('y_val.pkl','wb') as f:
    # dumpy the validation targets
    pickle.dump(y_val, f)

# save the outputs from test data
with open('y_test.pkl','wb') as f:
    # dumpy the test targets
    pickle.dump(y_test, f)

