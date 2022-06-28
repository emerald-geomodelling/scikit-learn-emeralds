import numpy as np
import pandas as pd
import sklearn
import pickle
import elnes
import math
from skl_emeralds.print import *

def test_train_split_balance_oversample_minority_byhole(arr, filt, test_size=0.2, random_state=None, verbose=False):
    training_wlabel = arr.loc[filt]

    if random_state is not None:
        np.random.seed(int(random_state))
        classes = np.unique(training_wlabel.label)
        print('Training classes: ', classes)

    if verbose:
        print('splitting by hole, then balancing')

    # All points from a borehole should either be in test OR in train
    data_train, data_test, label_train, label_test = train_test_split_byhole(arr,
                                                                             training_wlabel.label,
                                                                             test_size=test_size,
                                                                             random_state=random_state,
                                                                             test_size_byData=False,
                                                                             hole_id_name="title")

    for ID in data_train.title.unique():
        if ID in data_test.title.unique():
            print('Duplicate borehole in train / test split:', ID)

    label_train_counts = label_train.value_counts()
    label_train_deficits = label_train_counts.max() - label_train_counts
    for c in classes:
        if label_train_deficits.loc[c] == 0:
            continue

        index_train_c = label_train.loc[label_train == c].index
        n_concats = math.floor(label_train_deficits.loc[c] / label_train_counts.loc[c])
        n_random_sel = label_train_deficits.loc[c] % label_train_counts.loc[c]
        new_indices_c = np.concatenate((
            np.array(index_train_c.to_list() * (n_concats)),
            np.random.choice(index_train_c.to_numpy(), n_random_sel, replace=False)))

        data_train = pd.concat((data_train, data_train.loc[new_indices_c, :]))


    loc_label_ids_train = data_train.index.isin(new_indices_c)

    loc_label_ids_test = ~data_test.index.isin(new_indices_c)
    if verbose:
        print('----------------------------')
        print_label_info(label_train, 'Training labels')
        print('----------------------------')
        print_label_info(label_test, 'Testing labels')

    return loc_label_ids_train, loc_label_ids_test


def train_test_split_byhole(arr, label, test_size=0.2, hole_id_name='title', random_state=None, test_size_byData=None):
    IDs = arr.loc[:, hole_id_name].unique()
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(IDs)
    n_IDs = IDs.size
    index_IDs = np.arange(n_IDs)

    if not test_size_byData:  # default setting, where test_size is interpreted as fraction of holes

        n_IDs_train = int(round((test_size) * n_IDs))

        IDs_train = IDs[:n_IDs_train]
        IDs_test = IDs[n_IDs_train:]

        data_train = arr[arr.loc[:, hole_id_name].isin(IDs_train)]
        data_test = arr[arr.loc[:, hole_id_name].isin(IDs_test)]

        label_train = label[arr.loc[:, hole_id_name].isin(IDs_train)]
        label_test = label[arr.loc[:, hole_id_name].isin(IDs_test)]

    elif test_size_byData:  # test_size is interpreted as fraction of data points
        data_wLabel = arr
        data_wLabel.at[:, 'Label'] = label

        # make new dataframe, pointID shuffled as index, order index as one column, use this as lookup table
        lookup_ds = pd.Series(index=IDs, arr=index_IDs)
        for row_index, value in data.loc[:, hole_id_name].items():
            data_wLabel.at[row_index, 'ID_order'] = lookup_ds.loc[value].astype(np.int)
        data_wLabel = data_wLabel.sort_values(axis=0, by='ID_order')

        n_rows = data_wLabel.shape[0]
        n_train = round((1 - test_size) * n_rows)

        data_train = data_wLabel.iloc[:n_train, :]
        data_test = data_wLabel.iloc[n_train:, :]

        label_train = data_wLabel.iloc[:n_train, :].loc[:, 'Label']
        label_test = data_wLabel.iloc[n_train:, :].loc[:, 'Label']

    return data_train, data_test, label_train, label_test




