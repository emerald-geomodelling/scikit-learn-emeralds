import numpy as np
import pandas as pd
import sklearn
import pickle
import elnes
import math
from skl_emeralds.print import *

def test_train_split_balance_oversample_minority_stratify(arr, filt, test_size, verbose=None, random_state = None):
    print('Stratifying by label, then balancing')

    if verbose:
        print_label_info(filt, 'Input labels')

    if random_state is not None:
        np.random.seed(int(random_state))
    classes = np.unique(filt)

    data_train, data_test, label_train, label_test = sklearn.model_selection.train_test_split(
        arr, filt, test_size=test_size,
        random_state=random_state, stratify=filt)

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
        print('---------------------')
        print_label_info(label_train, 'Training labels')
        print('---------------------')
        print_label_info(label_test, 'Testing labels')

    return loc_label_ids_train, loc_label_ids_test
