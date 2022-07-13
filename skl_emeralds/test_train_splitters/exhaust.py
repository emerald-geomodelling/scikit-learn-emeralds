import numpy as np
import pandas as pd
import sklearn
import pickle
import elnes
import math
from skl_emeralds.print import *

def exhaust_class_(pointcloud, new_label, test_size, classes):

    class_counts = new_label.label.value_counts()

    n_train = (1 - test_size) * new_label.label.shape[0]
    n_even = n_train / classes.size

    label_train_deficits = np.round(n_even - class_counts)

    for c in classes:
        # Pandas series, single class e.g. Brittle, non-brittle
        label_c = new_label.label.loc[new_label.label == c]
        # Dataframe of above indexes
        data_c = pointcloud.loc[label_c.index]

        if n_even <= class_counts[c]:
            # Creating indices, array of indices
            new_indices_c = np.random.choice(label_c.index, round(n_even), replace=False)
            # 2 masks
            loc_label_ids_train = data_c.index.isin(new_indices_c)
            loc_label_ids_test = ~data_c.index.isin(new_indices_c)

        if n_even > class_counts[c]:
            n_concats = max(math.floor(label_train_deficits.loc[c] / class_counts.loc[c]), 0)
            n_random_sel = int(label_train_deficits.loc[c] % class_counts.loc[c])
            new_indices_c = np.concatenate((
                np.array(label_c.index.to_list() * (n_concats + 1)),
                np.random.choice(label_c.index.to_numpy(), n_random_sel, replace=False)))
            # 1 mask
            loc_label_ids_train = data_c.index.isin(new_indices_c)

    print('...Classes exhausted')
    return loc_label_ids_train, loc_label_ids_test

def test_train_split_balance_oversample_minority_exhaust(arr, filt, test_size=0.2, random_state=None, verbose=False):
    training_wlabel = arr.loc[filt]

    if random_state is not None:
        np.random.seed(int(random_state))
        classes = np.unique(training_wlabel.label)
        print('Classes in dataset are: ,', classes)

    train, test = exhaust_class_(arr, training_wlabel, test_size, classes)
    return train, test