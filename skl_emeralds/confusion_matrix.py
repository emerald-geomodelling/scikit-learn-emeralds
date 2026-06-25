import seaborn as sn
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_confusion_matrix(model, features, labels, label_names,
                          count=False,
                          cmap="viridis", norm = matplotlib.colors.LogNorm(),
                          ax = None,
                          **kw):
    """


    @param model: a trained classifier model from scikit-learn, for example, a RandomForestClassifier
    @param features: numpy array (size n_observations x n_features) of features to input to model
    @param labels: numpy array (size n_observations) of the true values as integers
    @param label_names: pandas.Series translating string names of the classes (index) to integer values like those in labels
    @param count: render category counts if True, proportions in False
    @param cmap: colormap to pass to matplotlib
    @param norm: norm to pass to matplotlib
    @param ax: matplotlib.pyplot.Axes object. If None, use the current axes available.
    @param kw: keywords to pass on to seaborn.heatmap
    @return:
    """
    if ax is None: ax = plt.gca()
    
    proba_layer = model.predict_proba(features)
    label_layer = model.classes_[np.argmax(proba_layer, axis=1)]

    if count:
        m = proba_layer.argmax(axis=1)
        p = np.zeros(proba_layer.shape).flatten()
        p[np.ravel_multi_index((np.arange(len(m)), m), proba_layer.shape)] = 1
        p = p.reshape(*proba_layer.shape)
        proba_layer = p
    
    # numpy 2 sorts the input to np.unique and raises on object label arrays that mix text with
    # NaN. Use pandas-native, NaN-safe equivalents (identical for clean numeric labels), keeping
    # a single explicit NaN label at the end of the axis when labels contain missing values.
    unique_labels = np.sort(pd.Series(labels).dropna().unique())
    if pd.isna(labels).any():
        unique_labels = np.append(unique_labels, np.nan)
    size1 = len(unique_labels)
    size2 = pd.Series(model.classes_).nunique(dropna=False)
    size3 = pd.Series(label_names).nunique(dropna=False)
    size = np.max([size1, size2, size3])

    res = pd.DataFrame().rename_axis(index='true_label', columns='predicted_label')#np.zeros((size, size))
    for label in unique_labels:
        res.loc[label,model.classes_] = proba_layer[labels == label, :].sum(axis=0)

    if not count:
        rowsum = res.sum(axis=1)
        rowsum = np.where(rowsum == 0, 1, rowsum)
        res = 100 * res / np.tile(np.array([rowsum]).transpose(), (1, res.shape[1]))
    format = ".0f"
    
    label_names_by_label = label_names.reset_index().set_index(0)["index"]
    label_names_by_label = label_names_by_label.loc[unique_labels]

    sn.heatmap(res, annot=True, annot_kws={"size": 10}, fmt=format,
               xticklabels = label_names_by_label,
               yticklabels = label_names_by_label,
               norm = norm,
               cmap = cmap,
               ax = ax,
               **kw
              )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
