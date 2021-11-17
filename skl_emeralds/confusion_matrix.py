import seaborn as sn
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(model, features, labels, label_names,
                          count=False,
                          cmap="viridis", norm = matplotlib.colors.LogNorm(),
                          ax = None,
                          **kw):
    
    if ax is None: ax = plt.gca()
    
    proba_layer = model.predict_proba(features)
    label_layer = model.classes_[np.argmax(proba_layer, axis=1)]

    if count:
        m = proba_layer.argmax(axis=1)
        p = np.zeros(proba_layer.shape).flatten()
        p[np.ravel_multi_index((np.arange(len(m)), m), proba_layer.shape)] = 1
        p = p.reshape(*proba_layer.shape)
        proba_layer = p
    
    size1 = np.max(labels)
    size2 = np.max(model.classes_)
    size3 = np.max(label_names)
    size = np.max([size1, size2, size3]) + 1

    res = np.zeros((size, size))
    for label in np.unique(labels):
        res[label,model.classes_] = proba_layer[labels == label, :].sum(axis=0)

    if not count:
        rowsum = res.sum(axis=1)
        rowsum = np.where(rowsum == 0, 1, rowsum)
        res = 100 * res / np.tile(np.array([rowsum]).transpose(), (1, res.shape[1]))
        format = ".1f"
    else:
        format = ".0f"
    
    label_names_by_label = label_names.reset_index().set_index(0)["index"]
    
    sn.heatmap(res, annot=True, annot_kws={"size": 16}, fmt=format,
               xticklabels = label_names_by_label,
               yticklabels = label_names_by_label,
               norm = norm,
               cmap = cmap,
               ax = ax,
               **kw
              )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
