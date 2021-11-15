import seaborn as sn

def plot_confusion_matrix(model, features, labels, label_names,
                          cmap="viridis", norm = matplotlib.colors.LogNorm(),
                          ax = None,
                          **kw):
    
    if ax is None: ax = plt.gca()
    
    proba_layer = model.predict_proba(features)
    label_layer = model.classes_[np.argmax(proba_layer, axis=1)]

    res = np.zeros((np.max(labels)+1, np.max(labels) + 1))
    for label in np.unique(labels):
        res[label,model.classes_] = proba_layer[labels == label, :].sum(axis=0)

    rowsum = res.sum(axis=1)
    rowsum = np.where(rowsum == 0, 1, rowsum)
    res = res / numpy.tile(np.array([rowsum]).transpose(), (1, res.shape[1]))

    label_names_by_label = label_names.reset_index().set_index(0)["index"]

    sn.heatmap(res, annot=True, annot_kws={"size": 16}, fmt=".3f",
               xticklabels = label_names_by_label,
               yticklabels = label_names_by_label,
               norm = norm,
               cmap = cmap,
               ax = ax,
               **kw
              )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
