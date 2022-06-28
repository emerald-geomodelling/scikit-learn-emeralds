def print_label_info(series, message):
    class_counts = series.value_counts()
    class_fractions = class_counts / series.size
    print(message,'\nClasses in the dataset, counts, fraction: \n', pd.concat((class_counts, class_fractions), axis=1))