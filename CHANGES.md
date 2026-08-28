# Changelog

## 0.0.5
2026-08-28

First release since 0.0.4.1 (2025-12-08). Everything here is a compatibility or correctness
fix; no API changes.

### Fixed
- pandas 2: `argrelextrema.filter_dfextrema` called `DataFrame.pivot` positionally, and those
  arguments became keyword-only in pandas 2.0. Now `pivot(index=, columns=, values=)`.
- numpy 2: numpy 2 sorts the input to `np.unique`, which raises `TypeError` on object label
  arrays that mix text with NaN. `confusion_matrix.plot_confusion_matrix` and the `exhaust`,
  `oversample` and `stratify` test/train splitters use NaN-safe pandas equivalents instead.
  Results are identical for clean numeric labels; when labels contain missing values a single
  explicit NaN entry is kept at the end of the axis. (#4)
- numpy 2: `np.int` -> `int` in `test_train_splitters.oversample.train_test_split_byhole`
  (`np.int` was removed in numpy 1.24; `int` is what it meant, so the cast is unchanged).
- `train_test_split_byhole`'s `test_size_byData` branch could not run at all: it passed `arr=`
  to `pandas.Series`, where the keyword is `data=`, and then indexed an undefined name `data`
  instead of `data_wLabel`.

### Added
- `tests/test_train_test_split_byhole.py`, covering both split modes, and
  `tests/test_numpy2_removed_names.py`, an AST-based source guard that fails if any numpy name
  the installed numpy no longer has is referenced anywhere in the package.
