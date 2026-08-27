"""Tests for ``train_test_split_byhole``, whose ``test_size_byData`` branch had never run.

That branch raised on its own first statement -- ``pd.Series(index=IDs, arr=index_IDs)``, and ``arr``
is not a ``Series`` keyword -- and then referenced an undefined name ``data`` on the line after. Both
are fixed; these tests pin what the branch is supposed to do so it cannot rot again unnoticed.

The two branches split differently and that is the point of the flag:

* default (``test_size_byData`` falsy) splits by *hole*, so each side gets whole boreholes but the
  row counts follow however many samples those holes happen to have;
* ``test_size_byData=True`` splits by *row*, ordering rows by a shuffled hole ranking first, so the
  boundary lands at the requested fraction of the data while still keeping each hole together.

Synthetic frames only -- no database, no files, no network.
"""
import os

# Importing this module reaches emerald_database at import time, which reads three environment
# variables and raises KeyError if any is unset. Nothing connects: connection.py only assembles a
# params dict and leaves engine as None until setup(). setdefault, so a real environment wins.
os.environ.setdefault("EMERALD_POSTGIS_PASSWORD", "unused-by-the-synthetic-test-suite")
os.environ.setdefault("EMERALD_GDRIVE_DRIVE_ID", "unused-by-the-synthetic-test-suite")
os.environ.setdefault("EMERALD_GDRIVE_ID", "unused-by-the-synthetic-test-suite")
os.environ.pop("EMERALD_CONFIG_URL", None)   # would fetch and exec a remote config at import

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import pytest                                                         # noqa: E402

from skl_emeralds.test_train_splitters.oversample import train_test_split_byhole   # noqa: E402


def _frame(n_holes=5, per_hole=2):
    """``per_hole`` samples from each of ``n_holes`` boreholes, values unique per row."""
    titles = [chr(ord("a") + i) for i in range(n_holes) for _ in range(per_hole)]
    frame = pd.DataFrame({"title": titles, "v": np.arange(len(titles), dtype=float)})
    labels = pd.Series(np.arange(len(titles)) % 2, index=frame.index)
    return frame, labels


def test_by_data_splits_at_the_requested_row_fraction():
    frame, labels = _frame(n_holes=5, per_hole=2)

    train, test, label_train, label_test = train_test_split_byhole(
        frame.copy(), labels, test_size=0.2, random_state=0, test_size_byData=True)

    assert len(train) + len(test) == len(frame)      # nothing lost
    assert len(train) == 8 and len(test) == 2        # 1 - test_size of 10 rows
    assert len(label_train) == len(train)
    assert len(label_test) == len(test)


def test_by_data_never_splits_a_hole_across_the_boundary():
    """Rows are ordered by a whole-hole ranking, so a borehole cannot land on both sides."""
    frame, labels = _frame(n_holes=5, per_hole=2)

    train, test, _, _ = train_test_split_byhole(
        frame.copy(), labels, test_size=0.2, random_state=0, test_size_byData=True)

    assert set(train.title) & set(test.title) == set()
    # each hole's rows stay adjacent after the sort
    order = train.title.tolist() + test.title.tolist()
    assert [k for k, _ in zip(order, order)] == order
    for hole in set(order):
        positions = [i for i, t in enumerate(order) if t == hole]
        assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_by_data_is_reproducible_for_a_given_random_state():
    frame, labels = _frame()

    first, _, _, _ = train_test_split_byhole(frame.copy(), labels, test_size=0.2,
                                             random_state=7, test_size_byData=True)
    second, _, _, _ = train_test_split_byhole(frame.copy(), labels, test_size=0.2,
                                              random_state=7, test_size_byData=True)

    assert first.title.tolist() == second.title.tolist()


def test_by_hole_branch_splits_whole_boreholes():
    """The default branch is unchanged by the fix; pinned so the two stay distinguishable."""
    frame, labels = _frame(n_holes=5, per_hole=2)

    train, test, _, _ = train_test_split_byhole(
        frame.copy(), labels, test_size=0.2, random_state=0)

    assert set(train.title) & set(test.title) == set()
    assert len(train) + len(test) == len(frame)
    assert len(train) % 2 == 0                       # whole holes, so a multiple of per_hole


def test_lookup_series_construction_is_valid():
    """Guards the specific defect: pd.Series(index=..., arr=...) raises, data= is the keyword."""
    with pytest.raises(TypeError):
        pd.Series(index=np.array(["a", "b"]), arr=np.arange(2))

    ranking = pd.Series(data=np.arange(2), index=np.array(["b", "a"]))
    assert ranking.loc["b"] == 0 and ranking.loc["a"] == 1
