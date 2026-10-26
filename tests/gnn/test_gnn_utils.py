"""Tests for predicators/gnn/gnn_utils.py."""

from predicators.gnn.gnn_utils import GraphTransC2DDataset


def test_transition_dataset_keeps_order():
    """The dataset keeps the order it is given: the approach splits and
    shuffles the data with its own seeded generator, and a reshuffle with
    the global numpy generator made runs with the same seed differ."""
    data = [{"index": i} for i in range(50)]
    dataset = GraphTransC2DDataset(list(data))
    assert [dataset[i] for i in range(len(dataset))] == data
