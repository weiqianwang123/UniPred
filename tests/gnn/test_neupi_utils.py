"""Tests for predicators/gnn/neupi_utils.py."""

import torch

from predicators.gnn import neupi_utils


def test_train_predicate_model_keeps_best_weights(monkeypatch):
    """The returned weights are a copy of the weights with the lowest
    validation loss, not the live (final) parameters."""
    model = torch.nn.Linear(2, 1)
    val_losses = iter([0.1, 0.5, 0.9])
    validated_weights = []

    def _fake_losses(model_, data, super_label, device):
        del data, super_label, device  # unused
        if torch.is_grad_enabled():  # training: move the weights
            return model_.weight.sum(), torch.tensor(0.0)
        validated_weights.append(model_.weight.detach().clone())
        return torch.tensor(next(val_losses)), torch.tensor(0.0)

    monkeypatch.setattr(neupi_utils, "_effect_losses", _fake_losses)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    best, best_loss = neupi_utils.train_predicate_model(0,
                                                        model, [None], [None],
                                                        optimizer, {},
                                                        num_epochs=3)
    assert abs(best_loss - 0.1) < 1e-6
    assert torch.equal(best["weight"], validated_weights[0])
    assert not torch.equal(best["weight"], model.weight.detach())


def test_validation_after_last_epoch(monkeypatch):
    """The last epoch is validated even if val_freq does not divide the
    number of epochs."""
    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(
        neupi_utils, "_effect_losses", lambda m, d, s, dev:
        (m.weight.sum() * 0, torch.tensor(0.0)))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    best, best_loss = neupi_utils.train_predicate_model(0,
                                                        model, [None], [None],
                                                        optimizer, {},
                                                        num_epochs=2,
                                                        val_freq=10)
    assert best and best_loss == 0.0


def test_equiv_effect():
    """A transition matches an effect vector entry ([add, delete]) only if
    the atom changes as the entry says."""
    zero, one = torch.tensor(0.0), torch.tensor(1.0)
    no_effect, add, delete = (torch.tensor([0, 0]), torch.tensor([1, 0]),
                              torch.tensor([0, 1]))
    assert neupi_utils.equiv_effect(one, one, no_effect)
    assert not neupi_utils.equiv_effect(one, one, add)
    assert neupi_utils.equiv_effect(zero, one, add)
    assert not neupi_utils.equiv_effect(zero, one, delete)
    assert neupi_utils.equiv_effect(one, zero, delete)
