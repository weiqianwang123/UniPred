"""Tests for predicators/gnn/neupi.py."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from predicators.gnn.neupi import quaternion_to_matrix


def test_quaternion_to_matrix():
    """Tests for quaternion_to_matrix()."""
    # Compare against scipy, which orders quaternions scalar-last.
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(64, 4))  # (w, x, y, z), not normalized
    expected = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    actual = quaternion_to_matrix(torch.tensor(quats)).numpy()
    assert actual.shape == (64, 3, 3)
    assert np.allclose(actual, expected, atol=1e-10)
    # Results are proper rotations.
    assert np.allclose(actual @ actual.transpose(0, 2, 1), np.eye(3))
    assert np.allclose(np.linalg.det(actual), 1.0)
    # Identity and a 90 degree rotation about the z axis.
    identity = quaternion_to_matrix(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(identity, torch.eye(3))
    half = float(np.sqrt(0.5))
    rot_z = quaternion_to_matrix(torch.tensor([half, 0.0, 0.0, half]))
    assert torch.allclose(rot_z,
                          torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0]]),
                          atol=1e-6)
    # Batch dimensions are preserved.
    assert quaternion_to_matrix(torch.ones(2, 5, 4)).shape == (2, 5, 3, 3)
