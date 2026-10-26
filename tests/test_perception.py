"""Tests for the crop encoders (no network weights needed)."""

import numpy as np
import pytest

from predicators import utils
from unipred.perception.features import CropEncoder, resolve_backend


def test_fake_encoder_is_deterministic_and_cached():
    """The fake backend gives fixed features that depend on the crop."""
    encoder = CropEncoder("fake")
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((16, 16, 3), dtype=np.uint8)
    blue[..., 2] = 255
    features = encoder.encode([red, blue, red])
    assert features.shape == (3, encoder.dim)
    assert np.allclose(features[0], features[2])
    assert not np.allclose(features[0], features[1])
    assert np.allclose(encoder.encode([red])[0], features[0])


def test_missing_dinov3_weights_name_the_path(tmp_path):
    """DINOv3 without installed weights fails with the expected path."""
    utils.reset_config({"perception_weights_dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "dinov3")):
        CropEncoder("dinov3")
    utils.reset_config({"perception_weights_dir": ""})
    with pytest.raises(FileNotFoundError, match="UNIPRED_WEIGHTS_DIR"):
        CropEncoder("dinov3")


def test_auto_backend(tmp_path):
    """auto is DINOv3 when its weights are installed, else ResNet-18."""
    utils.reset_config({"perception_weights_dir": str(tmp_path)})
    assert resolve_backend("auto") == "resnet18"
    (tmp_path / "dinov3").mkdir()
    assert resolve_backend("auto") == "dinov3"
    assert resolve_backend("fake") == "fake"
    with pytest.raises(ValueError, match="Unknown encoder backend"):
        CropEncoder("random")
