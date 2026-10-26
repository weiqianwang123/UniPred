"""Per-object crop features: a frozen image encoder maps each object's crop
to a feature vector, which becomes the object's state in image domains.

Backends:

- "dinov3": DINOv3 (the paper's encoder), loaded with transformers from a
  local copy of a DINOv3 checkpoint (scripts/install.sh downloads
  facebook/dinov3-vits16-pretrain-lvd1689m); the feature is the class token
  of the last layer;
- "resnet18": torchvision's ResNet-18 with ImageNet weights (downloaded by
  torchvision on first use); the feature is the pooled penultimate layer;
- "fake": no network weights; each crop is averaged over a 4 x 4 grid of
  cells (48 numbers). It is deterministic and only meant for tests.

"auto" means "dinov3" if its weights are installed, else "resnet18".
Missing weights raise an error that names the expected path; an encoder is
never initialized with random weights.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from predicators.settings import CFG

# Where scripts/install.sh puts the DINOv3 checkpoint, under
# CFG.perception_weights_dir.
DINOV3_SUBDIR = "dinov3"
_FAKE_GRID = 4
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def dinov3_weights_path() -> Path:
    """The directory of the installed DINOv3 checkpoint."""
    return Path(CFG.perception_weights_dir) / DINOV3_SUBDIR


def resolve_backend(backend: str) -> str:
    """The backend that "auto" stands for, or the given one."""
    if backend != "auto":
        return backend
    if CFG.perception_weights_dir and dinov3_weights_path().is_dir():
        return "dinov3"
    return "resnet18"


class CropEncoder:
    """A frozen encoder of object crops (RGB uint8 arrays of any size).

    Features of a crop are cached by the crop's bytes, so re-encoding an
    unchanged object is free.
    """

    def __init__(self,
                 backend: str,
                 device: str = "cpu",
                 image_size: int = 224,
                 cache_size: int = 4096) -> None:
        self.backend = resolve_backend(backend)
        self._device = torch.device(device if torch.cuda.is_available()
                                    or device == "cpu" else "cpu")
        self._image_size = image_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._model: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        if self.backend == "fake":
            self.dim = 3 * _FAKE_GRID * _FAKE_GRID
        elif self.backend == "resnet18":
            self._model, self.dim = self._load_resnet18()
        elif self.backend == "dinov3":
            self._model, self.dim = self._load_dinov3()
        else:
            raise ValueError(f"Unknown encoder backend: {backend!r} (use "
                             "dinov3, resnet18, fake or auto).")

    def encode(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """The features of the crops, an array of shape (len(crops),
        self.dim)."""
        keys = [self._key(crop) for crop in crops]
        missing = [i for i, key in enumerate(keys) if key not in self._cache]
        if missing:
            new = self._encode_uncached([crops[i] for i in missing])
            for i, features in zip(missing, new):
                self._cache[keys[i]] = features
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        for key in keys:
            self._cache.move_to_end(key)
        return np.stack([self._cache[key] for key in keys]).astype(np.float32)

    @staticmethod
    def _key(crop: np.ndarray) -> str:
        return hashlib.sha1(
            np.ascontiguousarray(crop).tobytes() +
            str(crop.shape).encode()).hexdigest()

    def _encode_uncached(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if self._model is None:
            return np.stack([_grid_means(crop) for crop in crops])
        batch = torch.from_numpy(
            np.stack([self._preprocess(crop)
                      for crop in crops])).to(self._device)
        with torch.no_grad():
            return self._model(batch).float().cpu().numpy()

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize to a square image and normalize as ImageNet models
        expect; channels first."""
        image = Image.fromarray(crop.astype(np.uint8)).convert("RGB").resize(
            (self._image_size, self._image_size), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return ((pixels - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)

    def _load_resnet18(
            self) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
        import torchvision  # pylint: disable=import-outside-toplevel
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model.eval().to(self._device)
        return model, 512

    def _load_dinov3(
            self) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
        path = dinov3_weights_path()
        if not CFG.perception_weights_dir or not path.is_dir():
            raise FileNotFoundError(
                f"DINOv3 weights not found at {path}. Run scripts/install.sh "
                "or set --perception_weights_dir (default: the "
                "UNIPRED_WEIGHTS_DIR environment variable) to the directory "
                "that contains dinov3/.")
        from transformers import \
            AutoModel  # pylint: disable=import-outside-toplevel
        model = AutoModel.from_pretrained(str(path)).eval().to(self._device)

        def _class_token(batch: torch.Tensor) -> torch.Tensor:
            return model(pixel_values=batch).last_hidden_state[:, 0]

        return _class_token, int(model.config.hidden_size)


def _grid_means(crop: np.ndarray) -> np.ndarray:
    """Mean color of each cell of a 4 x 4 grid over the crop, in [0, 1]."""
    rows = np.array_split(crop.astype(np.float32) / 255.0, _FAKE_GRID, axis=0)
    return np.concatenate([
        cell.reshape(-1, 3).mean(axis=0) for row in rows
        for cell in np.array_split(row, _FAKE_GRID, axis=1)
    ])
