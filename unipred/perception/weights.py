"""Where the perception models' weights are installed.

scripts/install.sh downloads each model into a subdirectory of
CFG.perception_weights_dir (default: the UNIPRED_WEIGHTS_DIR environment
variable):

- grounding_dino: IDEA-Research/grounding-dino-tiny,
- sam2: facebook/sam2.1-hiera-tiny,
- dinov3: facebook/dinov3-vits16-pretrain-lvd1689m.

A model is only loaded from there; if it is missing, loading fails with the
path, never with randomly initialized weights.
"""

from pathlib import Path

from predicators.settings import CFG

MODELS = ("grounding_dino", "sam2", "dinov3")


def weights_path(model: str) -> Path:
    """The directory of a model's installed weights."""
    assert model in MODELS, f"Unknown perception model {model!r}."
    return Path(CFG.perception_weights_dir) / model


def is_installed(model: str) -> bool:
    """Whether the model's weights are installed (its config is there)."""
    return bool(CFG.perception_weights_dir) and (weights_path(model) /
                                                 "config.json").is_file()


def require(model: str) -> Path:
    """The directory of a model's weights; raises FileNotFoundError with
    the expected path if they are not installed."""
    if not is_installed(model):
        raise FileNotFoundError(
            f"{model} weights not found at {weights_path(model)}. Run "
            "scripts/install.sh or set --perception_weights_dir (default: "
            f"the UNIPRED_WEIGHTS_DIR environment variable) to the directory "
            f"that contains {model}/.")
    return weights_path(model)
