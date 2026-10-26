# Third-party notices

UniPred (this repository, `unipred/` and its scripts and configs) is under
the MIT License (`LICENSE`). It includes or downloads the following
third-party software and models, under their own licenses.

## Included in this repository

| Component | Where | License |
|---|---|---|
| predicators (Chitnis, Silver et al.) and IVNTR (Li, Silver, Scherer, Gray), forked | `third_party/IVNTR` | MIT (`third_party/IVNTR/LICENSE.md`) |
| Fast Downward planner | `third_party/IVNTR/ext/downward` (git submodule) | GPL-3.0 (`third_party/IVNTR/ext/downward/LICENSE.md`) |
| Fast Downward PDDL translator, vendored in predicators | `third_party/IVNTR/predicators/third_party/fast_downward_translator` | GPL-3.0 |

Fast Downward is a separate program that predicators runs as a subprocess;
the vendored translator is used to ground PDDL tasks.

## Downloaded by `scripts/install.sh --weights`

| Model | Source | License |
|---|---|---|
| GroundingDINO (Swin-T) | `IDEA-Research/grounding-dino-tiny` | Apache-2.0 |
| SAM 2.1 (Hiera tiny) | `facebook/sam2.1-hiera-tiny` | Apache-2.0 |
| DINOv3 (ViT-S/16, LVD-1689M) | `facebook/dinov3-vits16-pretrain-lvd1689m` | DINOv3 License (Meta); accept it on the model page before downloading |

ResNet-18 ImageNet weights are downloaded by torchvision (BSD-3-Clause)
when Packing runs without DINOv3.

## Python dependencies

Installed from PyPI by `pip install -e`, under their own licenses; among
them PyTorch and torchvision (BSD-3-Clause), Hugging Face transformers
(Apache-2.0), pyperplan (GPL-3.0, used for planning heuristics), z3-solver
(MIT), the OpenAI and Google client libraries (Apache-2.0), NumPy, SciPy
and scikit-learn (BSD-3-Clause).
