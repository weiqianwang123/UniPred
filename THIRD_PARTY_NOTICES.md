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

## Downloaded by `scripts/install.sh --sim`

Fetched by `python -m unipred.robot_sim.assets` into `sim_assets/` (not
committed), at pinned commits:

| Asset | Source | License |
|---|---|---|
| Franka Research 3 (`franka_fr3`) | google-deepmind/mujoco_menagerie | Apache-2.0 |
| Robotiq 2F-85 (`robotiq_2f85`) | google-deepmind/mujoco_menagerie | BSD-2-Clause (ROS-Industrial) |
| Google Scanned Objects (eight toys, a wooden tray, a sponge) as MuJoCo models | kevinzakka/mujoco_scanned_objects, from Downs et al., "Google Scanned Objects", ICRA 2022 | CC-BY-4.0 (the scans); MIT (the conversion) |

The gripper's mount on the arm follows R2S2R's `fr3_robotiq` (MIT). The
simulator can take its grasps from GraspGenX (NVIDIA), served over HTTP by
a separate server; nothing of GraspGenX is included or downloaded here.

`scripts/sam3_server.py` serves Meta's SAM3 (github.com/facebookresearch/sam3,
the SAM License; its checkpoint is gated on Hugging Face and downloaded by
the user) and is adapted from PhysCoder's perception servers
(github.com/Jaraxxus-Me/physcoder, MIT License, Copyright (c) 2025 Bowen
Li).

## Python dependencies

Installed from PyPI by `pip install -e`, under their own licenses; among
them PyTorch and torchvision (BSD-3-Clause), Hugging Face transformers
(Apache-2.0), pyperplan (GPL-3.0, used for planning heuristics), z3-solver
(MIT), the OpenAI and Google client libraries (Apache-2.0), NumPy, SciPy
and scikit-learn (BSD-3-Clause).
