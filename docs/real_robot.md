# Real-robot template

The real-robot part of UniPred is a template: no robot drivers, ROS or
learned skills, but the whole path from recordings to a deployed model,
for Table Clean Real and for a domain of your own. The commands are in
[examples/real_robot/README.md](../examples/real_robot/README.md); this
page describes the pieces.

## From recordings to features

A recording is a session folder: keyframe images (one before each skill
and one after the last), the objects with their types, the goal, the
skills executed, and optional annotations of the true atoms (see the raw
format in the examples README). `examples/real_robot/process_demos.py`
perceives every keyframe with `unipred/perception/scene.py`:

1. **Detection** (`detector.py`): GroundingDINO finds boxes for the text
   prompt of each object type (`prompts` in `configs/real_robot.yaml`), and
   SAM2 refines them into masks.
2. **Identity**: detections are matched to the session's object names, left
   to right in the first keyframe, then by proximity to the previous box;
   an undetected object keeps its previous box, and boxes in the recording
   override detection (for objects the detector misses).
3. **Crops** (`crops.py`): a padded square around each box, keeping the
   surroundings that relational predicates depend on (optionally masked).
4. **Features** (`features.py`): the class token of DINOv3 for each crop.

The output (`processed.pkl` and the images) is what the domain's
environment reads (`--table_clean_real_data_dir`).

Weights: `scripts/install.sh --weights` downloads GroundingDINO (tiny),
SAM2.1 (tiny) and DINOv3 (ViT-S/16; gated on Hugging Face, accept its
licence first) into `$UNIPRED_WEIGHTS_DIR`. A missing model fails with the
path it expected. The `fake` detector and encoder need no weights and run
the synthetic fixture.

## Learning and evaluation

`unipred/envs/offline_real.py` builds the environment from the processed
data: train tasks are the recorded sessions (their demonstrations are the
dataset), test tasks the held-out scenes. There is no simulator, so
approaches plan skeletons only (`bilevel_plan_without_sim`), and
`unipred/eval/offline_planning.py` scores a test scene as solved if its
skeleton is valid under the ground-truth operators from the scene's
annotated atoms. Table III's methods run with `scripts/run.sh`:

| Table III row | Config (`configs/table_clean_real/`) |
|---|---|
| UniPred | `unipred.yaml` |
| IVNTR w/ derived | `ivntr_derived.yaml` |
| Pix2Pred w/ GT preds. and partial GT labels | `pix2pred_gt_labels.yaml` |
| Pix2Pred w/ GT preds. | `pix2pred_gt_predicates.yaml` |
| Pix2Pred | `pix2pred.yaml` |
| ViLa-zero-shot / few-shot / HPE | `vila_zero_shot.yaml`, `vila_few_shot.yaml`, `vila_hpe.yaml` |

The given predicates (goals such as ToyInBox) are read from the
annotations, which demonstrations and test scenes must have. On a new image
there are none: a small classifier per given predicate, trained on the
annotated keyframes, evaluates them.

## Deployment

`unipred/deploy.py` loads a learned model with the flags of its training
run and the perception of `configs/real_robot.yaml`; it checks that the
encoder is the one the training data was processed with.
`examples/real_robot/plan_from_image.py` plans from one image.
`examples/real_robot/closed_loop_example.py` runs the loop of Section IV:
perceive, plan, execute the first skill, perceive again and replan if the
skill's expected effects did not happen. It talks to the robot through
`RobotInterface`:

```python
class RobotInterface(abc.ABC):
    def capture(self) -> np.ndarray:            # an RGB image of the scene
        ...
    def execute(self, option_name, objects):    # run a skill on named objects
        ...
```

`FakeRobotInterface` replays a recorded session so that the example runs
offline. On a real robot, `execute` dispatches to your skills (motion
planning for picks and places, learned policies for pushing, ...), and
`capture` reads the first camera in `cameras` of the robot config.

Table IV of the paper (execution on the real robot) is not reproduced: it
needs the hardware.
