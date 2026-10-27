# Real-robot template

From recorded demonstrations on your robot to a learned planning model
deployed in a closed loop, for Table Clean Real (the paper's Table III) or
a domain of your own (see [docs/adding_a_domain.md](../../docs/adding_a_domain.md)).

1. Record sessions in the raw format below.
2. `process_demos.py` detects, crops and encodes the objects of every
   keyframe (GroundingDINO, SAM2, DINOv3; `configs/real_robot.yaml`).
3. Train and evaluate with `scripts/run.sh` on the processed data: UniPred
   invents predicates from the crop features; test scenes are scored by
   offline planning against the ground-truth operators.
4. Deploy: `plan_from_image.py` plans from one image;
   `closed_loop_example.py` runs perceive → plan → execute → re-ground →
   replan through a `RobotInterface` that you implement.

Everything runs offline on the synthetic fixture in `fixture/raw` with the
fake perception backend (`--detector fake --encoder fake`), which needs no
weights. The commands below use it; drop the two flags (and run
`scripts/install.sh` for the weights) for real images.

## Raw format

One folder per session:

```
raw/
  train_00/
    session.json   {"split": "train", "objects": {"toy0": "toy", ...},
                    "goal": [["ToyInBox", ["toy0", "box"]], ...]}
    actions.json   [{"option": "PickToyFromTable",
                     "objects": ["robot", "toy0", "table"]}, ...]
    0.png 1.png …  keyframe k shows the scene before action k; the last
                   one shows it after the last action
    atoms.json     optional: per keyframe, the true ground-truth atoms,
                   [[["HandEmpty", ["robot"]], ...], ...]
    boxes.json     optional: per keyframe, boxes that replace detection,
                   [{"toy0": [x0, y0, x1, y1]}, ...]
    frames/        optional: frames/<k>/0.jpg 1.jpg … the camera's frames
                   during action k (any image format, in the order of their
                   numbers), through which a tracker follows the objects
  test_00/
    session.json   {"split": "test", ...}
    0.png          the scene
    atoms.json     its initial atoms (required for a test scene)
```

- `objects` names every object of the session with its type; the types,
  predicates and goal predicates are those of the domain spec
  (`configs/table_clean_real/domain.yaml`), the option names those of the
  domain's ground-truth options.
- Train sessions are the demonstrations; test sessions are held-out scenes
  (an image, its annotated atoms and a goal).
- Annotations: the given predicates (those not in `excluded_predicates`,
  e.g. ToyInBox, TableClean, WiperInBox) must be annotated in every
  keyframe of the demonstrations, all non-derived predicates in the
  first keyframe of a test scene. Derived predicates (NoToyOnTable) are
  computed and need no annotation. On new images, the given predicates are
  evaluated by classifiers trained on the annotations.
- Objects are matched to detections left to right in the first keyframe
  and by appearance and place afterwards; with a tracker (`tracker` in the
  robot config) and `frames/`, they are followed from keyframe to keyframe
  through the frames. An undetected object keeps its last box.

`fixture/make_fixture.py` regenerates the fixture from Table Clean Sim.

## Commands

Process the fixture:

```bash
python examples/real_robot/process_demos.py --raw examples/real_robot/fixture/raw --out outputs/table_clean_real/processed --domain configs/table_clean_real/domain.yaml --detector fake --encoder fake
```

Invent predicates and evaluate offline (4 demonstrations and 2 test scenes
in the fixture; the mock LLM needs no network):

```bash
LLM=mock scripts/run.sh configs/table_clean_real/unipred.yaml 0 --num_train_tasks 4 --num_test_tasks 2
```

The baselines of Table III run the same way:
`configs/table_clean_real/{ivntr_derived,vila_zero_shot,vila_few_shot,vila_hpe,pix2pred,pix2pred_gt_predicates,pix2pred_gt_labels}.yaml`.

Plan from the image of a test scene:

```bash
python examples/real_robot/plan_from_image.py --session examples/real_robot/fixture/raw/test_00 --detector fake --encoder fake --config configs/table_clean_real/unipred.yaml --config configs/llm/mock.yaml --seed 0 --num_train_tasks 4 --approach_dir outputs/table_clean_real/unipred/seed0/approach
```

Run the closed loop on a replayed session:

```bash
python examples/real_robot/closed_loop_example.py --session examples/real_robot/fixture/raw/train_00 --detector fake --encoder fake --config configs/table_clean_real/unipred.yaml --config configs/llm/mock.yaml --seed 0 --num_train_tasks 4 --approach_dir outputs/table_clean_real/unipred/seed0/approach
```

The deployment scripts take the flags of the training run, so they load
the same model.

## On your robot

Implement `RobotInterface` (`closed_loop_example.py`): `capture()` returns
an RGB image from your camera, `execute(option_name, objects)` runs your
skill for that option on the named objects, and optionally `video()`
returns the camera's frames during the last skill, for the tracker. Set
the detector prompts per object type, the thresholds and the tracker in
`configs/real_robot.yaml`.
