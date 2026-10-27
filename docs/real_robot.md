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
   prompt of each object type (`prompts` in `configs/real_robot.yaml`; a
   list of prompts for a type whose objects look different), and SAM2
   refines them into masks; or (`detector: sam3`) SAM3 segments the
   instances of each prompt, from a SAM3 server (`scripts/sam3_server.py`,
   at `$UNIPRED_SAM3_URL`). Overlapping detections of one type keep the best
   scored, and so do detections of different types that overlap almost
   entirely (the tray found as a tray and taken for a toy);
   `max_box_area` drops a type's detections larger than a fraction of the
   image.
2. **Identity**: detections are matched to the session's object names, left
   to right in the first keyframe, then by appearance (the encoding of the
   masked crop, as first and as last seen) and place (Hungarian matching).
   With a tracker (`tracker: sam3`, SAM3's video tracker on the SAM3
   server) and the frames recorded during the skills (`frames/<k>/`), the
   objects are also followed from keyframe to keyframe: in the next
   keyframe the detection that overlaps an object's followed mask is the
   object, or the followed mask itself when none does (a toy in the
   gripper that the detector misses); the objects the tracker loses are
   matched by appearance and place. An undetected object keeps its
   previous box, and boxes in the recording override detection (for
   objects the detector misses).
   `process_demos.py --fill_missing FILE` takes the box of an object that
   is missed and has no earlier box from a file of each session (the
   simulator's `gt_boxes.json`), as if it had been annotated.
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

`closed_loop_example.py --video loop.mp4` writes an annotated video of the
loop (`unipred/loop_video.py`). Each keyframe shows the camera image with
the boxes and names that perception found. Beside it is the plan, with the
skills done and the one about to run. Below the plan are the atoms the
robot believes. Each invented predicate is named after the hidden
ground-truth predicate it agrees with most on the annotated training
keyframes, with the agreement. The frames recorded while a skill runs
follow.

Table IV of the paper (execution on the real robot) is not reproduced: it
needs the hardware. The simulated robot below stands in for it.

## Simulated robot

The real-robot domains have MuJoCo stand-ins, to record demonstrations,
train and run the closed loop without hardware:
`unipred/robot_sim/table_clean_mujoco.py` (Table Clean Real) and
`cluttered_retrieval_mujoco.py` (Cluttered Retrieval Real), scenes of the
same Franka at a table (`franka_scene.py`: the robot, the cameras, the
scripted motions, grasping, the video). Table Clean:

- **Scene.** A Franka Research 3 with a Robotiq 2F-85 (mujoco_menagerie;
  the gripper on the flange as on DROID, as R2S2R's `fr3_robotiq`)
  at the edge of a table, with two or three toys, a wooden tray (the box)
  lengthwise on the robot's right, near or pushed out, a sponge (the wiper)
  at the tray's far end, and dirt. The toys, the tray and the sponge are
  Google Scanned Objects; each scene draws its toys from eight by its
  seed. A camera across the table renders 640x480 images.
- **Skills.** The seven skills of Table Clean are scripted motions: straight
  lines of the gripper, followed through inverse kinematics by the arm's
  position actuators while MuJoCo simulates. A pick grasps straight down
  across the object's narrow side (`grasps="top_down"`, the default), or
  takes its grasps from GraspGenX (`--grasps graspgenx`: a GraspGenX
  server, not included, on `localhost:8116` or `$UNIPRED_GRASPGENX_URL`;
  `unipred/robot_sim/graspgenx.py`), given the object's surface points and
  the rest of the scene as two depth cameras see it: it tries the reachable
  grasps from above, best first, until both pads close on the object (up
  to 3 plans of 6). A weld then holds the object. Pushing and pulling the
  tray press on its near wall and drag it; wiping moves the sponge over
  the dirt, which goes.
- **Constraints are physical.** Table Clean Sim's rules come from the
  scene, not from checks. Pushed out, the tray's far end with the sponge
  is beyond the arm's reach (straight down, it reaches x = 0.75 m on that
  side; the sponge is at 0.84), so the sponge can be taken out or put back
  only with the tray pulled in. The near tray stands on part of the dirt
  and the toys stand on it, so a wipe stops where the sponge meets the
  tray or a toy. A skill that cannot be carried out raises
  SimulationError (the closed loop sees its effects missing and replans);
  only what is impossible outright (picking with a full hand) does
  nothing.
- **Ground truth.** The scene keeps the places of the objects, the tray's
  side and whether the table is clean; `atoms()` are the true atoms, and
  `problems()` lists where the simulated objects disagree with them.

`examples/real_robot/collect_sim_demos.py` records sessions in the raw
format: train sessions are Table Clean Sim's oracle demonstrations executed
skill by skill, test scenes the initial scenes of its out-of-distribution
tasks; each keyframe also gets its true atoms and the objects' image boxes
(`gt_boxes.json`, to score detection), and the camera records every skill
(`frames/<k>/`, `--video_hz` frames per simulated second, 5 by default).
A session whose skill fails, or whose
objects end up where the atoms do not say, is simulated again with another
seed. Its `session.json` describes the scene, so that
`closed_loop_example.py --robot sim` starts the loop from it: the skills of
the plan run on the simulated Franka, and the end says whether the goal
holds in the simulator. `configs/sim_robot.yaml` has the perception for
the simulator's images: SAM3 with a prompt per type, and its video tracker
for the toys.

Cluttered Retrieval (`collect_sim_demos.py --domain cluttered_retrieval`,
perception in `configs/sim_robot_cluttered_retrieval.yaml`):

- **Scene.** The Franka puts snack bars (the targets) into a black basket
  (the box). In front of them, seen from a low camera across the table,
  stand obstacles: scanned bottles, and a gift box and a lunch box on its
  side (too wide to grasp). An obstacle blocks a target if, on its own, it
  hides at least a fifth of the target's pixels from the camera; a target
  no obstacle blocks is all but fully in view. Positions are drawn until
  the camera shows the scene's blocks of the Cluttered Retrieval Sim task
  so, and every bulky obstacle has a side to be pushed to where it hides
  nothing.
- **Blocked targets cannot be picked, physically.** As the real robot's
  grasps come from the object's segmented point cloud, a pick sees the
  target through the camera: one less than 90 percent in view is not
  segmented well enough for a grasp (the pick fails), and grasps are
  planned on the target's points in view.
- **Skills.** PickTarget and PlaceTargetInBox pick a target in view and lay
  it in the basket; RemoveObstacle picks a bottle, carries
  it high over the others and stands it upright in a corner behind every
  target; PushObstacle lowers the closed fingers beside a bulky obstacle
  (which slides rather than tips) and pushes it sideways as far as it
  takes.
