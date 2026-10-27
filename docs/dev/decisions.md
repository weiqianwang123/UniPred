# Design decisions

Choices made where the paper (arXiv 2512.17992) does not specify the
design. Each entry says what was chosen and why.

## Table Clean Sim

- **Options.** The seven controllers PickToyFromTable(robot, toy, table),
  PlaceToyInBox(robot, toy, box), PickWiperFromBox(robot, wiper, box),
  PlaceWiperInBox(robot, wiper, box), PushBoxOut(robot, box),
  PullBoxIn(robot, box) and WipeTable(robot, wiper, table, box).
  WipeTable takes the box because its precondition "the box is on the far
  side" is about the box, and an operator can only mention its parameters.
- **Pushing and pulling the box** work with the wiper in hand. With only
  these seven options, a robot that needed an empty hand to push the box
  could never hold the wiper while the box is on the far side, which
  wiping needs.
- **Goal: the toys in the box, the table clean and the wiper back in the
  box.** The paper's goal leaves out the wiper. Without it, the
  demonstrations from the standard start (two toys on the table, the wiper
  in the box, the box on the near side) never use PullBoxIn or
  PlaceWiperInBox, so no operator can be learned for them, while the
  unseen test starts (wiper in hand, box on the far side) need both.
  Putting the wiper back makes every demonstration use all seven options,
  and every test start solvable with the learned operators. This also
  gives the paper's constraint "storing the towel requires the box on the
  near side" a role in every task.
- **State.** Per-object features as listed in the paper: 2D poses plus
  in_box, on_table and held for toys, in_box and held for the wiper, side
  for the box (0 near, 1 far), is_clean for the table and gripper for the
  robot.
- **Given predicates** are the goal predicates (ToyInBox, TableClean,
  WiperInBox); HandEmpty, HoldingToy, HoldingWiper, ToyOnTable, BoxNear and
  BoxFar are invented, and NoToyOnTable is the derived predicate ForAll
  ?toy. Not ToyOnTable(?toy, ?table).

## Packing

- **Scene.** A row of boxes, each split by a movable divider into a left
  and a right region that hold one item each. Items are small or large;
  a large item fits only a region widened by shifting the divider, which
  is possible only while both regions of the box are empty. This keeps
  the paper's two features of Packing: the item and region properties
  that matter (size, width, occupancy) are visible only in the images,
  and solving a task sometimes needs an action (Shift) whose only purpose
  is to enable a later placement.
- **Options.** Pick(robot, item), PlaceSmall and PlaceLarge(robot, item,
  region), and Shift(robot, region, other). The paper names placing skills
  per item size; Shift takes both regions of the box so that its effects
  (one region wide, both regions no longer centered) mention only its
  parameters.
- **State.** Poses and the robot's gripper and held flags, plus each item's
  and region's crop feature vector from a frozen encoder. What only the
  images show is kept in State.simulator_state for the ground-truth
  predicates, so `allow_state_allclose_comparison_despite_simulator_state`
  is set.
- **Encoder.** DINOv3 (the paper's) when its weights are installed, else
  ResNet-18 with ImageNet weights (`--perception_encoder auto`); either can
  be forced. A "fake" encoder (mean colors of a 4 x 4 grid) exists only for
  tests. No encoder is ever used with random weights.
- **Given predicates** are the goal predicate Packed and the static facts
  IsSmall, IsLarge and SameBox; HandEmpty, Holding, OnTable, In, Wide and
  Centered are invented, and Free (no item in the region) is derived.
- **Task sizes.** Train: two boxes, one large and two small items. Test:
  three boxes, two large and two small items. Larger test tasks made
  ground-truth planning time out within the 10 s budget without changing
  what is being tested (more boxes and more large items than in training).
- **50 demonstrations**, as in the paper.

## Planning with derived predicates

- **Delete relaxation.** Derived atoms are never operator effects, so the
  delete relaxation cannot tell when they become true. While an atoms
  closure is installed, reachability checks and the delete-relaxation
  heuristics (lmcut, hff, ...) assume every derived atom is reachable.
  This is optimistic, so it never rules out a plan, and it lets A* use the
  usual lmcut heuristic in every domain instead of goal_count.
- **Fast Downward** cannot call the closure, so planning with fdopt/fdsat
  while a closure is installed is an error rather than a silent fallback.

## Learning

- **Loss feedback scale.** The LLM sees each earlier proposal with a score
  in [0, 100] that is linear in log(validation loss): 100 at the
  acceptance threshold (`loss_thresh`), 0 at a loss of 1. Losses span
  orders of magnitude, and a fixed mapping keeps a proposal's score
  unchanged as new proposals arrive. With `unipred_loss_feedback: false`
  the LLM sees only "accepted" or "rejected".
- **Classifier seeds.** Each classifier is seeded from (seed, predicate,
  iteration, proposal index), so the same seed learns the same predicates
  whatever order the parallel training processes finish in.
- **JS loss.** The fork's divergence on unchanged atoms is kept as it is
  (not the full Bernoulli Jensen-Shannon divergence), because the
  acceptance threshold 0.005 is calibrated to it.
- **IVNTR configs** share the predicate types of the UniPred configs
  (configs/<domain>/ivntr_predicates.yaml includes predicates.yaml), so
  the two methods search the same predicate space. They differ in how
  candidates are proposed and in when the search of a type stops: IVNTR,
  as upstream, runs its whole budget and keeps every vector that passes,
  for selection to choose from; UniPred's LLM loop stops once
  `max_num_vectors` vectors pass (the paper's LLM proposals are few and
  semantically targeted). When more vectors pass than may be kept, the
  ones with the lowest validation loss are kept. With a cap of one, IVNTR
  had kept "Calibrate adds Sees" over "MoveTo adds Sees" in one run and
  solved no Satellites task.

- **Derived-aware selection** (unipred/derived/selection.py, the fork's
  hooks in bilevel_learning_approach.py). IVNTR keeps a negated or
  quantified variant only if it is a consistent effect, so "no toy is on
  the table" (picking a toy makes it true only for the last toy) never
  reached selection, and Table Clean Sim's learned wiping had no such
  precondition. UniPred also keeps a variant that is a precondition of some
  option (holding before all its unchanged transitions, not before every
  transition), scores a basic candidate together with each of its variants
  (the toy-table predicate helps only through "no toy is on the table"),
  goes on below the selection threshold while a candidate lowers the score
  by 1 percent, and after operator learning drops each derived
  precondition whose removal keeps the score within 1 percent (the
  demonstrations, all in one order, put "no toy is on the table" before
  every later operator, which made tasks starting with the wiper in hand
  unsolvable). A rule preferring variants that keep a single-object type
  (the table) free was tried and dropped: "the hand holds no wiper" must
  quantify the wiper to be a precondition of picking a toy.
- **Where derived preconditions go.** The demonstrations cannot tell
  whether "no toy is on the table" belongs to taking the wiper or to
  wiping, and the planning effort on them prefers the former (it even
  saves search there); tasks that start with the wiper in hand then plan
  wiping first. A derived precondition is therefore dropped from an
  operator none of whose parameters has a type of its basic predicate
  (taking the wiper is about the robot, the wiper and the box) before the
  score-based pruning. Seed 0, cached LLM replies: Table Clean Sim 50 of 50
  test tasks (46 without the rule, with 50 skeletons), with the default 8
  skeletons; the simulated robot's sessions with true boxes 9 of 10 (6).
- **Sampler inputs.** Learned samplers see every object feature but those
  `domain_sampler_data_filter` leaves out: Satellites' ids
  (object.id, calibration_obj_id, read_obj_id) are labels, and test tasks
  have an id no training task has. A sampler keeps the filter it was
  trained with.

## Baselines

- **GNN and transformer policies** use the ground-truth NSRTs' samplers to
  turn a predicted option and objects into continuous parameters, as the
  predicators baselines do.
- **Grammar search** uses the cluster-and-intersect operator learner and a
  bounded budget (at most 50 predicates, 20 demonstrations for scoring and
  a 0.1 s planning timeout while scoring), so a run ends in minutes rather
  than days.
- **Code as Policy** (Table VI) gets what the paper lists (object states,
  the skills, a description of the goal and constraints) plus the same
  three demonstrations UniPred's prompts show, rendered as low-level
  states, skills with parameters and the features each skill changed, so
  both methods learn from the same data. The policy picks the next skill
  (name, objects, parameters) whenever the current one finishes;
  parameters outside a skill's range are clipped. Only errors (an
  exception, a malformed return value, unknown names or types) are sent
  back for repair, at most three times; failing to reach a goal is not.
  Prompts show at most twelve features per object, so crop features of
  image domains do not flood them; the policy receives all features.

## Paper tables

- **Domains of the averaged tables.** Tools is not part of this release.
  Table V (paper: Satellites, Blocks, Tools, Packing) averages Satellites,
  Blocks and Packing; Table VII (paper: "the three simulated domains",
  whose IVNTR numbers are those of Satellites, Blocks and Tools) averages
  Satellites and Blocks.
- **Averages** are taken per seed over the domains, then reported as mean
  ± sample standard deviation over the seeds.
- **Run names.** scripts/run.sh names an output directory after the config
  and its overlays (unipred+no_semantic), so ablations never overwrite the
  full method's results; the table scripts skip seeds that have results.

## Real-robot template (Table Clean Real)

- **Data.** examples/real_robot/process_demos.py turns raw sessions (one
  folder each: keyframe images, the objects with their types, the goal,
  the actions, optional annotated atoms and boxes) into processed.pkl:
  per keyframe, each object's crop features and the annotated atoms. The
  env (unipred/envs/offline_real.py) builds the train tasks, the recorded
  demonstrations (--offline_data_method recorded) and the test scenes
  from it; it has no simulator.
- **Fork hooks.** BaseEnv.get_recorded_dataset() and
  BaseEnv.evaluate_offline(approach) let an env without a simulator
  provide its data and score test tasks itself; get_last_nsrt_plan() is
  available to every planning approach.
- **Evaluation** (Table III protocol): the approach plans a skeleton
  without a simulator; the skeleton is valid if every step matches a
  ground-truth operator whose preconditions hold in the annotated atoms
  (with derived atoms recomputed by the ground-truth closure) and the goal
  holds at the end. In-domain tests plan from the first keyframes of the
  train sessions.
- **Given predicates** (the paper assumes goal and static predicates are
  provided) are read from the annotations, which demonstrations and test
  scenes must have. At deployment there are none, so each given predicate
  is evaluated by an MLP over its arguments' crop features, trained on the
  annotated keyframes.
- **Recorded options** have no parameters: the middle of the parameter
  space stands for them, and samplers are not learned
  (sampler_learner: random). A recorded option counts as initiable and
  terminal, since the options' own functions read features that crops do
  not have.
- **Object identity.** In the first keyframe, the detections of a type
  are assigned to the session's objects of that type from left to right;
  afterwards each object takes the nearest detection (Hungarian matching
  on box centres). An undetected object keeps its previous box (a toy in
  the gripper); boxes in the raw data override detection.
- **Crops** keep the object's surroundings (padding 0.15 of the box) so
  that relational predicates can be learned; masking with SAM2 is an
  option.
- **Extensibility.** Nothing in perception, processing, the env base
  class, evaluation or deployment names a table-clean type or option. A
  second real domain needs a config directory (domain.yaml, prompts in
  real_robot.yaml, approach configs) and a ground-truth package (options,
  operators, and a five-line OfflineRealEnv subclass that names the env
  and its data-directory setting).
- **Fixture.** Generated from Table Clean Sim: oracle demonstrations and
  out-of-distribution test starts, drawn top-down with one colour per
  object type, which the fake detector finds; the fake encoder averages
  colours on a 4 x 4 grid.
- **ViLa.** zero_shot gets no examples (the original ViLa prompt's
  generic in-context examples are not reproduced); few_shot adds the
  first image, goal and plan of three recorded sessions; hpe adds the
  hand-written constraints of configs/table_clean_real/vila_constraints.txt.
- **Pix2Pred.** Predicates are proposed from the first four keyframes of
  the first demonstration; each image is labelled once with the true atoms
  of all predicates; operators come from cluster-and-intersect. With
  ground-truth predicates, the hidden predicates of the env (including
  NoToyOnTable, labelled directly) replace invention; partial GT labels
  take the hand-state predicates (HandEmpty, HoldingToy, HoldingWiper)
  from the annotations.

## Simulated robot (unipred/robot_sim)

- **Why.** Table IV needs the real robot; the simulator gives Table Clean
  Real's pipeline (record, perceive, invent, plan, closed loop) images of
  a real-looking scene and a robot to execute on, with true atoms.
- **Assets.** The FR3 and the 2F-85 are mujoco_menagerie's, the gripper
  mounted as CDG's and R2S2R's fr3_robotiq (flange 0.107 m, turned 90
  degrees, pads at R2S2R's friction and squeeze). Toys, tray and sponge
  are Google Scanned Objects chosen to fit the gripper (narrow side under
  7 cm) and the tray (long side under 12 cm); a stacking toy that toppled
  was dropped. Fetched at pinned commits, not committed.
- **Skills** are scripted Cartesian lines through damped least squares IK
  on the arm's position actuators. A grasp closes the real fingers and is
  then held by a weld, so that a demonstration never fails on a slip (the
  demonstrations are what is learned from, not grasping). Likewise the
  tray slides along x only, a position drive keeping it with the gripper
  that presses on its wall, rather than being pushed by contact. The
  skills' conditions are Table Clean Sim's, so its oracle plans are the
  demonstrations.
- **GraspGenX** gets the object's mesh vertices as its points (the
  cameras saw ~170 points of a toy at the far end and none of the sponge's
  sides in the tray, and GraspGenX then gave no grasp) and the rest of the
  scene from two depth cameras for its collision filter (5 mm). Grasps
  from above (within 50 degrees) that the arm reaches are tried best
  first until both pads touch the object; grasps whose open pads would
  land on the object are skipped. 24 of 24 full demonstrations succeed.
- **Layout.** The sponge lies along the tray, so that the fingers close
  across it clear of the walls; across, GraspGenX's collision filter
  removed every grasp of its narrow side and only pinches of its ends
  were left.
- **Names.** The raw format names the objects of a type left to right in
  the first keyframe; the collector renames the toys so, or the labels of
  one toy would go with another's crops.
- **Stains** are painted into the texture of a thin layer over the table
  (the table's own texture with soft brown smears), hidden once the table
  is wiped. 3D blots, stacked or translucent, were detected as toys (a
  held toy that went undetected was then matched to a stain for the rest
  of its session) or cast stippled shadows.
- **Toys** that SAM3 handles poorly were replaced: a car of separately
  coloured parts (split into several "toys") and a stacking toy that
  toppled.
- **Perception.** SAM3 (CDG's server) is the simulator config's detector:
  it found 258 of 259 toys where GroundingDINO found 189 of 215, but also
  takes the arm, tray and sponge for toys; detections of different types
  that overlap almost entirely (IoU > 0.7) keep the best scored, which
  removes those. Its robot is the whole arm (no prompt found the gripper
  alone), so the robot's true box is the whole robot's; the prompt is
  "robotic arm": "robot" also matches the toys that are robot figures
  (Android), which then lost to it, and adding "robot arm" found parts of
  the arm (the robot right in 452 of 510 keyframes, against 500 of 510).
- **Tracking is the bottleneck of the real-robot pipeline on the
  simulator.** With true boxes, a linear classifier tells a toy on the
  table from one elsewhere by its ResNet-18 features without error, and
  UniPred solves 9 of 10 test scenes (20 demonstrations); with detection
  and matching it was 95 percent, too noisy for the classifiers' loss
  threshold, and UniPred did not invent the toy-table predicate.
