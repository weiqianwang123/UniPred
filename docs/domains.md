# Domains

Every domain has a directory under `configs/` with a `base.yaml` (the
environment and its task sizes), one config per approach, and
`predicates.yaml` (what UniPred knows and learns: the domain description,
the known predicates with their known effects, and the neural predicate
types to invent). The predicates listed as hidden are excluded
(`excluded_predicates`): the learning approaches must invent them; the
oracle uses all predicates.

| Domain | `--env` | State | Demos | Train tasks | Test tasks |
|---|---|---|---|---|---|
| Satellites | `satellites` | SE2 poses, attributes | 500 | 2 satellites, 2 objects | 3 satellites, 3 objects |
| Blocks | `blocks_onclear` | 3D poses | 500 | 4-5 blocks | 6-7 blocks |
| Packing | `packing` | poses + image crop features | 50 | 2 boxes, 3 items (1 large) | 3 boxes, 4 items (2 large) |
| Table Clean Sim | `table_clean_sim` | SE2 poses, flags | 50 | 2 toys, standard start | 3 toys, varied starts |
| Table Clean Real | `table_clean_real` | image crop features | recorded | recorded sessions | held-out scenes |

Each experiment tests on 50 tasks of the test distribution (and, with
`--in_domain_test True`, 50 of the training distribution).

## Satellites

Satellites carry instruments (camera, infrared sensor, Geiger counter) and
take readings of objects in their view; a camera reading needs the object
shot with Chemical X first, an infrared reading Chemical Y, and every
reading a calibrated instrument.

- Options: Calibrate, MoveTo, MoveAway, ShootChemX, ShootChemY, UseCamera,
  UseGeiger, UseInfraRed (each on a satellite and an object).
- Given: CameraReadingTaken, GeigerReadingTaken, InfraredReadingTaken
  (goals); CalibrationTarget, HasCamera, HasGeiger, HasInfrared,
  ShootsChemX, ShootsChemY (static).
- Hidden: Sees, ViewClear, IsCalibrated, HasChemX, HasChemY.
- Planner: Fast Downward (`fdsat`), 5 s per task.

## Blocks

A robot arm builds two-level towers, a given block on another given block;
AchieveGoal confirms a finished tower (the IVNTR and UniPred papers'
"Blocks").

- Options: PickFromTable, Unstack, Stack, PutOnTable, AchieveGoal.
- Given: GoalAchieved. Hidden: On, OnTable, GripperOpen, Holding, Clear.
- Planner: A* (Fast Downward finds unrefinable skeletons here), 5 s.

## Packing

Items of two sizes go into boxes split by a movable divider into a left
and a right region with room for one item each. A large item fits only a
region widened by shifting the divider, which is possible only while both
regions of the box are empty. Region width, occupancy and item size are
visible only in the top-down image crops, whose features (DINOv3 if
installed, else ResNet-18; `--perception_encoder`) are part of the state.

- Options: Pick(robot, item), PlaceSmall and PlaceLarge(robot, item,
  region), Shift(robot, region, other region).
- Given: Packed (goal); IsSmall, IsLarge, SameBox (static).
- Hidden: HandEmpty, Holding, OnTable, In, Wide, Centered, and the derived
  Free(region) = ForAll ?item. Not In(?item, ?region).
- Planner: A* with the closure of Free, 10 s.

## Table Clean Sim

A robot puts every toy into the box, wipes the table with a towel (the
wiper) and puts the towel back. It holds one object at a time; wiping is
impossible while a toy is on the table or the box is on the near side; the
towel leaves and enters the box only while the box is near.

- Options: PickToyFromTable, PlaceToyInBox, PushBoxOut, PullBoxIn,
  PickWiperFromBox, PlaceWiperInBox, WipeTable.
- Given: ToyInBox, TableClean, WiperInBox (goals).
- Hidden: HandEmpty, HoldingToy, HoldingWiper, ToyOnTable, BoxNear, BoxFar,
  and the derived NoToyOnTable(table) = ForAll ?toy. Not
  ToyOnTable(?toy, ?table), a precondition of WipeTable.
- Test starts: toys may already be in the box, a toy or the towel may be in
  hand, the box may be on the far side.
- Planner: A* with the closure of NoToyOnTable, 10 s.

## Table Clean Real

The same types, options and operators as Table Clean Sim, built from
recordings of a real robot (see [real_robot.md](real_robot.md)): an
object's state is the feature vector of its image crop, train tasks are
the recorded sessions, test tasks held-out scenes, and approaches are
scored by offline planning. A synthetic fixture
(`examples/real_robot/fixture`) runs the chain without a robot.

## Approaches per domain

| Config | Approach | Satellites | Blocks | Packing | Table Clean Sim | Table Clean Real |
|---|---|---|---|---|---|---|
| `unipred` | UniPred | x | x | x | x | x |
| `ivntr` | IVNTR | x | x | x | x | |
| `ivntr_derived` | IVNTR with derived-aware selection | | | x | x | x |
| `oracle` | ground-truth operators | x | x | x | x | x |
| `gnn`, `transformer` | behaviour cloning with shooting | x | x | x | x | |
| `grammar` | grammar-based predicate invention | x | x | x | x | |
| `random` | random options | x | x | x | x | |
| `code_as_policy` | LLM-written policy | x | x | x | x | |
| `vila_*` | VLM planning from images | | | | | x |
| `pix2pred*` | VLM predicates and labels | | | | | x |
