# Progress

Kept short and factual: what is done, the commands and their numbers, the
bugs fixed, and what remains. Design choices are in decisions.md.

## Commits

Root (refactor/paper-release), oldest first:

- 56c2cd1 unipred package on top of the IVNTR fork; b616b04 gymnasium.
- 78604b6 Table Clean Sim with the paper's seven options.
- 2eac151 Prompts in the paper's four parts; every LLM request logged.
- 8d1afb9 Baseline configs per domain; 4e7d858 Packing and crop encoders.
- d48bef3 Same seed + mock LLM = identical predicates (regression test).
- adc134c Test that each ablation changes only what it names.
- 6b270de Code as Policy (Table VI).
- f9461d3 summarize.py, scripts/paper/table{2,5,6,7,8}.sh, Qwen/Gemini
  overlays; run.sh names runs after config + overlays, LLM default codex
  if installed else openai.
- 651f3ba Perception (GroundingDINO + SAM2 detection, crops, scenes).
- 3ade3be Real-robot template: table_clean_real, offline planning
  evaluation, examples/real_robot (process, plan from image, closed loop),
  synthetic fixture.
- 32aca3e ViLa and Pix2Pred (Table III).
- 4a64bb3 One torch thread in the main process (--torch_num_threads).
- c067a08, f9192f3 Tests: every env, every config, every baseline.
- 534e168 Dependencies match the imports.
- install.sh; docs (architecture, domains, adding a domain, real robot),
  THIRD_PARTY_NOTICES, README codebase sections.
- 60db615 derived-aware selection; 9dfb04b derived preconditions only on
  operators about their types; 8e952e3 Packing selection with the
  visited-state set; fa4361c Packing's Shift on one box only.
- Simulated robot: 61adfc8, 13d2906 (scene, stains, names), 199ec04 SAM3
  and identity by appearance, cefa196 robot prompt, c4783c3 SAM3 video
  tracker between keyframes, 738a2ff Franka base class of the scenes.
- 06d7598 Cluttered Retrieval (Sim, Real, MuJoCo scene); 7090656 image
  boxes as features (optional); 408360d annotated closed-loop video.
- 8cf4204 the scenes' constraints are physical; 6cecf15 cameras;
  fede043 clutter prompts; 177e300 derived preconditions pruned only if
  the score is the same without them; 7012b09 real domains use the
  paper's image thresholds; 1879311, 2bc0291 perception fixes; 9f3bf70,
  e74fd5d, 6c35eb8, 9a0f40f closed loop and its video.

Fork (third_party/IVNTR, unipred), after a406fa2 (later: fdad623
derived-aware hooks, 6bed62c, 8ab1712 Satellites samplers, 0eef079):

- c5ab702 No unseeded global random numbers.
- d20f00b Hooks: BaseEnv.get_recorded_dataset, BaseEnv.evaluate_offline.
- 9692b27 Spawned training processes share the cores.
- a810ecb GNN features cover all known predicates.
- f5afa62 Fast Downward leaves the working directory alone.
- 115e561 Dead code, commented-out code and unused names removed (2 000
  lines); 5b17b60 notebooks run again, README updated.

## Results so far

All with `scripts/run.sh` unless noted.

- Oracle, 5 train / 10 test tasks, OOD and in-domain: 10/10 on
  satellites, blocks_onclear, table_clean_sim, packing.
- Every config of every simulated domain runs train -> OOD test ->
  in-domain test from the saved model with the mock LLM (5 train / 2 test
  tasks, tiny budgets; 36 configs, all exit 0 after the fixes below).
- Table Clean Real on the fixture (4 sessions, 2 test scenes, fake
  perception): oracle 2/2 OOD, 4/4 in-domain; unipred, ivntr_derived,
  vila x3, pix2pred x3 run end to end with the mock LLM (0/2: tiny
  budgets, fake LLM); plan_from_image and closed_loop_example run.
- Real LLM (local Codex CLI, gpt-5.6-luna, low reasoning effort), 1 seed,
  from snapshots of earlier code (before the thread, FD and determinism
  fixes), on a machine with load ~90 during testing:
  - table_clean_sim unipred: OOD 32/50, in-domain 50/50, learning 1984 s,
    24 LLM calls; failures: max skeletons reached (learned operators).
  - satellites unipred (500 demos): OOD 13/50, in-domain 49/50, learning
    12593 s, 51 LLM calls; the learned operators match the ground-truth
    ones; failures are refinement timeouts (5 s) and FD skeletons not
    refinable with the learned samplers.
  - satellites ivntr (500 demos): OOD 20/50, in-domain 48/50, learning
    13374 s (random_options: 0/50).
  Final-code reruns (snapshot run3, before the fixes below): satellites
  unipred OOD 20/50, in-domain 47/50; table_clean_sim unipred 32/50,
  50/50; satellites ivntr (uncapped) 20/50, 48/50; ablation
  no_derived_selection on table_clean_sim 1/50, 0/50; the quickstart
  (mock LLM, 50 demos) 0/10, as expected of the mock.
- After the Satellites sampler fix (fork 8ab1712; snapshot satfix, cached
  LLM replies of run3): satellites unipred OOD 48/50, in-domain 50/50
  (the paper: 95.2%). The ids-free samplers swapped into the saved run1
  IVNTR model: 49/50.
- After the derived-aware selection fixes (60db615; cached replies):
  table_clean_sim unipred OOD 34/50 with 8 skeletons, 46/50 with 50 (now
  the setting), in-domain 50/50; the same replies without the fixes 22/50
  with 50 skeletons (the paper: 93.4%).
- IVNTR on Satellites after the sampler fix (snapshot satfix): OOD 47/50,
  in-domain 50/50 (20/50 before).
- Simulated robot, Table Clean (unipred/robot_sim): 60 sessions with
  skill videos (50 demonstrations, 10 test scenes), processed with SAM3,
  ResNet-18 and the tracker. Tracking (per keyframe, a toy's box right):
  held 100/103 (84 without the tracker), in the tray 694/709 (680), on
  the table 212/218. UniPred (codex, cached replies), 20 demonstrations:
  OOD 5/10, in-domain 10/10 (with true boxes 9/10; with the earlier
  perception 1/10); the failures plan wiping first, as the toy-table
  predicate is not invented. Closed loop with the annotated video on
  test_01: the goal holds in the simulator.
- Simulated robots' constraints made physical (8cf4204): the far tray's
  sponge is out of reach, the near tray and the toys stand on the dirt,
  a hidden target is not segmented for a grasp; tests check each, and ten
  GraspGenX demonstrations of each domain reach the goal.
- Recollected with these constraints (raw_v3; seed 0, cached LLM replies,
  processed with SAM3, ResNet-18, the tracker and image boxes):
  - Table Clean (50 demonstrations, 10 test scenes of 3 toys): UniPred
    OOD 6/10, in-domain 10/10 (3/10 and 4/10 before 177e300, which keeps
    "no toy is on the table" on wiping; with 20 demonstrations the
    toy-table predicate is not invented: its classifier passes the effect
    check for none of 10 seeds). Closed loop on the simulated Franka with
    annotated videos: 9 of 10 test scenes reach the goal (test_01: no
    plan, as offline); this needed the arm not to be taken for a toy
    (1879311), a toy hidden in the gripper at the start to get its box
    (--first_boxes, 9f3bf70) and to be matched once it shows (2bc0291).
  - Cluttered Retrieval (47 demonstrations; test scenes have 3 targets,
    one behind two obstacles): perception with the new prompts (fede043)
    boxes 640 of 642 targets right. UniPred in-domain 10/10 (5/10 with
    the state thresholds, see 7012b09), OOD 0/10: the obstacle-target
    predicate agrees with Blocks on 62 percent of the training groundings
    (it is "this obstacle has not been moved", which holds for every
    target), while the target predicate is Clear on all of them. Closed
    loop on 6 unseen scenes of the training distribution (seed 1): 6/6.
- IVNTR's effect check on these image features: the non-change loss is
  JS divergence alone, diluted over every unchanged grounding, so a
  state never labelled by a change (a toy in the tray, a target in the
  basket) can be classified either way. A confidence (entropy) term and a
  heavier non-change weight were tried in the fork and dropped: Table
  Clean 0/10, and the change rows' own labels lost. The features
  separate ToyOnTable linearly (accuracy 0.997), so more demonstrations
  (50 instead of 20) were the remedy.
- Cluttered Retrieval Sim: oracle 50/50 OOD and in-domain, random options
  50/50 and 48/50 (no dead ends). UniPred (codex): 0/50 OOD, 18/50
  in-domain, the same with 500 demonstrations; Blocks fails the
  consistency check on the options without an obstacle (Pick, Place).
- Both failures (Packing's In, Cluttered Retrieval's Blocks) are binary
  predicates checked on the options that lack one of their types: the
  check draws an object of that type and requires the value not to
  change, and those groundings are trained only with the JS non-change
  loss, which leaves outputs that flip at the 0.5 boundary. The fork's
  learning functions are upstream IVNTR's (except seeded sampling), so
  this is IVNTR's behaviour on these domains, not a refactoring bug.
- Packing: random options OOD 3/50, in-domain 24/50 (the paper: 1.6% and
  12.4%; tighter train tasks give 1-2%). UniPred (codex): OOD 0/50; "item
  in region" (proposed by the LLM) fails the consistency check on Pick
  (0.28; threshold 0.05), so Free is never derived and placing has no
  region precondition. Removing the items' held flag (a suspected
  shortcut) left the same 0.28 and 0/50, and was reverted. With 500
  demonstrations In still fails the check (0.11 on Pick, 0.30 on another
  option; stopped after 5 h, log in outputs/packing/unipred_500_stopped).
- Fork README commands: oracle satellites 5/5, blocks_onclear 5/5; GNN on
  satellites with 50 demos 0/50 OOD.

Tests: root `pytest tests` (191) and fork `pytest tests` (41) pass;
earlier (141 and 40) also in a fresh clone installed by scripts/install.sh
(311 s).

## Bugs fixed

- gym Box failed to unpickle in spawned training processes (fork 8210559).
- Classifier training returned the last weights, not the best (48cd656).
- equiv_effect accepted unchanged atoms for any effect entry (48cd656).
- The training data was reshuffled with the unseeded global numpy
  generator, so the same seed learned different predicates (c5ab702).
- Spawned training processes and the main process used a torch thread per
  core: tens of times slower under load; grammar search on Packing went
  from > 1 h to 43 s (9692b27, 4a64bb3).
- Fast Downward wrote and deleted sas_plan in the working directory, so
  concurrent runs deleted each other's plans (f5afa62).
- GNN crashed on test atoms unseen in training (a810ecb).
- Packing's place sampler crashed when the item did not fit (08b4b85).
- Grammar search asserted on Packing's visited-state set (28325f2) and
  needed a belief-free operator learner (cluster_and_intersect).
- Packing test tasks timed out: closure groundings cached, lmcut with the
  optimistic relaxation, visited-state set.
- IVNTR configs capped the effect vectors kept per type (inherited from
  the UniPred configs), so IVNTR kept the first vector that passed; a
  final-code Satellites run kept a useless one and solved 0/50. IVNTR now
  keeps every passing vector, as upstream; capped types keep the best
  (fork 6dbccae, b5d863d).
- A 2 x 1 pixel sliver detected as an object gave NaN crop features and
  diverged UniPred's training on a 20-session fixture (7204a1d); that fix
  resized every crop of the fake encoder, which hid Packing's divider and
  failed test_packing (fbe69d2: only crops smaller than the grid).
- The fixture kept Table Clean Sim's object names instead of naming toys
  left to right in the first keyframe, so one toy's labels went with the
  other's crops (30375d6; a test checks held toys' boxes).
- Satellites' learned samplers saw the objects' ids, labels whose test
  values no training task has: OOD 20/50 instead of 48/50 (fork 8ab1712).
- Derived predicates that are preconditions but not consistent effects
  ("no toy is on the table") were never candidates, selection stopped
  before trying them, and the fixed demonstration order spread them over
  every later operator (fork fdad623, root 60db615).
- The planning-effort score skipped the last demonstration (fork
  6bed62c, inherited).

- Packing's Shift on regions of two boxes moved the divider of the box
  halfway between them (fa4361c).
- Derived-precondition pruning dropped a precondition whenever planning
  on the demonstrations got cheaper without it, which it always does, so
  "no toy is on the table" left wiping (177e300).
- The real domains used the simulated domains' effect-check and loss
  thresholds, which image-crop classifiers do not reach (7012b09).
- SAM3's "toy" boxed the arm holding the sponge above "robotic arm"
  (1879311); a toy hidden in the gripper at the start took the fingers'
  appearance and was not matched once placed (2bc0291).
- Identity in the simulated robot's recordings: toys swapped behind the
  arm or in the gripper (c4783c3: SAM3 video tracker); the arm's prompt
  "robot" matched toys that are robot figures (cefa196).

## Remaining

- The Table III methods on the simulated robots' sessions.
- Packing: UniPred 0/50 (the paper: 100%); the binary predicate In is not
  consistent enough to be selected (see Results). The task matches the
  paper's description; its exact distributions are not in the paper.
- Cluttered Retrieval on the simulated robot: OOD 0/10 (the
  obstacle-target predicate is not relational, see Results).
- Table Clean on the simulated robot: test_01 finds no plan.
- The real-robot fixture cannot express where the box is (the fake
  encoder sees the same crop near and far), so UniPred cannot solve its
  test scenes; it is a smoke test. The simulated robot's sessions are the
  realistic check.
- GroundingDINO tiny and SAM2.1 tiny are downloaded and tested on the
  simulator's images; DINOv3 is gated (accept its licence, `hf auth
  login`), so ResNet-18 features stand in.
- The paper tables need five seeds of every experiment (days of compute);
  the table scripts run them.
