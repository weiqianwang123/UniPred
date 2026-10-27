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

Fork (third_party/IVNTR, unipred), after a406fa2:

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
- Simulated robot (unipred/robot_sim): 24/24 full demonstrations with
  GraspGenX grasps; 60 sessions collected (50 demonstrations, 10 test
  scenes), processed with SAM3 and ResNet-18; UniPred on them pending.
- Fork README commands: oracle satellites 5/5, blocks_onclear 5/5; GNN on
  satellites with 50 demos 0/50 OOD.

Tests: root `pytest tests` (151) and fork `pytest tests` (41) pass;
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

## Remaining

- IVNTR on Satellites retrained with the sampler fix (running), and the
  Table III methods on the simulated robot's sessions.
- Packing: review of the environment's tasks (the owner suspects problems
  in how they are written).
- The real-robot fixture cannot express where the box is (the fake
  encoder sees the same crop near and far), so UniPred cannot solve its
  test scenes; it is a smoke test. The simulated robot's sessions are the
  realistic check.
- GroundingDINO tiny and SAM2.1 tiny are downloaded and tested on the
  simulator's images; DINOv3 is gated (accept its licence, `hf auth
  login`), so ResNet-18 features stand in.
- The paper tables need five seeds of every experiment (days of compute);
  the table scripts run them.
