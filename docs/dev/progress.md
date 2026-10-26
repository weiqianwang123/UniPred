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
  Reruns from the final code (snapshot run3) are running one after the
  other; their numbers replace these.
- Fork README commands: oracle satellites 5/5, blocks_onclear 5/5; GNN on
  satellites with 50 demos 0/50 OOD.

Tests: root `pytest tests` (141) and fork `pytest tests` (40) pass,
also in a fresh clone installed by scripts/install.sh (311 s).

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

## Remaining

- Numbers of the final runs (above) and of the README commands that train
  (quickstart, the ablation example, the real-robot template with full
  budgets).
- Satellites OOD success is far below the paper's (UniPred 13/50, IVNTR
  20/50 vs ~95%) while in-domain success is 48-49/50. Diagnosis on the
  saved UniPred model: the learned operators match the ground truth; the
  plans fail at MoveTo, where the learned sampler (a Gaussian fitted to
  the demonstrations' ring of positions around the object) puts the
  satellite where even the ground-truth Sees does not hold (0/30 samples
  at the failing steps of 6 tasks), so refinement backtracks until the
  timeout; a 20 s budget gives the same 5/20 as 5 s. The environment, the
  ground-truth sampler, sampler learning and low-level search are the
  same as upstream IVNTR's. Open: why upstream's samplers generalise to
  three satellites and ours do not.
- Perception with real weights is untested: GroundingDINO tiny, SAM2.1
  tiny and the gated DINOv3 ViT-S/16 need a download (scripts/install.sh
  --weights).
- The paper tables need five seeds of every experiment (days of compute);
  the table scripts run them.
