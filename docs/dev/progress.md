# Progress

Kept short and factual: what is done, the commands and their numbers, the
bugs fixed, and what remains. Design choices are in decisions.md.

## Commits

Root (refactor/paper-release):

- 56c2cd1 unipred package on top of the IVNTR fork.
- b616b04 gymnasium spaces.
- 78604b6 Table Clean Sim with the paper's seven options.
- 2eac151 Prompts in the paper's four parts; every LLM request logged.
- d77d888, 8e3d8ee, 349bd98 fork pointer updates (below).
- 8d1afb9 Baseline configs per domain.
- 4e7d858 Packing domain and crop encoders.

Fork (third_party/IVNTR, unipred):

- 8210559 gymnasium Box (gym's Box did not unpickle).
- 48cd656 Best classifier weights kept; every classifier seeded; wandb
  removed.
- ec24654 Offline-only pipeline (online learning, MDP data, task JSON and
  caches removed).
- d19c89f Baselines use the GT NSRTs; one save/load path for IVNTR.
- a406fa2 Derived atoms in the delete relaxation (lmcut/hff with a
  closure).

## Results so far

Oracle, `--num_train_tasks 5 --num_test_tasks 10`, OOD and
`--in_domain_test True`: 10/10 on satellites, blocks_onclear,
table_clean_sim and packing (fake and ResNet-18 encoders).

Baseline smoke (10 train / 3 test tasks, minutes each): random, gnn and
transformer run end to end on satellites, blocks and table_clean_sim.

## Bugs fixed

- gym Box failed to unpickle in spawned training processes (fork 8210559).
- Classifier training returned the last weights, not the best (48cd656).
- equiv_effect accepted unchanged atoms for any effect entry (48cd656).
- A stray `@dataclass` decorated `sesame_plan` after a deletion (ec24654).
- Grammar search crashed without a belief learner: it now uses
  cluster_and_intersect.
- Packing test tasks timed out: closure groundings cached, lmcut with the
  optimistic relaxation, visited-state set.

## Remaining

See the plan in the task description: determinism test; real-LLM runs on
satellites and table_clean_sim; code_as_policy; LLM ablation configs;
summarize.py and scripts/paper; the real-robot template (perception,
table_clean_real, offline planning, vila, pix2pred); dead-code pass, lint,
install.sh, README and docs.
