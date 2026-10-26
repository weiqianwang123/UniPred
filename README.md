# IVNTR for UniPred

This repository is the bilevel planning and predicate learning library that
[UniPred](https://github.com/weiqianwang123/UniPred) builds on. It is a fork of
[IVNTR](https://github.com/centaur-ai/IVNTR) ("Bilevel Learning for Bilevel
Planning", Li, Silver, Scherer and Gray, RSS 2025), which itself is based on
[predicators](https://github.com/Learning-and-Intelligent-Systems/predicators).
The Python package is still called `predicators`.

It contains:

- SeSamE bilevel planning (`predicators/planning.py`) with A* or Fast Downward
  as the task planner;
- the IVNTR approach, which learns neural predicates and operators from
  demonstrations (`--approach ivntr`);
- the simulated baselines used in the UniPred paper;
- the two IVNTR domains that UniPred evaluates on, Satellites and Blocks;
- a small set of extension hooks (see [Extension hooks](#extension-hooks)),
  which UniPred uses to add its approaches, domains, settings and derived
  predicates without editing this package.

| Approach | `--approach` |
|---|---|
| IVNTR | `ivntr` |
| Oracle (ground-truth NSRTs) | `oracle` |
| GNN behavior cloning | `gnn_nsrt_policy` |
| Transformer behavior cloning | `transformer_nsrt_policy` |
| Grammar-based predicate invention | `grammar_search_invention` |
| Random options / random NSRTs | `random_options` / `random_nsrt` |

| Domain | `--env` | Train / test tasks |
|---|---|---|
| Satellites | `satellites` | 2 satellites, 2 objects / 3 satellites, 3 objects |
| Blocks (the "Blocks" domain of the IVNTR and UniPred papers) | `blocks_onclear` | 4-5 blocks / 6-7 blocks |

`blocks` is the original predicators Blocks domain, which `blocks_onclear`
extends.

## Changes from upstream IVNTR

Upstream base: centaur-ai/IVNTR at commit f12b1ae.

- **Pruned.** Everything that UniPred does not use is removed: Spot and PyBullet
  code, ikfast, the other IVNTR domains (Measure-Mul, Climb-Measure,
  Climb-Transport, Engrave), unrelated predicators environments and approaches,
  explorers, LLM/VLM interfaces, point-cloud models and the corresponding
  settings. The package no longer depends on bosdyn, pybullet, open3d or
  pytorch3d (`quaternion_to_matrix` is now plain torch).
- **Strict registries.** A missing dependency in a submodule raises instead of
  silently removing an env or approach, and ground-truth options and NSRTs are
  looked up through each factory's `get_env_names()`.
- **No wandb.** The Weights & Biases logging of predicate training and its
  hard-coded API keys are removed.
- **Fixes in the IVNTR approach.** Saved model artifacts (guidance and effect
  vectors) are found by file name only, so directory names that contain
  "model" work; predicate selection no longer loops forever or crashes when no
  candidate predicate is left; models trained on a GPU load on a CPU-only
  machine; neural predicate learning is seeded; the debug flag `--neupi_bug`,
  which filtered candidates with ground-truth indices, is removed; an optional
  `max_num_vectors` key in a predicate config caps the number of effect vectors
  kept per predicate type; selection logs the candidate counts.
- **Other fixes.** The operator scoring in predicate selection skips a
  demonstration (counting it as unsolved) when no heuristic can be built for
  it; learned samplers load with torch >= 2.6; `main.py` runs outside a git
  checkout. Parallel classifier training (`--neupi_parallel_invention`)
  starts its processes with the "spawn" method, because forked processes
  cannot use CUDA once the parent has, and a failed training process raises
  an error instead of leaving the parent waiting forever.
- **Execution monitoring.** `--execution_monitor expected_atoms` now also works
  for plans found with simulation: the atoms expected after each option are
  checked when the option terminates, and the agent replans if they do not
  hold.
- **Clearer configuration.** The predicate config names options and
  predicates instead of using 0/1 lists in sorted-option order, and shares
  training settings through a `defaults` block. The selection settings named
  after a venue are renamed (`domain_aaai_thresh` -> `neupi_selection_thresh`,
  `neupi_aaai_metric` -> `neupi_selection_metric`,
  `neupi_aaai_expected_nodes_upper_bound` ->
  `neupi_selection_nodes_upper_bound`), and scripts no longer pass flags
  that equal their defaults.
- **Packaging.** `pyproject.toml` with pinned core dependencies (Python 3.10,
  numpy < 2, gym 0.23.1, torch >= 2.1) and Fast Downward as a git submodule.

## Installation

```bash
git clone --recurse-submodules <this repository> IVNTR
cd IVNTR
conda create -n ivntr python=3.10
conda activate ivntr
# Install torch for your platform first (https://pytorch.org), then:
pip install -e .
# Build Fast Downward (cmake and a C++ compiler are needed).
scripts/build_downward.sh
export FD_EXEC_PATH=$PWD/ext/downward
```

If the repository was cloned without `--recurse-submodules`,
`scripts/build_downward.sh` fetches the submodule (`ext/downward`, pinned to
aibasel/downward commit a6b98adb). `FD_EXEC_PATH` must point at the
`ext/downward` directory whenever `--sesame_task_planner` is `fdopt` (the
default) or `fdsat`; A* (`astar`) does not need it. `PYTHONHASHSEED=0` must be
set for `predicators/main.py`. On macOS, install `coreutils` for `gtimeout`.

Check the installation with the oracle planner:

```bash
export PYTHONHASHSEED=0
python predicators/main.py --env satellites --approach oracle --seed 0 \
    --num_train_tasks 2 --num_test_tasks 5
python predicators/main.py --env blocks_onclear --approach oracle --seed 0 \
    --num_train_tasks 2 --num_test_tasks 5 --sesame_task_planner astar
```

Run the unit tests with `pip install -e ".[develop]"` and `pytest tests`.

## Running IVNTR and the baselines

All scripts run from the repository root, write logs to `logs/` and models to
`saved_approaches/`, and loop over seeds 0-4.

| Script | What it runs |
|---|---|
| `scripts/train/satellites/satellites_biplan.sh` | trains IVNTR on Satellites (500 demonstrations) and tests it out of distribution |
| `scripts/train/blocks_pos/blocks_pos_biplan.sh` | the same for Blocks (`blocks_onclear`) |
| `scripts/test/<domain>/<domain>_ivntr.sh` | tests saved IVNTR models (out of distribution and in domain) |
| `scripts/test/<domain>/<domain>_gnn_policy.sh` | trains and tests the GNN baseline |
| `scripts/test/<domain>/<domain>_tf_policy.sh` | trains and tests the Transformer baseline |
| `scripts/test/<domain>/<domain>_no_learning.sh` | tests the random options and random NSRT baselines |

The GNN, Transformer and random NSRT baselines plan with the samplers learned
by IVNTR (`--ivntr_nsrt_path`), so train IVNTR first. The IVNTR test scripts
load JSON models; convert trained models with
`python scripts/biplan_pkl2json.py saved_approaches/final/<domain>/ivntr_<seed>`.
Pre-trained IVNTR models and logs are linked from the
[upstream README](https://github.com/centaur-ai/IVNTR).

The grammar baseline has no script; for example:

```bash
python predicators/main.py --env satellites --approach grammar_search_invention \
    --seed 0 --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
    --num_train_tasks 50 --sesame_task_planner astar --timeout 5
```

The neural predicate types IVNTR searches over, their network and training
hyperparameters and the search thresholds are set in
`predicators/config/<domain>/pred.yaml` (`--neupi_pred_config`):

```yaml
final_options: [UseCamera, UseGeiger, UseInfraRed]  # options that only achieve goals
static_predicates: [HasCamera, ...]  # given predicates that never change
defaults: {batch_size: 512, loss_thresh: 0.005, ...}  # merged into every type
predicate_types:
  - name: neural_b_p3
    types: [satellite, object]
    ent_idx: [0, 0]  # which option argument of each type
    epochs: 200
    num_iter: 70  # search iterations, batch_vect_num vectors each
    max_num_vectors: 1  # optional: stop once this many vectors are kept
```

`load_predicate_config()` in `predicators/approaches/bilevel_learning_approach.py`
documents every key. For a quick run, copy the file and reduce `epochs`,
`num_iter` and `batch_vect_num` in `defaults`.

## Extension hooks

Packages built on this library (such as UniPred) import their own approaches,
environments and ground-truth models, which register themselves as subclasses,
and then call `predicators.main.main()`. They can use the following hooks.

**Settings** (`predicators/settings.py`)

- `register_settings(settings_cls)` adds the public attributes of
  `settings_cls` to `GlobalSettings` and `CFG`. They are then accepted as
  command line flags by `utils.parse_args()` and reset by
  `utils.reset_config()`. A name that is already a setting or command line
  argument raises a `ValueError`.
- `register_arg_specific_settings(fn)` merges `fn(args)` into the
  arg-specific settings, for example defaults that depend on `--env`.

**Atoms closure for derived predicates** (`predicators/hooks.py`)

- `set_atoms_closure(fn)` installs `fn(atoms, objects) -> atoms`, which
  recomputes derived atoms (atoms of predicates that are defined from other
  predicates and that no operator changes directly); `set_atoms_closure(None)`
  removes it. `atoms_closure(atoms, objects)` applies it (identity when none is
  installed), `closure_active()` tells whether one is installed, and
  `temporary_atoms_closure(fn)` installs one for a `with` block.
- A* task planning applies the closure to the initial atoms and to every
  successor, including the successors proposed by an abstract policy, with all
  task objects. While a closure is installed, the delete-relaxation
  reachability filter and goal check are skipped, because derived atoms are
  invisible to them, and Fast Downward raises an error, because it cannot apply
  the closure. Use `--sesame_task_planner astar`. Delete-relaxation heuristics
  (such as `lmcut` and `hadd`) also ignore derived atoms that no operator adds,
  so `goal_count` can be the better heuristic with derived preconditions.
- `BaseEnv.derived_predicates` (empty by default) declares the derived
  predicates of an environment, and
  `utils.create_pddl_domain(..., derived_predicates=...)` leaves them out of
  operator effects.

**Predicate selection** (`predicators/approaches/bilevel_learning_approach.py`)

- `_OperatorBeliefScoreFunction.evaluate(..., operator_predicates=None,
  atoms_closure=None)`: with `operator_predicates`, only these effect
  predicates become operator effects (the others can still be preconditions);
  with `atoms_closure`, that closure is used while planning with the learned
  operators.
- `BilevelLearningApproach` methods that subclasses can override:
  `gen_ae_vectors4pred` and `gen_sat_vec` (proposal and training of effect
  vectors; the `symbolic_model` passed to `gen_sat_vec` can be any object with
  `propose()` and `update_value(vector, guidance)`),
  `_expand_candidate_pool(selected, pool)` (called after each predicate that
  hill-climbing adds; it can append `PredicateCandidate`s),
  `_operator_predicates(predicates)` (the effect predicates allowed in operator
  effects, while scoring and when learning the final NSRTs; all by default),
  `_atoms_closure(predicates)` (the closure installed while scoring a
  predicate set and while planning with the learned predicates; none by
  default) and `_model_artifact_path(model_path, artifact)` (paths of the
  files saved next to a predicate model).
- `learn_nsrts_from_data(..., derived_predicates=...)` keeps the given
  predicates out of operator effects (they can still be preconditions).
- A predicate config entry can set `generate_derived: False` to skip the
  negated and quantified variants of that predicate type, and
  `max_num_vectors` to cap the number of effect vectors kept.

**Execution monitoring.** `BilevelPlanningApproach.get_execution_monitoring_info()`
returns the expected atoms of the current plan, for plans found with or without
simulation, so `--execution_monitor expected_atoms` replans in both cases.

## Tutorials

The notebooks in `docs/` (from upstream IVNTR) walk through the domains, the
planner, demonstration data and bilevel learning.

## License and citation

MIT (see `LICENSE.md`; predicators is (c) 2022 Rohan Chitnis and Tom Silver).
Fast Downward (`ext/downward`) and the vendored translator in
`predicators/third_party/fast_downward_translator` are GPL-3.0.

If you use IVNTR, please cite:

```
@INPROCEEDINGS{Li2025IVNTR,
  title={{Bilevel Learning for Bilevel Planning}},
  author={Li, Bowen and Silver, Tom and Scherer, Sebastian and Gray, Alexander},
  booktitle={Proceedings of the Robotics: Science and Systems (RSS)},
  year={2025}
}
```
