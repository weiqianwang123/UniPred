# Adding a domain

A domain needs an environment, its ground-truth options (the skills the
robot has) and operators (for the oracle, demonstrations and evaluation),
and configs. Use Table Clean Sim (simulated) or Table Clean Real (recorded)
as the template.

## A simulated domain

1. **Environment** (`unipred/envs/<domain>.py`): a subclass of
   `predicators.envs.BaseEnv` with `get_name()`, the types, the predicates
   (with classifiers on states), the goal predicates, `simulate()`, the
   action space, `_generate_train_tasks()` and `_generate_test_tasks()`
   (the latter honouring `CFG.in_domain_test`), and `render_state_plt()`.
   Declare predicates defined from others (e.g. "no toy on the table") in
   `derived_predicates`. Import the class in `unipred/envs/__init__.py`.
   Settings of the domain go into `unipred/settings.py`.
2. **Ground-truth models** (`unipred/ground_truth_models/<domain>/`): a
   `GroundTruthOptionFactory` (the parameterized options) and a
   `GroundTruthNSRTFactory` (operators with samplers), each returning the
   env name from `get_env_names()`. If the domain has derived predicates,
   the NSRT factory's `get_atoms_closure()` returns their closure, built
   with `unipred.derived.closure.make_closure` from `DerivedRule`s. Import
   the package in `unipred/ground_truth_models/__init__.py`.
3. **Configs** (`configs/<domain>/`):
   - `base.yaml`: `env`, `excluded_predicates` (the predicates to invent),
     task counts, `timeout`, the task planner (`astar` if there are derived
     predicates);
   - `predicates.yaml`: the domain description for the LLM, the known
     predicates with their known effects (`effects: {Option: add}`) and
     preconditions, the static predicates, and the neural predicate types
     to invent (`predicate_types`, with their argument types and training
     settings; see `configs/satellites/predicates.yaml`);
   - `ivntr_predicates.yaml`: includes `predicates.yaml` and sets IVNTR's
     search budget;
   - one config per approach, each including `base.yaml`: `unipred.yaml`
     (with `unipred_derived_selection: true` if there are derived
     predicates), `ivntr.yaml`, `oracle.yaml` (no excluded predicates),
     `gnn.yaml`, `transformer.yaml`, `grammar.yaml`, `random.yaml`,
     `code_as_policy.yaml`.
4. **Check**: the oracle solves train and test tasks
   (`LLM=mock scripts/run.sh configs/<domain>/oracle.yaml 0 --num_train_tasks 5 --num_test_tasks 10`),
   then `LLM=mock scripts/run.sh configs/<domain>/unipred.yaml 0` with few
   tasks. Add the domain to `tests/test_envs.py` and the configs are
   checked by `tests/test_configs.py`.

## A real domain

Everything specific to the domain is in its config directory and its
ground-truth package; the pipeline (perception, processing, the offline
environment, evaluation and deployment) stays unchanged.

1. **Ground-truth package** (`unipred/ground_truth_models/<domain>/`): the
   option and NSRT factories as above, with the real env's name in
   `get_env_names()`. Options are never executed by the pipeline (the robot
   executes them through `RobotInterface`), so their policies can be
   placeholders; the operators define the offline evaluation. If a
   simulated version exists, share its package (as Table Clean does).
2. **Environment declaration**: a subclass of
   `unipred.envs.offline_real.OfflineRealEnv` that names the env and the
   setting of its data directory, e.g.

   ```python
   class ClutterRealEnv(OfflineRealEnv):
       """Pick a target in clutter, from --clutter_real_data_dir."""
       data_dir_setting = "clutter_real_data_dir"

       @classmethod
       def get_name(cls) -> str:
           return "clutter_real"
   ```

   with `clutter_real_data_dir` registered as a setting
   (`predicators.settings.register_settings`), imported where the package
   is imported.
3. **Configs** (`configs/<domain>/`): `domain.yaml` (the object types, the
   ground-truth predicates with their argument types, the derived and goal
   predicates), `predicates.yaml`, and the approach configs, as in
   `configs/table_clean_real/`; detector prompts for the new object types
   in `configs/real_robot.yaml`.
4. **Data**: record sessions in the raw format of
   [examples/real_robot/README.md](../examples/real_robot/README.md), then
   process, train and deploy as described there.
