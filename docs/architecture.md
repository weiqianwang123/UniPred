# Architecture

UniPred is a package (`unipred/`) on top of a fork of IVNTR
(`third_party/IVNTR`, Python package `predicators`). The fork provides
bilevel planning, IVNTR's neural predicate learning, the baselines and the
Satellites and Blocks domains; UniPred adds its approach, the other domains,
the LLM client, perception, evaluation and the real-robot template. UniPred
changes the fork's behaviour only through the fork's extension hooks.

## Running an experiment

`python -m unipred.main --config FILE ... [flags]` expands the YAML configs
into command line flags (`unipred/config.py`; later configs and explicit
flags win), registers UniPred's settings, approaches, environments and
ground-truth models (importing `unipred`), and calls `predicators.main`:
create the environment and its train tasks, generate the demonstrations
with the oracle (or load recorded ones), learn, and test. `scripts/run.sh`
runs one experiment: train and test on the out-of-distribution test tasks,
then test the saved model on tasks from the training distribution.

## UniPred (Section IV of the paper)

`unipred/approaches/unipred.py` is IVNTR's `BilevelLearningApproach` with
the effect vectors proposed by an LLM instead of a tree search.

1. **Data.** Demonstrations (state, option, next state). In image domains,
   an object's state is the feature vector of its image crop
   (`unipred/perception`).
2. **PDDL completion** (`unipred/llm/pddl.py`). The partial PDDL domain
   (types, options as operators, the known predicates and their known
   effects and preconditions) and a few demonstrations go to the LLM, which
   completes the domain. Each new predicate becomes a seed effect vector
   over the options (add, delete or no effect) for the predicate type with
   its argument types.
3. **LLM-in-the-loop learning** (`unipred/llm/effect_proposer.py`). For each
   predicate type, an `EffectProposer` offers the seeds and then asks the
   LLM for new effect vectors, showing every earlier proposal with a score
   in [0, 100] derived from its classifier's validation loss. IVNTR trains a
   classifier per proposal with the effect loss and keeps those with a
   validation loss below `loss_thresh`. The proposer implements the
   interface of IVNTR's tree searcher (`propose()`, `update_value()`).
4. **Derived-aware selection** (`unipred/derived/`, Algorithm 1).
   Hill-climbing selects the predicates that best explain the
   demonstrations' skeletons. Negated and quantified forms of a basic
   predicate enter the candidate pool only once the basic predicate is
   selected; they are computed by a closure from the basic atoms at every
   search state and are never operator effects.
5. **Operators and samplers.** Operators by intersection of the selected
   predicates' effects; samplers learned from the demonstrations.
6. **Planning and execution.** Task planning with the learned operators
   (A* with the closure, or Fast Downward without derived predicates),
   refinement with the samplers, closed-loop execution that replans when
   the expected atoms do not hold (`--execution_monitor expected_atoms`).

Every stage is a setting (`unipred/settings.py`); the ablations of the
paper turn one off (`configs/ablations/`). Every LLM request and reply is
appended to `<approach save path>.llm_log.jsonl`.

## LLM client

`unipred/llm/client.py` is the one client for all providers: `openai`,
`qwen` (DashScope's OpenAI-compatible API), `gemini`, `codex` (the local
Codex CLI) and `mock` (rule-based answers for tests, `unipred/llm/mock.py`).
Keys come from the environment or `.env.local`. Replies are cached on disk,
keyed by the request, so reruns are free and reproducible. Requests can
attach images (for the VLM baselines). Prompts are templates in
`unipred/llm/prompts/`.

## Fork hooks used by UniPred

- `settings.register_settings`: UniPred's settings become flags.
- `hooks.AtomsClosure`, `set_atoms_closure`, `temporary_atoms_closure`:
  derived predicates in task planning (the delete relaxation treats derived
  atoms as reachable while a closure is installed).
- `BaseEnv.derived_predicates`, and `get_atoms_closure()` of a ground-truth
  NSRT factory: an environment's derived predicates and their closure.
- Overridable methods of `BilevelLearningApproach`: `gen_ae_vectors4pred`,
  `gen_sat_vec` (the proposer), `_expand_candidate_pool`,
  `_operator_predicates` and `_atoms_closure` (derived-aware selection).
- Optional arguments of `_OperatorBeliefScoreFunction.evaluate`
  (`operator_predicates`, `atoms_closure`).
- `BaseEnv.get_recorded_dataset()` and `BaseEnv.evaluate_offline()`:
  environments built from recordings (real robots).

## Other approaches

- `ivntr_derived`: IVNTR with derived-aware selection.
- `code_as_policy`: the LLM writes a Python policy over low-level states.
- `vila`, `pix2pred`: VLM baselines on real images, scored by offline
  planning.
- From the fork: `ivntr`, `oracle`, `gnn_nsrt_policy`,
  `transformer_nsrt_policy`, `grammar_search_invention`, `random_options`.

## Evaluation

- Simulated domains: `predicators.main` executes the policy of each test
  task in the simulator; success means the goal holds.
- Real domains: `unipred/eval/offline_planning.py` checks the skeleton
  planned for each test scene against the ground-truth operators from the
  scene's annotated atoms (the paper's Table III).
- `unipred/eval/summarize.py` turns the results of `scripts/run.sh` into
  a table.

## Layout

```
configs/            experiment configs per domain, ablation overlays, LLMs
scripts/            run.sh, install.sh, sam3_server.py
unipred/
  approaches/       unipred, ivntr_derived, code_as_policy, vila, pix2pred
  derived/          derived predicate rules, closure, derived-aware selection
  envs/             packing, table_clean_sim, offline_real, table_clean_real
  ground_truth_models/  options and operators of packing and table_clean
  llm/              client, PDDL completion, effect proposer, prompts
  perception/       detector, crops, crop encoders, scenes, weights
  eval/             offline planning, result tables
  deploy.py         plan from images with a learned model
examples/real_robot/  processing recordings, planning from images, closed loop
third_party/IVNTR/  the predicators fork (with Fast Downward in ext/)
```
