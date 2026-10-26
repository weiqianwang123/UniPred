"""Contains global, immutable settings.

Anything that varies between runs should be a command-line arg
(args.py).
"""

from collections import defaultdict
from types import FunctionType, SimpleNamespace
from typing import Any, Callable, Dict, List, Set

from predicators.args import create_arg_parser

# Functions registered by extensions that return extra arg-specific settings.
_ARG_SPECIFIC_SETTINGS_FNS: List[Callable[[Dict[str, Any]], Dict[str,
                                                                   Any]]] = []
# Settings classes registered by extensions.
_REGISTERED_SETTINGS_CLASSES: List[type] = []


class GlobalSettings:
    """Unchanging settings."""
    # global parameters
    device = "cuda:0"
    debug = False
    num_train_tasks = 50
    num_test_tasks = 50
    # Perform online learning for this many cycles or until this many
    # transitions have been collected, whichever happens first.
    num_online_learning_cycles = 10
    # Optional {type name: [feature names]} of object features hidden from
    # the neural predicate and GNN inputs. None keeps all features.
    exclude_domain_feat = None
    # Optional {type name: [feature indices]} of object features used as
    # sampler inputs. None keeps all features.
    domain_sampler_data_filter = None
    online_learning_max_transitions = float("inf")
    # Maximum number of training tasks to give a demonstration for, if the
    # offline_data_method is demo-based.
    max_initial_demos = float("inf")
    # If this is False, then environment interactions can only take place
    # on tasks that have no demonstrations.
    allow_interaction_in_demo_tasks = True
    # Maximum number of steps to run an InteractionRequest policy.
    max_num_steps_interaction_request = 100
    # Whether to pretty print predicates and NSRTs when NSRTs are loaded.
    pretty_print_when_loading = False
    # Used for random seeding in test environment.
    test_env_seed_offset = 10000
    # The method to use for segmentation. By default, segment using options.
    # If you are learning options, you should change this via the command line.
    segmenter = "option_changes"
    # The method to use for generating demonstrations: "oracle" or "human".
    demonstrator = "oracle"
    force_create_dataset = True
    in_domain_test = False
    verbose_pddl_matrix = True # print the PDDL matrix using oracle approach
    # DPI for rendering the state. Increase this if video quality is poor.
    # Note that for unit testing, we use a much smaller value by default,
    # which is set in utils.reset_config(). If you want higher-quality videos
    # in unit tests, make sure to pass in a value for `render_state_dpi` into
    # your call to utils.reset_config().
    render_state_dpi = 150
    approach_wrapper = None
    # Normally, excluding goal predicates does not make sense, because then
    # there is no goal for the agent to plan towards. This is intended to be
    # used by VLM predicate invention, where we want to invent goal predicates
    # and different task goals are provided to the agent and the demonstrator.
    allow_exclude_goal_predicates = False
    # Normally, State.allclose() raises an error if the simulator state of
    # either of its arguments is not None.
    allow_state_allclose_comparison_despite_simulator_state = False

    # mdp offline dataset settings
    all_skeleton_found = 1
    max_num_pos_mdp_traj = 1
    max_num_neg_mdp_traj = 1
    mdp_hl_search_timeout = 30 # this is big as we want to get more data for one task
    mdp_ll_search_timeout = 5
    mdp_ll_step_fail_rw = -0.0
    mdp_bad_hl_step = -0.0
    mdp_fail_last_step = -0.0
    mdp_good_hl_step_fail_rw = -0.0
    mdp_bad_hl_step_fail_rw = -0.0

    # blocks env parameters
    blocks_num_blocks_train = [4, 5]
    blocks_num_blocks_test = [6, 7]
    blocks_holding_goals = False
    blocks_block_size = 0.045  # use 0.0505 for real with panda

    # satellites env parameters
    satellites_num_sat_train = [2, 2]
    satellites_num_obj_train = [2, 2]
    # default
    # satellites_num_sat_train = [2, 3]
    # satellites_num_obj_train = [3, 4]
    satellites_num_sat_test = [3, 3]
    satellites_num_obj_test = [3, 3]

    # parameters for random options approach
    random_options_max_tries = 100

    # option model parameters
    option_model_terminate_on_repeat = True
    option_model_use_gui = False

    # parameters for abstract GNN approach
    # keep fair with neupi, nn size, bs, epochs
    ivntr_nsrt_path = ''
    gnn_num_message_passing = 3
    gnn_num_layers = 2 # after the first linear, how many gnn_layer_size layers
    gnn_layer_size = 512
    gnn_learning_rate = 1e-3
    gnn_num_epochs = 800
    gnn_batch_size = 512
    gnn_do_normalization = False  # performs worse in Cover when True
    gnn_use_validation_set = True

    # parameters for GNN option policy approach
    gnn_option_policy_solve_with_shooting = True
    random_policy_solve_with_shooting = True
    random_policy_shooting_max_samples = 100
    gnn_option_policy_shooting_max_samples = 100

    # SeSamE parameters
    sesame_task_planner = "fdopt"  # "astar" or "fdopt" or "fdsat"
    sesame_task_planning_heuristic = "lmcut"
    sesame_allow_noops = True  # recommended to keep this False if using replays
    sesame_check_expected_atoms = True
    sesame_use_necessary_atoms = False # Note, all high-level atoms must be satisfied at every step
    sesame_use_visited_state_set = False
    # The algorithm used for grounding the planning problem. Choices are
    # "naive" or "fd_translator". The former does a type-aware cross product
    # of operators and objects to obtain ground operators, while the latter
    # calls Fast Downward's translator to produce an SAS task, then extracts
    # the ground operators from that. The latter is preferable when grounding
    # is a bottleneck in your environment, but will not work when operators
    # with no effects need to be part of the ground planning problem, like the
    # OpenLid() operator in painting. So, we'll keep the former as the
    # default.
    sesame_grounder = "naive"
    sesame_check_static_object_changes = False
    sesame_static_object_change_tol = 1e-3
    # If True, then bilevel planning approaches will run task planning only,
    # and then greedily sample and execute in the environment. This avoids the
    # need for a simulator. In the future, we could check to see if the
    # observed states match (at the abstract level) the expected states, and
    # replan if not. But for now, we just execute each step without checking.
    bilevel_plan_without_sim = False

    # evaluation parameters
    results_dir = "results"
    eval_trajectories_dir = "eval_trajectories"
    approach_dir = "saved_approaches"
    data_dir = "saved_datasets"
    video_dir = "videos"
    image_dir = "images"
    video_fps = 2
    failure_video_mode = "longest_only"

    # dataset parameters
    # For learning-based approaches, the data collection timeout for planning.
    # If -1, defaults to CFG.timeout.
    offline_data_planning_timeout = -1
    # If "default", defaults to CFG.task_planning_heuristic.
    offline_data_task_planning_heuristic = "default"
    # If -1, defaults to CFG.sesame_max_skeletons_optimized.
    offline_data_max_skeletons_optimized = -1
    # Default to bilevel_plan_without_sim.
    offline_data_bilevel_plan_without_sim = None

    # NSRT learning parameters
    min_data_for_nsrt = 0
    min_perc_data_for_nsrt = 0
    data_orderings_to_search = 1  # NSRT learning data ordering parameters
    # STRIPS learning algorithm. See get_name() functions in the directory
    # nsrt_learning/strips_learning/ for valid settings.
    strips_learner = "belief_and_intersect"
    strips_operator_eq_option = False
    disable_harmlessness_check = True  # some methods may want this to be True
    enable_harmless_op_pruning = False  # some methods may want this to be True
    backchaining_check_intermediate_harmlessness = False
    pnad_search_without_del = False
    pnad_search_timeout = 10.0
    compute_sidelining_objective_value = False
    clustering_learner_true_pos_weight = 10
    clustering_learner_false_pos_weight = 1
    cluster_and_intersect_prederror_max_groundings = 10
    cluster_and_search_inner_search_max_expansions = 2500
    cluster_and_search_inner_search_timeout = 30
    cluster_and_search_score_func_max_groundings = 10000
    cluster_and_search_var_count_weight = 0.1
    cluster_and_search_precon_size_weight = 0.01
    cluster_and_intersect_prune_low_data_pnads = False
    # If cluster_and_intersect_prune_low_data_pnads is set to True, PNADs must
    # have at least this fraction of the segments produced by the option that is
    # associated with their PNAD in order to not be pruned during operator
    # learning.
    cluster_and_intersect_min_datastore_fraction = 0.0

    # torch GPU usage setting
    use_torch_gpu = False

    # torch model parameters
    learning_rate = 1e-3
    weight_decay = 0
    mlp_regressor_max_itr = 10000
    mlp_regressor_hid_sizes = [32, 32]
    mlp_regressor_clip_gradients = False
    mlp_regressor_gradient_clip_value = 5
    mlp_classifier_hid_sizes = [128, 128]
    mlp_classifier_balance_data = True
    neural_gaus_regressor_hid_sizes = [128, 128]
    neural_gaus_regressor_max_itr = 1000
    mlp_classifier_n_iter_no_change = 5000
    implicit_mlp_regressor_max_itr = 10000
    implicit_mlp_regressor_num_negative_data_per_input = 5
    implicit_mlp_regressor_num_samples_per_inference = 100
    implicit_mlp_regressor_temperature = 1.0
    implicit_mlp_regressor_inference_method = "derivative_free"
    implicit_mlp_regressor_derivative_free_num_iters = 3
    implicit_mlp_regressor_derivative_free_sigma_init = 0.33
    implicit_mlp_regressor_derivative_free_shrink_scale = 0.5
    implicit_mlp_regressor_grid_num_ticks_per_dim = 100

    # ml training parameters
    pytorch_train_print_every = 1000

    # sampler learning parameters
    sampler_learner = "neural"  # "neural" or "random" or "oracle"
    max_rejection_sampling_tries = 100
    sampler_mlp_classifier_max_itr = 10000
    sampler_mlp_classifier_n_reinitialize_tries = 1
    sampler_learning_use_goals = False
    sampler_disable_classifier = False
    sampler_learning_regressor_model = "neural_gaussian"
    sampler_learning_max_negative_data = 100000

    # option learning parameters
    option_learning_action_converter = "identity"

    # refinement data collection parameters
    refinement_data_include_execution_cost = True
    refinement_data_low_level_execution_cost = 0.05  # per action cost to add

    # grammar search invention parameters
    grammar_search_grammar_includes_givens = True
    grammar_search_grammar_includes_foralls = True
    grammar_search_grammar_use_diff_features = False
    grammar_search_grammar_use_euclidean_dist = False
    grammar_search_use_handcoded_debug_grammar = False
    grammar_search_pred_selection_approach = "score_optimization"
    grammar_search_pred_clusterer = "oracle"
    grammar_search_true_pos_weight = 10
    grammar_search_false_pos_weight = 1
    grammar_search_bf_weight = 1
    grammar_search_operator_complexity_weight = 0.0
    grammar_search_pred_complexity_weight = 1e-4
    grammar_search_max_predicates = 200
    grammar_search_predicate_cost_upper_bound = 6
    grammar_search_prune_redundant_preds = True
    grammar_search_score_function = "expected_nodes_created"
    grammar_search_heuristic_based_weight = 10.
    grammar_search_max_demos = float("inf")
    grammar_search_max_nondemos = 50
    grammar_search_energy_based_temperature = 10.
    grammar_search_task_planning_timeout = 1.0
    grammar_search_search_algorithm = "hill_climbing"  # hill_climbing or gbfs
    grammar_search_hill_climbing_depth = 0
    grammar_search_parallelize_hill_climbing = False
    grammar_search_gbfs_num_evals = 1000
    grammar_search_off_demo_count_penalty = 1.0
    grammar_search_on_demo_count_penalty = 10.0
    grammar_search_suspicious_state_penalty = 10.0
    grammar_search_expected_nodes_upper_bound = 1e5
    grammar_search_expected_nodes_optimal_demo_prob = 1 - 1e-5
    grammar_search_expected_nodes_backtracking_cost = 1e3
    grammar_search_expected_nodes_allow_noops = True
    grammar_search_classifier_pretty_str_names = ["?x", "?y", "?z"]

    # grammar search clustering algorithm parameters
    grammar_search_clustering_gmm_num_components = 10

    # neural predicate classifier parameters
    # assume we know the number and types of predicates
    neupi_load_pretrained = ''
    load_neupi_from_json = False
    neupi_pred_config = 'predicators/config/satellites/pred.yaml'
    neupi_ae_matrix_channel = 2 # 2 is full [non, add, del], 00: non, 01: add, 10: del (no 11). # 1 is add/non
    neupi_super_label = {
        'ignore': -1, # wrong type match for the predicate
        'change_neg': 0,
        'change_pos': 1,
        'non_change_1': 2, # the predicate is not in the effect, and the objects (pairs) are not operated
        'non_change_2': 3, # the predicate is in the effect, but the objects are not operated
        'non_change_3': 4, # the predicate is not in the effect, but the objects (pairs) are operated
    }
    neupi_save_path = "saved_approaches/debug"
    neupi_do_normalization = True  # the satellites scripts turn this off
    neupi_entropy_w = 0.5
    neupi_loss_w = 0.5
    neupi_given_pred_effect_thresh = 0.8  # useful for noisy domains
    neupi_entropy_entry_min = 0.0
    neupi_entropy_entry_max = 1.0
    neupi_gt_ae_matrix = False
    neupi_w_negation = True
    neupi_w_quantifiers = True
    # Predicate selection scores a predicate set by the planning effort on
    # the demonstrations (num_nodes_expanded or num_nodes_created; lower is
    # better), counting a demonstration that cannot be solved as
    # neupi_selection_nodes_upper_bound. Hill-climbing stops once the score
    # is below neupi_selection_thresh.
    neupi_selection_metric = 'num_nodes_expanded'
    neupi_selection_nodes_upper_bound = 1e5
    neupi_selection_thresh = 1e6
    neupi_equ_dataset = 0.05
    neupi_learning_dataset = 1.0 # for predicate training and NSRT learning
    neupi_pred_search_dataset = 1.0
    neupi_quantify_dataset = 1.0
    neupi_max_neural_nets = 50
    neupi_parallel_invention = True
    neupi_save_init_atom_dataset = True
    neupi_gt_sampler = False
    neupi_cache_input_graph = False
    precond_thresh = 0.999
    max_normalizer_data = -1

    @classmethod
    def get_arg_specific_settings(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """A workaround for global settings that are derived from the
        experiment-specific args."""

        settings = dict(
            # The method used for perception.
            perceiver="trivial",
            # When checking if a policy solves a task, we run the policy for at
            # most this many steps.
            horizon=100,
            # Maximum number of steps to roll out an option policy.
            max_num_steps_option_rollout=1000,
            # In SeSamE, when to propagate failures back up to the high level
            # search. Choices are: {"after_exhaust", "immediately", "never"}.
            sesame_propagate_failures="immediately",
            # For learning-based approaches, the data collection strategy.
            offline_data_method="demo",
            # The name of the option model used by the agent.
            option_model_name="oracle",
            # In SeSamE, the maximum number of skeletons optimized before
            # giving up. If 1, can only solve downward refinable tasks.
            sesame_max_skeletons_optimized=defaultdict(
                lambda: 8,
                {
                    # For these environments, allow more skeletons.
                    "satellites": 50,
                })[args.get("env", "")],
            # In SeSamE, the maximum effort put into refining a single skeleton.
            # Concretely, this effort refers to the maximum number of calls to
            # the sampler on each step before backtracking.
            sesame_max_samples_per_step=10,
            # Maximum number of skeletons used by ExpectedNodesScoreFunction.
            # If -1, defaults to CFG.sesame_max_skeletons_optimized.
            grammar_search_expected_nodes_max_skeletons=-1,
            # Factor to divide feature range by when instantiating predicates
            # of the form |t1.f1 - t2.f2| < c to indicate that t1.f1 and
            # t2.f2 are "touching" or close.
            grammar_search_diff_features_const_multiplier=1e-6,
            # Feature names to use as part of the EuclideanPredicateGrammar.
            # Each entry is (type1_feature1name, type1_feature2name,
            # type2_feature1name, type2_feature2name)
            grammar_search_euclidean_feature_names=[("x", "y", "x", "y")],
            # Factor to divide feature range by when instantiating euclidean
            # predicates of the form
            # (t1.f1 - t2.f1)^2 + (t1.f2 - t2.f2)^2 < c^2 to indicate that
            # the euclidean distance between f1 and f2 is close enough that.
            # the two objects are "touching".
            grammar_search_euclidean_const_multiplier=1e-6,
        )
        for settings_fn in _ARG_SPECIFIC_SETTINGS_FNS:
            settings.update(settings_fn(args))
        return settings


def get_allowed_query_type_names() -> Set[str]:
    """Get the set of names of query types that the teacher is allowed to
    answer, computed based on the configuration CFG."""
    if CFG.option_learner == "direct_bc":
        return {"PathToStateQuery"}
    if CFG.approach == "unittest":
        return {
            "GroundAtomsHoldQuery",
            "DemonstrationQuery",
            "PathToStateQuery",
            "_MockQuery",
        }
    return set()


def register_settings(settings_cls: type) -> None:
    """Add the public attributes of settings_cls as new global settings.

    Extensions of predicators define their settings in a class with the
    same layout as GlobalSettings and register it once at import time.
    The new names are then accepted as command line flags by
    utils.parse_args() and restored to their defaults by
    utils.reset_config(), exactly like the built-in settings.
    Registering a name that is already a setting or a command line arg
    raises a ValueError. Registering the same class twice is a no-op.
    """
    if settings_cls in _REGISTERED_SETTINGS_CLASSES:
        return
    new_settings = {
        name: value
        for name, value in vars(settings_cls).items()
        if not name.startswith("_") and not isinstance(
            value, (classmethod, staticmethod, FunctionType))
    }
    arg_names = {
        action.dest
        for action in create_arg_parser(env_required=False,
                                        approach_required=False,
                                        seed_required=False)._actions  # pylint: disable=protected-access
    }
    for name in new_settings:
        if hasattr(GlobalSettings, name) or name in arg_names or \
                hasattr(CFG, name):
            raise ValueError(f"Cannot register setting {name} of "
                             f"{settings_cls.__name__}: the name is taken.")
    for name, value in new_settings.items():
        setattr(GlobalSettings, name, value)
        setattr(CFG, name, value)
    _REGISTERED_SETTINGS_CLASSES.append(settings_cls)


def register_arg_specific_settings(
        settings_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Register a function that maps the parsed args to extra arg-specific
    settings (for example, defaults that depend on the env).

    Its output is merged into GlobalSettings.get_arg_specific_settings(),
    after the built-in values, so it can override them. Values passed on
    the command line still take precedence.
    """
    if settings_fn not in _ARG_SPECIFIC_SETTINGS_FNS:
        _ARG_SPECIFIC_SETTINGS_FNS.append(settings_fn)


_attr_to_value = {}
for _attr, _value in GlobalSettings.__dict__.items():
    if _attr.startswith("_"):
        continue
    assert _attr not in _attr_to_value  # duplicate attributes
    _attr_to_value[_attr] = _value
CFG = SimpleNamespace(**_attr_to_value)
