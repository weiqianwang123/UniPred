"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar."""

from __future__ import annotations

import os
import copy
import glob
import time
import yaml
import itertools
import queue as queue_lib
import json
import logging
from z3 import Bool, Not, Or, Solver, is_true, sat
from tqdm import tqdm
from gymnasium.spaces import Box
from typing import Any, Callable, Collection, Dict, FrozenSet, List, \
    NamedTuple, Optional, Sequence, Set, Tuple, TypeVar

import numpy as np
import dill as pkl
import torch
import torch.nn
import torch.optim
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader

from predicators import hooks, utils
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.gnn.neupi import setup_neupi_mlp_net, setup_neupi_optimizer, HierachicalMCTSearcher
from predicators.gnn.neupi_utils import train_predicate_model, get_single_neupi_prediction, \
    compute_guidance_vector, distill_learned_ae_vector, check_learned_ap_vector, \
    gen_pretty_pred_name, two2one, one2two, prob_three2two, inferece_dataloader, name2quantifier_types, \
    parse_basic_pred_name
from predicators.gnn.gnn_utils import GraphTransC2DDataset, compute_normalizers, action_graph_batch_collate, \
    normalize_graph
from predicators.ground_truth_models import get_dummy_nsrts
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.settings import CFG
from predicators.predicate_search_score_functions import _OperatorBeliefScoreFunction
from predicators.structs import NSRT, Action, Dataset, DummyPredicate, \
    GroundAtom, GroundAtomTrajectory, LowLevelTrajectoryReward, \
    NeuralPredicate, Object, ParameterizedOption, Predicate, State, Task, \
    Type, _Option

_Output = TypeVar("_Output")  # a generic type for the output of this GNN


class PredicateCandidate(NamedTuple):
    """An effect predicate that predicate selection can add, with its column
    of the action-effect (AE) matrix, of shape (num options, channels), and
    the indices of its arguments among the option arguments."""
    predicate: Predicate
    ae_column: torch.Tensor
    ent_idx: List[int]


def _load_tensor(path: str) -> Any:
    """Load a tensor or state dict saved by predicate learning onto the CPU,
    so that models trained on a GPU can be loaded without one."""
    return torch.load(path, map_location="cpu")

################################################################################
#                          Programmatic classifiers                            #
################################################################################

_PREDICATE_TYPE_DEFAULTS = {
    "decision_b": 0.5,
    "skip_train": False,
    "search_tree_max_level": 1,
}


def read_predicate_config(path: str) -> Dict:
    """Read a predicate config file, resolving its include.

    With ``include: other.yaml`` (relative to the file), the other file is
    read first; then the defaults of this file update its defaults, a
    predicate type of this file updates the type with the same name (or is
    added), and the other keys of this file replace its keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if "include" not in config:
        return config
    base = read_predicate_config(
        os.path.join(os.path.dirname(path), config.pop("include")))
    merged = {**base, **config}
    merged["defaults"] = {
        **base.get("defaults", {}),
        **config.get("defaults", {})
    }
    types = {t["name"]: dict(t) for t in base.get("predicate_types", [])}
    for entry in config.get("predicate_types", []):
        types[entry["name"]] = {**types.get(entry["name"], {}), **entry}
    merged["predicate_types"] = list(types.values())
    return merged


def load_predicate_config(
    path: str, options: Sequence[ParameterizedOption]
) -> Tuple[List[Dict], List[int], List[str]]:
    """Load the predicate config of IVNTR (--neupi_pred_config).

    The YAML file has these keys::

        include: other.yaml  # optional, see read_predicate_config()
        final_options: [UseCamera]  # options that only achieve goals
        static_predicates: [HasCamera]  # given predicates, never effects
        defaults: {epochs: 100, ...}  # merged into every predicate type
        predicate_types:  # the neural predicate types to invent
          - name: neural_b_p3
            types: [satellite, object]
            ent_idx: [0, 0]  # which option argument of each type
            search_options: [MoveTo, MoveAway]  # optional, default all
            ...  # training and search settings

    Other top-level keys are left to extensions. Returns the predicate
    types (with the defaults merged in), a 0/1 flag per option marking the
    final options, and the names of the static predicates.
    """
    config = read_predicate_config(path)
    option_names = [option.name for option in options]
    final_options = config.get("final_options", [])
    unknown = set(final_options) - set(option_names)
    for entry in config["predicate_types"]:
        unknown |= set(entry.get("search_options", [])) - set(option_names)
    if unknown:
        raise ValueError(f"{path}: unknown options {sorted(unknown)}; the "
                         f"options are {option_names}.")
    predicate_types = [{
        **_PREDICATE_TYPE_DEFAULTS,
        **config.get("defaults", {}),
        **entry
    } for entry in config["predicate_types"]]
    is_final = [int(name in final_options) for name in option_names]
    return predicate_types, is_final, list(
        config.get("static_predicates", []))


def _train_val_model_process(settings: Dict[str, Any], queue: Queue,
                             args: Tuple) -> None:
    """Entry point of a training process for parallel invention.

    The processes are started with the "spawn" method, because CUDA
    cannot be used in a forked process once the parent has used it. A
    spawned process starts from scratch, so the global settings and the
    logging are restored first. Each process trains one model with
    train_val_model_single() and puts its (model path, val loss) on the
    queue.
    """
    CFG.__dict__.update(settings)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if CFG.log_file:
        handlers.append(logging.FileHandler(CFG.log_file, mode="a"))
    logging.basicConfig(level=CFG.loglevel,
                        format="%(message)s",
                        handlers=handlers,
                        force=True)
    queue.put(train_val_model_single(*args, num_workers=0))


def train_val_model_single(curr_pred, ent_idx, pred_save_path, ae_vector, iteration, pred_config, \
                                ae_row_name_dict, node_feature_to_index, edge_feature_to_index, \
                                train_dataset, val_dataset, n, wandb_run_name='', num_workers=4):
    # Initialize W&B run with a unique name for this process
    # assert n == 0, "Single Process"
    logging.info(f"Predicat {curr_pred.arity} Iteration {iteration} | Starting process {n}")
    if wandb_run_name:
        logging.info("Starting W&B run. This makes training slower due to logging.")
        utils.init_wandb_run(
            f"{wandb_run_name}_pred_{pred_config['name']}_init_{n}",
            utils.get_important_cfg(CFG, pred_config))

    train_dataloader = DataLoader(train_dataset,
                                    batch_size=pred_config['batch_size'],
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=num_workers > 0,
                                    persistent_workers=num_workers > 0,
                                    collate_fn=action_graph_batch_collate)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=pred_config['batch_size'],
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=num_workers > 0,
                                persistent_workers=num_workers > 0,
                                collate_fn=action_graph_batch_collate)
    model = setup_neupi_mlp_net(train_dataset,
                                curr_pred.arity,
                                pred_config['architecture'],
                                node_feat2inx=node_feature_to_index,
                                edge_feat2inx=edge_feature_to_index)

    optimizer, scheduler = setup_neupi_optimizer(model, \
                                                 pred_config['optimizer'], \
                                                 pred_config['lr_scheduler'])
    # # Train the model
    if "quick_skip" not in pred_config:
        pred_config["quick_skip"] = None
    best_model_dict, best_val_loss = train_predicate_model(
        iteration,  # replace with your iteration management
        model,
        train_dataloader,
        val_dataloader,
        optimizer=optimizer,
        super_label=CFG.neupi_super_label,
        num_epochs=pred_config["epochs"],
        gumbel_temperature=pred_config["gumbel_temp"],
        val_freq=pred_config['val_freq'],
        device=CFG.device,
        scheduler=scheduler,
        quick_skip=pred_config["quick_skip"]
    )
    logging.info("Model {} Trained".format(n))
    logging.info("Corresponding AE Vector: {}".format(ae_vector[:, 0]))
    if ae_vector.shape[-1] == 2:
        logging.info("Corresponding AE Vector (Del): {}".format(ae_vector[:, 1]))
    model.load_state_dict(best_model_dict)
    # for basic predicates, always using 0.5 as decision boundary
    learned_ae_vector = distill_learned_ae_vector(val_dataloader, \
            pred_config['gumbel_temp'], 0.5, model, curr_pred, ent_idx, \
            ae_row_name_dict, node_feature_to_index, CFG.device)
    learned_guidance = compute_guidance_vector(
                                        learned_ae_vector,
                                        ae_vector,
                                        min_prob=CFG.neupi_entropy_entry_min,
                                        max_prob=CFG.neupi_entropy_entry_max,
                                        entropy_w=CFG.neupi_entropy_w,
                                        loss_w=CFG.neupi_loss_w)
    logging.info("Model {}".format(n))
    logging.info("Learned AE Guidance (Lower better): {}".format(learned_guidance))
    save_path_model = os.path.join(pred_save_path, f"iter_{iteration}_{n}_model.pth")
    torch.save(best_model_dict, save_path_model)
    save_path_guidance = os.path.join(pred_save_path, f"iter_{iteration}_{n}_guidance.pth")
    torch.save(learned_guidance, save_path_guidance)
    save_path_ae_vector = os.path.join(pred_save_path, f"iter_{iteration}_{n}_ae_vector.pth")
    torch.save(ae_vector, save_path_ae_vector)
    # Put result in the queue
    return save_path_model, best_val_loss

################################################################################
#                                 Approach                                     #
################################################################################


class BilevelLearningApproach(NSRTLearningApproach):
    """An approach that invents predicates by learn a GNN that maps continous Graph 
    to discrete space. Using Action Effect Theorem. """

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._sorted_options = sorted(self._initial_options,
                                      key=lambda o: o.name)
        for option in self._sorted_options:
            logging.info("Options (Clusters): {}, Arguments: {}".format(option.name, option.types))
        self.option_appeared = [0 for _ in range(len(self._sorted_options))]
        self._obj_types = {}
        self._opt_type_id = {}
        self._max_type_id = {} # this will be 1 larger than actual ent_id
        for type_ in self._types:
            self._max_type_id[type_.name] = 0
        for option in self._sorted_options:
            self._opt_type_id[option.name] = {}
            types = option.types
            for t in types:
                if t.name not in self._obj_types:
                    self._obj_types[t.name] = t
                if t.name not in self._opt_type_id[option.name]:
                    self._opt_type_id[option.name][t.name] = [0]
                    self._max_type_id[t.name] = 1
                else:
                    curr_id = len(self._opt_type_id[option.name][t.name])
                    self._opt_type_id[option.name][t.name].append(curr_id)
                    if curr_id + 1 > self._max_type_id[t.name]:
                        self._max_type_id[t.name] = curr_id + 1

        self.gt_ae_matrix: bool = CFG.neupi_gt_ae_matrix
        self.learned_ae_pred_info: Dict = {}

        self.pred_config, self.final_op, self._static_predicate_names = \
            load_predicate_config(CFG.neupi_pred_config,
                                  self._sorted_options)

        for pred_info in self.pred_config:
            pred_name = pred_info['name']
            types_str = pred_info['types']
            types = [self._obj_types[t] for t in types_str]
            dummy_pred = DummyPredicate(name=pred_name, types=types)
            if 'ent_idx' in pred_info:
                # which entity in the option to be used as predicate argument
                assert len(pred_info['ent_idx']) == len(types), "Ent Index Not Match"
                ent_idx = pred_info['ent_idx']
            else:
                logging.info("WARNING: Ent Index Not Provided, Use First Entities for Predicate {}".format(pred_name))
                ent_idx = [0 for _ in range(len(types))]
            for idx, t in enumerate(types):
                logging.info("Using: {}_{}".format(t.name, ent_idx[idx]))
            logging.info("Predicate Type TOBE Invented: {}".format(pred_name))
            logging.info("Learning Conifg: {}".format(pred_info))
            gt_ae_vectors = []
            if 'gt' in pred_info and pred_info['gt']:
                logging.info("GT Provided for Predicate {}!".format(pred_name))
                gt_ae_list = pred_info['gt']
                for ae in gt_ae_list:
                    logging.info("GT AE Vector: {}".format(ae))
                    ae_vector = torch.tensor(ae)
                    ae_vector_two = one2two(ae_vector, CFG.neupi_ae_matrix_channel)
                    gt_ae_vectors.append(ae_vector_two)
            search_region = [
                int(option.name in pred_info['search_options'])
                for option in self._sorted_options
            ] if 'search_options' in pred_info else [1] * len(
                self._sorted_options)
            self.learned_ae_pred_info[dummy_pred] = {
                'example_data': None,
                'quantifiers': [], # if this is from a higher-arity predicate
                'constraints': [],
                'ae_vecs': [],
                'gt_ae_vecs': gt_ae_vectors,
                'search_region': search_region,
                'scores': [],
                'model_weights': [],
                'ent_idx': [ent_idx],
                'learned': False,
                'provided': False
            }

        # provided predicates
        # this is used to generate the ae matrix
        for pred in sorted(self._initial_predicates):
            if pred.name in self._static_predicate_names:
                # Static predicates are never effects.
                continue
            # Note that we are using belief learner, so "grounding" is not learned
            pred_types = pred.types
            ent_idx = []
            type_idx_counter = {}
            for t in pred_types:
                if t.name not in type_idx_counter:
                    type_idx_counter[t.name] = 0
                else:
                    type_idx_counter[t.name] += 1
                ent_idx.append(type_idx_counter[t.name])
            logging.info("Belived Ent Idx for Predicate: {}".format(pred.name))
            for idx, t in enumerate(pred_types):
                logging.info("Using: {}_{}".format(t.name, ent_idx[idx]))
            # needs to distinguish provided and learned predicates
            self.learned_ae_pred_info[pred] = {
                'constraints': [],
                'ae_vecs': [],
                'learned': True,
                'ent_idx': [ent_idx],
                'search_region': [1] * len(self._sorted_options), # dummy
                'provided': True
            }
            logging.info("Provided Predicate (No Learning): {}".format(pred.name))

        self.ae_row_names = list(self._sorted_options)
        self.ae_row_names_dict = {} # input to parallel function
        for i, option in enumerate(self.ae_row_names):
            self.ae_row_names_dict[i] = {
                'name': option.name,
                'arity': len(option.types),
                'types': [t.name for t in option.types]
            }
        # input
        self._node_feature_to_index: Dict[Any, int] = {}
        self._node_is_rot: List[bool] = [] # Track rot feat, they should not be normalized
        self._edge_feature_to_index: Dict[Any, int] = {}
        self._edge_is_rot: List[bool] = [] # Track rot feat, they should not be normalized
        self._input_normalizers = None

        self._learned_predicates: Set[NeuralPredicate] = set()
        # Seed torch so that neural predicate learning is reproducible.
        torch.manual_seed(self._seed)

    @classmethod
    def get_name(cls) -> str:
        return "ivntr"

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    @staticmethod
    def _model_artifact_path(model_path: str, artifact: str) -> str:
        """Return the path of a file saved next to a predicate model.

        A model is saved as iter_<i>_<n>_model.pth, renamed to
        iter_<i>_<n>_model_good.pth once it is kept. Its artifacts follow
        the same pattern: artifact "guidance" of a kept model is
        iter_<i>_<n>_guidance_good.pth, and artifact "model_good" of an
        unkept model is the path it is renamed to. Only the file name is
        changed, so directory names may contain "model".
        """
        directory, filename = os.path.split(model_path)
        stem, sep, suffix = filename.rpartition("_model")
        assert sep, f"Not a predicate model path: {model_path}"
        return os.path.join(directory, f"{stem}_{artifact}{suffix}")

    def _expand_candidate_pool(self, selected: List[Predicate],
                               pool: List[PredicateCandidate]) -> None:
        """Called by predicate selection after each predicate it adds.

        Subclasses can append new candidates to the pool in place, for
        example variants of the predicate that was just added. Candidates
        added after an imagined predicate are removed again if that
        predicate is taken back. By default, the pool is not changed.
        """
        del selected, pool  # unused

    def _operator_predicates(
            self, predicates: List[Predicate]) -> Collection[Predicate]:
        """Return the effect predicates that may appear in operator effects
        when scoring a predicate set during selection.

        The others can still appear in preconditions. The same split is
        used when learning the final NSRTs. By default, all effect
        predicates are used.
        """
        return predicates

    def _atoms_closure(
            self,
            predicates: Sequence[Predicate]) -> Optional[hooks.AtomsClosure]:
        """Return the atoms closure (see predicators/hooks.py) that
        recomputes the derived predicates among the given ones, or None.

        It is installed while scoring a predicate set during selection
        and while planning with the learned predicates. By default, there
        are no derived predicates.
        """
        del predicates  # unused
        return None

    def add_general_matrix_constraints(self, solver: Solver, local_entities: List, \
                                width: int, height: int, channels: int, col_ent: List[List[int]]) -> Solver:
        # Non-zero constraints for rows
        assert width > 1, "Width should be larger than 1 for a matrix, otherwise use vector"
        for row in range(height):
            exist_non_zero = Or([local_entities[row * width * channels + col * channels + channel] \
                                 for col in range(width) for channel in range(channels)])
            solver.add(exist_non_zero)

        # Ensure any two pairs of rows are different in at least one column and one channel
        for row1 in range(height):
            for row2 in range(row1 + 1, height):
                diff_constraints = []
                for col in range(width):
                    for channel in range(channels):
                        diff_constraints.append(local_entities[row1 * width * channels + col * channels + channel] \
                                                != local_entities[row2 * width * channels + col * channels + channel])
                solver.add(Or(diff_constraints))

        # Ensure any two pairs of columns are different in at least one row and one channel
        for col1 in range(width):
            for col2 in range(col1 + 1, width):
                if not torch.equal(torch.tensor(col_ent[col1]), torch.tensor(col_ent[col2])):
                    # the column pair has different grounding types, skip them
                    continue
                diff_constraints = []
                for row in range(height):
                    for channel in range(channels):
                        diff_constraints.append(local_entities[row * width * channels + col1 * channels + channel] \
                                                != local_entities[row * width * channels + col2 * channels + channel])
                solver.add(Or(diff_constraints))
        
        return solver
    
    def add_general_col_constraints(self, solver: Solver, local_entities: List, \
                                width: int, height: int, channels: int) -> Solver:
        # Non-zero constraints for the column
        assert width == 1, "Width should be 1 for a vector, otherwise use matrix"
        for col in range(width):
            exist_non_zero = Or([local_entities[row * width * channels + col * channels + channel] \
                                 for row in range(height) for channel in range(channels)])
            solver.add(exist_non_zero)

        # Across all the chanels of each entry, there exist at most one 1
        for row in range(height):
            for col in range(width):
                variables = [local_entities[row * width * channels + col * channels + channel] for channel in range(channels)]
                for i in range(len(variables)):
                    for j in range(i + 1, len(variables)):
                        # Add the constraint that at most one of the two variables can be true
                        solver.add(Or(Not(variables[i]), Not(variables[j])))
        
        return solver
    
    def check_matrix_consistency(self, ae_matrix: torch.Tensor, col_ent: List[List[int]]) -> bool:
        """
        Check the consistency of the given matrix.
        """
        height, width, channels = ae_matrix.shape
        if width <= 1:
            # one column won't sat
            return False
        solver = Solver()
        local_entities = [Bool(f'x_{row}_{col}_{channel}') for row in range(height) \
                          for col in range(width) for channel in range(channels)]
        solver = self.add_general_matrix_constraints(solver, local_entities, width, height, channels, col_ent)
        for row in range(height):
            for col in range(width):
                for channel in range(channels):
                    if ae_matrix[row, col, channel] == 1:
                        solver.add(local_entities[row * width * channels + col * channels + channel] == True)
                    elif ae_matrix[row, col, channel] == 0:
                        solver.add(local_entities[row * width * channels + col * channels + channel] == False)
                    else:
                        raise ValueError("Should be 0 or 1, got {}".format(ae_matrix[row, col, channel]))
        
        return solver.check() == sat

    def initialize_ae_constraints(self, data: List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], \
                                   _Option, str]], channels: int=1) -> torch.Tensor:
        """
        Generate the initial binary action effect matrix for all the predicates.
        The initial global hard constraints are also returned.
        """
        columns = list(self.learned_ae_pred_info.keys())
        width = len(columns)
        height = len(self.ae_row_names)

        assert channels <= 2, "Only support at most 2 channels"

        # Read the data and pose "zero" constraints by comparing the continous states
        # **Type Constraints for Typed Predicates**
        # Distill intial constraints from transition data
        all_zero_ae_matrix = torch.zeros((height, width, channels), dtype=int)
        for state, _, state_, _, action, _ in data:
            action_index = self.ae_row_names.index(action.parent)
            operated_objs = action.objects
            for affected_obj in operated_objs:
                s = state[affected_obj]
                s_ = state_[affected_obj]
                diff = s_ - s
                if not np.all(diff == 0):
                    for pred in columns:
                        arity = pred.arity
                        col_index = columns.index(pred)
                        if arity == 1:
                            types = pred.types
                            if types[0] == affected_obj.type:
                                all_zero_ae_matrix[action_index, col_index, :] = 1
                        elif arity == 2:
                            types = pred.types
                            if types[0] == affected_obj.type or types[1] == affected_obj.type:
                                all_zero_ae_matrix[action_index, col_index, :] = 1
        for row in range(height):
            if all_zero_ae_matrix[row].sum() == 0:
                # this action never appears
                continue
            for col in range(width):
                for channel in range(channels):
                    if all_zero_ae_matrix[row, col, channel] == 0:
                        rule = ('position', row, 0, channel, False) # this is for a column vector
                        pred = columns[col]
                        logging.info("Low-level feat not changed for Predicate {} in Row {}".format(pred.name, row))
                        if rule not in self.learned_ae_pred_info[pred]['constraints']:
                            self.learned_ae_pred_info[pred]['constraints'].append(rule)

        # Read the data and pose constraints for the given predicates (we can know it from data)
        all_ae_matrix_dummy = torch.zeros((height, width, channels), dtype=int) - 1
        action_counts = torch.zeros((height, width, channels), dtype=int)
        for pred in columns:
            pred_index = columns.index(pred)
            if not self.learned_ae_pred_info[pred]['provided']:
                continue
            all_ae_matrix_dummy[:, pred_index, :] = 0 # assmues no effect
        # first, use data to get the add and delete effects
        for _, atom, _, atom_, action, _ in data:
            action_index = self.ae_row_names.index(action.parent)
            action_counts[action_index, :, :] += 1
            for at in atom_:
                if at.predicate.arity not in [1, 2]:
                    continue
                if at.predicate not in columns:
                    continue
                if at not in atom:
                    # this is an add effect in this transition
                    pred_index = columns.index(at.predicate)
                    assert all_ae_matrix_dummy[action_index, pred_index, 0] >= 0, "Should be zero"
                    all_ae_matrix_dummy[action_index, pred_index, 0] += 1
            if channels == 2:
                for at in atom:
                    if at.predicate.arity not in [1, 2]:
                        continue
                    if at.predicate not in columns:
                        continue
                    if at not in atom_:
                        # this is a delete effect in this transition
                        pred_index = columns.index(at.predicate)
                        assert all_ae_matrix_dummy[action_index, pred_index, 0] >= 0, "Should be zero"
                        all_ae_matrix_dummy[action_index, pred_index, 1] += 1
        # round the values to 0 or 1
        # some actions may not appear
        high_bar = CFG.neupi_given_pred_effect_thresh
        low_bar = 1 - CFG.neupi_given_pred_effect_thresh
        action_counts[action_counts == 0] = 1
        all_ae_matrix_dummy = all_ae_matrix_dummy / action_counts
        all_ae_matrix_dummy[all_ae_matrix_dummy < 0] = -1
        all_ae_matrix_dummy[all_ae_matrix_dummy >= high_bar] = 1
        all_ae_matrix_dummy[(0 <= all_ae_matrix_dummy) & (all_ae_matrix_dummy < low_bar)] = 0
        # check all the matrix entries are -1 1 or 0
        assert ((all_ae_matrix_dummy == 0) | (all_ae_matrix_dummy == 1) \
                | (all_ae_matrix_dummy == -1)).all(), "Should be -1 0 or 1, check data noise"
        # We don't invent the goal predicate, so final op row should be 0 (for invented predicates)
        for row in range(height):
            if self.final_op[row] == 0:
                # this action may have invented predicate effects
                continue
            postive_mask = all_ae_matrix_dummy[row] == 1
            if not postive_mask.any():
                # this action never has positive effects
                logging.info("(Final) Action {} has no positive effects, it is not appearing in data?".\
                                format(self.ae_row_names[row].name))
            negtive_mask = all_ae_matrix_dummy[row] == -1
            # this is the final op, should have no invented predicate effects
            all_ae_matrix_dummy[row, negtive_mask] = 0

        for col in range(width):
            pred_name = columns[col]
            if all_ae_matrix_dummy[:, col].sum() == 0:
                # this effect never appears
                raise ValueError("This effect predicate never appears in data, bug!")
            for row in range(height):
                if all_ae_matrix_dummy[row, col, :].sum() == 0:
                    # this action never appears
                    logging.info("Action {} Definitely has no effect for pred {}".\
                                 format(self.ae_row_names[row].name, pred_name.name))
                    for channel in range(channels):
                        rule = ('position', row, 0, channel, False)
                        if rule not in self.learned_ae_pred_info[pred_name]['constraints']:
                            self.learned_ae_pred_info[pred_name]['constraints'].append(rule)
                    continue
                if all_ae_matrix_dummy[row, col, :].sum() == -1 * channels:
                    # this effect predicate is not provided
                    logging.info("Predicate {} Not Provided for action {} in Data".format(pred_name.name, \
                                                                self.ae_row_names[row].name))
                    continue
                for channel in range(channels):
                    if all_ae_matrix_dummy[row, col, channel] == 0:
                        # note that the rule is for a vector, so column==0
                        rule = ('position', row, 0, channel, False)
                        if rule not in self.learned_ae_pred_info[pred_name]['constraints']:
                            self.learned_ae_pred_info[pred_name]['constraints'].append(rule)
                    elif all_ae_matrix_dummy[row, col, channel] == 1:
                        if channel == 0:
                            logging.info("Action {} Definitely has add effect for pred {}".\
                                            format(self.ae_row_names[row].name, pred_name.name))
                        else:
                            logging.info("Action {} Definitely has delete effect for pred {}".\
                                            format(self.ae_row_names[row].name, pred_name.name))
                        rule = ('position', row, 0, channel, True)
                        if rule not in self.learned_ae_pred_info[pred_name]['constraints']:
                            self.learned_ae_pred_info[pred_name]['constraints'].append(rule)
                    else:
                        assert (all_ae_matrix_dummy[:, col, :] >= 0).any(), "Should have at least one positive effect"

        # Type Checking, if the type is impossible to be matched, the effect is sure to be 0
        non_effect_matrix = torch.zeros((height, width, channels), dtype=int) - 1
        for pred in columns:
            types_pred = {}
            pred_ent_idx = self.learned_ae_pred_info[pred]['ent_idx'][0]
            for a, t in enumerate(pred.types):
                if t.name not in types_pred:
                    types_pred[t.name] = pred_ent_idx[a]
                else:
                    curr_num = types_pred[t.name]
                    types_pred[t.name] = max(curr_num, pred_ent_idx[a])
            pred_index = columns.index(pred)
            if self.learned_ae_pred_info[pred]['provided']:
                # provided predicate no need to add this constraint
                continue
            for row in range(height):
                possible = True
                action_type = {}
                for t in self.ae_row_names[row].types:
                    if t.name not in action_type:
                        action_type[t.name] = 0
                    else:
                        action_type[t.name] += 1
                for tp in types_pred.keys():
                    if tp not in action_type:
                        # this action type cannot match the predicate type
                        possible = False
                        break
                    else:
                        if types_pred[tp] > action_type[tp]:
                            possible = False
                if not possible:
                    logging.info("Impossible Type Match for Predicate {} in Row {}".format(pred.name, row))
                    non_effect_matrix[row, pred_index, :] = 0
                    for channel in range(channels):
                        rule = ('position', row, 0, channel, False)
                        if rule not in self.learned_ae_pred_info[pred]['constraints']:
                            self.learned_ae_pred_info[pred]['constraints'].append(rule)

        # Check search region
        for pred in columns:
            search_region = self.learned_ae_pred_info[pred]['search_region']
            if sum(search_region) == len(self.ae_row_names):
                # search everywehre
                continue
            else:
                for row in range(height):
                    if not search_region[row]:
                        for channel in range(channels):
                            logging.info("Search Region Not Match for Predicate {} in Row {}".format(pred.name, row))
                            rule = ('position', row, 0, channel, False)
                            if rule not in self.learned_ae_pred_info[pred]['constraints']:
                                self.learned_ae_pred_info[pred]['constraints'].append(rule)

    def gen_ae_vectors4pred(self, iteration: int, curr_pred: DummyPredicate, ent_idx: List[int], pred_save_path: str, \
                                    pred_config: Dict, data: List[Tuple[State, Set[GroundAtom], \
                                    State, Set[GroundAtom], _Option, str]], symbolic_model: HierachicalMCTSearcher) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a bunch of initial binary action effect vector for the tgt predicate.
        The scores for these initial vectors are also returned.
        """
        logging.info("Generate {} AE Vectors-Score Pairs for Predicate {}"\
                     .format(pred_config["batch_vect_num"], curr_pred.name))

        # 1. Use symbolic model to generate AE vectors
        if self.gt_ae_matrix:
            if iteration == 0:
                assert len(self.learned_ae_pred_info[curr_pred]['gt_ae_vecs']), "GT AE Vec Not Provided"
            else:
                assert not len(self.learned_ae_pred_info[curr_pred]['gt_ae_vecs']), \
                    "Iteration > 0, GT AE Vec Should Not Be []"
            logging.info("GT AE Vec Provided, No Need to Generate AE Vectors.")
            sat_vectors = copy.deepcopy(self.learned_ae_pred_info[curr_pred]['gt_ae_vecs'])
            self.learned_ae_pred_info[curr_pred]['gt_ae_vecs'] = []
        else:
            sat_vectors = self.gen_sat_vec(curr_pred, pred_config["batch_vect_num"], \
                            pred_config["matrix_vec_try"], symbolic_model)
            
        if not len(sat_vectors):
            logging.info("No more sat matrixes can be generated at iteration {}!".format(iteration))
            return torch.tensor([]), torch.tensor([]), [], []
        curr_ae_vectors = sat_vectors
        # 2. Use neural model to compute the scores for the selected AE vectors
        logging.info(f"Optimizing {len(curr_ae_vectors)} Neural Models with AE Vector from BO...")
        ae_vecs = []
        scores = []
        val_losses = []
        model_weight_paths = []
        train_datasets = []
        val_datasets = []

        for n, sat_vector in enumerate(curr_ae_vectors):
            logging.info(f"*******Vec {n} ({curr_pred.name})*******")
            logging.info("(Add): \n{}".format(sat_vector[:, 0]))
            if sat_vector.shape[-1] == 2:
                logging.info("(Del): \n{}".format(sat_vector[:, 1]))
            train_dataset, val_dataset = self.gen_graph_data(data, 
                                                        curr_pred, 
                                                        ent_idx,
                                                        sat_vector)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        if len(curr_ae_vectors) == 1 or (not CFG.neupi_parallel_invention):
            for i in range(len(sat_vectors)):
                logging.info(f"Training Neural Model {i}...")
                model_path, val_loss = train_val_model_single(curr_pred, ent_idx, pred_save_path, \
                                    sat_vectors[i], iteration, pred_config, self.ae_row_names_dict, \
                                    self._node_feature_to_index, self._edge_feature_to_index, \
                                    train_datasets[i], val_datasets[i], i, CFG.wandb_run_name)
                model_weight_paths.append(model_path)
                val_losses.append(val_loss)
                ae_vecs.append(sat_vectors[i])
                scores.append(
                    _load_tensor(
                        self._model_artifact_path(model_path, "guidance")))
        else:
            if CFG.neupi_parallel_invention:
                context = torch.multiprocessing.get_context("spawn")
                queue = context.Queue()
                settings = {
                    name: value
                    for name, value in vars(CFG).items()
                    if not isinstance(value, (classmethod, staticmethod))
                }
                processes = []
                for i in range(len(sat_vectors)):
                    args = (curr_pred, ent_idx, pred_save_path,
                            sat_vectors[i], iteration, pred_config,
                            self.ae_row_names_dict,
                            self._node_feature_to_index,
                            self._edge_feature_to_index, train_datasets[i],
                            val_datasets[i], i, CFG.wandb_run_name)
                    p = context.Process(target=_train_val_model_process,
                                        args=(settings, queue, args))
                    p.start()
                    processes.append(p)
                # Results are small, so read them before joining; a failed
                # process puts nothing, so check the exit codes.
                model_paths_loss = []
                while len(model_paths_loss) < len(processes):
                    try:
                        model_paths_loss.append(queue.get(timeout=10))
                    except queue_lib.Empty:
                        failed = [p.pid for p in processes
                                  if p.exitcode not in (None, 0)]
                        if failed:
                            raise RuntimeError(
                                f"Training processes {failed} failed; see "
                                "the log above.")
                for p in processes:
                    p.join()
                    logging.info(f"Process {p.pid} finished")
                # Keep the order of the vectors (n is in the file name).
                model_paths_loss.sort(
                    key=lambda r: int(os.path.basename(r[0]).split("_")[2]))
                for i in range(len(sat_vectors)):
                    model_weight_paths.append(model_paths_loss[i][0])
                    val_losses.append(model_paths_loss[i][1])
                    scores.append(
                        _load_tensor(
                            self._model_artifact_path(model_paths_loss[i][0],
                                                      "guidance")))
                    ae_vecs.append(
                        _load_tensor(
                            self._model_artifact_path(model_paths_loss[i][0],
                                                      "ae_vector")))
        return ae_vecs, scores, val_losses, model_weight_paths

    def gen_sat_vec(self, pred: DummyPredicate, \
                    max_num: int, \
                    max_samples: int, \
                    searcher: Optional[HierachicalMCTSearcher]=None) \
                    -> List[torch.Tensor]:
        """
        Generate the action effect matrix guided by entropy.
        """
        constraints = self.learned_ae_pred_info[pred]['constraints']
        # it is already clamped, do not normalize with max
        # n_row x (n_channel + 1)
        width = 1
        height = len(self.ae_row_names)
        channels = CFG.neupi_ae_matrix_channel
        max_num = min(max_num, CFG.neupi_max_neural_nets)
        sat_vectors = []
        for i in range(max_samples):
            if len(sat_vectors) >= max_num:
                break
            logging.info(f"Sampling {i}/{max_samples} AE Vectors (Tgt {max_num})")
            # check if the sampled matrix is satisfying all constraints
            solver = Solver()
            local_entities = [Bool(f'x_{row}_{col}_{channel}') for row in range(height) \
                          for col in range(width) for channel in range(channels)]
            
            solver = self.add_general_col_constraints(solver, local_entities, width, height, channels)
            constraints = self.learned_ae_pred_info[pred]['constraints']
            for rule in constraints:
                c_type = rule[0]
                if c_type == 'position':
                    row, col, channel, value = rule[1], rule[2], rule[3], rule[4]
                    assert col == 0, "Should be 0 for a vector"
                    solver.add(local_entities[row * width * channels + col * channels + channel] == value)
                else:
                    raise ValueError('Unknown constraint type')
            
            if searcher is not None:
                symbolic_proposal = searcher.propose()
                if symbolic_proposal is None:
                    return sat_vectors
                vector_sampled = one2two(symbolic_proposal, channels)
                ae_vec_flipped = vector_sampled.clone()
                    
                # add all the entry in flipped matrix
                for row in range(height):
                    for channel in range(channels):
                        if ae_vec_flipped[row, channel] == 1:
                            solver.add(local_entities[row * width * channels + 0 * channels + channel] == True)
                        elif ae_vec_flipped[row, channel] == 0:
                            solver.add(local_entities[row * width * channels + 0 * channels + channel] == False)
                        else:
                            raise ValueError("Should be 0 or 1, got {}".format(ae_vec_flipped[row, channel]))
            
            # sat?
            if solver.check() == sat:
                logging.info(f"Found a satisfying AE Vector with Guidance after {i} tries.")
                model = solver.model()
                matrix = torch.zeros((height, width, channels), dtype=int)
                for row in range(height):
                    for col in range(width):
                        for channel in range(channels):
                            if is_true(model.evaluate(local_entities[row * width * channels + col * channels + channel])):
                                matrix[row, col, channel] = 1
                sat_vectors.append(matrix[:, 0])
            else:
                assert searcher is not None, "No satisfying vector found."
                state = symbolic_proposal.copy()
                guidance = np.array([np.inf for _ in range(len(self.ae_row_names))])
                # unsatisfiable node, update searcher
                searcher.update_value(state, guidance)
                logging.info(f"Vector not satisfiable.")

        return sat_vectors
            
    def _setup_input_fields(
        self, data: List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], \
                                 _Option, str]]
    ) -> None:
        """
        Graph input only has continous object features.
        """
        obj_types_set = set()
        obj_attrs_set = set()

        # Go through the data, identifying the types/attributes.
        # Note that we don't use any Atom data even if they exist.
        for state, _, _, _, _, _ in data:
            for obj in state:
                obj_types_set.add(f"type_{obj.type.name}")
                for feat in obj.type.feature_names:
                    obj_attrs_set.add(f"feat_{feat}")

        obj_types = sorted(obj_types_set)
        obj_attrs = sorted(obj_attrs_set)

        # Initialize input node features.
        self._node_feature_to_index = {}
        index = 0
        for obj_type in obj_types:
            self._node_feature_to_index[obj_type] = index
            self._node_is_rot.append(False)
            index += 1
        for obj_attr in obj_attrs:
            self._node_feature_to_index[obj_attr] = index
            if "_q" in obj_attr:
                self._node_is_rot.append(True)
            else:
                self._node_is_rot.append(False)
            index += 1

        # Initialize input edge features. Concat of two node features.
        self._edge_feature_to_index = {}
        index = 0
        for obj_feat1 in self._node_feature_to_index.keys():
            self._edge_feature_to_index["ent1-{}".format(obj_feat1)] = index
            if ("_q" in obj_feat1) or ("_norm" in obj_feat1):
                self._edge_is_rot.append(True)
            else:
                self._edge_is_rot.append(False)
            index += 1
        for obj_feat2 in self._node_feature_to_index.keys():
            self._edge_feature_to_index["ent2-{}".format(obj_feat2)] = index
            if ("_q" in obj_feat2) or ("_norm" in obj_feat2):
                self._edge_is_rot.append(True)
            else:
                self._edge_is_rot.append(False)
            index += 1

    def _generate_data_from_dataset(
        self, dataset: Dataset
    ) -> Tuple[List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], _Option, str]], List]:
        data = []
        # Initialization only takes positive trajs, negative ones do not satisfy AET
        max_action_arity = 0
        segmented_trajs = []
        ground_atom_dataset = []
        num_trajs = int(len(dataset.trajectories) * CFG.neupi_learning_dataset)
        for traj in tqdm(dataset.trajectories[:num_trajs]):
            segment_traj = segment_trajectory(traj, self._initial_predicates)
            segmented_trajs.append(segment_traj)
        for segment_traj, ll_traj in zip(segmented_trajs,
                                         dataset.trajectories[:num_trajs]):
            if not ll_traj.is_demo:
                continue
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            atoms_traj = []
            for segment in segment_traj:
                state = segment.states[0]  # the segment's initial state
                atoms = segment.init_atoms  # the segment's initial atoms
                atoms_traj.append(atoms)
                state_ = segment.states[-1]  # the segment's final state
                atoms_ = segment.final_atoms
                action = segment.get_option()  # the segment's option
                action_id = self.ae_row_names.index(action.parent)
                self.option_appeared[action_id] = 1
                action_arity = len(action.objects)
                max_action_arity = max(max_action_arity, action_arity)
                data.append((state, atoms, state_, atoms_, action, '{}_{}'.format(action_id, action_arity)))
            atoms_traj.append(atoms_) # the last state atoms
            ground_atom_dataset.append((ll_traj, atoms_traj))
        self.max_action_arity = max_action_arity
        return data, dataset.trajectories[:num_trajs], ground_atom_dataset

    # Optimized vectorized version
    def add_edge_features_fast(self, 
                               node_features, 
                               object_to_node, 
                               node_feature_to_index, 
                               edge_feature_to_index):
        num_objects = len(object_to_node)
        num_edge_features = max(len(edge_feature_to_index), 1)
        
        # Create mapping arrays for node indices
        node_indices = np.array([object_to_node[obj] for obj in object_to_node])
        
        # Create mapping arrays for feature indices
        ent1_feature_mapping = np.array([
            (edge_feature_to_index[f"ent1-{feat}"], node_feature_to_index[feat])
            for feat in node_feature_to_index
        ])
        ent2_feature_mapping = np.array([
            (edge_feature_to_index[f"ent2-{feat}"], node_feature_to_index[feat])
            for feat in node_feature_to_index
        ])
        
        # Initialize edge features array
        all_edge_features = np.zeros((num_objects, num_objects, num_edge_features))
        
        # Create mask for valid edges (excluding self-loops)
        mask = ~np.eye(num_objects, dtype=bool)
        
        # Compute entity 1 features
        for edge_feat_idx, node_feat_idx in ent1_feature_mapping:
            # Broadcasting node features to all edges
            all_edge_features[..., edge_feat_idx] = node_features[:, node_feat_idx:node_feat_idx+1]
        
        # Compute entity 2 features
        for edge_feat_idx, node_feat_idx in ent2_feature_mapping:
            # Broadcasting node features to all edges
            all_edge_features[..., edge_feat_idx] = node_features[:, node_feat_idx:node_feat_idx+1].T
        
        # Zero out self-loops
        all_edge_features[~mask] = 0
        
        return all_edge_features

    def _graphify_single_input(self, state: State) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)

        graph = {}

        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        node_features = np.zeros((num_objects, num_node_features))
        for feat_name, feat_index in self._node_feature_to_index.items():
            if "_qw" in feat_name:
                # rot initialize to qw=1
                node_features[:, feat_index] = 1

        ## Add node features for obj types.
        for obj in state:
            obj_index = object_to_node[obj]
            type_index = self._node_feature_to_index[f"type_{obj.type.name}"]
            node_features[obj_index, type_index] = 1

        ## Add node features for state.
        for obj in state:
            obj_index = object_to_node[obj]
            obj_type = obj.type.name
            for feat in obj.type.feature_names:
                if CFG.exclude_domain_feat is not None:
                    if obj_type in CFG.exclude_domain_feat.keys():
                        if feat in CFG.exclude_domain_feat[obj_type]:
                            continue
                feat_index = self._node_feature_to_index[f"feat_{feat}"]
                node_features[obj_index, feat_index] = state[obj][obj.type.feature_names.index(feat)]

        graph["nodes"] = node_features

        # Deal with edge case (pun).
        num_edge_features = max(num_edge_features, 1)

        # Add edges (one between each pair of objects) and edge features.
        all_edge_features = self.add_edge_features_fast(
            node_features, object_to_node, self._node_feature_to_index, self._edge_feature_to_index)

        # Organize into expected representation.
        # this should be an all-connected graph
        adjacency_mat = np.any(all_edge_features, axis=2)
        receivers, senders, edges = [], [], []
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)

        n_edge = len(edges)
        graph["edges"] = np.reshape(edges, [n_edge, num_edge_features])
        graph["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
        graph["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
        graph["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)

        # Add global features.
        graph["globals"] = None

        return graph, object_to_node

    def _graphify_single_output(self, action: _Option,
                                action_label: str,
                                curr_pred: Predicate,
                                ent_idx: List[int],
                                ae_vector: torch.Tensor,
                                next_state: int,
                                graph_input: Dict,
                                object_to_node: Dict) -> Dict:
        # First, copy over all unchanged fields.
        graph_target = {
            "n_node": graph_input["n_node"],
        }
        action_index = self.ae_row_names.index(action.parent)
        ae_vector_value = ae_vector[action_index]
        num_objects = len(object_to_node.keys())

        # add node features, assume all are ignored
        node_features = np.zeros((graph_target["n_node"], 1)) \
            + CFG.neupi_super_label['ignore']
        
        # pesudo unary predicates
        if curr_pred.arity == 1:
            type = curr_pred.types[0]
            ent = ent_idx[0]
            # get the ent id of each type in the action
            action_objects = action.objects
            action_ent = {
                type: []
            }
            for obj in action_objects:
                if obj.type == type:
                    action_ent[obj.type].append(obj)
            # find the operated object that is the argument in the pred
            # case one, action only has one object with the type
            if len(action_ent[type]) == 1:
                pred_type_obj = action_ent[type][0]
            # case two, action has multiple objects with the type
            elif len(action_ent[type]) > 1:
                # find the specified ent_idx object
                pred_type_obj = action_ent[type][ent]
            else:
                # action objs do not have enough objects for this predicate
                assert (ae_vector_value==0).all(), "Should be zero"
                pred_type_obj = len(object_to_node.keys()) + 1

            if ae_vector_value[0] == 1:
                # add effect
                # for operated object
                if next_state:
                    node_features[object_to_node[pred_type_obj]] \
                        = CFG.neupi_super_label['change_pos']
                else:
                    node_features[object_to_node[pred_type_obj]] \
                        = CFG.neupi_super_label['change_neg']
                # for non operated object
                for obj in object_to_node.keys():
                    if obj not in action_objects:
                        node_features[object_to_node[obj]] \
                            = CFG.neupi_super_label['non_change_2']
            elif ae_vector.shape[1] == 2 and ae_vector_value[1] == 1:
                # delete effect
                assert ae_vector_value[0] == 0, "Should be zero"
                if next_state:
                    node_features[object_to_node[pred_type_obj]] \
                        = CFG.neupi_super_label['change_neg']
                else:
                    node_features[object_to_node[pred_type_obj]] \
                        = CFG.neupi_super_label['change_pos']
                # for non operated object
                for obj in object_to_node.keys():
                    if obj not in action_objects:
                        node_features[object_to_node[obj]] \
                            = CFG.neupi_super_label['non_change_2']
            else:
                # non effect
                assert ae_vector_value[0] == 0, "Should be zero"
                if ae_vector.shape[1] == 2:
                    assert ae_vector_value[1] == 0, "Should be zero"
                for obj in object_to_node.keys():
                    if obj not in action_objects:
                        # non operated object
                        node_features[object_to_node[obj]] \
                            = CFG.neupi_super_label['non_change_1']
                    else:
                        # operated object, but not the operated type
                        node_features[object_to_node[obj]] \
                            = CFG.neupi_super_label['non_change_3']

            for obj in object_to_node.keys():
                # do not use type supervision, as predicates will automaticly handel this
                if obj.type != type:
                    node_features[object_to_node[obj]] \
                        = CFG.neupi_super_label['ignore']
        graph_target["nodes"] = node_features

        # add edge features
        all_edge_features = np.zeros(
            (num_objects, num_objects, 1)) \
            + CFG.neupi_super_label['ignore']
        
        ## Add edge features for pseudo binary atoms.
        if curr_pred.arity == 2:
            type0, type1 = curr_pred.types
            ent0, ent1 = ent_idx[0], ent_idx[1]
            action_objects = action.objects
            action_ent = {
                type0: [],
                type1: []
            }
            for obj in action_objects:
                if obj.type in action_ent:
                    action_ent[obj.type].append(obj)
            # find the operated object that is the argument in the pred
            if len(action_ent[type0]) <= ent0 or len(action_ent[type1]) <= ent1:
                # action objs do not have enough objects for this predicate
                assert (ae_vector_value==0).all(), "Should be zero"
                pred_type_obj0 = None
                pred_type_obj1 = None
            else:
                pred_type_obj0 = object_to_node[action_ent[type0][ent0]]
                pred_type_obj1 = object_to_node[action_ent[type1][ent1]]
            # note, here we don't consider "reverse" as this is specified in ent_idx
            if ae_vector_value[0] == 1:
                # add effect
                # for operated object pair
                if next_state:
                    all_edge_features[pred_type_obj0, \
                            pred_type_obj1] \
                        = CFG.neupi_super_label['change_pos']
                else:
                    all_edge_features[pred_type_obj0, \
                            pred_type_obj1] \
                        = CFG.neupi_super_label['change_neg']
                # for non operated object pairs
                for obj1 in object_to_node.keys():
                    for obj2 in object_to_node.keys():
                        # non operated object pairs
                        if (object_to_node[obj1] != pred_type_obj0) or \
                            (object_to_node[obj2] != pred_type_obj1):
                            all_edge_features[object_to_node[obj1], object_to_node[obj2]] \
                                = CFG.neupi_super_label['non_change_2']
            elif ae_vector.shape[1] == 2 and ae_vector_value[1] == 1:
                assert ae_vector_value[0] == 0, "Should be zero"
                # delete effect
                if next_state:
                    all_edge_features[pred_type_obj0, \
                            pred_type_obj1] \
                        = CFG.neupi_super_label['change_neg']
                else:
                    all_edge_features[pred_type_obj0, \
                            pred_type_obj1] \
                        = CFG.neupi_super_label['change_pos']
                # for non operated object pairs
                for obj1 in object_to_node.keys():
                    for obj2 in object_to_node.keys():
                        if (object_to_node[obj1] != pred_type_obj0) \
                            or (object_to_node[obj2] != pred_type_obj1):
                            all_edge_features[object_to_node[obj1], object_to_node[obj2]] \
                                = CFG.neupi_super_label['non_change_2']
            else:
                # non effect
                assert ae_vector_value[0] == 0, "Should be zero"
                if ae_vector.shape[1] == 2:
                    assert ae_vector_value[1] == 0, "Should be zero"
                for obj1 in object_to_node.keys():
                    for obj2 in object_to_node.keys():
                        if (obj1 not in action_objects) or (obj2 not in action_objects):
                            # non operated object pairs
                            all_edge_features[object_to_node[obj1], object_to_node[obj2]] \
                                = CFG.neupi_super_label['non_change_1']
                        else:
                            # operated object pairs
                            all_edge_features[object_to_node[obj1], object_to_node[obj2]] \
                                = CFG.neupi_super_label['non_change_3']
            
            for obj0 in object_to_node.keys():
                for obj1 in object_to_node.keys():
                    # do not use type supervision, as predicates will automaticly handel this
                    if obj0.type != type0 or obj1.type != type1:
                        all_edge_features[object_to_node[obj0], object_to_node[obj1]] \
                        = CFG.neupi_super_label['ignore']

        # Organize into expected representation for batched training.
        # this should be an all-connected graph
        adjacency_mat = np.any(all_edge_features, axis=2)
        receivers, senders, edges = [], [], []
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)

        n_edge = len(edges)
        graph_target["edges"] = np.reshape(edges, [n_edge, 1])
        graph_target["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
        graph_target["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
        graph_target["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)

        # Add global features.
        graph_target["globals"] = None

        # action info
        action_id, action_arity = action_label.split('_')
        action_id = int(action_id)
        action_arity = int(action_arity)
        assert action_arity <= self.max_action_arity, "Action arity should be less than max"
        # note that action is using tensor
        # padding, to avoid future bug if use -1
        action_node_ids = np.zeros(int(self.max_action_arity)).astype(np.int64) + num_objects + 1 
        for i, obj in enumerate(action.objects):
            action_node_ids[i] = object_to_node[obj]
        action_info = {
            "action_id": action_id,
            "action_arity": action_arity,
            "action_node_ids": action_node_ids
        }

        return graph_target, action_info
    
    def generate_classifier(self, pred: DummyPredicate, 
                            pred_model: torch.nn.Module,
                            pred_gumbel_temp: float,
                            pred_decision_b: float,
                            quantifier: str,
                            quantified_types: List[str],
                            quantified_types_idx: List[int],
                            negated: bool) -> Callable[[State, Sequence[Object]], bool]:
        def specific_function(state: State, objects: Sequence[Object]) -> bool:
            # Implement your function logic here
            state_graph, obj2node = self._graphify_single_input(state)
            if CFG.neupi_do_normalization:
                assert self._input_normalizers is not None, "Should have normalizers"
                state_graph = normalize_graph(state_graph, self._input_normalizers)
            if quantifier == None:
                # basic predicates always use 0.5 as threshold
                pred_binary_graph = get_single_neupi_prediction(pred_model, state_graph, \
                                                            pred_gumbel_temp, \
                                                            0.5, \
                                                            CFG.device)
            else:
                # quantified predicates use higher threshold
                pred_binary_graph = get_single_neupi_prediction(pred_model, state_graph, \
                                                            pred_gumbel_temp, \
                                                            pred_decision_b, \
                                                            CFG.device)
            if pred.arity == 1:
                assert len(objects) == 1, "Predicate arity should be 1"
                assert not quantified_types, "Should not have quantified types"                
                # a unary basic predicate
                obj = objects[0]
                node_index = obj2node[obj]
                attribute_value = pred_binary_graph["nodes"][node_index]
            elif pred.arity == 2:
                # get the specific predicate
                pred_type0, pred_type1 = pred.types
                pred_type0 = pred_type0.name
                pred_type1 = pred_type1.name
                if not quantified_types:
                    # a binary basic predicate
                    assert len(objects) == 2, "Predicate arity should be 2"
                    obj0, obj1 = objects
                    node0_index = [obj2node[obj0]]
                    node1_index = [obj2node[obj1]]
                else:
                    if len(quantified_types) == 1:
                        assert len(objects) == 1, "Predicate arity should be 1"
                        # use id, instead of type name, to avoid same type name
                        quantified_type_idx = quantified_types_idx[0]
                        node0_index = []
                        node1_index = []
                        if quantified_type_idx == 0:
                            for obj in state:
                                if obj.type.name == pred_type0:
                                    node0_index.append(obj2node[obj])
                            node1_index = [obj2node[objects[0]]]
                        elif quantified_type_idx == 1:
                            for obj in state:
                                if obj.type.name == pred_type1:
                                    node1_index.append(obj2node[obj])
                            node0_index = [obj2node[objects[0]]]
                        else:
                            raise ValueError("Unknown quantified type")
                    else:
                        assert len(quantified_types) == 2, "Should have two quantified types"
                        node0_index = []
                        node1_index = []
                        for obj in state:
                            if obj.type.name == pred_type0:
                                node0_index.append(obj2node[obj])
                            elif obj.type.name == pred_type1:
                                node1_index.append(obj2node[obj])

                selected_edges = []
                for n0, n1 in itertools.product(node0_index, node1_index):
                    if n0 == n1:
                        continue
                    selected_edges.append(pred_binary_graph["edges"][n0, n1])
                # add negation
                if len(selected_edges) == 0:
                    return False
                if negated:
                    selected_edges = 1 - np.stack(selected_edges)
                else:
                    selected_edges = np.stack(selected_edges)
                # add quantifier
                if quantifier in [None, 'ForAll']:
                    attribute_value = np.all(selected_edges, axis=0)
                else:
                    # Exists x P(x, y) = any comb P(x, y)
                    assert quantifier == 'Exists', "Should be Exists"
                    attribute_value = np.any(selected_edges, axis=0)
            return bool(attribute_value)
        return specific_function
            
    def gen_graph_data(self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom], _Option]],
                          current_pred: Predicate,
                          ent_idx: List[int],
                          ae_vector: torch.Tensor) -> List[Dict]:
        all_data = []
        transition_id = 0
        logging.info("Generating Graph Data with Current AE matrix...")
        if CFG.neupi_do_normalization:
            assert self._input_normalizers is not None, "Should have normalizers"
        if CFG.neupi_cache_input_graph:
            if not hasattr(self, 'cached_input_graph'):
                self.cached_input_graph = {}
        for state, atoms, state_, atoms_, action, action_label in tqdm(data):
            if CFG.neupi_cache_input_graph:
                if transition_id in self.cached_input_graph:
                    input_graph, object_to_node, input_graph_, object_to_node_ = \
                    self.cached_input_graph[transition_id]
                else:
                    # state s
                    input_graph, object_to_node = self._graphify_single_input(state)
                    if CFG.neupi_do_normalization:
                        input_graph = normalize_graph(input_graph, self._input_normalizers)
                    # state s'
                    input_graph_, object_to_node_ = self._graphify_single_input(state_)
                    if CFG.neupi_do_normalization:
                        input_graph_ = normalize_graph(input_graph_, self._input_normalizers)
                    assert object_to_node_ == object_to_node
                    self.cached_input_graph[transition_id] = (input_graph, object_to_node, \
                                                            input_graph_, object_to_node_)
            else:
                # state s
                input_graph, object_to_node = self._graphify_single_input(state)
                if CFG.neupi_do_normalization:
                    input_graph = normalize_graph(input_graph, self._input_normalizers)
                # state s'
                input_graph_, object_to_node_ = self._graphify_single_input(state_)
                if CFG.neupi_do_normalization:
                    input_graph_ = normalize_graph(input_graph_, self._input_normalizers)
                assert object_to_node_ == object_to_node
            # partitioned discrete graph (invented predicates)
            target_graph, action_info = self._graphify_single_output(
                action, action_label, current_pred, ent_idx, ae_vector, 0, input_graph, object_to_node)
            # partitioned discrete graph (invented predicates)
            target_graph_, _ = self._graphify_single_output(
                action, action_label, current_pred, ent_idx, ae_vector, 1, input_graph_, object_to_node)
            data_dict = {
                'input': input_graph,
                'target': target_graph,
                'input_': input_graph_,
                'target_': target_graph_,
                'action_info': action_info
            }
            all_data.append(data_dict)
            transition_id += 1
        if not CFG.neupi_cache_input_graph:
            self._rng.shuffle(all_data)
        train_dataset = GraphTransC2DDataset(all_data[:int(len(all_data)*0.8)])
        val_dataset = GraphTransC2DDataset(all_data[int(len(all_data)*0.8):])

        return train_dataset, val_dataset

    def learn_neural_predicates(
        self, dataset: Dataset
    ) -> Tuple[List[GroundAtomTrajectory], Dict[Predicate, float]]:
        """Learn the Neural predicates by Action Effect Martix Identification."""
        logging.info("Constructing NeuPi Data...")
        # 1. Generate data from the dataset. This is general
        data, trajectories, init_atom_traj = self._generate_data_from_dataset(dataset)
        # 2. Setup the input fields for the neural predicate, this is general
        self._setup_input_fields(data)
        # 3. Get the initial constraints (for all types of predicates)
        self.initialize_ae_constraints(data, CFG.neupi_ae_matrix_channel)
        # compute normalizer if needed (for all types of predicates)
        if CFG.neupi_do_normalization:
            graph_inputs = []
            logging.info("Computing Normalizer for Inouts...")
            if self._input_normalizers is not None:
                logging.info("Normalizer already computed!")
            else:
                if CFG.max_normalizer_data == -1:
                    # use_all
                    num_data = len(data)
                else:
                    logging.info(f"Using {CFG.max_normalizer_data} data for normalizer computation")
                    num_data = CFG.max_normalizer_data
                for state, atoms, state_, atoms_, action, action_label in tqdm(data[:num_data]):
                    input_graph, object_to_node = self._graphify_single_input(state)
                    input_graph_, object_to_node_ = self._graphify_single_input(state_)
                    graph_inputs.append(input_graph)
                    graph_inputs.append(input_graph_)
                self._input_normalizers = compute_normalizers(graph_inputs, normalize_nodes=self._node_is_rot, \
                                                            normalize_edges=self._edge_is_rot,
                                                            normalize_globals=False)
        # 4. Start Learning the typed columns one by one
        for curr_pred in list(sorted(self.learned_ae_pred_info.keys(), key=lambda p: p.arity, reverse=True)):
            if self.learned_ae_pred_info[curr_pred]['provided']:
                # this is learned/provided, directly generate the vectors
                logging.info(f"Skipping learning for {curr_pred.name} since it is already provided! Generate vectors for it!")
                ae_vectors = self.gen_sat_vec(curr_pred, 1, 100)
                assert len(ae_vectors) == 1, "Should find exactly one vector for provided predicate"
                ae_vector = ae_vectors[0]
                logging.info(f"AE Vector (Add): {ae_vector[:, 0]}")
                if ae_vector.shape[1] == 2:
                    logging.info(f"AE Vector (Del): {ae_vector[:, 1]}")
                self.learned_ae_pred_info[curr_pred]['ae_vecs'].append(ae_vector.clone())
                continue
            pred_config = {}
            for config in self.pred_config:
                if config["name"] == curr_pred.name:
                    pred_config = config
            assert pred_config, "Should have the config for the current predicate"
            logging.info(f"**************Learning Typed Predicate: {curr_pred.name}**************")
            logging.info(f"Learning Config: {pred_config}")
            assert len(self.learned_ae_pred_info[curr_pred]['ent_idx']) == 1
            ent_idx = self.learned_ae_pred_info[curr_pred]['ent_idx'][0]
            # 4.1 Setup the Neural Model and Optimizers
            # state s
            if CFG.exclude_domain_feat is not None:
                logging.info("Excluding Domain Features: {}".format(CFG.exclude_domain_feat))
            example_input, example_object_to_node = self._graphify_single_input(
                data[0][0])
            # state s'
            example_input_, example_object_to_node_ = self._graphify_single_input(
                data[0][2])
            # mapping should be the same for the same transition pair
            # but could be different for different transition pairs/trajs (entities can be different)
            assert example_object_to_node_ == example_object_to_node
            # This is a dummy vector, should be replaced by the real one
            dummy_vector = torch.zeros((len(self.ae_row_names), 2))
            # expected output for s s', action a
            example_target, example_action_info = self._graphify_single_output(
                                                        data[0][4],
                                                        data[0][5],
                                                        curr_pred,
                                                        ent_idx,
                                                        dummy_vector,
                                                        0,
                                                        example_input,
                                                        example_object_to_node)
            example_target_, _ = self._graphify_single_output(
                                                    data[0][4],
                                                    data[0][5],
                                                    curr_pred,
                                                    ent_idx,
                                                    dummy_vector,
                                                    1,
                                                    example_input_,
                                                    example_object_to_node)
            data_example = {
                'input': example_input,
                'target': example_target,
                'input_': example_input_,
                'target_': example_target_,
                'action_info': example_action_info
            }

            example_dataset = GraphTransC2DDataset([data_example])
            self.learned_ae_pred_info[curr_pred]['example_dataset'] = copy.deepcopy(example_dataset)
            predicate_neural_model = setup_neupi_mlp_net(example_dataset,
                                        curr_pred.arity,
                                        pred_config['architecture'],
                                        self._node_feature_to_index,
                                        self._edge_feature_to_index)
            symbolic_search_model = HierachicalMCTSearcher(
                                            len(self.ae_row_names),
                                            pred_config['search_tree_max_level'], \
                                            pred_config['guidance_thresh'])
            self.learned_ae_pred_info[curr_pred]['model'] = predicate_neural_model # save the model template
            # The predicate is trained and will be skipped
            # Or, we directly load it
            if os.path.exists(CFG.neupi_load_pretrained) or pred_config["skip_train"]:
                iteration = -1
                logging.info(f"Skipping training for {curr_pred}")
                if os.path.exists(CFG.neupi_load_pretrained):
                    logging.info(f"Loading pretrained model from {CFG.neupi_load_pretrained}")
                    pred_model_paths = os.path.join(CFG.neupi_load_pretrained, curr_pred.name, "*_model_good.pth")
                else:
                    assert os.path.exists(os.path.join(CFG.neupi_save_path, curr_pred.name)), "Should have the saved model"
                    logging.info(f"Loading pretrained model from {CFG.neupi_save_path}")
                    pred_model_paths = os.path.join(CFG.neupi_save_path, curr_pred.name, "*_model_good.pth")
                if len(glob.glob(pred_model_paths)) == 0:
                    logging.info(f"WRINING!!! Pretrained model not found for {curr_pred}, skipping...")
                    self.learned_ae_pred_info[curr_pred]['learned'] = True
                else:
                    # importantly, ordering the filenames as predicates
                    def parse_filename(filename):
                        # Extracting the integers from a filename like: iter_m_n_model_good.pth
                        base = os.path.basename(filename)
                        # Assuming the filename always follows the pattern: iter_<m>_<n>_model_good.pth
                        parts = base.split('_')
                        # parts = ['iter', 'm', 'n', 'model', 'good.pth']
                        m = int(parts[1])
                        n = int(parts[2])
                        return m, n
                    sorted_paths = sorted(glob.glob(pred_model_paths), key=parse_filename)
                    for pred_model_path in sorted_paths:
                        logging.info(f"Loading pretrained model from {pred_model_path}")
                        ae_vector = _load_tensor(
                            self._model_artifact_path(pred_model_path,
                                                      "ae_vector"))
                        ae_vector_guidance = _load_tensor(
                            self._model_artifact_path(pred_model_path,
                                                      "guidance"))
                        self.learned_ae_pred_info[curr_pred]['ae_vecs'].append(ae_vector.clone())
                        self.learned_ae_pred_info[curr_pred]['scores'].append(ae_vector_guidance.clone())
                        self.learned_ae_pred_info[curr_pred]['model_weights'].append(pred_model_path)
                    self.learned_ae_pred_info[curr_pred]['learned'] = True
            else:
                # 4.2 Start the Bi-level Optimization Process for Training
                logging.info("Training from scratch.")
                ## Launch training code.
                pred_save_path = os.path.join(CFG.neupi_save_path, curr_pred.name)
                if not os.path.exists(pred_save_path):
                    os.makedirs(pred_save_path)
                # Step1: Get a bunch of initial ae vector - score pairs
                logging.info(f"***************Bi-level Optimizing ({curr_pred.name})***************")
                # Optional cap on the number of vectors kept for this type.
                max_num_vectors = pred_config.get("max_num_vectors")
                for iteration in range(pred_config['num_iter']):
                    if max_num_vectors is not None and len(
                            self.learned_ae_pred_info[curr_pred]
                        ['ae_vecs']) >= max_num_vectors:
                        logging.info(f"Kept {max_num_vectors} vectors for "
                                     f"{curr_pred.name}; stopping.")
                        break
                    logging.info(f"-----Iteration: {iteration} ({curr_pred.name})------")
                    s_time = time.time()
                    iter_ae_vectors, iter_guidance_vecs, iter_val_loss, model_weight_paths = \
                        self.gen_ae_vectors4pred(iteration, curr_pred, ent_idx, pred_save_path, \
                                                pred_config, data, symbolic_search_model)
                    if len(model_weight_paths) == 0:
                        logging.info(f"Early Stopping at Iteration {iteration}!")
                        break
                    # Do we have a low-objective ae vector?
                    logging.info(f"Learning Done in {time.time()-s_time} sec, Checking if exists low-objective ae vector...")
                    all_iter_objective = torch.stack(iter_guidance_vecs, dim=0)
                    all_iter_objective = all_iter_objective.sum(dim=1)
                    all_iter_losses = torch.tensor(iter_val_loss)
                    # checking both objective and loss, loss also consideres non-change groundings
                    all_pass_mask_obj = all_iter_objective < pred_config["guidance_thresh"]
                    all_pass_mask_loss = all_iter_losses < pred_config["loss_thresh"]
                    all_pass_mask = all_pass_mask_obj & all_pass_mask_loss
                    i = 0
                    if all_pass_mask.any():
                        # find out their idxes and check if they pass the preconditions
                        logging.info(f"Found {all_pass_mask.sum()} init low-objective ae vectors!")
                        min_idxes = torch.where(all_pass_mask)[0]
                        for min_idx in min_idxes:
                            if max_num_vectors is not None and len(
                                    self.learned_ae_pred_info[curr_pred]
                                ['ae_vecs']) >= max_num_vectors:
                                logging.info(
                                    f"Kept {max_num_vectors} vectors for "
                                    f"{curr_pred.name}; skipping the "
                                    "remaining passed vectors.")
                                break
                            logging.info(f"Checking AE Vector {min_idx}...")
                            predicate_neural_model.load_state_dict(
                                _load_tensor(model_weight_paths[min_idx]))
                            ae_vector = iter_ae_vectors[min_idx].clone()
                            _, val_dataset = self.gen_graph_data(data, 
                                                                curr_pred, 
                                                                ent_idx,
                                                                ae_vector)
                            val_dataloader = DataLoader(val_dataset,
                                                        batch_size=pred_config['batch_size'],
                                                        shuffle=False,
                                                        num_workers=1,
                                                        collate_fn=action_graph_batch_collate)
                            # basic predicate always use 0.5 as threshold
                            precond_pass = check_learned_ap_vector(val_dataloader, \
                            pred_config['gumbel_temp'], 0.5, predicate_neural_model, curr_pred, ent_idx, \
                                self.ae_row_names_dict, ae_vector.clone(), copy.deepcopy(self._node_feature_to_index), CFG.device, \
                                thresh=CFG.precond_thresh)
                            if precond_pass:
                                i += 1
                                logging.info(f"Precondition Passed for AE Vector {min_idx}! Add to the final list.")
                                logging.info(f"AE Vector {min_idx} (Add): {ae_vector[:, 0]}")
                                if ae_vector.shape[1] == 2:
                                    logging.info(f"AE Vector {min_idx} (Del): {ae_vector[:, 1]}")
                                self.learned_ae_pred_info[curr_pred]['ae_vecs'].append(ae_vector.clone())
                                self.learned_ae_pred_info[curr_pred]['scores'].append(iter_guidance_vecs[min_idx].clone())
                                self.learned_ae_pred_info[curr_pred]['model_weights'].append(model_weight_paths[min_idx])
                    else:
                        logging.info(f"No low-objective ae vectors found for this iteration..")
                    # save the neural feedback for symbolic training
                    # convert to categorical
                    for curr_ae_vector, learned_guidance in zip(iter_ae_vectors, iter_guidance_vecs):
                        data_x_new = two2one(curr_ae_vector)
                        state = data_x_new.numpy()
                        value = learned_guidance.numpy()
                        symbolic_search_model.update_value(state, value)

            logging.info(f"******************Bi-level Optimization Done for {curr_pred.name}! Summary:******************")
            logging.info(f"After {iteration} iterations, we got {len(self.learned_ae_pred_info[curr_pred]['ae_vecs'])} basic vectors:")
            self.learned_ae_pred_info[curr_pred]['learned'] = True
            predicate_vars_basic = []
            for i, t in enumerate(curr_pred.types):
                # ent_idx is the index of the operated object, in the operator
                predicate_vars_basic.append(f"?{t.name[:3].lower()}{ent_idx[i]}")
            # group the learned predicates
            num_basic_pred = len(self.learned_ae_pred_info[curr_pred]['model_weights'])
            all_ae_vectors = copy.deepcopy(self.learned_ae_pred_info[curr_pred]['ae_vecs'])
            all_ae_scores = copy.deepcopy(self.learned_ae_pred_info[curr_pred]['scores'])
            basic_ent_idx = copy.deepcopy(ent_idx)
            self.learned_ae_pred_info[curr_pred]['ae_vecs'] = [[] for _ in range(num_basic_pred)]
            self.learned_ae_pred_info[curr_pred]['scores'] = [[] for _ in range(num_basic_pred)]
            self.learned_ae_pred_info[curr_pred]['quantifiers'] = [[] for _ in range(num_basic_pred)]
            self.learned_ae_pred_info[curr_pred]['ent_idx'] = [[] for _ in range(num_basic_pred)]
            # group the learned predicates
            for i in range(num_basic_pred):
                curr_ae_vector = all_ae_vectors[i]
                self.learned_ae_pred_info[curr_pred]['ae_vecs'][i].append(curr_ae_vector)
                curr_guidance = all_ae_scores[i]
                self.learned_ae_pred_info[curr_pred]['scores'][i].append(curr_guidance)
                basic_pred_name = f"{curr_pred.name}_{i}({','.join(predicate_vars_basic)})"
                logging.info(f"******************Predicate {basic_pred_name}******************")
                logging.info(f"AE Vector {i} (Add): {curr_ae_vector[:, 0]}")
                if curr_ae_vector.shape[1] == 2:
                    logging.info(f"AE Vector {i} (Del): {curr_ae_vector[:, 1]}")
                logging.info(f"Guidance Score: {curr_guidance}") 
                self.learned_ae_pred_info[curr_pred]['quantifiers'][i].append(('', '', basic_pred_name))
                self.learned_ae_pred_info[curr_pred]['ent_idx'][i].append(basic_ent_idx)
                # rename the model weights paths with "good" marker
                if iteration >= 0:
                    saved_model_path = self.learned_ae_pred_info[curr_pred]['model_weights'][i]
                    for artifact in ("model", "guidance", "ae_vector"):
                        os.rename(
                            self._model_artifact_path(saved_model_path,
                                                      artifact),
                            self._model_artifact_path(saved_model_path,
                                                      f"{artifact}_good"))
                    self.learned_ae_pred_info[curr_pred]['model_weights'][i] = \
                        self._model_artifact_path(saved_model_path,
                                                  "model_good")
                # 4.3 Check the Negated / Quantified version (a predicate
                # config can turn this off with generate_derived: False)
                if (CFG.neupi_w_negation or CFG.neupi_w_quantifiers) and \
                        pred_config.get("generate_derived", True):
                    if curr_pred.arity == 1:
                        logging.info("Skipping Negated / Quantified version for unary predicate...")
                        # only support binary + quantifier now
                        continue
                    if os.path.exists(self.learned_ae_pred_info[curr_pred]['model_weights'][i].replace("_model_good.pth", "quantified.pkl")):
                        logging.info("Negated / Quantified version already exists, loading...")
                        with open(self.learned_ae_pred_info[curr_pred]['model_weights'][i].replace("_model_good.pth", "quantified.pkl"), 'rb') as f:
                            quantified_info = pkl.load(f)
                        for m in range(1, len(quantified_info['ent_idx'])):
                            logging.info(f"Negated / Quantified version {m}...")
                            logging.info(f"Quantifier: {quantified_info['quantifiers'][m]}")
                            logging.info(f"AE Vector {m}: {two2one(quantified_info['ae_vecs'][m])}")
                            # skipping the first one, as it is the original one
                            self.learned_ae_pred_info[curr_pred]['ent_idx'][i].append(quantified_info['ent_idx'][m])
                            self.learned_ae_pred_info[curr_pred]['ae_vecs'][i].append(quantified_info['ae_vecs'][m])
                            self.learned_ae_pred_info[curr_pred]['scores'][i].append(quantified_info['scores'][m])
                            self.learned_ae_pred_info[curr_pred]['quantifiers'][i].append(quantified_info['quantifiers'][m])
                        continue
                    logging.info("Checking Negated / Quantified version...")
                    curr_pred_types = [types.name for types in curr_pred.types]
                    typed_quantifier = []
                    w_negation = [False] if not CFG.neupi_w_negation else [True, False]
                    quantifiers = [''] if not CFG.neupi_w_quantifiers else ['', 'ForAll', 'Exists']
                    pred_model_weights = _load_tensor(
                        self.learned_ae_pred_info[curr_pred]['model_weights'][i])
                    pred_model = setup_neupi_mlp_net(example_dataset,
                                curr_pred.arity,
                                pred_config['architecture'],
                                self._node_feature_to_index,
                                self._edge_feature_to_index)
                    pred_model.load_state_dict(pred_model_weights)
                    if CFG.neupi_quantify_dataset != 1.0:
                        logging.info(f"Quantifying Dataset: {CFG.neupi_quantify_dataset}")
                        self._rng.shuffle(data)
                    quantify_data = data[:int(len(data) * CFG.neupi_quantify_dataset)]
                    _, val_dataset = self.gen_graph_data(quantify_data, curr_pred, ent_idx, \
                    curr_ae_vector.clone())
                    val_dataloader = DataLoader(val_dataset,
                        batch_size=1, # always use 1 for memory
                        shuffle=False,
                        num_workers=4,
                        pin_memory=False,
                        collate_fn=action_graph_batch_collate)
                    # just inference once and use the outputs
                    logging.info("Inference for the current predicate model...")
                    data_results = inferece_dataloader(val_dataloader, pred_model, \
                                                       pred_config['gumbel_temp'], \
                                                       pred_config['decision_b'], \
                                                       CFG.device)
                    quantified_threshold = pred_config['quantified_guidance_thresh'] if "quantified_guidance_thresh" \
                                            in pred_config else pred_config['guidance_thresh']
                    for neg in w_negation:
                        logging.info(f"Negation: {neg}")
                        for quantifier in quantifiers:
                            if quantifier == '':
                                local_ent_idx = ent_idx
                                logging.info(f"No Quantifier, using the original ent idx: {local_ent_idx}")
                                if not neg:
                                    continue
                                type_comb = []
                                local_name = gen_pretty_pred_name(basic_pred_name, type_comb, \
                                                                quantifier, neg)
                                logging.info(f" ++++{local_name}++++")
                                typed_quantifier.append((local_name))
                                learned_ae_vector = distill_learned_ae_vector(data_results, \
                                    pred_config['gumbel_temp'], pred_config['decision_b'], pred_model, curr_pred, local_ent_idx, \
                                    self.ae_row_names_dict, copy.deepcopy(self._node_feature_to_index), \
                                        CFG.device, neg, quantifier, type_comb)
                                # no target vector will use entropy
                                learned_guidance = compute_guidance_vector(learned_ae_vector)
                                logging.info(f"Guidance Score: {learned_guidance}")
                                if learned_guidance.sum() < quantified_threshold:
                                    final_ae_vec = prob_three2two(learned_ae_vector)
                                    if (final_ae_vec==0).all():
                                        logging.info(f"AE Vector is all zeros, skip...")
                                        continue
                                    logging.info(f"Checking AP Vector for {local_name}...")
                                    precond_pass = check_learned_ap_vector(data_results, \
                                    pred_config['gumbel_temp'], pred_config['decision_b'], predicate_neural_model, curr_pred, local_ent_idx, \
                                        self.ae_row_names_dict, final_ae_vec, copy.deepcopy(self._node_feature_to_index), \
                                        CFG.device, neg, quantifier, type_comb, thresh=CFG.precond_thresh)
                                    if precond_pass:
                                        logging.info(f"Passed!!! Add to the final list.")
                                        self.learned_ae_pred_info[curr_pred]['ent_idx'][i].append(local_ent_idx)
                                        self.learned_ae_pred_info[curr_pred]['ae_vecs'][i].append(final_ae_vec.clone())
                                        self.learned_ae_pred_info[curr_pred]['scores'][i].append(learned_guidance.clone())
                                        quantifier_str = local_name.split(', ')[0]
                                        neg_str = local_name.split(', ')[1].split('-')[0]
                                        basic_pred_str = local_name.split(', ')[1].split('-')[1]
                                        self.learned_ae_pred_info[curr_pred]['quantifiers'][i].append((quantifier_str, \
                                                                        neg_str, basic_pred_str))
                            else:
                                # use idx, instead of content, to handel same name type
                                combinded_types_idx = []
                                curr_pred_types_ids = list(range(len(curr_pred_types)))
                                for r in range(1, len(curr_pred_types) + 1):
                                    combinded_types_idx += list(itertools.combinations(curr_pred_types_ids, r))
                                for type_comb_idx in combinded_types_idx:
                                    # who to quantify
                                    type_comb = [curr_pred_types[t] for t in type_comb_idx]
                                    predicate_vars = []
                                    possible_ent_idx = []
                                    for k, t in enumerate(curr_pred.types):
                                        if k in type_comb_idx:
                                            # this type is in quantifier, no id needed
                                            predicate_vars.append(f"?{t.name[:3].lower()}")
                                        else:
                                            # This type is not quantified, need to specify the ent id in operator
                                            # ent_idx is the index of the operated object, in the operator
                                            predicate_vars.append(f"?{t.name[:3].lower()}N")
                                            possible_ent_idx = list(range(self._max_type_id[t.name]))
                                    if len(possible_ent_idx) == 0:
                                        logging.info(f"The unquantified type has no choice")
                                        basic_pred_name = f"{curr_pred.name}_{i}({','.join(predicate_vars)})"
                                        # reset the local ent idx, as this is a new predicate
                                        local_ent_idx = []
                                        assert "N" not in basic_pred_name, "Should not have N"
                                        local_name = gen_pretty_pred_name(basic_pred_name, type_comb, \
                                                                        quantifier, neg)
                                        logging.info(f" ++++{local_name}++++")
                                        typed_quantifier.append((local_name))
                                        learned_ae_vector = distill_learned_ae_vector(data_results, \
                                                pred_config['gumbel_temp'], pred_config['decision_b'], pred_model, curr_pred, local_ent_idx, \
                                                self.ae_row_names_dict, copy.deepcopy(self._node_feature_to_index), \
                                                    CFG.device, neg, quantifier, type_comb_idx)
                                        # no target vector will use entropy
                                        learned_guidance = compute_guidance_vector(learned_ae_vector)
                                        logging.info(f"Guidance Score: {learned_guidance}")
                                        if learned_guidance.sum() < quantified_threshold:
                                            final_ae_vec = prob_three2two(learned_ae_vector)
                                            logging.info(f"Checking AP Vector for {local_name}...")
                                            precond_pass = check_learned_ap_vector(data_results, \
                                            pred_config['gumbel_temp'], pred_config['decision_b'], predicate_neural_model, curr_pred, local_ent_idx, \
                                            self.ae_row_names_dict, final_ae_vec, copy.deepcopy(self._node_feature_to_index),
                                            CFG.device, neg, quantifier, type_comb_idx, thresh=CFG.precond_thresh)
                                            if (final_ae_vec==0).all():
                                                logging.info(f"AE Vector is all zeros, skip...")
                                                precond_pass = False
                                            if precond_pass:
                                                logging.info(f"Passed!!! Add to the final list.")
                                                self.learned_ae_pred_info[curr_pred]['ent_idx'][i].append(local_ent_idx)
                                                self.learned_ae_pred_info[curr_pred]['ae_vecs'][i].append(final_ae_vec.clone())
                                                self.learned_ae_pred_info[curr_pred]['scores'][i].append(learned_guidance.clone())
                                                quantifier_str = local_name.split(', ')[0]
                                                neg_str = local_name.split(', ')[1].split('-')[0]
                                                basic_pred_str = local_name.split(', ')[1].split('-')[1]
                                                self.learned_ae_pred_info[curr_pred]['quantifiers'][i].append((quantifier_str, \
                                                                                neg_str, basic_pred_str))
                                    else:
                                        logging.info(f"The unquantified type has >= one choice, checking each one of them...")
                                        for loc_ent in possible_ent_idx:
                                            # reset the local ent idx, as this is a new predicate
                                            # note that ent idx is the index in the action
                                            local_ent_idx = [loc_ent]
                                            basic_pred_name = f"{curr_pred.name}_{i}({','.join(predicate_vars)})"
                                            assert "N" in basic_pred_name, "Should have N"
                                            basic_pred_name = basic_pred_name.replace('N', str(loc_ent))
                                            local_name = gen_pretty_pred_name(basic_pred_name, type_comb, \
                                                                            quantifier, neg)
                                            logging.info(f" ++++{local_name}++++")
                                            typed_quantifier.append((local_name))
                                            learned_ae_vector = distill_learned_ae_vector(data_results, \
                                                    pred_config['gumbel_temp'], pred_config['decision_b'], pred_model, curr_pred, local_ent_idx, \
                                                    self.ae_row_names_dict, copy.deepcopy(self._node_feature_to_index), \
                                                        CFG.device, neg, quantifier, type_comb_idx)
                                            # no target vector will use entropy
                                            learned_guidance = compute_guidance_vector(learned_ae_vector)
                                            logging.info(f"Guidance Score: {learned_guidance}")
                                            if learned_guidance.sum() < quantified_threshold:
                                                final_ae_vec = prob_three2two(learned_ae_vector)
                                                logging.info(f"Checking AP Vector for {local_name}...")
                                                precond_pass = check_learned_ap_vector(data_results, \
                                                pred_config['gumbel_temp'], pred_config['decision_b'], pred_model, curr_pred, local_ent_idx, \
                                                self.ae_row_names_dict, final_ae_vec, copy.deepcopy(self._node_feature_to_index),
                                                CFG.device, neg, quantifier, type_comb_idx, thresh=CFG.precond_thresh)
                                                if (final_ae_vec==0).all():
                                                    logging.info(f"AE Vector is all zeros, skip...")
                                                    precond_pass = False
                                                if precond_pass:
                                                    logging.info(f"Passed!!! Add to the final list.")
                                                    self.learned_ae_pred_info[curr_pred]['ent_idx'][i].append(local_ent_idx)
                                                    self.learned_ae_pred_info[curr_pred]['ae_vecs'][i].append(final_ae_vec.clone())
                                                    self.learned_ae_pred_info[curr_pred]['scores'][i].append(learned_guidance.clone())
                                                    quantifier_str = local_name.split(', ')[0]
                                                    neg_str = local_name.split(', ')[1].split('-')[0]
                                                    basic_pred_str = local_name.split(', ')[1].split('-')[1]
                                                    self.learned_ae_pred_info[curr_pred]['quantifiers'][i].append((quantifier_str, \
                                                                                    neg_str, basic_pred_str))
                    logging.info(f"Savings Negated / Quantified version for {curr_pred.name}...")
                    with open(self.learned_ae_pred_info[curr_pred]['model_weights'][i].replace("_model_good.pth", "quantified.pkl"), 'wb') as f:
                        pkl.dump({'ent_idx': self.learned_ae_pred_info[curr_pred]['ent_idx'][i],
                                  'ae_vecs': self.learned_ae_pred_info[curr_pred]['ae_vecs'][i],
                                  'scores': self.learned_ae_pred_info[curr_pred]['scores'][i],
                                  'quantifiers': self.learned_ae_pred_info[curr_pred]['quantifiers'][i]}, f)
        return trajectories, init_atom_traj

    def _get_predicate_identifier(
        self, traj: List ,predicate: Predicate
    ) -> FrozenSet[Tuple[int, int, FrozenSet[Tuple[Object, ...]]]]:
        """Returns frozenset identifiers for each data point."""
        raw_identifiers = set()
        # Get atoms for this predicate alone on the dataset, and then
        # go through the entire dataset.
        atom_dataset = utils.create_ground_atom_dataset(
            traj, {predicate})
        for traj_idx, (_, atom_traj) in enumerate(atom_dataset):
            for t, atoms in enumerate(atom_traj):
                atom_args = frozenset(tuple(a.objects) for a in atoms)
                raw_identifiers.add((traj_idx, t, atom_args))
        return frozenset(raw_identifiers)
    
    def _get_invented_pruned_predicates(self, traj: List) -> List[Predicate]:
        colind2pred_info = {}
        huge_ae_matrix_invented = []
        # track equivalence
        seen: Dict[FrozenSet[Tuple[int, int, FrozenSet[Tuple[Object, ...]]]],
                   Predicate] = {}  # keys are from _get_predicate_identifier()
        for pred in self.learned_ae_pred_info.keys():
            assert self.learned_ae_pred_info[pred]['learned']
            ae_vec_groups = self.learned_ae_pred_info[pred]['ae_vecs']
            if self.learned_ae_pred_info[pred]['provided']:
                ae_vec = ae_vec_groups[0]
                huge_ae_matrix_invented.append(ae_vec.clone())
                pred_ent_idx = self.learned_ae_pred_info[pred]['ent_idx'][0]
                colind2pred_info[len(huge_ae_matrix_invented)-1] = (pred, pred_ent_idx, True, (pred, 0, 0))
                continue
            example_dataset = self.learned_ae_pred_info[pred]['example_dataset']
            for item in self.pred_config:
                if item['name'] == pred.name:
                    pred_config = copy.deepcopy(item)

            for i, ae_vec_group in enumerate(ae_vec_groups):
                for m, ae_vec in enumerate(ae_vec_group):
                    huge_ae_matrix_invented.append(ae_vec.clone())
                    # construct the real predicate!
                    name = ""
                    quantifier_str, neg_str, basic_pred_str = \
                        self.learned_ae_pred_info[pred]['quantifiers'][i][m]
                    original_types = pred.types
                    original_type_names = [t.name for t in original_types]
                    quantified_type_idx, local_ent_idx = parse_basic_pred_name(basic_pred_str, \
                                                                    original_type_names)
                    assert (np.array(local_ent_idx)==np.array(self.learned_ae_pred_info[pred]['ent_idx'][i][m])).all()
                    quantifier, quantified_types = None, None
                    neg = False
                    if "?" in quantifier_str:
                        quantifier, quantified_types = name2quantifier_types(quantifier_str, list(self._obj_types.keys()))
                        name += quantifier.lower()
                        name += '_'
                        for r, t in enumerate(quantified_types):
                            pos_id = quantified_type_idx[r]
                            # no pos id to avoid confusion
                            name += f"{t[:3].lower()}_"
                    if neg_str:
                        neg = True
                        name += neg_str + "_"
                    name += basic_pred_str
                    logging.info(f"Constructing Neural Predicate: {name}")
                    logging.info(f"Its Effect Vector: {two2one(ae_vec.clone())}")
                    original_types = pred.types
                    pred_model = setup_neupi_mlp_net(example_dataset,
                                pred.arity,
                                pred_config['architecture'],
                                self._node_feature_to_index,
                                self._edge_feature_to_index)
                    # same group shares the same model
                    weights = _load_tensor(
                        self.learned_ae_pred_info[pred]['model_weights'][i])
                    pred_model.load_state_dict(weights)
                    # note that grounding this predicate do not need the operatr ent idx, just ground all
                    specific_function = self.generate_classifier(pred, pred_model, pred_config['gumbel_temp'], \
                                                                 pred_config['decision_b'], \
                                                        quantifier, quantified_types, quantified_type_idx, neg)
                    types = []
                    if quantified_types is not None:
                        # quantified types are no longer input types
                        for q, t in enumerate(original_types):
                            if q in quantified_type_idx:
                                continue
                            types.append(t)
                    else:
                        types = original_types
                    # do a renaming as FD has naming requirements
                    basic_name = name.split('(')[0]
                    arguments = name.split('(')[1].split(')')[0].split(',')
                    new_arguments = [arg.replace('?', '') for arg in arguments]
                    name = basic_name + '_' + '_'.join(new_arguments)
                    neual_pred = NeuralPredicate(name=name, types=types, 
                                                _classifier=specific_function)
                    pred_id = self._get_predicate_identifier(traj, neual_pred)
                    if pred_id in seen:
                        logging.info(f"Pruning {name} b/c equal to "
                                f"{seen[pred_id]}")
                        continue
                    seen[pred_id] = name
                    colind2pred_info[len(huge_ae_matrix_invented)-1] = (neual_pred, local_ent_idx, False, (pred, i, m))

        huge_ae_matrix_invented = torch.stack(huge_ae_matrix_invented, dim=1)
        return huge_ae_matrix_invented, colind2pred_info

    def _get_invented_predicates_from_info(self, info: Dict, from_json=False) -> \
        Tuple[Dict, List, torch.Tensor, List]:
        dummy2real = {}
        ae_col_names = []
        ae_matrix = []
        ae_ent_idx = []
        for pred_temp in info.keys():
            if pred_temp in self._initial_predicates:
                ae_col_names.append(pred_temp)
                ae_vec_one = self.learned_ae_pred_info[pred_temp]['ae_vecs'][0]
                ae_vec_two = one2two(torch.tensor(ae_vec_one), 2)
                ae_matrix.append(ae_vec_two)
                ae_ent_idx.append(self.learned_ae_pred_info[pred_temp]['ent_idx'][0])
                continue
            assert pred_temp in self.learned_ae_pred_info, "Should have learned this predicate"
            # There could be multiple specific predicate in this template
            logging.info(f"Template {pred_temp} has {len(info[pred_temp])} specific variants.")
            for k in range(len(info[pred_temp])):
                i, m = info[pred_temp][k][1], info[pred_temp][k][2]
                for item in self.pred_config:
                    if item['name'] == pred_temp.name:
                        pred_config = copy.deepcopy(item)
                # construct the real predicate!
                name = ""
                quantifier_str, neg_str, basic_pred_str = \
                    self.learned_ae_pred_info[pred_temp]['quantifiers'][i][m]
                original_types = pred_temp.types
                original_type_names = [t.name for t in original_types]
                quantified_type_idx, local_ent_idx = parse_basic_pred_name(basic_pred_str, \
                                                                original_type_names)
                assert (np.array(local_ent_idx)==np.array(self.learned_ae_pred_info[pred_temp]['ent_idx'][i][m])).all()
                ae_ent_idx.append(self.learned_ae_pred_info[pred_temp]['ent_idx'][i][m])
                quantifier, quantified_types = None, None
                neg = False
                if "?" in quantifier_str:
                    quantifier, quantified_types = name2quantifier_types(quantifier_str, list(self._obj_types.keys()))
                    name += quantifier.lower()
                    name += '_'
                    for r, t in enumerate(quantified_types):
                        pos_id = quantified_type_idx[r]
                        # no pos id to avoid confusion
                        name += f"{t[:3].lower()}_"
                if neg_str:
                    neg = True
                    name += neg_str + "_"
                name += basic_pred_str
                original_types = pred_temp.types
                if 'ae_vecs' in self.learned_ae_pred_info[pred_temp]:
                    logging.info(f"Constructing Neural Predicate: {name}")
                    ae_vec = self.learned_ae_pred_info[pred_temp]['ae_vecs'][i][m]
                    logging.info(f"Its Effect Vector: {ae_vec}")
                    ae_matrix.append(one2two(torch.tensor(ae_vec), 2))
                if 'example_dataset' in self.learned_ae_pred_info[pred_temp]:
                    example_dataset = self.learned_ae_pred_info[pred_temp]['example_dataset']
                    pred_model = setup_neupi_mlp_net(example_dataset,
                                pred_temp.arity,
                                pred_config['architecture'],
                                self._node_feature_to_index,
                                self._edge_feature_to_index)
                else:
                    assert 'neural_dim' in self.learned_ae_pred_info[pred_temp]
                    pred_model = setup_neupi_mlp_net(None,
                                pred_temp.arity,
                                pred_config['architecture'],
                                self._node_feature_to_index,
                                self._edge_feature_to_index,
                                dims=self.learned_ae_pred_info[pred_temp]['neural_dim'])
                # same group shares the same model
                weights = _load_tensor(
                    self.learned_ae_pred_info[pred_temp]['model_weights'][i])
                pred_model.load_state_dict(weights)
                # note that grounding this predicate do not need the operatr ent idx, just ground all
                specific_function = self.generate_classifier(pred_temp, pred_model, pred_config['gumbel_temp'], \
                                                                pred_config['decision_b'], \
                                                    quantifier, quantified_types, quantified_type_idx, neg)
                types = []
                if quantified_types is not None:
                    # quantified types are no longer input types
                    for m, t in enumerate(original_types):
                        if m in quantified_type_idx:
                            continue
                        types.append(t)
                else:
                    types = original_types
                # do a renaming as FD has naming requirements
                basic_name = name.split('(')[0]
                arguments = name.split('(')[1].split(')')[0].split(',')
                new_arguments = [arg.replace('?', '') for arg in arguments]
                name = basic_name + '_' + '_'.join(new_arguments)
                neual_pred = NeuralPredicate(name=name, types=types, 
                                            _classifier=specific_function)
                ae_col_names.append(neual_pred)
                # assert name.split('(')[0] == info[pred_temp][k][0].name, "Should have the same name"
                dummy2real[info[pred_temp][k][0]] = neual_pred
                logging.info(f"Dummy Predicate: {info[pred_temp][k][0]} -> Real Predicate: {neual_pred} Done!")
                assert len(ae_col_names) == len(ae_matrix)
                assert len(ae_col_names) == len(ae_ent_idx)
        if from_json:
            return dummy2real, ae_col_names, torch.stack(ae_matrix, dim=1), ae_ent_idx
        else:
            return dummy2real
    
    def _select_predicates_by_score_search(
        self, huge_ae_matrixe: torch.Tensor,
        all_predicates_info: Dict[int, Tuple[Predicate, List[int], bool,
                                             Tuple]],
        score_function: _OperatorBeliefScoreFunction
    ) -> Tuple[Set[Predicate], torch.Tensor, List[Predicate], List[List[int]]]:
        """Select effect predicates by hill-climbing on the score function.

        Provided effect predicates are always selected. On each level,
        every candidate in the pool is scored together with the selected
        predicates and the best one is added if it improves the score. After
        each added predicate, _expand_candidate_pool() can add candidates.
        If no candidate improves the score while it is still above
        CFG.neupi_selection_thresh, a random candidate is added ("imagined").
        If that does not lead to an improvement on the next level, it is
        taken back (with the candidates added after it) and another one is
        imagined, until no candidates are left.

        Returns the selected predicates as a set and as an ordered list,
        with their AE matrix and argument indices.
        """
        provided: List[PredicateCandidate] = []
        pool: List[PredicateCandidate] = []
        for colind, info in all_predicates_info.items():
            candidate = PredicateCandidate(info[0], huge_ae_matrixe[:, colind],
                                           info[1])
            if info[2]:
                logging.info(f"Provided Predicate: {candidate.predicate}")
                provided.append(candidate)
            else:
                logging.info(f"Possible Predicate: {candidate.predicate}")
                pool.append(candidate)
        # The predicates that are not effect predicates can still appear in
        # preconditions, so they are part of every scored set.
        provided_effect_predicates = {c.predicate for c in provided}
        precondition_predicates = [
            pred for pred in sorted(self._initial_predicates)
            if pred not in provided_effect_predicates
        ]
        logging.info(
            f"Pre-selection predicate counts: provided effect "
            f"{len(provided)}, candidates {len(pool)}, precondition only "
            f"{len(precondition_predicates)}")

        def _stack(candidates: List[PredicateCandidate]) -> torch.Tensor:
            if not candidates:
                num_rows, _, num_channels = huge_ae_matrixe.shape
                return huge_ae_matrixe.new_zeros((num_rows, 0, num_channels))
            return torch.stack([c.ae_column for c in candidates], dim=1)

        selected = list(provided)
        best_score = float("inf")
        # Imagined predicates tried on the current level, and the pool from
        # before the last imagined predicate was added (None if the last
        # added predicate was not imagined).
        imagined: List[Predicate] = []
        pool_before_imagined: Optional[List[PredicateCandidate]] = None
        level = 0
        while True:
            logging.info(f"Search Level {level}...")
            selected_predicates = [c.predicate for c in selected]
            selected_ent_idx = [c.ent_idx for c in selected]
            selected_consistent = self.check_matrix_consistency(
                _stack(selected), selected_ent_idx)
            best_candidate: Optional[PredicateCandidate] = None
            for candidate in pool:
                if candidate.predicate in selected_predicates:
                    continue
                try_matrix = _stack(selected + [candidate])
                try_effect_predicates = selected_predicates + [
                    candidate.predicate
                ]
                try_ent_idx = selected_ent_idx + [candidate.ent_idx]
                if selected_consistent and not self.check_matrix_consistency(
                        try_matrix, try_ent_idx):
                    logging.info("Matrix is not consistent, skip...")
                    continue
                start_time = time.time()
                score = score_function.evaluate(
                    try_matrix,
                    try_effect_predicates + precondition_predicates,
                    try_ent_idx,
                    CFG.strips_learner,
                    operator_predicates=self._operator_predicates(
                        try_effect_predicates),
                    atoms_closure=self._atoms_closure(try_effect_predicates))
                logging.info(f"Predicate set {try_effect_predicates} has "
                             f"score {score} in {time.time()-start_time} "
                             "seconds.")
                if score < best_score:
                    logging.info(f"New smallest score: {score}")
                    best_score = score
                    best_candidate = candidate
            if best_candidate is not None:
                logging.info(f"New best score at level {level}: {best_score}")
                logging.info(f"Adding Predicate {best_candidate.predicate}...")
                selected.append(best_candidate)
                imagined = []
                pool_before_imagined = None
                self._expand_candidate_pool([c.predicate for c in selected],
                                            pool)
                if best_score < CFG.neupi_selection_thresh:
                    logging.info(f"Search Done at level {level} with score "
                                 f"{best_score}")
                    break
                level += 1
                continue
            if best_score <= CFG.neupi_selection_thresh:
                break
            logging.info(f"Hill-Climbing Got Stuck at level {level} and not "
                         "good enough...Force to Next Level")
            if pool_before_imagined is not None:
                logging.info("Imagined Search, Backtrack one step...")
                selected.pop()
                pool[:] = pool_before_imagined
                level -= 1
            selected_predicates = [c.predicate for c in selected]
            remaining = [
                c for c in pool if c.predicate not in selected_predicates
                and c.predicate not in imagined
            ]
            if not remaining:
                logging.info("No predicates left to imagine; stopping.")
                break
            candidate = remaining[self._rng.integers(len(remaining))]
            logging.info(f"Adding Imagined Predicate {candidate.predicate}...")
            imagined.append(candidate.predicate)
            pool_before_imagined = list(pool)
            selected.append(candidate)
            self._expand_candidate_pool([c.predicate for c in selected], pool)
            level += 1
        final_predicates = [c.predicate for c in selected]
        return set(final_predicates), _stack(selected), final_predicates, \
            [c.ent_idx for c in selected]

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        #  Step 1: Initializing individual NN predicates by Bi-Level Learning
        if not CFG.load_neupi_from_json:
            s = time.time()
            learning_trajectories, init_atom_traj = self.learn_neural_predicates(dataset)
            logging.info(f"Neural Predicate Invention Done in {time.time()-s} seconds.")
            if CFG.neupi_save_init_atom_dataset:
                save_path = utils.get_approach_save_path_str()
                with open(f"{save_path}_.init_atom_traj", "wb") as f:
                    pkl.dump(init_atom_traj, f)
            # Purne Equivalence Classes
            # Step 2: Get all the columns into one big matrix and their info, including provided
            num_traj = len(learning_trajectories)
            traj4equavalence = [learning_trajectories[i] for i in range(int(num_traj * CFG.neupi_equ_dataset))]
            huge_matrix, colind2info = self._get_invented_pruned_predicates(traj4equavalence)
            all_pred = set(info[0] for info in colind2info.values())
            new_predicate = set()
            for pred in list(all_pred):
                if pred in self._initial_predicates:
                    continue
                new_predicate.add(pred)
            # Step 3: Select predicates by the planning-effort score of each candidate set
            # 1. construct atom dataset for all the predicates
            traj4search = [init_atom_traj[i] for i in range(int(num_traj * CFG.neupi_pred_search_dataset))]
            s = time.time()
            atom_dataset_part = utils.add_ground_atom_dataset(
                            traj4search,
                            new_predicate)
            # remeber to add the initial predicates, which may not in the effect matrix (unchanged)
            all_pred = all_pred | self._initial_predicates
            # 2. Score Function Def
            # Note that need to use full _train_tasks
            score_function = _OperatorBeliefScoreFunction(
                    atom_dataset_part, self._train_tasks, self.ae_row_names,
                    CFG.neupi_selection_metric)
            self._learned_predicates, self.ae_matrix_tgt, self.ae_col_names_all, self.ae_col_ent_idx = \
                    self._select_predicates_by_score_search(
                    huge_matrix, colind2info, score_function)
            logging.info(f"Predicate Selection Done in {time.time()-s} seconds.")
            logging.info("Final AE matrix (Add): {}".format(self.ae_matrix_tgt[:, :, 0]))
            if self.ae_matrix_tgt.shape[-1] == 2:
                logging.info("Final AE matrix (Del): {}".format(self.ae_matrix_tgt[:, :, 1]))
            logging.info("Final (Ordered) Effect Predicates: {}".format(self.ae_col_names_all))
            # For efficiency, just prune it, no need to ground again
            # Note, remeber to keep the precondition only predicates in initial predicates
            if CFG.neupi_pred_search_dataset == 1:
                # only needs to prune the atom dataset
                atom_dataset = utils.prune_ground_atom_dataset(
                    atom_dataset_part,
                    self._learned_predicates | self._initial_predicates)
            else:
                # needs to ground again from the beginning
                atom_dataset = utils.add_ground_atom_dataset(
                    init_atom_traj,
                    self._learned_predicates)
            logging.info("Learning NSRT from the learned predicates.")
        else:
            logging.info("Loading Neural Predicates from JSON...")
            self.load_from_json()
            if not CFG.neupi_gt_sampler:
                logging.info("Re-Learning NSRT (sampler) from the dataset.")
                learning_trajectories = dataset.trajectories[:int(len(dataset.trajectories) * \
                                                                  CFG.neupi_learning_dataset)]
                logging.info(f"Learning NSRT from {len(learning_trajectories)} trajectories.")
                atom_dataset = utils.create_ground_atom_dataset(
                    learning_trajectories, self._learned_predicates | self._initial_predicates
                )
        # Finally, learn NSRTs via superclass, using all the kept predicates.
        annotations = None
        if dataset.has_annotations:
            annotations = dataset.annotations
        logging.info("Learning NSRTs from the provided atom dataset.")
        self._learn_nsrts(learning_trajectories,
                          atom_dataset,
                          annotations)
        if CFG.load_neupi_from_json:
            # no need to save anything
            return
        # save everything we get
        save_path = utils.get_approach_save_path_str()
        saved_info = {"learned_ae_pred_info": copy.deepcopy(self.learned_ae_pred_info)
                      }
        saved_info['node_feature_to_index'] = self._node_feature_to_index
        saved_info['edge_feature_to_index'] = self._edge_feature_to_index
        saved_info['node_is_rot'] = self._node_is_rot
        saved_info['edge_is_rot'] = self._edge_is_rot
        saved_info['input_normalizers'] = self._input_normalizers
        # we will save the selected pred info instead of the predicate, since it is neural-related classifier
        selected_pred = {}
        selected_pred2dummy = {}
        for pred in self._learned_predicates:
            for col, info in colind2info.items():
                if info[0] == pred:
                    # this predicate is a template
                    dummy_pred = info[-1][0]
                    name = pred.name
                    types = pred.types
                    # this predicate is specifically for this one
                    new_dummy_pred = DummyPredicate(name=name, types=types)
                    selected_pred2dummy[pred] = new_dummy_pred
                    # importanty, each dummy pred could have multiple real preds
                    if dummy_pred in selected_pred:
                        selected_pred[dummy_pred].append((new_dummy_pred, info[-1][1], info[-1][2]))
                    else:
                        selected_pred[dummy_pred] = [(new_dummy_pred, info[-1][1], info[-1][2])]
                    break
        saved_info['selected_pred'] = selected_pred
        # we will save the "dummy" nsrts with the neural predicates replaced by dummy ones with the same name
        dummy_nsrts = utils.replace_nsrt_predicates(self._nsrts, selected_pred2dummy)
        saved_info['dummy_nsrts'] = dummy_nsrts
        # finally, save the info
        with open(f"{save_path}.neupi_info", "wb") as f:
            pkl.dump(saved_info, f)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        if CFG.load_neupi_from_json:
            self.load_from_json()
        else:
            self.load_from_pkl()

    def load_from_json(self) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}.json", "r") as f:
            content_info = json.load(f)
        # easy stats
        self._node_feature_to_index = content_info['node_feature_to_index']
        self._edge_feature_to_index = content_info['edge_feature_to_index']
        self._node_is_rot = content_info['node_is_rot']
        self._edge_is_rot = content_info['edge_is_rot']
        input_normalizers_list = content_info['input_normalizers']
        self._input_normalizers = {}
        for k, v in input_normalizers_list.items():
            min_v = np.array(v[0])
            max_v = np.array(v[1])
            self._input_normalizers[k] = (min_v, max_v)
        # learned_ae_pred_info and selected_pred info
        self.learned_ae_pred_info, selected_pred = utils.parse_json_ivntr_info(
            content_info,
            self._initial_predicates,
            self._obj_types)
        selected_pred_names2dummy = {pred.name: selected_pred[pred][0][0] \
                                     for pred in list(selected_pred.keys())}
        # construct neural predicates
        logging.info("Loading Neural Predicates...")
        dummy2real_pred, self.ae_col_names_all, self.ae_matrix_tgt, self.ae_col_ent_idx = \
            self._get_invented_predicates_from_info(selected_pred, from_json=True)
        for k, v in dummy2real_pred.items():
            self._learned_predicates.add(v)
        # also add the initial predicates if they are in the selected pred
        for pred in self._initial_predicates:
            if pred.name in selected_pred_names2dummy.keys():
                dummy2real_pred[selected_pred_names2dummy[pred.name]] = pred
        # construct nsrts with neural predicates and dummy nsrts
        logging.info("Loading NSRTs...")
        dummy_nsrts = content_info["dummy_nsrts"]
        if CFG.neupi_gt_sampler:
            load_sampler = False
        else:
            load_sampler = True
        self._nsrts = utils.parse_str2nsrt(dummy_nsrts, dummy2real_pred,
                                           self._initial_predicates,
                                           self._initial_options,
                                           self._obj_types,
                                           load_sampler)

        if CFG.pretty_print_when_loading:
            preds, _ = utils.extract_preds_and_types(self._nsrts)
            name_map = {}
            logging.info("Invented predicates:")
            for idx, pred in enumerate(
                    sorted(set(preds.values()) - self._initial_predicates)):
                vars_str, body_str = pred.pretty_str()
                logging.info(f"\tP{idx+1}({vars_str}) ≜ {body_str}")
                name_map[body_str] = f"P{idx+1}"
        logging.info("\n\nLoaded NSRTs:")
        for nsrt in sorted(self._nsrts):
            if CFG.pretty_print_when_loading:
                logging.info(nsrt.pretty_str(name_map))
            else:
                logging.info(nsrt)
        logging.info("")
        # Seed the option parameter spaces after loading.
        for nsrt in self._nsrts:
            nsrt.option.params_space.seed(CFG.seed)

    def load_from_pkl(self) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}.neupi_info", "rb") as f:
            content_info = pkl.load(f)
        self.learned_ae_pred_info = content_info["learned_ae_pred_info"]
        self._node_feature_to_index = content_info['node_feature_to_index']
        self._edge_feature_to_index = content_info['edge_feature_to_index']
        self._node_is_rot = content_info['node_is_rot']
        self._edge_is_rot = content_info['edge_is_rot']
        self._input_normalizers = content_info['input_normalizers']
        selected_pred = content_info["selected_pred"]
        selected_pred_names2dummy = {pred.name: selected_pred[pred][0][0] for pred in list(selected_pred.keys())}
        # construct neural predicates
        logging.info("Loading Neural Predicates...")
        dummy2real_pred = self._get_invented_predicates_from_info(selected_pred, from_json=False)
        for k, v in dummy2real_pred.items():
            self._learned_predicates.add(v)
        # also add the initial predicates if they are in the selected pred
        for pred in self._initial_predicates:
            if pred.name in selected_pred_names2dummy.keys():
                dummy2real_pred[selected_pred_names2dummy[pred.name]] = pred
        # construct nsrts with neural predicates and dummy nsrts
        logging.info("Loading NSRTs...")
        dummy_nsrts = content_info["dummy_nsrts"]
        self._nsrts = utils.replace_nsrt_predicates(dummy_nsrts, dummy2real_pred)

        if CFG.pretty_print_when_loading:
            preds, _ = utils.extract_preds_and_types(self._nsrts)
            name_map = {}
            logging.info("Invented predicates:")
            for idx, pred in enumerate(
                    sorted(set(preds.values()) - self._initial_predicates)):
                vars_str, body_str = pred.pretty_str()
                logging.info(f"\tP{idx+1}({vars_str}) ≜ {body_str}")
                name_map[body_str] = f"P{idx+1}"
        logging.info("\n\nLoaded NSRTs:")
        for nsrt in sorted(self._nsrts):
            if CFG.pretty_print_when_loading:
                logging.info(nsrt.pretty_str(name_map))
            else:
                logging.info(nsrt)
        logging.info("")
        # Seed the option parameter spaces after loading.
        for nsrt in self._nsrts:
            nsrt.option.params_space.seed(CFG.seed)

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectoryReward],
                     atom_dataset: Optional[List[GroundAtomTrajectory]],
                     annotations: Optional[List[Any]]) -> None:
        logging.info("Using the provided ground atom dataset.")
        ground_atom_dataset = atom_dataset

        # Learn NSRTs from the ground atom dataset, with belief assumptions.
        # check if every option is appeared

        # Only the operator predicates become effects; the other selected
        # effect predicates (e.g., derived ones) stay available as
        # preconditions.
        operator_predicates = self._operator_predicates(
            self.ae_col_names_all)
        keep = [
            i for i, pred in enumerate(self.ae_col_names_all)
            if pred in operator_predicates
        ]
        operator_belief = {
            'row_names': self.ae_row_names,
            'col_names': [self.ae_col_names_all[i] for i in keep],
            'ae_matrix': self.ae_matrix_tgt[:, keep].cpu().numpy(),
            'col_ent_idx': [self.ae_col_ent_idx[i] for i in keep]
        }

        self._nsrts, self._segmented_trajs, self._seg_to_nsrt = \
            learn_nsrts_from_data(trajectories,
                                  self._train_tasks,
                                  self._get_current_predicates(),
                                  self._initial_options,
                                  self._action_space,
                                  ground_atom_dataset,
                                  sampler_learner=CFG.sampler_learner,
                                  annotations=annotations,
                                  operator_belief=operator_belief,
                                  derived_predicates=set(
                                      self.ae_col_names_all) -
                                  set(operator_predicates))

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # Unless the learned samplers are used, plan with the ground-truth
        # samplers of the matching dummy NSRTs.
        nsrts = self._get_current_nsrts()
        if not CFG.neupi_gt_sampler and all(n._sampler is not None
                                            for n in nsrts):
            dummy_nsrts = set()
        else:
            dummy_nsrts = get_dummy_nsrts(CFG.env, self._initial_predicates,
                                          self._initial_options)
        new_nsrts = []
        for nsrt in nsrts:
            if (not CFG.neupi_gt_sampler) and (nsrt._sampler is not None):
                logging.info(f"Using fully learned NSRT for {nsrt.name}")
                new_nsrts.append(nsrt)
                continue
            for dummy_nsrt in dummy_nsrts:
                if nsrt.option == dummy_nsrt.option:
                    logging.info(f"Matched {nsrt.name} with GT Sampler {dummy_nsrt.name}")
                    new_nsrt = NSRT(name=nsrt.name, parameters=nsrt.parameters, preconditions=nsrt.preconditions,
                        add_effects=nsrt.add_effects, delete_effects=nsrt.delete_effects,
                        ignore_effects=nsrt.ignore_effects, option=nsrt.option,
                        option_vars=nsrt.option_vars, _sampler=dummy_nsrt.sampler)
                    new_nsrt.option.params_space.seed(CFG.seed)
                    new_nsrts.append(new_nsrt)
                    break
        self._nsrts = set(new_nsrts)
        closure = self._atoms_closure(sorted(self._learned_predicates))
        if closure is None:
            return super()._solve(task, timeout)
        with hooks.temporary_atoms_closure(closure):
            return super()._solve(task, timeout)
