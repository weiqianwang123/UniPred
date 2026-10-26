"""PDDL completion (stage 2 of UniPred): seed predicates from an LLM.

The robot's options become the actions of a partial PDDL domain that
contains only the known (goal and static) predicates and what the domain's
predicate config says about them. An LLM completes the domain, given a few
demonstrations, and every new predicate it writes is turned into an effect
vector over the options. These vectors seed the LLM-in-the-loop search.

The known predicates are listed in the predicate config::

    known_predicates:
      - {name: CameraReadingTaken, effects: {UseCamera: add}}
      - {name: HasCamera, precondition_of: [UseCamera]}

Names must be predicates and options of the environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from string import Template
from typing import Collection, Dict, FrozenSet, List, Optional, Sequence, \
    Set, Tuple, Union

import numpy as np

from predicators import utils
from predicators.structs import LowLevelTrajectory, ParameterizedOption, \
    Predicate, Task, Type
from unipred.llm.client import LLMClient
from unipred.llm.names import NameMap, pddl_symbol

# Codes of an effect vector entry, as in IVNTR (see two2one in
# predicators/gnn/neupi_utils.py).
NO_EFFECT, ADD, DELETE = 0, 1, 2

SExpr = Union[str, List["SExpr"]]


def load_prompt(name: str) -> Template:
    """Load a prompt template from unipred/llm/prompts."""
    text = resources.files("unipred.llm").joinpath("prompts", name)
    return Template(text.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PredicateHints:
    """What the predicate config says about a known predicate."""
    effects: Tuple[Tuple[ParameterizedOption, int], ...]  # (option, ADD/DELETE)
    preconditions: FrozenSet[ParameterizedOption]


def load_known_predicates(
        entries: Sequence[Dict], predicates: Collection[Predicate],
        options: Collection[ParameterizedOption]
) -> Dict[Predicate, PredicateHints]:
    """Check the known_predicates entries of a predicate config against the
    environment and return what they say about each predicate."""
    pred_by_name = {p.name: p for p in predicates}
    option_by_name = {o.name: o for o in options}

    def _option(name: str) -> ParameterizedOption:
        if name not in option_by_name:
            raise ValueError(f"known_predicates: unknown option {name}; the "
                             f"options are {sorted(option_by_name)}.")
        return option_by_name[name]

    hints: Dict[Predicate, PredicateHints] = {}
    for entry in entries:
        name = entry["name"]
        if name not in pred_by_name:
            raise ValueError(
                f"known_predicates: {name} is not a given predicate; the "
                f"given predicates are {sorted(pred_by_name)}.")
        unknown_keys = set(entry) - {"name", "effects", "precondition_of"}
        if unknown_keys:
            raise ValueError(f"known_predicates: unknown keys "
                             f"{sorted(unknown_keys)} for {name}.")
        effects = []
        for option_name, kind in entry.get("effects", {}).items():
            if kind not in ("add", "del"):
                raise ValueError(f"known_predicates: the effect of "
                                 f"{option_name} on {name} must be 'add' or "
                                 f"'del', not {kind!r}.")
            effects.append(
                (_option(option_name), ADD if kind == "add" else DELETE))
        preconditions = frozenset(
            _option(o) for o in entry.get("precondition_of", []))
        hints[pred_by_name[name]] = PredicateHints(tuple(effects),
                                                   preconditions)
    return hints


def _typed_vars(types: Sequence[Type]) -> List[Tuple[str, Type]]:
    """PDDL variables for the given types, numbered per type."""
    counts: Dict[str, int] = {}
    variables = []
    for t in types:
        idx = counts.get(t.name, 0)
        counts[t.name] = idx + 1
        variables.append((f"?{pddl_symbol(t.name)}{idx}", t))
    return variables


def _params(option: ParameterizedOption) -> List[Tuple[str, Type]]:
    return _typed_vars(option.types)


def _atom_str(pred: Predicate, option: ParameterizedOption,
              names: NameMap) -> str:
    """Bind each argument of the predicate to the first unused parameter of
    the option with the same type."""
    params = _params(option)
    used: Set[int] = set()
    args = []
    for t in pred.types:
        match = next((i for i, (_, pt) in enumerate(params)
                      if i not in used and pt == t), None)
        if match is None:
            raise ValueError(f"Cannot bind {pred.name} to the parameters of "
                             f"{option.name}: no unused {t.name} argument.")
        used.add(match)
        args.append(params[match][0])
    return f"({' '.join([names.predicate(pred)] + args)})"


def build_partial_domain(options: Sequence[ParameterizedOption],
                         predicates: Collection[Predicate],
                         hints: Dict[Predicate, PredicateHints],
                         names: NameMap) -> str:
    """The partial PDDL domain shown to the LLM."""
    types = sorted({t.name for o in options for t in o.types}
                   | {t.name for p in predicates for t in p.types})
    pred_lines = []
    for pred in sorted(predicates, key=names.predicate):
        args = "".join(f" {var} - {pddl_symbol(t.name)}"
                       for var, t in _typed_vars(pred.types))
        pred_lines.append(f"    ({names.predicate(pred)}{args})")
    action_blocks = []
    for option in sorted(options, key=names.option):
        params = " ".join(f"{var} - {pddl_symbol(t.name)}"
                          for var, t in _params(option))
        preconditions = []
        effects = []
        for pred in sorted(hints, key=names.predicate):
            if option in hints[pred].preconditions:
                preconditions.append(_atom_str(pred, option, names))
            for effect_option, code in hints[pred].effects:
                if effect_option == option:
                    atom = _atom_str(pred, option, names)
                    effects.append(atom if code == ADD else f"(not {atom})")
        action_blocks.append(
            f"  (:action {names.option(option)}\n"
            f"    :parameters ({params})\n"
            f"    :precondition (and {' '.join(preconditions)})\n"
            f"    :effect (and {' '.join(effects)}))")
    return ("(define (domain robot)\n"
            "  (:requirements :strips :typing)\n"
            f"  (:types {' '.join(pddl_symbol(t) for t in types)})\n"
            "  (:predicates\n" + "\n".join(pred_lines) + ")\n" +
            "\n".join(action_blocks) + ")")


def _option_sequence(traj: LowLevelTrajectory) -> List:
    options = []
    for action in traj.actions:
        option = action.get_option()
        if not options or option is not options[-1]:
            options.append(option)
    return options


def render_demos(trajectories: Sequence[LowLevelTrajectory],
                 tasks: Sequence[Task], predicates: Collection[Predicate],
                 names: NameMap, num_demos: int) -> str:
    """Describe the first demonstrations by their initial known atoms, goal
    and action sequence."""

    def _atoms(atoms) -> str:
        return " ".join(
            sorted(f"({' '.join([names.predicate(a.predicate)] + [o.name for o in a.objects])})"
                   for a in atoms))

    demos = [t for t in trajectories if t.is_demo][:num_demos]
    blocks = []
    for k, traj in enumerate(demos):
        task = tasks[traj.train_task_idx]
        init = utils.abstract(traj.states[0], set(predicates))
        goal = {a for a in task.goal if a.predicate in predicates}
        actions = " ".join(
            f"({' '.join([names.option(o.parent)] + [obj.name for obj in o.objects])})"
            for o in _option_sequence(traj))
        objects = " ".join(
            sorted(f"{o.name} - {pddl_symbol(o.type.name)}"
                   for o in traj.states[0]))
        blocks.append(f"Demonstration {k + 1}:\n  objects: {objects}\n"
                      f"  initial state: {_atoms(init)}\n"
                      f"  goal: {_atoms(goal)}\n  actions: {actions}")
    return "\n".join(blocks)


def parse_sexpr(text: str) -> List[SExpr]:
    """Parse the top-level s-expressions of a text; comments (;) are
    dropped and unbalanced trailing text is ignored."""
    lines = [line.split(";", 1)[0] for line in text.splitlines()]
    tokens = " ".join(lines).replace("(", " ( ").replace(")", " ) ").split()
    stack: List[List[SExpr]] = [[]]
    for token in tokens:
        if token == "(":
            stack.append([])
        elif token == ")":
            if len(stack) == 1:
                continue
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token.lower())
    return stack[0]


def _typed_list(items: Sequence[SExpr]) -> List[Tuple[str, str]]:
    """Parse "?a ?b - type ?c - other" into [(var, type)]."""
    result: List[Tuple[str, str]] = []
    pending: List[str] = []
    i = 0
    while i < len(items):
        item = items[i]
        if item == "-" and i + 1 < len(items):
            result.extend((v, str(items[i + 1])) for v in pending)
            pending = []
            i += 2
            continue
        if isinstance(item, str):
            pending.append(item)
        i += 1
    result.extend((v, "object") for v in pending)
    return result


@dataclass(frozen=True)
class ParsedAction:
    """The parameters and effects of an action in a PDDL domain."""
    parameters: Tuple[Tuple[str, str], ...]
    add_effects: FrozenSet[str]  # predicate names
    delete_effects: FrozenSet[str]


@dataclass(frozen=True)
class ParsedDomain:
    """The predicates and actions of a PDDL domain."""
    predicates: Dict[str, Tuple[str, ...]]  # name -> argument types
    actions: Dict[str, ParsedAction]


def _collect_effects(expr: SExpr, adds: Set[str], deletes: Set[str]) -> None:
    if not isinstance(expr, list) or not expr:
        return
    head = expr[0]
    if head == "and":
        for sub in expr[1:]:
            _collect_effects(sub, adds, deletes)
    elif head == "not":
        if len(expr) == 2 and isinstance(expr[1], list) and expr[1]:
            deletes.add(str(expr[1][0]))
    elif isinstance(head, str) and not head.startswith(":") and head not in (
            "forall", "when", "exists", "or", "imply", "increase",
            "decrease"):
        adds.add(head)


def parse_domain(text: str) -> Optional[ParsedDomain]:
    """Parse a PDDL domain; None if the text has no domain definition."""
    domain = next(
        (e for e in parse_sexpr(text)
         if isinstance(e, list) and e and e[0] == "define"), None)
    if domain is None:
        return None
    predicates: Dict[str, Tuple[str, ...]] = {}
    actions: Dict[str, ParsedAction] = {}
    for section in domain[1:]:
        if not isinstance(section, list) or not section:
            continue
        if section[0] == ":predicates":
            for decl in section[1:]:
                if isinstance(decl, list) and decl and isinstance(
                        decl[0], str):
                    predicates[decl[0]] = tuple(
                        t for _, t in _typed_list(decl[1:]))
        elif section[0] == ":action" and len(section) > 1:
            fields = dict(zip(section[2::2], section[3::2]))
            parameters = fields.get(":parameters", [])
            adds: Set[str] = set()
            deletes: Set[str] = set()
            _collect_effects(fields.get(":effect", []), adds, deletes)
            actions[str(section[1])] = ParsedAction(
                tuple(_typed_list(parameters if isinstance(parameters, list)
                                  else [])), frozenset(adds - deletes),
                frozenset(deletes - adds))
    return ParsedDomain(predicates, actions)


@dataclass(frozen=True)
class ProposedPredicate:
    """A new predicate of a completed domain and its effect vector."""
    name: str
    types: Tuple[str, ...]
    codes: Tuple[int, ...]  # one entry per option row: NO_EFFECT/ADD/DELETE


def extract_proposals(domain: ParsedDomain,
                      rows: Sequence[ParameterizedOption],
                      names: NameMap) -> List[ProposedPredicate]:
    """Effect vectors of the new predicates that some action changes."""
    known = names.known_predicate_symbols()
    row_of = {option: i for i, option in enumerate(rows)}
    proposals = []
    for pred_name, types in sorted(domain.predicates.items()):
        if pred_name in known:
            continue
        codes = [NO_EFFECT] * len(rows)
        for action_name, action in domain.actions.items():
            try:
                option = names.option_from_symbol(action_name)
            except KeyError:
                continue
            if option not in row_of:
                continue
            if pred_name in action.add_effects:
                codes[row_of[option]] = ADD
            elif pred_name in action.delete_effects:
                codes[row_of[option]] = DELETE
        if any(codes):
            proposals.append(ProposedPredicate(pred_name, types, tuple(codes)))
    return proposals


def complete_domain(client: LLMClient, partial_domain: str, demos: str,
                    description: str, signatures: Collection[Tuple[str, ...]],
                    rows: Sequence[ParameterizedOption],
                    names: NameMap) -> List[ProposedPredicate]:
    """Ask the LLM to complete the domain and return the new predicates whose
    argument types are among the given signatures."""
    system = load_prompt("pddl_completion_system.txt").substitute()
    signature_lines = "\n".join(f"- ({', '.join(s)})"
                                for s in sorted(signatures))
    user = load_prompt("pddl_completion_user.txt").substitute(
        description=description or "(none)",
        domain=partial_domain,
        signatures=signature_lines,
        demos=demos or "(none)")
    for attempt in range(client.retry_attempts):
        reply = client.complete(system, user, "pddl_completion", attempt)
        domain = parse_domain(reply)
        if domain is None:
            logging.info("PDDL completion: the reply has no domain.")
            continue
        proposals = [
            p for p in extract_proposals(domain, rows, names)
            if tuple(pddl_symbol(t) for t in p.types) in signatures
        ]
        if proposals:
            for proposal in proposals:
                logging.info(f"PDDL completion proposes {proposal.name}"
                             f"{proposal.types}: {proposal.codes}")
            return proposals
        logging.info("PDDL completion: no new predicate with an allowed "
                     "signature is an effect of an action.")
    logging.warning("PDDL completion produced no seed predicates.")
    return []


def proposals_to_seeds(
    proposals: Sequence[ProposedPredicate],
    targets: Collection[Predicate],
) -> Dict[Predicate, List[np.ndarray]]:
    """Group the proposals by the target predicate type with the same
    argument types."""
    seeds: Dict[Predicate, List[np.ndarray]] = {t: [] for t in targets}
    for proposal in proposals:
        for target in targets:
            if tuple(pddl_symbol(t.name) for t in target.types) == tuple(
                    pddl_symbol(t) for t in proposal.types):
                seeds[target].append(np.array(proposal.codes, dtype=int))
    return seeds
