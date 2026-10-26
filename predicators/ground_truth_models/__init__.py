"""Implements ground-truth NSRTs and options."""
import abc
from typing import Dict, List, Optional, Sequence, Set
from typing import Type as TypingType
from typing import TypeVar, Union

from gymnasium.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.hooks import AtomsClosure
from predicators.settings import CFG
from predicators.structs import NSRT, ParameterizedOption, Predicate, Type


class GroundTruthOptionFactory(abc.ABC):
    """Parent class for ground-truth option definitions."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds options for."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Create options for the given env name."""
        raise NotImplementedError("Override me!")


class GroundTruthNSRTFactory(abc.ABC):
    """Parent class for ground-truth NSRT definitions."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds NSRTs for."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_nsrts(cls, env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        """Create NSRTs for the given env name."""
        raise NotImplementedError("Override me!")

    @classmethod
    def get_atoms_closure(
            cls, env_name: str,
            predicates: Dict[str, Predicate]) -> Optional[AtomsClosure]:
        """Return the atoms closure (see predicators/hooks.py) that computes
        the derived predicates among the given ones, if the NSRTs use any.

        By default, the NSRTs use no derived predicates.
        """
        del env_name, predicates  # unused
        return None


_FactoryT = TypeVar("_FactoryT")


def _get_factory(base_cls: TypingType[_FactoryT],
                 env_name: str) -> _FactoryT:
    """Return the unique registered factory of the given kind for an env."""
    matches = [
        cls for cls in utils.get_all_subclasses(base_cls)
        if not cls.__abstractmethods__ and env_name in cls.get_env_names()
    ]
    if not matches:
        raise NotImplementedError(f"No {base_cls.__name__} registered for "
                                  f"env: {env_name}")
    if len(matches) > 1:
        names = sorted(cls.__name__ for cls in matches)
        raise ValueError(f"Multiple {base_cls.__name__} classes registered "
                         f"for env {env_name}: {names}")
    return matches[0]()


def get_gt_options(env_name: str) -> Set[ParameterizedOption]:
    """Create ground truth options for an env."""
    env = get_or_create_env(env_name)
    factory = _get_factory(GroundTruthOptionFactory, env_name)
    types = {t.name: t for t in env.types}
    predicates = {p.name: p for p in env.predicates}
    options = factory.get_options(env_name, types, predicates,
                                  env.action_space)
    # Seed the options for reproducibility.
    for option in options:
        option.params_space.seed(CFG.seed)
    return options


def get_gt_nsrts(env_name: str, predicates_to_keep: Set[Predicate],
                 options_to_keep: Set[ParameterizedOption]) -> Set[NSRT]:
    """Create ground truth NSRTs for an env, keeping only the given
    predicates and the NSRTs whose options are kept."""
    env = get_or_create_env(env_name)
    env_options = get_gt_options(env_name)
    assert predicates_to_keep.issubset(env.predicates)
    assert options_to_keep.issubset(env_options)
    factory = _get_factory(GroundTruthNSRTFactory, env_name)
    # Give all predicates and options, then filter based on kept ones at the
    # end of this function. This is easier than filtering within the factory.
    types = {t.name: t for t in env.types}
    predicates = {p.name: p for p in env.predicates}
    options = {o.name: o for o in env_options}
    nsrts = factory.get_nsrts(env_name, types, predicates, options)
    final_nsrts = set()
    for nsrt in nsrts:
        if nsrt.option not in options_to_keep:
            continue
        nsrt = nsrt.filter_predicates(predicates_to_keep)
        final_nsrts.add(nsrt)
    return final_nsrts


def get_gt_atoms_closure(
        env_name: str,
        predicates_to_keep: Set[Predicate]) -> Optional[AtomsClosure]:
    """The atoms closure of the ground-truth NSRTs of an env, or None."""
    factory = _get_factory(GroundTruthNSRTFactory, env_name)
    return factory.get_atoms_closure(env_name,
                                     {p.name: p
                                      for p in predicates_to_keep})


def parse_config_included_options(env: BaseEnv) -> Set[ParameterizedOption]:
    """Parse the CFG.included_options string, given an environment.

    Return the set of included oracle options.

    Note that "all" is not implemented because setting the option_learner flag
    to "no_learning" is the preferred way to include all options.
    """
    if not CFG.included_options:
        return set()
    env_options = get_gt_options(env.get_name())
    included_names = set(CFG.included_options.split(","))
    assert included_names.issubset({option.name for option in env_options}), \
        "Unrecognized option in included_options!"
    included_options = {o for o in env_options if o.name in included_names}
    return included_options


# Convenience functions used by external scripts, tests, and other oracles.


# Find the factories.
utils.import_submodules(__path__, __name__)
