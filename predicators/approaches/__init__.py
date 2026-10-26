"""Handle creation of approaches."""

from typing import List, Set
from typing import Type as TypingType

from gymnasium.spaces import Box

from predicators import utils
from predicators.approaches.base_approach import ApproachFailure, \
    ApproachTimeout, BaseApproach
from predicators.structs import ParameterizedOption, Predicate, Task, Type

__all__ = ["BaseApproach", "ApproachTimeout", "ApproachFailure"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def _get_approach_cls_from_name(name: str) -> TypingType[BaseApproach]:
    for cls in utils.get_all_subclasses(BaseApproach):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            return cls
    raise NotImplementedError(f"Unknown approach: {name}")


def create_approach(name: str, initial_predicates: Set[Predicate],
                    initial_options: Set[ParameterizedOption],
                    types: Set[Type], action_space: Box,
                    train_tasks: List[Task]) -> BaseApproach:
    """Create an approach given its name."""
    cls = _get_approach_cls_from_name(name)
    return cls(initial_predicates, initial_options, types, action_space,
               train_tasks)
