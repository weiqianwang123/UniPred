"""IVNTR with derived-aware predicate selection ("IVNTR w/ derived select"
in Table VIII of the paper): IVNTR's tree search proposes the effect
vectors, and selection treats negated and quantified forms as derived
predicates (see unipred/derived/selection.py)."""

from predicators.approaches.bilevel_learning_approach import \
    BilevelLearningApproach
from unipred.derived.selection import DerivedSelectionMixin


class IvntrDerivedApproach(DerivedSelectionMixin, BilevelLearningApproach):
    """IVNTR with derived-aware predicate selection."""

    @classmethod
    def get_name(cls) -> str:
        return "ivntr_derived"
