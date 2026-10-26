"""UniPred approaches; importing this package registers them with the
predicators approach registry."""

from unipred.approaches.ivntr_derived import IvntrDerivedApproach
from unipred.approaches.unipred import UniPredApproach

__all__ = ["IvntrDerivedApproach", "UniPredApproach"]
