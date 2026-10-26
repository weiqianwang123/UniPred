"""UniPred approaches; importing this package registers them with the
predicators approach registry."""

from unipred.approaches.code_as_policy import CodeAsPolicyApproach
from unipred.approaches.ivntr_derived import IvntrDerivedApproach
from unipred.approaches.unipred import UniPredApproach

__all__ = ["CodeAsPolicyApproach", "IvntrDerivedApproach", "UniPredApproach"]
