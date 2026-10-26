"""UniPred approaches; importing this package registers them with the
predicators approach registry."""

from unipred.approaches.code_as_policy import CodeAsPolicyApproach
from unipred.approaches.ivntr_derived import IvntrDerivedApproach
from unipred.approaches.pix2pred import Pix2PredApproach
from unipred.approaches.unipred import UniPredApproach
from unipred.approaches.vila import VilaApproach

__all__ = [
    "CodeAsPolicyApproach", "IvntrDerivedApproach", "Pix2PredApproach",
    "UniPredApproach", "VilaApproach"
]
