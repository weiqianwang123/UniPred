"""Ground-truth options and NSRTs of the UniPred environments; importing
this package registers them with predicators."""

from unipred.ground_truth_models import cluttered_retrieval, packing, \
    table_clean

__all__ = ["cluttered_retrieval", "packing", "table_clean"]
