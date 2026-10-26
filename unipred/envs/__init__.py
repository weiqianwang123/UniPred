"""UniPred environments; importing this package registers them with the
predicators environment registry."""

from unipred.envs.packing import PackingEnv
from unipred.envs.table_clean_real import TableCleanRealEnv
from unipred.envs.table_clean_sim import TableCleanSimEnv

__all__ = ["PackingEnv", "TableCleanRealEnv", "TableCleanSimEnv"]
