"""Cluttered Retrieval Real: Cluttered Retrieval on a real robot, built
offline from processed recordings (see unipred/envs/offline_real.py), with
the types, options and ground-truth operators of Cluttered Retrieval Sim."""

from unipred.envs.offline_real import OfflineRealEnv


class ClutteredRetrievalRealEnv(OfflineRealEnv):
    """Cluttered Retrieval Real, from --cluttered_retrieval_real_data_dir."""
    data_dir_setting = "cluttered_retrieval_real_data_dir"

    @classmethod
    def get_name(cls) -> str:
        return "cluttered_retrieval_real"
