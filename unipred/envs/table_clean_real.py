"""Table Clean Real: Table Clean on a real robot, built offline from
processed recordings (see unipred/envs/offline_real.py), with the types,
options and ground-truth operators of Table Clean Sim."""

from unipred.envs.offline_real import OfflineRealEnv


class TableCleanRealEnv(OfflineRealEnv):
    """Table Clean Real, from --table_clean_real_data_dir."""
    data_dir_setting = "table_clean_real_data_dir"

    @classmethod
    def get_name(cls) -> str:
        return "table_clean_real"
