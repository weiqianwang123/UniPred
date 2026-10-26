"""The simulator's assets, fetched once into a directory outside git.

- mujoco_menagerie: the Franka Research 3 (franka_fr3) and the Robotiq
  2F-85 gripper (robotiq_2f85);
- mujoco_scanned_objects: Google Scanned Objects as MuJoCo models (the toys,
  the tray that is the box and the sponge that is the wiper).

Both are sparse, shallow checkouts of a pinned commit, in
$UNIPRED_SIM_ASSETS_DIR (default: sim_assets/ in the repository):

    python -m unipred.robot_sim.assets

A scene is only built from there; if an asset is missing, building fails
with its path.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

MENAGERIE = "mujoco_menagerie"
SCANNED_OBJECTS = "mujoco_scanned_objects"
# Repository, commit and the paths checked out of it.
_SOURCES: Dict[str, Tuple[str, str, List[str]]] = {
    MENAGERIE: ("https://github.com/google-deepmind/mujoco_menagerie.git",
                "0059d4335f8156206f63a35662313385f7ad6d74",
                ["franka_fr3", "robotiq_2f85"]),
    SCANNED_OBJECTS:
    ("https://github.com/kevinzakka/mujoco_scanned_objects.git",
     "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c", ["/LICENSE", "/README.md"] + [
         f"models/{name}" for name in [
             "Android_Figure_Orange",
             "Android_Figure_Panda",
             "BABY_CAR",
             "Dino_4",
             "FIRE_TRUCK",
             "SCHOOL_BUS",
             "Schleich_Lion_Action_Figure",
             "Toysmith_Windem_Up_Flippin_Animals_Dog",
             "Design_Ideas_Drawer_Store_Organizer",
             "Big_O_Sponges_Assorted_Cellulose_12_pack",
         ]
     ]),
}


def assets_dir() -> Path:
    """Where the assets are (see module doc)."""
    default = Path(__file__).resolve().parents[2] / "sim_assets"
    return Path(os.environ.get("UNIPRED_SIM_ASSETS_DIR", default))


def require(source: str, path: str) -> Path:
    """An asset's path (relative to its source's checkout); raises
    FileNotFoundError with that path if it is missing."""
    full = assets_dir() / source / path
    if not full.exists():
        raise FileNotFoundError(
            f"{full} not found. Run `python -m unipred.robot_sim.assets` "
            "(or scripts/install.sh --sim), or set UNIPRED_SIM_ASSETS_DIR to "
            f"the directory that contains {source}/.")
    return full


def fetch() -> None:
    """Check out every source's paths at its pinned commit."""
    for name, (url, commit, paths) in _SOURCES.items():
        target = assets_dir() / name

        def git(*args: str, target: Path = target) -> None:
            subprocess.run(["git", "-C", str(target), *args], check=True)

        if not (target / ".git").is_dir():
            target.mkdir(parents=True, exist_ok=True)
            git("init", "-q")
            git("remote", "add", "origin", url)
            git("config", "core.sparseCheckout", "true")
        git("sparse-checkout", "set", "--no-cone", *paths)
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin",
            commit)
        git("checkout", "-q", commit)
        print(f"{name} ({commit[:7]}) in {target}")


if __name__ == "__main__":
    fetch()
