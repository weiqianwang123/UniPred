"""Process recorded sessions into the data of an offline real-robot domain.

Every keyframe image of every session (see README.md for the raw format) is
perceived: the objects are detected, matched to the session's object names,
cropped and encoded (unipred/perception/scene.py). The output directory
receives a copy of the images and processed.pkl, from which the domain's
environment (e.g. table_clean_real, with --table_clean_real_data_dir set to
the output directory) builds its train tasks, recorded demonstrations and
test scenes.

    python examples/real_robot/process_demos.py \\
        --raw examples/real_robot/fixture/raw --out outputs/real/processed \\
        --domain configs/table_clean_real/domain.yaml \\
        --robot_config configs/real_robot.yaml --detector fake --encoder fake

An object that the detector misses and that has no earlier box (say, a toy
in the first keyframe) stops the processing, unless --fill_missing names a
file of every session with boxes per keyframe (the format of boxes.json),
from which such an object's box is then taken, as if annotated: e.g.
gt_boxes.json of simulated recordings (collect_sim_demos.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

from predicators import utils
from unipred.perception.scene import PerceptionError, SceneObservation, \
    ScenePerceiver


def _read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_atoms(atoms: List, domain: Dict, objects: Dict[str, str],
                 where: str) -> List:
    """Check annotated atoms against the domain's predicates and the
    session's objects."""
    for name, args in atoms:
        if name not in domain["predicates"]:
            raise ValueError(f"{where}: unknown predicate {name}.")
        types = [objects.get(a) for a in args]
        if types != domain["predicates"][name]:
            raise ValueError(f"{where}: {name}{tuple(args)} needs arguments "
                             f"of types {domain['predicates'][name]}.")
    return [(name, tuple(args)) for name, args in atoms]


def process_session(directory: str,
                    out_dir: str,
                    domain: Dict,
                    perceiver: ScenePerceiver,
                    fill_missing: Optional[str] = None) -> Dict:
    """Perceive and check one recorded session."""
    name = os.path.basename(os.path.normpath(directory))
    session = _read_json(os.path.join(directory, "session.json"))
    if session is None:
        raise FileNotFoundError(f"{directory} has no session.json.")
    objects: Dict[str, str] = session["objects"]
    unknown = sorted(set(objects.values()) - set(domain["types"]))
    if unknown:
        raise ValueError(f"{name}: unknown object types {unknown}.")
    actions = _read_json(os.path.join(directory, "actions.json"), [])
    for step in actions:
        if any(o not in objects for o in step["objects"]):
            raise ValueError(f"{name}: {step} uses unknown objects.")
    atoms = _read_json(os.path.join(directory, "atoms.json"))
    boxes = _read_json(os.path.join(directory, "boxes.json"))
    fill = None
    if fill_missing is not None:
        fill = _read_json(os.path.join(directory, fill_missing))
        if fill is None:
            raise FileNotFoundError(f"{name} has no {fill_missing}.")
    num_keyframes = len(actions) + 1
    images = [os.path.join(directory, f"{i}.png") for i in range(num_keyframes)]
    missing = [p for p in images if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{name}: {len(actions)} actions need the "
                                f"keyframes 0..{len(actions)}; missing "
                                f"{missing}.")
    if session["split"] == "test" and not atoms:
        raise ValueError(f"{name}: a test scene needs annotated atoms of its "
                         "first keyframe (atoms.json).")
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    keyframes = []
    previous: Optional[SceneObservation] = None
    for i, path in enumerate(images):
        image = np.asarray(Image.open(path).convert("RGB"))
        given = dict(boxes[i]) if boxes is not None and i < len(boxes) else {}
        while True:
            try:
                observation = perceiver.perceive(image, objects, previous,
                                                 given)
                break
            except PerceptionError as error:
                missed = error.object_name
                if fill is None or missed is None or missed in given:
                    raise
                logging.info(f"{name} keyframe {i}: {missed} not detected; "
                             f"its box from {fill_missing}.")
                given[missed] = fill[i][missed]
        previous = observation
        image_path = os.path.join(name, f"{i}.png")
        shutil.copyfile(path, os.path.join(out_dir, image_path))
        keyframe_atoms = None
        if atoms is not None and i < len(atoms):
            keyframe_atoms = _check_atoms(atoms[i], domain, objects,
                                          f"{name} keyframe {i}")
        keyframes.append({
            "image": image_path,
            "features": observation.features,
            "boxes": observation.boxes,
            "atoms": keyframe_atoms,
        })
    return {
        "name": name,
        "split": session["split"],
        "objects": objects,
        "goal": _check_atoms(session["goal"], domain, objects,
                             f"{name} goal"),
        "actions": [(a["option"], tuple(a["objects"])) for a in actions],
        "keyframes": keyframes,
    }


def main() -> None:
    """Process the sessions given on the command line (see module doc)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--raw", required=True,
                        help="directory with one folder per session")
    parser.add_argument("--out", required=True)
    parser.add_argument("--domain", required=True,
                        help="domain spec, e.g. "
                        "configs/table_clean_real/domain.yaml")
    parser.add_argument("--robot_config", default="configs/real_robot.yaml")
    parser.add_argument("--detector", help="override the config's detector")
    parser.add_argument("--encoder", help="override the config's encoder")
    parser.add_argument("--perception_weights_dir",
                        help="default: $UNIPRED_WEIGHTS_DIR")
    parser.add_argument("--fill_missing",
                        help="a file of each session (e.g. gt_boxes.json) "
                        "with the boxes of objects the detector misses")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = {}
    if args.perception_weights_dir:
        settings["perception_weights_dir"] = args.perception_weights_dir
    utils.reset_config(settings)
    with open(args.domain, "r", encoding="utf-8") as f:
        domain = yaml.safe_load(f)
    perceiver = ScenePerceiver.from_config(args.robot_config, args.detector,
                                           args.encoder)
    os.makedirs(args.out, exist_ok=True)
    sessions = []
    for entry in sorted(os.listdir(args.raw)):
        directory = os.path.join(args.raw, entry)
        if os.path.isdir(directory) and not entry.startswith("."):
            sessions.append(
                process_session(directory, args.out, domain, perceiver,
                                args.fill_missing))
            logging.info(f"Processed {entry} ({sessions[-1]['split']}, "
                         f"{len(sessions[-1]['keyframes'])} keyframes)")
    processed = {
        "domain": domain,
        "encoder": perceiver.encoder.backend,
        "feature_dim": perceiver.encoder.dim,
        "sessions": sessions,
    }
    with open(os.path.join(args.out, "processed.pkl"), "wb") as f:
        pickle.dump(processed, f)
    num_train = sum(s["split"] == "train" for s in sessions)
    logging.info(f"Wrote {num_train} train sessions and "
                 f"{len(sessions) - num_train} test scenes to {args.out}.")


if __name__ == "__main__":
    main()
