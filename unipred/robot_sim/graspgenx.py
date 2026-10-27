"""Grasps from GraspGenX (NVIDIA), from a GraspGenX server (optional: the
scenes take grasps straight from above by default, grasps="top_down").

The server is not part of this repository; it is expected on port 8116, or
at $UNIPRED_GRASPGENX_URL. A request is a scene's points (Z up, metres) with the mask of
the object to grasp; the answer is grasps of the Robotiq 2F-85, best first,
in the points' frame: GraspGenX's gripper frame, z the approach, x the
closing direction, the origin on the gripper's mounting face.
"""

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any, Tuple

import numpy as np

GRIPPER = "robotiq_2f_85"  # GraspGenX's name for the 2F-85
DEFAULT_URL = "http://localhost:8116"


def _encode(array: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, array, allow_pickle=False)
        return base64.b64encode(f.getvalue()).decode("ascii")


def _decode(text: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(text)) as f:
        return np.load(f, allow_pickle=False)


def plan_grasps(points: np.ndarray, mask: np.ndarray,
                **options: Any) -> Tuple[np.ndarray, np.ndarray]:
    """The grasps (N, 4, 4) and scores (N,) of the object at points[mask],
    clear of the other points (see module doc); options are GraspGenX's
    (planner, num_grasps, topk_num_grasps, collision_threshold, ...)."""
    url = os.environ.get("UNIPRED_GRASPGENX_URL", DEFAULT_URL).rstrip("/")
    payload = {
        "scene_points_base64": _encode(np.asarray(points, np.float32)),
        "object_mask_base64": _encode(np.asarray(mask, bool)),
        "gripper": GRIPPER,
        **options,
    }
    request = urllib.request.Request(f"{url}/plan_segmented_scene",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GraspGenX at {url} refused the request: "
                           f"{e.read().decode(errors='replace')}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"GraspGenX is not reachable at {url} ({e.reason}). Start a "
            "GraspGenX server, set UNIPRED_GRASPGENX_URL, or use top-down "
            "grasps (--grasps top_down).") from e
    return (_decode(answer["grasps_base64"]).reshape(-1, 4, 4),
            _decode(answer["scores_base64"]).reshape(-1))
