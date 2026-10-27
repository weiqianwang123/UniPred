"""SAM3 text-prompted segmentation and video tracking, as CDG's perception
servers serve them.

CDG (cd envs/perception && pixi run serve --only sam) serves SAM3 over
HTTP on port 8114; $UNIPRED_SAM3_URL points at a server elsewhere. A
segmentation request is an RGB image and a text prompt; the answer is every
instance of the prompt with its mask, box (x0, y0, x1, y1 in pixels) and
score, best first (the server drops scores below 0.3). A track follows one
instance from the mask it starts with through the frames sent to it, one
at a time: each answer is the instance's mask in that frame and the
probability that it is in view (the mask empty when not). The server keeps
16 tracks at once, dropping the least recently used.
"""

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

DEFAULT_URL = "http://localhost:8114"

Instance = Tuple[Tuple[float, ...], float, np.ndarray]  # box, score, mask


def url() -> str:
    """The SAM3 server's address."""
    return os.environ.get("UNIPRED_SAM3_URL", DEFAULT_URL).rstrip("/")


def segment_text(image: np.ndarray, prompt: str) -> List[Instance]:
    """The instances of a text prompt in an RGB image: (box, score, mask)
    each, best first."""
    answer = _post("/segment", {
        "image_base64": _encode_image(image),
        "text_prompt": prompt
    })
    return [_instance(result) for result in answer["results"]]


def track_start(image: np.ndarray, mask: np.ndarray,
                session_id: str) -> Instance:
    """Start following the instance that a mask (H x W booleans) covers in
    an RGB image, as track session_id (replacing one of that name)."""
    return _instance(
        _post(
            "/track_start", {
                "session_id": session_id,
                "image_base64": _encode_image(image),
                "mask_base64": base64.b64encode(
                    np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
                ).decode("ascii"),
                "shape": list(mask.shape)
            }))


def track(image: np.ndarray, session_id: str) -> Instance:
    """The followed instance in the next frame of track session_id: its
    box, the probability that it is in view, and its mask."""
    return _instance(
        _post("/track", {
            "session_id": session_id,
            "image_base64": _encode_image(image)
        }))


def track_reset(session_id: str) -> None:
    """End track session_id (an unknown one is fine)."""
    _post("/track_reset", {"session_id": session_id})


def _encode_image(image: np.ndarray) -> str:
    with io.BytesIO() as f:
        Image.fromarray(image.astype(np.uint8)).convert("RGB").save(f, "PNG")
        return base64.b64encode(f.getvalue()).decode("ascii")


def _instance(result: Dict) -> Instance:
    mask = np.frombuffer(base64.b64decode(result["mask_base64"]),
                         dtype=np.uint8).reshape(result["shape"]) > 0
    return (tuple(float(v) for v in result["box"]), float(result["score"]),
            mask)


def _post(path: str, body: Dict) -> Dict:
    request = urllib.request.Request(
        f"{url()}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"SAM3 at {url()} refused the request: "
                           f"{e.read().decode(errors='replace')}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"SAM3 is not reachable at {url()} ({e.reason}). Start CDG's "
            "perception servers (cd envs/perception && pixi run serve --only "
            "sam) or set UNIPRED_SAM3_URL.") from e
