"""SAM3 text-prompted segmentation, as CDG's perception servers serve it.

CDG (cd envs/perception && pixi run serve --only sam) serves SAM3 over
HTTP on port 8114; $UNIPRED_SAM3_URL points at a server elsewhere. A request
is an RGB image and a text prompt; the answer is every instance of the
prompt with its mask, box (x0, y0, x1, y1 in pixels) and score, best first
(the server drops scores below 0.3).
"""

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import List, Tuple

import numpy as np
from PIL import Image

DEFAULT_URL = "http://localhost:8114"


def url() -> str:
    """The SAM3 server's address."""
    return os.environ.get("UNIPRED_SAM3_URL", DEFAULT_URL).rstrip("/")


def segment_text(
        image: np.ndarray,
        prompt: str) -> List[Tuple[Tuple[float, ...], float, np.ndarray]]:
    """The instances of a text prompt in an RGB image: (box, score, mask)
    each, best first."""
    with io.BytesIO() as f:
        Image.fromarray(image.astype(np.uint8)).convert("RGB").save(f, "PNG")
        encoded = base64.b64encode(f.getvalue()).decode("ascii")
    request = urllib.request.Request(
        f"{url()}/segment",
        data=json.dumps({
            "image_base64": encoded,
            "text_prompt": prompt
        }).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"SAM3 at {url()} refused the request: "
                           f"{e.read().decode(errors='replace')}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"SAM3 is not reachable at {url()} ({e.reason}). Start CDG's "
            "perception servers (cd envs/perception && pixi run serve --only "
            "sam) or set UNIPRED_SAM3_URL.") from e
    found = []
    for result in answer["results"]:
        mask = np.frombuffer(base64.b64decode(result["mask_base64"]),
                             dtype=np.uint8).reshape(result["shape"]) > 0
        found.append((tuple(float(v)
                            for v in result["box"]), float(result["score"]),
                      mask))
    return found
