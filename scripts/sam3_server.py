"""A SAM3 server for UniPred's perception (unipred/perception/sam3.py).

It serves Meta's SAM3 over HTTP: every instance of a text prompt in an
image (POST /segment), and an instance followed from frame to frame with
SAM3's tracker (POST /track_start, /track, /track_reset). The simulated
robot's recordings and the closed loop use it (configs/sim_robot.yaml,
configs/sim_robot_cluttered_retrieval.yaml); the real-robot config can.

    pip install "sam3 @ git+https://github.com/facebookresearch/sam3" \\
        fastapi uvicorn huggingface_hub
    hf auth login  # SAM3 is gated: accept its licence on Hugging Face
    python scripts/sam3_server.py --port 8114

UNIPRED_SAM3_URL tells UniPred where the server is (default
http://localhost:8114). It needs a CUDA GPU (about 7 GB).

Every request and answer is JSON. An image is a base64 PNG
(image_base64); a mask is its raw uint8 bytes, base64 (mask_base64),
beside its shape. An instance is {mask_base64, shape, box (x0, y0, x1, y1
pixels), score, label}; instances scoring below 0.3 are dropped. A track
answers with the instance in the frame and, as its score, the probability
that it is in view (its mask empty when it is not); the server keeps 16
tracks, dropping the least recently used.

Tracking streams the image model's own tracker (its interactive
predictor's, SAM2's design) one frame a call: its memory is the first
frame's mask and the last frames' (as many as it attends to), so a long
track stays bounded. Adapted from PhysCoder's perception servers
(github.com/Jaraxxus-Me/physcoder, MIT License, Copyright (c) 2025 Bowen
Li).
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

MIN_SCORE = 0.3
MAX_TRACKS = 16

Instance = Tuple[np.ndarray, float, np.ndarray]  # mask, score, box


class Sam3:
    """SAM3's image model with its tracker."""

    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        # pylint: disable=import-outside-toplevel
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = build_sam3_image_model(checkpoint_path=checkpoint,
                                            load_from_HF=False,
                                            enable_inst_interactivity=True,
                                            device=device)
        self.processor = Sam3Processor(self.model,
                                       device=device,
                                       confidence_threshold=MIN_SCORE)
        self.tracker = self.model.inst_interactive_predictor.model
        self._autocast = ("cuda" if "cuda" in device else "cpu",
                          torch.bfloat16)
        self._tracks: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    @torch.inference_mode()
    def segment_text(self, image: np.ndarray,
                     prompt: str) -> List[Instance]:
        """The instances of a prompt, best first."""
        with torch.autocast(*self._autocast):
            state = self.processor.set_image(Image.fromarray(image))
            output = self.processor.set_text_prompt(state=state,
                                                    prompt=prompt)
        if any(output.get(k) is None for k in ("masks", "boxes", "scores")):
            return []
        masks = _numpy(output["masks"])
        if masks.ndim == 4:  # (N, 1, H, W)
            masks = masks[:, 0]
        found = [(mask > 0, float(score), np.asarray(box, float))
                 for mask, box, score in zip(masks, _numpy(output["boxes"]),
                                             _numpy(output["scores"]))]
        return sorted(found, key=lambda instance: -instance[1])

    @torch.inference_mode()
    def track_start(self, image: np.ndarray, mask: np.ndarray,
                    session_id: str) -> Instance:
        """Start following the instance a mask covers (replacing the track
        of that name)."""
        if mask.shape != image.shape[:2] or not mask.any():
            raise ValueError("the mask must be the image's size and cover "
                             "something")
        self.track_reset(session_id)
        height, width = image.shape[:2]
        with torch.autocast(*self._autocast):
            state = self.tracker.init_state(video_height=height,
                                            video_width=width,
                                            num_frames=1)
            state["cached_features"] = {0: self._features(image)}
            self.tracker.add_new_mask(state,
                                      frame_idx=0,
                                      obj_id=1,
                                      mask=torch.from_numpy(mask))
            self.tracker.propagate_in_video_preflight(state)
        while len(self._tracks) >= MAX_TRACKS:
            self._tracks.popitem(last=False)
        self._tracks[session_id] = {"state": state, "frame": 0}
        return _with_box(mask.copy(), 1.0)

    @torch.inference_mode()
    def track(self, image: np.ndarray, session_id: str) -> Instance:
        """The followed instance in the next frame."""
        # pylint: disable=protected-access
        track = self._tracks.get(session_id)
        if track is None:
            raise ValueError(f"no track {session_id!r}: start it first")
        state, frame = track["state"], track["frame"] + 1
        size = (state["video_height"], state["video_width"])
        if image.shape[:2] != size:
            raise ValueError(f"track {session_id!r} is of {size} images, not "
                             f"{image.shape[:2]}")
        self._tracks.move_to_end(session_id)
        tracker = self.tracker
        with torch.autocast(*self._autocast):
            state["num_frames"] = frame + 1
            state["cached_features"] = {frame: self._features(image)}
            out, prediction = tracker._run_single_frame_inference(
                inference_state=state,
                output_dict=state["output_dict"],
                frame_idx=frame,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True)
            state["output_dict"]["non_cond_frame_outputs"][frame] = out
            tracker._add_output_per_object(state, frame, out,
                                           "non_cond_frame_outputs")
            _, video = tracker._get_orig_video_res_output(state, prediction)
        # The tracker attends to its last frames' masks and object pointers
        # only: older frames go.
        kept = max(tracker.num_maskmem,
                   getattr(tracker, "max_obj_ptrs_in_encoder", 0))
        for outputs in (state["output_dict"],
                        *state["output_dict_per_obj"].values()):
            for old in [
                    f for f in outputs["non_cond_frame_outputs"]
                    if f < frame - kept
            ]:
                del outputs["non_cond_frame_outputs"][old]
        track["frame"] = frame
        score = float(torch.sigmoid(out["object_score_logits"].float()).mean())
        mask = _numpy(video)[0, 0] > 0 if score >= 0.5 else \
            np.zeros(image.shape[:2], bool)
        return _with_box(mask, score)

    def track_reset(self, session_id: str) -> None:
        """Forget a track (an unknown one is fine)."""
        self._tracks.pop(session_id, None)

    def _features(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """An image's features as the tracker takes them: the image
        model's, projected as the tracker's own encoder would."""
        out = self.processor.set_image(
            Image.fromarray(image))["backbone_out"]["sam2_backbone_out"]
        features = {
            "backbone_fpn": list(out["backbone_fpn"]),
            "vision_pos_enc": list(out["vision_pos_enc"]),
        }
        # The tracker never looks at the image itself, only its features.
        return torch.zeros(1, 3, 8, 8, device=self.tracker.device), features


def _with_box(mask: np.ndarray, score: float) -> Instance:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    box = np.array([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], float) \
        if len(rows) else np.zeros(4)
    return mask, score, box


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _image(text: str) -> np.ndarray:
    with Image.open(io.BytesIO(base64.b64decode(text))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _instance(instance: Instance, label: str = "") -> Dict[str, Any]:
    mask, score, box = instance
    return {
        "mask_base64": base64.b64encode(
            np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).decode(),
        "shape": list(mask.shape),
        "box": np.asarray(box, float).reshape(4).tolist(),
        "score": score,
        "label": label,
    }


def make_app(model: Sam3) -> Any:
    """The HTTP service of a model."""
    # pylint: disable=import-outside-toplevel
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="SAM3 for UniPred")
    lock = threading.Lock()  # one inference at a time on the GPU

    def _run(fn: Any, *args: Any) -> Any:
        with lock:
            try:
                return fn(*args)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:  # pylint: disable=broad-except
                torch.cuda.empty_cache()
                raise HTTPException(status_code=500,
                                    detail=f"inference failed: {e}") from e

    @app.get("/health")
    def health() -> Dict[str, bool]:
        return {"ready": True}

    @app.post("/segment")
    def segment(body: Dict[str, Any]) -> Dict[str, Any]:
        found = _run(model.segment_text, _image(body["image_base64"]),
                     body["text_prompt"])
        return {"results": [_instance(i, body["text_prompt"]) for i in found]}

    @app.post("/track_start")
    def track_start(body: Dict[str, Any]) -> Dict[str, Any]:
        mask = np.frombuffer(base64.b64decode(body["mask_base64"]),
                             dtype=np.uint8).reshape(body["shape"]) > 0
        return _instance(
            _run(model.track_start, _image(body["image_base64"]), mask,
                 body["session_id"]))

    @app.post("/track")
    def track(body: Dict[str, Any]) -> Dict[str, Any]:
        return _instance(
            _run(model.track, _image(body["image_base64"]),
                 body["session_id"]))

    @app.post("/track_reset")
    def track_reset(body: Dict[str, Any]) -> Dict[str, bool]:
        _run(model.track_reset, body["session_id"])
        return {"ok": True}

    return app


def main() -> None:
    """Serve SAM3 (see module doc)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--checkpoint",
                        help="sam3.pt (default: from Hugging Face, "
                        "facebook/sam3)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        # pylint: disable=import-outside-toplevel
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download("facebook/sam3", "sam3.pt")
    import uvicorn  # pylint: disable=import-outside-toplevel,import-error
    uvicorn.run(make_app(Sam3(checkpoint, args.device)),
                host=args.host,
                port=args.port)


if __name__ == "__main__":
    main()
