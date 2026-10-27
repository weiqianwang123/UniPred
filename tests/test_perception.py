"""Tests for perception: detection, crops, scenes and crop encoders (no
network weights needed)."""

import numpy as np
import pytest

from predicators import utils
from unipred.perception.crops import crop_object
from unipred.perception.detector import Detection, Detector, fake_colors
from unipred.perception.features import CropEncoder, resolve_backend
from unipred.perception.scene import PerceptionError, ScenePerceiver
from unipred.perception.tracker import Tracker


def test_fake_encoder_is_deterministic_and_cached():
    """The fake backend gives fixed features that depend on the crop."""
    encoder = CropEncoder("fake")
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((16, 16, 3), dtype=np.uint8)
    blue[..., 2] = 255
    features = encoder.encode([red, blue, red])
    assert features.shape == (3, encoder.dim)
    assert np.allclose(features[0], features[2])
    assert not np.allclose(features[0], features[1])
    assert np.allclose(encoder.encode([red])[0], features[0])


def test_missing_dinov3_weights_name_the_path(tmp_path):
    """DINOv3 without installed weights fails with the expected path."""
    utils.reset_config({"perception_weights_dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "dinov3")):
        CropEncoder("dinov3")
    utils.reset_config({"perception_weights_dir": ""})
    with pytest.raises(FileNotFoundError, match="UNIPRED_WEIGHTS_DIR"):
        CropEncoder("dinov3")


def test_auto_backend(tmp_path):
    """auto is DINOv3 when its weights are installed, else ResNet-18."""
    utils.reset_config({"perception_weights_dir": str(tmp_path)})
    assert resolve_backend("auto") == "resnet18"
    (tmp_path / "dinov3").mkdir()
    assert resolve_backend("auto") == "resnet18"  # an incomplete download
    (tmp_path / "dinov3" / "config.json").write_text("{}")
    assert resolve_backend("auto") == "dinov3"
    assert resolve_backend("fake") == "fake"
    with pytest.raises(ValueError, match="Unknown encoder backend"):
        CropEncoder("random")


def _scene_image(boxes):
    """An image with a filled rectangle of its type's fake colour per
    (type, box)."""
    colors = fake_colors(["robot", "toy"])
    image = np.full((60, 80, 3), 128, dtype=np.uint8)
    for object_type, (x0, y0, x1, y1) in boxes:
        image[y0:y1, x0:x1] = colors[object_type]
    return image


def test_fake_detector_and_crops():
    """The fake detector finds each region of a type's colour; crops are
    square, padded and clipped to the image."""
    detector = Detector("fake", {"robot": "robot gripper", "toy": "toy"})
    image = _scene_image([("toy", (10, 10, 20, 20)), ("toy", (50, 30, 60, 40)),
                          ("robot", (30, 45, 40, 55))])
    detections = sorted(detector.detect(image), key=lambda d: d.box)
    assert [(d.object_type, d.box) for d in detections] == [
        ("toy", (10.0, 10.0, 20.0, 20.0)), ("robot", (30.0, 45.0, 40.0, 55.0)),
        ("toy", (50.0, 30.0, 60.0, 40.0))
    ]
    assert detections[0].mask.sum() == 100
    crop = crop_object(image, detections[0].box, padding=0.5)
    assert crop.shape == (20, 20, 3)
    corner = crop_object(image, (0, 0, 10, 10), padding=0.5)
    assert corner.shape == (15, 15, 3)
    masked = crop_object(image, detections[0].box, padding=0.5,
                         mask=detections[0].mask)
    assert (masked[0, 0] == 0).all() and (masked[10, 10] != 0).any()


def test_scene_perceiver_tracks_objects():
    """Objects are named left to right in the first image, then follow the
    nearest detection; an undetected object keeps its box; given boxes
    replace detection."""
    perceiver = ScenePerceiver(
        Detector("fake", {"robot": "robot gripper", "toy": "toy"}),
        CropEncoder("fake"))
    objects = {"toy0": "toy", "toy1": "toy", "robot": "robot"}
    first = perceiver.perceive(
        _scene_image([("toy", (10, 10, 20, 20)), ("toy", (50, 30, 60, 40)),
                      ("robot", (30, 45, 40, 55))]), objects)
    assert first.boxes["toy0"][0] == 10 and first.boxes["toy1"][0] == 50
    assert set(first.features) == set(objects)
    # toy0 moved to the right of toy1's old place; toy1 is hidden.
    second = perceiver.perceive(
        _scene_image([("toy", (15, 30, 25, 40)), ("robot", (30, 45, 40, 55))]),
        objects, previous=first.boxes)
    assert second.boxes["toy0"] == (15.0, 30.0, 25.0, 40.0)
    assert second.boxes["toy1"] == first.boxes["toy1"]
    assert second.masks["toy1"] is None
    given = perceiver.perceive(_scene_image([("robot", (30, 45, 40, 55))]),
                               objects,
                               previous=first.boxes,
                               given={"toy0": [1, 2, 3, 4]})
    assert given.boxes["toy0"] == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(PerceptionError, match="toy1"):
        perceiver.perceive(_scene_image([("toy", (10, 10, 20, 20)),
                                         ("robot", (30, 45, 40, 55))]),
                           objects)


def test_missing_detector_weights_name_the_path(tmp_path):
    """GroundingDINO and SAM2 without installed weights fail with the
    expected path."""
    utils.reset_config({"perception_weights_dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError,
                       match=str(tmp_path / "grounding_dino")):
        Detector("groundingdino_sam2", {"toy": "toy"})


def test_tiny_crops_and_slivers():
    """The fake encoder gives finite features for crops of any size, and
    the fake detector ignores slivers of a few pixels."""
    encoder = CropEncoder("fake")
    for shape in ((1, 2, 3), (2, 1, 3), (3, 3, 3)):
        crop = np.full(shape, 200, dtype=np.uint8)
        assert np.all(np.isfinite(encoder.encode([crop])))
    detector = Detector("fake", {"robot": "robot gripper", "toy": "toy"})
    image = _scene_image([("toy", (10, 10, 20, 20)), ("toy", (40, 40, 42, 41))])
    assert len(detector.detect(image)) == 1


def test_max_box_area_drops_large_detections():
    """A detection larger than its type's max_box_area (a fraction of the
    image) is dropped; other types are not limited."""
    image = _scene_image([("toy", (10, 10, 20, 20)), ("toy", (30, 5, 60, 40)),
                          ("robot", (0, 45, 64, 64))])
    detector = Detector("fake", {
        "robot": "robot gripper",
        "toy": "toy"
    },
                        max_box_area={"toy": 0.1})
    assert sorted((d.object_type, d.box[0])
                  for d in detector.detect(image)) == [("robot", 0.0),
                                                       ("toy", 10.0)]
    with pytest.raises(ValueError, match="cup"):
        Detector("fake", {"toy": "toy"}, max_box_area={"cup": 0.1})


class _GivenDetections:
    """A detector that finds the toys at the given boxes (masks: the
    boxes)."""

    def __init__(self) -> None:
        self.boxes = []

    def detect(self, image):
        found = []
        for box in self.boxes:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[box[1]:box[3], box[0]:box[2]] = True
            found.append(Detection("toy", tuple(map(float, box)), 1.0, mask))
        return found


def _painted(regions):
    image = np.full((60, 80, 3), 128, dtype=np.uint8)
    for (x0, y0, x1, y1), color in regions:
        image[y0:y1, x0:x1] = color
    return image


def test_objects_are_matched_by_appearance():
    """After the first image, an object takes the detection that looks like
    it, not the nearest one; a detection that looks like none of them
    leaves the objects where they were."""
    detector = _GivenDetections()
    perceiver = ScenePerceiver(detector, CropEncoder("fake"))
    objects = {"toy0": "toy", "toy1": "toy"}
    red, blue, green = (220, 20, 20), (20, 20, 220), (20, 200, 20)
    detector.boxes = [(10, 10, 20, 20), (50, 10, 60, 20)]
    first = perceiver.perceive(
        _painted([((10, 10, 20, 20), red), ((50, 10, 60, 20), blue)]),
        objects)
    assert first.boxes["toy0"][0] == 10 and first.boxes["toy1"][0] == 50
    # The blue toy moves next to where the red one was, the red one away.
    detector.boxes = [(22, 10, 32, 20), (60, 30, 70, 40)]
    second = perceiver.perceive(
        _painted([((22, 10, 32, 20), blue), ((60, 30, 70, 40), red)]),
        objects, first)
    assert second.boxes["toy0"] == (60.0, 30.0, 70.0, 40.0)
    assert second.boxes["toy1"] == (22.0, 10.0, 32.0, 20.0)
    # By place only (boxes as the previous observation), they swap.
    by_place = perceiver.perceive(
        _painted([((22, 10, 32, 20), blue), ((60, 30, 70, 40), red)]),
        objects, first.boxes)
    assert by_place.boxes["toy0"] == (22.0, 10.0, 32.0, 20.0)
    detector.boxes = [(30, 40, 40, 50)]
    third = perceiver.perceive(_painted([((30, 40, 40, 50), green)]),
                               objects, second)
    assert third.boxes == second.boxes


def test_fake_tracker_follows_a_region_out_of_view_and_back():
    """The fake tracker follows an object's region as it moves; out of view
    it has no mask but is still followed, and found again where it was
    last seen."""
    tracker = Tracker("fake")
    image = _scene_image([("toy", (10, 10, 20, 20))])
    tracker.start(image, {"toy0": np.all(image == fake_colors(["robot", "toy"])
                                         ["toy"], axis=-1)})
    followed = tracker.follow(_scene_image([("toy", (14, 12, 24, 22))]))
    ys, xs = np.nonzero(followed["toy0"])
    assert (xs.min(), ys.min(), xs.max(), ys.max()) == (14, 12, 23, 21)
    assert tracker.follow(_scene_image([])) == {}
    assert tracker.followed == {"toy0"}
    assert "toy0" in tracker.follow(_scene_image([("toy", (16, 12, 26, 22))]))
    tracker.stop()
    assert not tracker.followed


def test_tracker_keeps_names_through_frames():
    """Two toys alike swap sides, each in a row of its own: matched by
    place they swap names; followed through the frames between, they keep
    them."""
    objects = {"toy0": "toy", "toy1": "toy"}

    def _image(x0, x1):
        return _scene_image([("toy", (x0, 5, x0 + 8, 13)),
                             ("toy", (x1, 45, x1 + 8, 53))])

    def _perceiver():
        return ScenePerceiver(Detector("fake", {
            "robot": "robot gripper",
            "toy": "toy"
        }),
                              CropEncoder("fake"),
                              tracker=Tracker("fake"))

    perceiver = _perceiver()
    first = perceiver.perceive(_image(4, 68), objects)
    assert first.boxes["toy0"][0] == 4 and first.boxes["toy1"][0] == 68
    last = _image(68, 4)
    by_place = _perceiver()
    by_place.perceive(_image(4, 68), objects)
    assert by_place.perceive(last, objects, first).boxes["toy0"][0] == 4
    frames = [_image(x, 72 - x) for x in range(8, 68, 4)]
    followed = perceiver.follow(first, frames + [last])
    assert set(followed) == {"toy0", "toy1"}
    second = perceiver.perceive(last, objects, first, followed=followed)
    assert second.boxes["toy0"] == (68.0, 5.0, 76.0, 13.0)
    assert second.boxes["toy1"] == (4.0, 45.0, 12.0, 53.0)
    # Only from the observation perceived last.
    assert perceiver.follow(first, [last]) is None


def test_followed_mask_stands_in_for_a_missed_detection():
    """A followed object that no detection overlaps is where its followed
    mask is; the detections go to the other objects."""
    detector = _GivenDetections()
    perceiver = ScenePerceiver(detector, CropEncoder("fake"))
    objects = {"toy0": "toy", "toy1": "toy"}
    image = _painted([((10, 10, 20, 20), (220, 20, 20)),
                      ((50, 10, 60, 20), (20, 20, 220))])
    detector.boxes = [(10, 10, 20, 20), (50, 10, 60, 20)]
    first = perceiver.perceive(image, objects)
    held = np.zeros(image.shape[:2], dtype=bool)
    held[30:36, 30:38] = True
    detector.boxes = [(50, 10, 60, 20)]
    second = perceiver.perceive(image, objects, first,
                                followed={"toy0": held})
    assert second.boxes["toy0"] == (30.0, 30.0, 38.0, 36.0)
    assert second.masks["toy0"] is held
    assert second.boxes["toy1"] == (50.0, 10.0, 60.0, 20.0)


def test_object_out_of_view_at_a_keyframe_stays_followed():
    """A toy hidden at a keyframe keeps its track there and is followed
    again once back in view; without frames to the keyframe, the tracks
    start afresh."""
    objects = {"toy0": "toy", "toy1": "toy"}
    toy0, toy1 = ("toy", (4, 5, 12, 13)), ("toy", (68, 45, 76, 53))
    perceiver = ScenePerceiver(
        Detector("fake", {
            "robot": "robot gripper",
            "toy": "toy"
        }), CropEncoder("fake"), tracker=Tracker("fake"))
    first = perceiver.perceive(_scene_image([toy0, toy1]), objects)
    hidden = _scene_image([toy1])
    followed = perceiver.follow(first, [hidden])
    assert set(followed) == {"toy1"}
    second = perceiver.perceive(hidden, objects, first, followed=followed)
    assert second.masks["toy0"] is None
    assert perceiver.tracker.followed == {"toy0", "toy1"}
    back = _scene_image([("toy", (6, 5, 14, 13)), toy1])
    assert set(perceiver.follow(second, [back])) == {"toy0", "toy1"}
    perceiver.perceive(back, objects, second)
    assert perceiver.tracker.followed == {"toy0", "toy1"}
    perceiver.perceive(hidden, objects, second)
    assert perceiver.tracker.followed == {"toy1"}
