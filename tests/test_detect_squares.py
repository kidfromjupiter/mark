import sys
from pathlib import Path
from detect_squares import find_answer_boxes
from warp import normalise_img

import cv2
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



DATASET_DIR = ROOT / "dataset/compressed/good/"
CROSSED_DATASET_DIR = ROOT / "dataset/compressed/crossed/"
UNCOMPRESSED_DATASET_DIR = ROOT / "dataset/uncompressed/"
CROSSED_COLORED_DATASET_DIR = ROOT / "dataset/compressed/crossed_and_colored_messy/"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_BOX_COUNT = 50


def dataset_images():
    return [
        p for p in DATASET_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def crossed_dataset_images():
    return [
        p for p in CROSSED_DATASET_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def crossed_colored_dataset_images():
    return [
        p for p in CROSSED_COLORED_DATASET_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def uncompressed__dataset_images():
    return [
        p for p in UNCOMPRESSED_DATASET_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

@pytest.mark.parametrize("img_path", dataset_images())
def test_find_answer_boxes_expected_count(img_path):
    img = cv2.imread(str(img_path))
    warped = normalise_img(img)

    assert warped is not None, ("Failed to normalise image")

    boxes = find_answer_boxes(warped)

    assert len(boxes) == EXPECTED_BOX_COUNT, (
        f"{img_path} detected {len(boxes)} boxes, "
        f"expected {EXPECTED_BOX_COUNT}"
    )

@pytest.mark.parametrize("img_path", uncompressed__dataset_images())
def test_find_uncompressed_answer_boxes_expected_count(img_path):
    img = cv2.imread(str(img_path))
    warped = normalise_img(img)

    assert warped is not None, ("Failed to normalise image")

    boxes = find_answer_boxes(warped)

    assert len(boxes) == EXPECTED_BOX_COUNT, (
        f"{img_path} detected {len(boxes)} boxes, "
        f"expected {EXPECTED_BOX_COUNT}"
    )

@pytest.mark.parametrize("img_path", crossed_dataset_images())
def test_crossed_find_answer_boxes_expected_count(img_path):
    img = cv2.imread(str(img_path))
    warped = normalise_img(img)

    assert warped is not None, ("Failed to normalise image")

    boxes = find_answer_boxes(warped)

    assert len(boxes) == EXPECTED_BOX_COUNT, (
        f"{img_path} detected {len(boxes)} boxes, "
        f"expected {EXPECTED_BOX_COUNT}"
    )

@pytest.mark.parametrize("img_path", crossed_colored_dataset_images())
def test_crossed_colored_find_answer_boxes_expected_count(img_path):
    img = cv2.imread(str(img_path))
    warped = normalise_img(img)

    assert warped is not None, ("Failed to normalise image")

    boxes = find_answer_boxes(warped)

    assert len(boxes) == EXPECTED_BOX_COUNT, (
        f"{img_path} detected {len(boxes)} boxes, "
        f"expected {EXPECTED_BOX_COUNT}"
    )



