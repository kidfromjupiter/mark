import sys
from pathlib import Path
from detect_squares import find_answer_boxes

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



DATASET_DIR = ROOT / "dataset/good/"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_BOX_COUNT = 50


def dataset_images():
    return [
        p for p in DATASET_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]




@pytest.mark.parametrize("img_path", dataset_images())
def test_find_answer_boxes_expected_count(img_path):
    img = cv2.imread(str(img_path))
    boxes = find_answer_boxes(img)

    assert len(boxes) == EXPECTED_BOX_COUNT, (
        f"{img_path} detected {len(boxes)} boxes, "
        f"expected {EXPECTED_BOX_COUNT}"
    )

