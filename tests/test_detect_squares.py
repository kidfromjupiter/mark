import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detect_squares import find_answer_boxes
from warp import normalise_img

DATASET_DIR = ROOT / "dataset/compressed/good/"
CROSSED_DATASET_DIR = ROOT / "dataset/compressed/crossed/"
UNCOMPRESSED_DATASET_DIR = ROOT / "dataset/uncompressed/"
CROSSED_COLORED_DATASET_DIR = ROOT / "dataset/compressed/crossed_and_colored_messy/"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_BOX_COUNT = 50


def dataset_images():
    return sorted(
        p for p in DATASET_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def crossed_dataset_images():
    return sorted(
        p for p in CROSSED_DATASET_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def crossed_colored_dataset_images():
    return sorted(
        p
        for p in CROSSED_COLORED_DATASET_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def uncompressed_dataset_images():
    return sorted(
        p
        for p in UNCOMPRESSED_DATASET_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_detection_count_test(img_path, dataset_name, record_detection_stats):
    img = cv2.imread(str(img_path))

    if img is None:
        record_detection_stats(
            dataset=dataset_name,
            img_path=img_path,
            expected=EXPECTED_BOX_COUNT,
            count=None,
            status="read_failed",
            reason="cv2.imread returned None",
        )
        pytest.fail(f"{img_path} failed to read image")

    warped = normalise_img(img)

    if warped is None:
        record_detection_stats(
            dataset=dataset_name,
            img_path=img_path,
            expected=EXPECTED_BOX_COUNT,
            count=None,
            status="normalise_failed",
            reason="normalise_img returned None",
        )
        pytest.fail(f"{img_path} failed to normalise image")

    boxes = find_answer_boxes(warped)
    count = len(boxes)

    status = "ok" if count == EXPECTED_BOX_COUNT else "wrong_count"

    record_detection_stats(
        dataset=dataset_name,
        img_path=img_path,
        expected=EXPECTED_BOX_COUNT,
        count=count,
        status=status,
    )

    assert count == EXPECTED_BOX_COUNT, (
        f"{img_path} detected {count} boxes, expected {EXPECTED_BOX_COUNT}"
    )


@pytest.mark.parametrize("img_path", dataset_images(), ids=lambda p: p.name)
def test_find_answer_boxes_expected_count(img_path, record_detection_stats):
    run_detection_count_test(
        img_path,
        "compressed/good",
        record_detection_stats,
    )


@pytest.mark.parametrize(
    "img_path", uncompressed_dataset_images(), ids=lambda p: p.name
)
def test_find_uncompressed_answer_boxes_expected_count(
    img_path, record_detection_stats
):
    run_detection_count_test(
        img_path,
        "uncompressed",
        record_detection_stats,
    )


@pytest.mark.parametrize("img_path", crossed_dataset_images(), ids=lambda p: p.name)
def test_crossed_find_answer_boxes_expected_count(img_path, record_detection_stats):
    run_detection_count_test(
        img_path,
        "compressed/crossed",
        record_detection_stats,
    )


@pytest.mark.parametrize(
    "img_path", crossed_colored_dataset_images(), ids=lambda p: p.name
)
def test_crossed_colored_find_answer_boxes_expected_count(
    img_path, record_detection_stats
):
    run_detection_count_test(
        img_path,
        "compressed/crossed_and_colored_messy",
        record_detection_stats,
    )
