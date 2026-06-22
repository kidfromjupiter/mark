from pathlib import Path

import cv2
import pytest

from warp import normalise_img, WIDTH, HEIGHT


DATASET_DIR = Path("dataset/good/")
BLUR_DATASET_DIR = Path("dataset/blur/")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_dataset_images():
    return [
        p for p in DATASET_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_blur_dataset_images():
    return [
        p for p in BLUR_DATASET_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


@pytest.mark.parametrize("img_path", get_dataset_images())
def test_normalise_img_detects_aruco_and_warps(img_path):
    img = cv2.imread(str(img_path))

    assert img is not None, f"Could not read image: {img_path}"

    warped = normalise_img(img)

    assert warped is not None, f"Failed to detect 4 ArUco markers in {img_path}"
    assert warped.shape[0] == HEIGHT
    assert warped.shape[1] == WIDTH
    assert warped.shape[2] == 3


@pytest.mark.parametrize("img_path", get_blur_dataset_images())
def test_normalise_blurred_img_detects_aruco_and_warps(img_path):
    img = cv2.imread(str(img_path))

    assert img is not None, f"Could not read image: {img_path}"

    warped = normalise_img(img)

    assert warped is not None, f"Failed to detect 4 ArUco markers in {img_path}"
    assert warped.shape[0] == HEIGHT
    assert warped.shape[1] == WIDTH
    assert warped.shape[2] == 3
