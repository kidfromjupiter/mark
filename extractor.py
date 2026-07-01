import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from detect_squares import find_answer_boxes
from warp import normalise_img

DEFAULT_SEED_DIR = Path("dataset/seed")
DEFAULT_OUTPUT_DIR = Path("dataset")
DEFAULT_VAL_RATIO = 0.2
DEFAULT_RANDOM_SEED = 7

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

LABELS_BY_STEM = {
    "cross_big": "crossed",
    "cross_small": "crossed",
    "empty": "empty",
    "scribble": "scribble",
}

LABELS_BY_DIR = {
    "cross": "crossed",
    "crossed": "crossed",
    "empty": "empty",
    "scribble": "scribble",
}


@dataclass(frozen=True)
class Crop:
    image: cv2.typing.MatLike
    label: str
    source: str
    box_index: int
    segment_index: int


@dataclass(frozen=True)
class SeedImage:
    path: Path
    label: str
    source: str


@dataclass(frozen=True)
class ExtractionIssue:
    path: Path
    label: str
    reason: str


def answer_segment_edges(x, w, segments=5):
    return [int(round(x + (w * i) / float(segments))) for i in range(segments + 1)]


def source_name(seed_dir, path):
    return "_".join(path.relative_to(seed_dir).with_suffix("").parts)


def iter_seed_images(seed_dir):
    for path in sorted(seed_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = None
        relative_parts = path.relative_to(seed_dir).parts

        if len(relative_parts) > 1:
            label = LABELS_BY_DIR.get(relative_parts[0])

        if label is None:
            label = LABELS_BY_STEM.get(path.stem)

        if label is None:
            continue

        yield SeedImage(path=path, label=label, source=source_name(seed_dir, path))


def iter_seed_crops(seed_dir, issues=None):
    for seed in iter_seed_images(seed_dir):
        path = seed.path
        img = cv2.imread(str(path))

        if img is None:
            if issues is not None:
                issues.append(
                    ExtractionIssue(path, seed.label, "cv2.imread returned None")
                )
                continue
            raise RuntimeError(f"Could not read seed image: {path}")

        try:
            warped = normalise_img(img)
        except Exception as exc:
            if issues is not None:
                issues.append(
                    ExtractionIssue(path, seed.label, f"normalise_img raised {exc!r}")
                )
                continue
            raise

        if warped is None:
            if issues is not None:
                issues.append(
                    ExtractionIssue(path, seed.label, "normalise_img returned None")
                )
                continue
            raise RuntimeError(f"Could not normalise seed image: {path}")

        boxes = find_answer_boxes(warped)

        if not boxes:
            if issues is not None:
                issues.append(
                    ExtractionIssue(path, seed.label, "no answer boxes found")
                )
                continue
            raise RuntimeError(f"No answer boxes found in seed image: {path}")

        for box_index, box in enumerate(boxes):
            segment_edges = answer_segment_edges(box.x, box.w)

            for segment_index in range(5):
                x1 = segment_edges[segment_index]
                x2 = segment_edges[segment_index + 1]
                crop = warped[box.y : box.y + box.h, x1:x2]

                if crop.size == 0:
                    raise RuntimeError(
                        "Empty crop from "
                        f"{path}, box {box_index}, segment {segment_index}"
                    )

                yield Crop(
                    image=crop,
                    label=seed.label,
                    source=seed.source,
                    box_index=box_index,
                    segment_index=segment_index,
                )


def reset_split_dirs(output_dir):
    for split in ("train", "val"):
        split_dir = output_dir / split

        if split_dir.exists():
            shutil.rmtree(split_dir)

        split_dir.mkdir(parents=True)


def write_crops(crops, output_dir, val_ratio, random_seed):
    by_label = defaultdict(list)

    for crop in crops:
        by_label[crop.label].append(crop)

    rng = random.Random(random_seed)
    counts = defaultdict(lambda: defaultdict(int))

    for label, label_crops in sorted(by_label.items()):
        rng.shuffle(label_crops)
        val_count = max(1, round(len(label_crops) * val_ratio))

        for index, crop in enumerate(label_crops):
            split = "val" if index < val_count else "train"
            class_dir = output_dir / split / label
            class_dir.mkdir(parents=True, exist_ok=True)

            filename = (
                f"{crop.source}_box{crop.box_index:03d}_seg{crop.segment_index + 1}.jpg"
            )
            destination = class_dir / filename

            if not cv2.imwrite(str(destination), crop.image):
                raise RuntimeError(f"Could not write crop: {destination}")

            counts[split][label] += 1

    return counts
