import argparse
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from detect_squares import find_answer_boxes
from warp import normalise_img

DEFAULT_SEED_DIR = Path("dataset/seed")
DEFAULT_OUTPUT_DIR = Path("dataset")
DEFAULT_VAL_RATIO = 0.2
DEFAULT_RANDOM_SEED = 7

LABELS_BY_STEM = {
    "cross_big": "crossed",
    "cross_small": "crossed",
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


def answer_segment_edges(x, w, segments=5):
    return [int(round(x + (w * i) / float(segments))) for i in range(segments + 1)]


def iter_seed_crops(seed_dir):
    for path in sorted(seed_dir.glob("*")):
        if path.stem not in LABELS_BY_STEM:
            continue

        label = LABELS_BY_STEM[path.stem]
        img = cv2.imread(str(path))

        if img is None:
            raise RuntimeError(f"Could not read seed image: {path}")

        warped = normalise_img(img)

        if warped is None:
            raise RuntimeError(f"Could not normalise seed image: {path}")

        boxes = find_answer_boxes(warped)

        if not boxes:
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
                    label=label,
                    source=path.stem,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract answer-square crops into classifier train/val folders."
    )
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    reset_split_dirs(args.output_dir)
    counts = write_crops(
        list(iter_seed_crops(args.seed_dir)),
        args.output_dir,
        args.val_ratio,
        args.random_seed,
    )

    for split in ("train", "val"):
        print(split)

        for label, count in sorted(counts[split].items()):
            print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
