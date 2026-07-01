from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUG_PREFIX = "aug_balance"


@dataclass(frozen=True)
class BalanceIssue:
    split: str
    class_name: str
    path: Path | None
    reason: str


@dataclass(frozen=True)
class BalanceResult:
    created: dict[str, int]
    issues: list[BalanceIssue]


def image_files(class_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def next_aug_index(class_dir: Path) -> int:
    indexes = []
    for path in image_files(class_dir):
        if not path.stem.startswith(AUG_PREFIX):
            continue
        suffix = path.stem.removeprefix(AUG_PREFIX).lstrip("_")
        if suffix.isdigit():
            indexes.append(int(suffix))
    return max(indexes, default=0) + 1


def fit_on_canvas(
    img: Image.Image,
    size: tuple[int, int],
    fill: tuple[int, int, int],
) -> Image.Image:
    canvas = Image.new("RGB", size, fill)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def translate_on_canvas(
    img: Image.Image,
    dx: int,
    dy: int,
    fill: tuple[int, int, int],
) -> Image.Image:
    canvas = Image.new("RGB", img.size, fill)
    canvas.paste(img, (dx, dy))
    return canvas


def affine_augment(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    width, height = img.size
    fill = tuple(int(c) for c in mean_color(img))

    angle = rng.uniform(-10.0, 10.0)
    scale = rng.uniform(0.92, 1.08)
    translate_x = int(rng.uniform(-0.06, 0.06) * width)
    translate_y = int(rng.uniform(-0.06, 0.06) * height)

    scaled = img.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    scaled = fit_on_canvas(scaled, (width, height), fill)
    augmented = scaled.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )
    augmented = translate_on_canvas(augmented, translate_x, translate_y, fill)

    brightness = rng.uniform(0.9, 1.1)
    contrast = rng.uniform(0.9, 1.15)
    augmented = ImageEnhance.Brightness(augmented).enhance(brightness)
    augmented = ImageEnhance.Contrast(augmented).enhance(contrast)

    if rng.random() < 0.25:
        augmented = augmented.filter(
            ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45))
        )

    return augmented


def mean_color(img: Image.Image) -> tuple[float, float, float]:
    pixels = img.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    if isinstance(pixels, int):
        return (pixels, pixels, pixels)
    return tuple(pixels[:3])


def load_augmented_image(path: Path, rng: random.Random) -> Image.Image:
    with Image.open(path) as img:
        return affine_augment(img, rng)


def balance_split(
    split_dir: Path,
    rng: random.Random,
    target: int | None,
) -> BalanceResult:
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    counts = {class_dir.name: len(image_files(class_dir)) for class_dir in class_dirs}
    target_count = target or max(counts.values(), default=0)
    created: dict[str, int] = {}
    issues: list[BalanceIssue] = []

    for class_dir in class_dirs:
        files = image_files(class_dir)
        source_files = list(files)
        missing = target_count - len(files)
        if missing <= 0:
            created[class_dir.name] = 0
            continue

        if not source_files:
            issues.append(
                BalanceIssue(
                    split=split_dir.name,
                    class_name=class_dir.name,
                    path=None,
                    reason="no source images available",
                )
            )
            created[class_dir.name] = 0
            continue

        aug_index = next_aug_index(class_dir)
        created_count = 0
        bad_sources: set[Path] = set()

        for _ in range(missing):
            available_sources = [p for p in source_files if p not in bad_sources]

            while available_sources:
                source = rng.choice(available_sources)

                try:
                    augmented = load_augmented_image(source, rng)
                    break
                except Exception as exc:
                    bad_sources.add(source)
                    issues.append(
                        BalanceIssue(
                            split=split_dir.name,
                            class_name=class_dir.name,
                            path=source,
                            reason=f"augmentation failed: {exc!r}",
                        )
                    )
                    available_sources = [
                        p for p in source_files if p not in bad_sources
                    ]
            else:
                issues.append(
                    BalanceIssue(
                        split=split_dir.name,
                        class_name=class_dir.name,
                        path=None,
                        reason="no usable source images available",
                    )
                )
                break

            output = (
                class_dir / f"{AUG_PREFIX}_{aug_index:04d}{source.suffix.lower()}"
            )
            augmented.save(output, quality=95)
            aug_index += 1
            created_count += 1

        created[class_dir.name] = created_count

    return BalanceResult(created=created, issues=issues)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance class folders by generating augmented images."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target images per class. Defaults to each split max.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_issues: list[BalanceIssue] = []

    for split in args.splits:
        split_dir = args.dataset / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        before = {
            class_dir.name: len(image_files(class_dir))
            for class_dir in sorted(split_dir.iterdir())
            if class_dir.is_dir()
        }
        result = balance_split(split_dir, rng, args.target)
        all_issues.extend(result.issues)
        after = {
            class_dir.name: len(image_files(class_dir))
            for class_dir in sorted(split_dir.iterdir())
            if class_dir.is_dir()
        }

        print(f"{split}:")
        for class_name in sorted(after):
            print(
                f"  {class_name}: {before[class_name]} -> {after[class_name]} "
                f"(created {result.created[class_name]})"
            )

    if all_issues:
        print("skipped")
        skipped_by_reason = defaultdict(int)

        for issue in all_issues:
            skipped_by_reason[(issue.split, issue.class_name, issue.reason)] += 1

        for (split, class_name, reason), count in sorted(skipped_by_reason.items()):
            print(f"  {split}/{class_name}: {reason}: {count}")


if __name__ == "__main__":
    main()
