import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SEED_DIR,
    DEFAULT_VAL_RATIO,
    iter_seed_crops,
    reset_split_dirs,
    write_crops,
)


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

    issues = []
    reset_split_dirs(args.output_dir)
    counts = write_crops(
        list(iter_seed_crops(args.seed_dir, issues)),
        args.output_dir,
        args.val_ratio,
        args.random_seed,
    )

    for split in ("train", "val"):
        print(split)

        for label, count in sorted(counts[split].items()):
            print(f"  {label}: {count}")

    if issues:
        print("skipped")
        skipped_by_reason = defaultdict(int)

        for issue in issues:
            skipped_by_reason[issue.reason] += 1

        for reason, count in sorted(skipped_by_reason.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
