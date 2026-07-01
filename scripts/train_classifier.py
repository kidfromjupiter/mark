import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MCQ mark classifier.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Train on CPU. By default training uses CUDA.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "empty", "mark"],
        default="all",
        help="Which stage to train. Defaults to both stages.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from classifier import train

    train(device="cpu" if args.cpu else "cuda", stage=args.stage)


if __name__ == "__main__":
    main()
