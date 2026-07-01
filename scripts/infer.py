import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EMPTY_MODEL_PATH = Path("mcq_empty_classifier.pt")
MARK_MODEL_PATH = Path("mcq_mark_type_classifier.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MCQ mark inference on a sheet image."
    )
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--empty-model", type=Path, default=EMPTY_MODEL_PATH)
    parser.add_argument("--mark-model", type=Path, default=MARK_MODEL_PATH)
    parser.add_argument(
        "--json", type=Path, help="Optional path for detailed JSON output."
    )
    parser.add_argument(
        "--normalised-image",
        "--normalized-image",
        dest="normalised_image",
        type=Path,
        help="Optional path for the normalised sheet image.",
    )
    parser.add_argument(
        "--annotated-image",
        type=Path,
        help="Optional path for the annotated inference image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional directory for per-image JSON, normalised, and annotated "
            "outputs. Required when saving artifacts for multiple images."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the annotated image window.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU. By default inference uses CUDA.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    is_batch = len(args.images) > 1

    if is_batch and (args.json or args.normalised_image or args.annotated_image):
        raise RuntimeError(
            "Use --output-dir for saved artifacts when processing multiple images."
        )

    if is_batch and not args.no_show:
        raise RuntimeError("Use --no-show when processing multiple images.")

    from inference_outputs import (
        output_paths_for_result,
        print_confidence_summary,
        print_summary,
        show_annotated_warped,
        write_result_outputs,
    )
    from inference_pipeline import InferencePipeline

    pipeline = InferencePipeline(
        args.empty_model,
        args.mark_model,
        use_cpu=args.cpu,
    )

    if is_batch:
        results = pipeline.predict_files(args.images)
    else:
        results = [pipeline.predict_file(args.images[0])]

    for result_index, result in enumerate(results):
        if is_batch:
            if result_index:
                print()
            print(result.image_id)

        print_summary(result.question_predictions)
        print_confidence_summary(result.question_predictions)

        if args.output_dir:
            write_result_outputs(
                result,
                **output_paths_for_result(args.output_dir, result),
            )
        elif not is_batch:
            write_result_outputs(
                result,
                json_path=args.json,
                normalised_image_path=args.normalised_image,
                annotated_image_path=args.annotated_image,
            )

    if not args.no_show:
        result = results[0]
        show_annotated_warped(result.warped, result.question_predictions)


if __name__ == "__main__":
    main()
