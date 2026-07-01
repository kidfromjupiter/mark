import json
from pathlib import Path

import cv2


def annotated_warped(warped, question_predictions):
    annotated = warped.copy()

    for question in question_predictions:
        for segment in question.segments:
            if segment.label == "empty":
                continue

            box = segment.box

            if segment.label == "crossed":
                color = (0, 180, 0)
            elif segment.label == "scribble":
                color = (0, 140, 255)
            else:
                continue

            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)
            cv2.putText(
                annotated,
                f"{segment.label} {segment.confidence:.2f}",
                (x + 2, max(12, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )

    return annotated


def show_annotated_warped(warped, question_predictions):
    annotated = annotated_warped(warped, question_predictions)
    cv2.imshow("Annotated MCQ", annotated)
    cv2.waitKey(0)
    cv2.destroyWindow("Annotated MCQ")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def write_result_outputs(
    result,
    json_path=None,
    normalised_image_path=None,
    annotated_image_path=None,
):
    if json_path:
        write_json(Path(json_path), result.json_payload)

    if normalised_image_path:
        write_image(Path(normalised_image_path), result.warped)

    if annotated_image_path:
        write_image(
            Path(annotated_image_path),
            annotated_warped(result.warped, result.question_predictions),
        )


def output_paths_for_result(output_dir, result):
    output_dir = Path(output_dir)
    stem = Path(result.image_id).stem if result.image_id else "image"

    return {
        "json_path": output_dir / f"{stem}_predictions.json",
        "normalised_image_path": output_dir / f"{stem}_normalised.jpg",
        "annotated_image_path": output_dir / f"{stem}_annotated.jpg",
    }


def print_summary(question_predictions):
    for question in question_predictions:
        answer = question.answer or "-"
        extras = []

        if len(question.crossed_options) > 1:
            extras.append("multiple")

        if question.scribble_options:
            extras.append("scribble:" + ",".join(question.scribble_options))

        suffix = f" ({'; '.join(extras)})" if extras else ""
        print(f"{question.question:02d}: {answer}{suffix}")


def print_confidence_summary(question_predictions):
    print("confidence")

    for question in question_predictions:
        segments = " ".join(
            f"{segment.option}:{segment.label}={segment.confidence:.3f}"
            for segment in question.segments
        )
        print(f"{question.question:02d}: {segments}")


def print_timing_summary(timings):
    print("timing")

    for name, seconds in timings:
        print(f"  {name}: {seconds * 1000.0:.2f} ms")
