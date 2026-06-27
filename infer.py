import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from classifier import EMPTY_MODEL_PATH, MARK_MODEL_PATH, load_trained_model
from detect_squares import find_answer_boxes
from extractor import answer_segment_edges
from warp import normalise_img

DEFAULT_OPTIONS = ["A", "B", "C", "D", "E"]


@dataclass(frozen=True)
class SegmentPrediction:
    option: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    empty_stage: dict[str, float]
    mark_stage: dict[str, float] | None
    box: dict[str, int]


@dataclass(frozen=True)
class QuestionPrediction:
    question: int
    answer: str
    crossed_options: list[str]
    scribble_options: list[str]
    segments: list[SegmentPrediction]


def preprocess_crop(crop, img_size):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    return torch.from_numpy(normalized).unsqueeze(0)


def predict_crops(model, classes, crops, img_size, device):
    if not crops:
        return []

    batch = torch.stack([preprocess_crop(crop, img_size) for crop in crops]).to(device)

    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1).cpu().numpy()

    predictions = []

    for row in probs:
        class_index = int(row.argmax())
        predictions.append(
            {
                "label": classes[class_index],
                "confidence": float(row[class_index]),
                "probabilities": {
                    label: float(probability)
                    for label, probability in zip(classes, row, strict=True)
                },
            }
        )

    return predictions


def predict_crops_two_stage(
    empty_model,
    empty_classes,
    empty_img_size,
    mark_model,
    mark_classes,
    mark_img_size,
    crops,
    device,
):
    empty_predictions = predict_crops(
        empty_model,
        empty_classes,
        crops,
        empty_img_size,
        device,
    )
    marked_indexes = [
        index
        for index, prediction in enumerate(empty_predictions)
        if prediction["label"] == "marked"
    ]
    mark_predictions_by_index = {}

    if marked_indexes:
        mark_predictions = predict_crops(
            mark_model,
            mark_classes,
            [crops[index] for index in marked_indexes],
            mark_img_size,
            device,
        )
        mark_predictions_by_index = dict(zip(marked_indexes, mark_predictions))

    predictions = []

    for index, empty_prediction in enumerate(empty_predictions):
        empty_probs = empty_prediction["probabilities"]
        empty_probability = empty_probs.get("empty", 0.0)
        marked_probability = empty_probs.get("marked", 0.0)
        mark_prediction = mark_predictions_by_index.get(index)

        if empty_prediction["label"] == "empty" or mark_prediction is None:
            probabilities = {
                "crossed": 0.0,
                "empty": empty_probability,
                "scribble": 0.0,
            }
            label = "empty"
            confidence = empty_probability
        else:
            mark_probs = mark_prediction["probabilities"]
            crossed_probability = marked_probability * mark_probs.get("crossed", 0.0)
            scribble_probability = marked_probability * mark_probs.get("scribble", 0.0)
            probabilities = {
                "crossed": crossed_probability,
                "empty": empty_probability,
                "scribble": scribble_probability,
            }
            label = "crossed" if crossed_probability >= scribble_probability else "scribble"
            confidence = probabilities[label]

        predictions.append(
            {
                "label": label,
                "confidence": confidence,
                "probabilities": probabilities,
                "empty_stage": empty_probs,
                "mark_stage": (
                    mark_prediction["probabilities"] if mark_prediction else None
                ),
            }
        )

    return predictions


def extract_segments(warped):
    boxes = find_answer_boxes(warped)
    segments = []

    for question_index, box in enumerate(boxes, start=1):
        edges = answer_segment_edges(box.x, box.w, segments=5)

        for option_index in range(5):
            x1 = edges[option_index]
            x2 = edges[option_index + 1]
            crop = warped[box.y : box.y + box.h, x1:x2]
            segments.append(
                {
                    "question": question_index,
                    "option_index": option_index,
                    "crop": crop,
                    "box": {
                        "x": x1,
                        "y": box.y,
                        "w": x2 - x1,
                        "h": box.h,
                    },
                }
            )

    return segments


def summarize_questions(segments, crop_predictions):
    by_question = {}

    for segment, prediction in zip(segments, crop_predictions, strict=True):
        question = segment["question"]
        option = DEFAULT_OPTIONS[segment["option_index"]]
        label = prediction["label"]
        probabilities = prediction["probabilities"]

        segment_prediction = SegmentPrediction(
            option=option,
            label=label,
            confidence=prediction["confidence"],
            probabilities=probabilities,
            empty_stage=prediction["empty_stage"],
            mark_stage=prediction["mark_stage"],
            box=segment["box"],
        )

        entry = by_question.setdefault(
            question,
            {
                "segments": [],
                "crossed_options": [],
                "scribble_options": [],
            },
        )
        entry["segments"].append(segment_prediction)

        if label == "crossed":
            entry["crossed_options"].append(option)

        if label == "scribble":
            entry["scribble_options"].append(option)

    question_predictions = []

    for question, entry in sorted(by_question.items()):
        crossed_options = entry["crossed_options"]

        if len(crossed_options) == 1:
            answer = crossed_options[0]
        elif len(crossed_options) > 1:
            answer = ",".join(crossed_options)
        else:
            answer = ""

        question_predictions.append(
            QuestionPrediction(
                question=question,
                answer=answer,
                crossed_options=crossed_options,
                scribble_options=entry["scribble_options"],
                segments=entry["segments"],
            )
        )

    return question_predictions


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MCQ mark inference on a sheet image."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--empty-model", type=Path, default=EMPTY_MODEL_PATH)
    parser.add_argument("--mark-model", type=Path, default=MARK_MODEL_PATH)
    parser.add_argument(
        "--json", type=Path, help="Optional path for detailed JSON output."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU. By default inference uses CUDA.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timings = []
    started_at = perf_counter()

    device = "cpu" if args.cpu else "cuda"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --cpu.")

    stage_started_at = perf_counter()
    image = cv2.imread(str(args.image))
    timings.append(("read_image", perf_counter() - stage_started_at))

    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    stage_started_at = perf_counter()
    warped = normalise_img(image)
    timings.append(("normalise_image", perf_counter() - stage_started_at))

    if warped is None:
        raise RuntimeError(f"Could not normalise image: {args.image}")

    stage_started_at = perf_counter()
    empty_model, empty_classes, empty_img_size = load_trained_model(
        args.empty_model,
        device,
    )
    mark_model, mark_classes, mark_img_size = load_trained_model(
        args.mark_model,
        device,
    )
    timings.append(("load_model", perf_counter() - stage_started_at))

    stage_started_at = perf_counter()
    segments = extract_segments(warped)
    timings.append(("extract_segments", perf_counter() - stage_started_at))

    if not segments:
        raise RuntimeError(f"No answer boxes found in image: {args.image}")

    stage_started_at = perf_counter()
    crop_predictions = predict_crops_two_stage(
        empty_model,
        empty_classes,
        empty_img_size,
        mark_model,
        mark_classes,
        mark_img_size,
        [segment["crop"] for segment in segments],
        device,
    )
    timings.append(("classify_crops", perf_counter() - stage_started_at))

    stage_started_at = perf_counter()
    question_predictions = summarize_questions(segments, crop_predictions)
    timings.append(("summarize", perf_counter() - stage_started_at))
    timings.append(("total_inference", perf_counter() - started_at))

    print_summary(question_predictions)
    print_confidence_summary(question_predictions)
    print_timing_summary(timings)

    if args.json:
        payload = [asdict(question) for question in question_predictions]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    show_annotated_warped(warped, question_predictions)


if __name__ == "__main__":
    main()
