from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from classifier import EMPTY_MODEL_PATH, MARK_MODEL_PATH, load_trained_model
from detect_squares import find_answer_boxes
from extractor import answer_segment_edges
from inference_types import InferenceResult, QuestionPrediction, SegmentPrediction
from warp import normalise_img

DEFAULT_OPTIONS = ["1", "2", "3", "4", "5"]


class InferencePipeline:
    @dataclass(frozen=True)
    class _ModelBundle:
        empty_model: torch.nn.Module
        empty_classes: list[str]
        empty_img_size: int
        mark_model: torch.nn.Module
        mark_classes: list[str]
        mark_img_size: int
        device: str

    @dataclass(frozen=True)
    class _PreparedImage:
        image_id: str | None
        warped: np.ndarray
        segments: list[dict]
        timings: list[tuple[str, float]]
        started_at: float

    def __init__(
        self,
        empty_model_path=EMPTY_MODEL_PATH,
        mark_model_path=MARK_MODEL_PATH,
        device=None,
        use_cpu=False,
    ):
        if device is None:
            device = self.resolve_device(use_cpu)

        self.models = self._load_models(
            empty_model_path,
            mark_model_path,
            device,
        )

    @staticmethod
    def resolve_device(use_cpu=False):
        device = "cpu" if use_cpu else "cuda"

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available. Use --cpu.")

        return device

    def predict(self, image_bytes, image_id=None):
        prepared = self._prepare_image(image_bytes, image_id=image_id)

        stage_started_at = perf_counter()
        crop_predictions = self._classify_segments(prepared.segments)
        classify_timing = perf_counter() - stage_started_at

        return self._build_result(prepared, crop_predictions, classify_timing)

    def predict_many(self, images, image_ids=None):
        images = list(images)

        if image_ids is None:
            image_ids = [None] * len(images)
        else:
            image_ids = list(image_ids)

        if len(images) != len(image_ids):
            raise ValueError("images and image_ids must have the same length.")

        prepared_images = [
            self._prepare_image(image_bytes, image_id=image_id)
            for image_bytes, image_id in zip(images, image_ids, strict=True)
        ]
        all_segments = [
            segment
            for prepared in prepared_images
            for segment in prepared.segments
        ]

        stage_started_at = perf_counter()
        all_crop_predictions = self._classify_segments(all_segments)
        classify_timing = perf_counter() - stage_started_at

        results = []
        prediction_offset = 0

        for prepared in prepared_images:
            next_offset = prediction_offset + len(prepared.segments)
            crop_predictions = all_crop_predictions[prediction_offset:next_offset]
            prediction_offset = next_offset
            results.append(
                self._build_result(prepared, crop_predictions, classify_timing)
            )

        return results

    def predict_file(self, image_path):
        image_path = Path(image_path)
        return self.predict(image_path.read_bytes(), image_id=str(image_path))

    def predict_files(self, image_paths):
        image_paths = [Path(image_path) for image_path in image_paths]
        return self.predict_many(
            [image_path.read_bytes() for image_path in image_paths],
            image_ids=[str(image_path) for image_path in image_paths],
        )

    def _load_models(self, empty_model_path, mark_model_path, device):
        empty_model, empty_classes, empty_img_size = load_trained_model(
            empty_model_path,
            device,
        )
        mark_model, mark_classes, mark_img_size = load_trained_model(
            mark_model_path,
            device,
        )

        return self._ModelBundle(
            empty_model=empty_model,
            empty_classes=empty_classes,
            empty_img_size=empty_img_size,
            mark_model=mark_model,
            mark_classes=mark_classes,
            mark_img_size=mark_img_size,
            device=device,
        )

    def _prepare_image(self, image_bytes, image_id=None):
        timings = []
        started_at = perf_counter()

        stage_started_at = perf_counter()
        image = self._decode_image(image_bytes)
        timings.append(("decode_image", perf_counter() - stage_started_at))

        source = image_id or "input image"

        if image is None:
            raise RuntimeError(f"Could not decode image: {source}")

        stage_started_at = perf_counter()
        warped = normalise_img(image)
        timings.append(("normalise_image", perf_counter() - stage_started_at))

        if warped is None:
            raise RuntimeError(f"Could not normalise image: {source}")

        stage_started_at = perf_counter()
        segments = self._extract_segments(warped)
        timings.append(("extract_segments", perf_counter() - stage_started_at))

        if not segments:
            raise RuntimeError(f"No answer boxes found in image: {source}")

        return self._PreparedImage(
            image_id=image_id,
            warped=warped,
            segments=segments,
            timings=timings,
            started_at=started_at,
        )

    @staticmethod
    def _decode_image(image_bytes):
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        if image_array.size == 0:
            return None

        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    def _extract_segments(self, warped):
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

    def _classify_segments(self, segments):
        crops = [segment["crop"] for segment in segments]
        return self._predict_crops_two_stage(crops)

    def _predict_crops_two_stage(self, crops):
        models = self.models
        empty_predictions = self._predict_crops(
            models.empty_model,
            models.empty_classes,
            crops,
            models.empty_img_size,
        )
        marked_indexes = [
            index
            for index, prediction in enumerate(empty_predictions)
            if prediction["label"] == "marked"
        ]
        mark_predictions_by_index = {}

        if marked_indexes:
            mark_predictions = self._predict_crops(
                models.mark_model,
                models.mark_classes,
                [crops[index] for index in marked_indexes],
                models.mark_img_size,
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
                crossed_probability = marked_probability * mark_probs.get(
                    "crossed", 0.0
                )
                scribble_probability = marked_probability * mark_probs.get(
                    "scribble", 0.0
                )
                probabilities = {
                    "crossed": crossed_probability,
                    "empty": empty_probability,
                    "scribble": scribble_probability,
                }
                label = (
                    "crossed"
                    if crossed_probability >= scribble_probability
                    else "scribble"
                )
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

    def _predict_crops(self, model, classes, crops, img_size):
        if not crops:
            return []

        batch = torch.stack(
            [self._preprocess_crop(crop, img_size) for crop in crops]
        ).to(self.models.device)

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

    @staticmethod
    def _preprocess_crop(crop, img_size):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        return torch.from_numpy(normalized).unsqueeze(0)

    def _build_result(self, prepared, crop_predictions, classify_timing):
        stage_started_at = perf_counter()
        question_predictions = self._summarize_questions(
            prepared.segments,
            crop_predictions,
        )
        timings = [
            *prepared.timings,
            ("classify_crops", classify_timing),
            ("summarize", perf_counter() - stage_started_at),
            ("total_inference", perf_counter() - prepared.started_at),
        ]

        return InferenceResult(
            image_id=prepared.image_id,
            warped=prepared.warped,
            question_predictions=question_predictions,
            timings=timings,
        )

    def _summarize_questions(self, segments, crop_predictions):
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
