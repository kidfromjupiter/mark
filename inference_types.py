from dataclasses import asdict, dataclass

import numpy as np


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


@dataclass(frozen=True)
class InferenceResult:
    image_id: str | None
    warped: np.ndarray
    question_predictions: list[QuestionPrediction]
    timings: list[tuple[str, float]]

    @property
    def json_payload(self):
        return [asdict(question) for question in self.question_predictions]
