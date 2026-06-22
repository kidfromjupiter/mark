from dataclasses import dataclass

import cv2
import numpy as np

from warp import normalise_img


MIN_W = 80
MIN_H = 15
MIN_AREA = 1800
MAX_AREA = 2200


@dataclass
class CandidateBox:
    x: int
    y: int
    w: int
    h: int


def find_answer_boxes(img):
    warped = normalise_img(img)

    if warped is None:
        return []

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th_clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        th_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        if len(approx) != 4:
            continue

        aspect = w / float(h)

        if aspect < 3.0 or aspect > 8.0:
            continue

        if w < MIN_W or h < MIN_H:
            continue

        candidates.append(CandidateBox(x, y, w, h))

    return sorted(candidates, key=lambda r: (r.x, r.y))


