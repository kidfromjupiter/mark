from dataclasses import dataclass
import cv2
import numpy as np
from warp import normalise_img

MIN_W = 80
MIN_H = 15
MAX_W = 120
MAX_H = 32
MIN_AREA = 2000
MAX_AREA = 2600
LOWER_MARGIN = 480
UPPER_MARGIN = 140

RECTANGULARITY_THRESHOLD = 0.35
RECOVERY_MIN_PASSED = 5
RECOVERY_WIDTH_TOL = 0.05
RECOVERY_HEIGHT_TOL = 0.25
RECOVERY_AREA_TOL = 0.30
RECOVERY_MAX_IOU = 0.30


@dataclass
class CandidateBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class InspectionLog:
    """Stores the bounding box and failure reason for EVERY detected contour."""

    x: int
    y: int
    w: int
    h: int
    reason: str  # "PASSED" or why it failed


# Global tracking variables for mouse callbacks
mouse_x, mouse_y = -1, -1
last_hovered_reason = None


def mouse_hover_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, last_hovered_reason
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

        # Look through our inspection history logs (passed via 'param')
        inspection_history = param
        found_hover = False

        for item in inspection_history:
            # Check if mouse is inside this specific contour's bounding rect
            if item.x <= mouse_x <= (item.x + item.w) and item.y <= mouse_y <= (
                item.y + item.h
            ):
                found_hover = True
                # Only print if it's a NEW reason to prevent terminal spamming
                if last_hovered_reason != item.reason:
                    print(
                        f"Hover Box at ({item.x}, {item.y}) -> Status/Failure: {item.reason}"
                    )
                    last_hovered_reason = item.reason
                break

        if not found_hover:
            last_hovered_reason = None


def box_iou(a, b):
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h

    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih
    union = (a.w * a.h) + (b.w * b.h) - intersection

    if union == 0:
        return 0.0

    return intersection / union


def find_answer_boxes(warped, debug=False):

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if debug:
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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

    if debug:
        cv2.imshow("cleaned", th_clean)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Extract mostly horizontal rectangle borders
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal = cv2.morphologyEx(
        th_clean,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    # Extract mostly vertical rectangle borders
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    vertical = cv2.morphologyEx(
        th_clean,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )

    # Combine border lines only
    box_lines = cv2.bitwise_or(horizontal, vertical)

    # Reconnect corners
    corner_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    box_lines = cv2.morphologyEx(
        box_lines,
        cv2.MORPH_CLOSE,
        corner_kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        box_lines,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if debug:
        cv2.imshow("grid", box_lines)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    candidates = []
    rectangularity_failures = []
    inspection_history = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = w * h

        if bbox_area == 0:
            inspection_history.append(
                InspectionLog(x, y, w, h, "Failed Zero Area Bounding Box")
            )
            continue

        contour_area = cv2.contourArea(c)
        rectangularity = contour_area / float(bbox_area)

        box = CandidateBox(x, y, w, h)

        if debug:
            # Draw all inspected boxes in blue
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Fail check 1: Positioning
        if y < UPPER_MARGIN or y > LOWER_MARGIN:
            inspection_history.append(
                InspectionLog(
                    x,
                    y,
                    w,
                    h,
                    f"Failed Positioning: Got y:{y}. Allowed: {UPPER_MARGIN}-{LOWER_MARGIN}",
                )
            )
            continue

        # Fail check 2: Aspect Ratio
        if h == 0:
            inspection_history.append(InspectionLog(x, y, w, h, "Failed Zero Height"))
            continue

        aspect = w / float(h)

        if aspect < 3.0 or aspect > 8.0:
            inspection_history.append(
                InspectionLog(
                    x,
                    y,
                    w,
                    h,
                    f"Failed Aspect Ratio: {aspect:.2f} Allowed: 3.0-8.0",
                )
            )
            continue

        # Fail check 3: Dimensions
        if w < MIN_W or h < MIN_H or h > MAX_H or w > MAX_W:
            inspection_history.append(
                InspectionLog(
                    x,
                    y,
                    w,
                    h,
                    f"Failed Dimensions: {w}x{h}. "
                    f"Min: {MIN_W}x{MIN_H}. Max: {MAX_W}x{MAX_H}",
                )
            )
            continue

        # Fail check 4: Rectangularity
        # Important: these are saved for second-pass recovery.
        if rectangularity < RECTANGULARITY_THRESHOLD:
            if debug:
                cv2.drawContours(annotated, [c], -1, (0, 0, 255), 1)

            rectangularity_failures.append(box)

            inspection_history.append(
                InspectionLog(
                    x,
                    y,
                    w,
                    h,
                    f"Failed Rectangularity: Got {rectangularity:.3f} "
                    f"< {RECTANGULARITY_THRESHOLD}. "
                    f"contour_area:{contour_area:.1f}, bbox_area:{bbox_area}",
                )
            )
            continue

        candidates.append(box)

        inspection_history.append(InspectionLog(x, y, w, h, "PASSED"))

    # Second pass:
    # Recover boxes that failed only rectangularity but match the normal box size.
    recovered = []

    if len(candidates) >= RECOVERY_MIN_PASSED:
        widths = np.array([b.w for b in candidates], dtype=np.float32)
        heights = np.array([b.h for b in candidates], dtype=np.float32)
        areas = np.array([b.w * b.h for b in candidates], dtype=np.float32)

        med_w = float(np.median(widths))
        med_h = float(np.median(heights))
        med_area = float(np.median(areas))

        for b in rectangularity_failures:
            area = b.w * b.h

            if med_w == 0 or med_h == 0 or med_area == 0:
                continue

            width_diff = abs(b.w - med_w) / med_w
            height_diff = abs(b.h - med_h) / med_h
            area_diff = abs(area - med_area) / med_area

            width_ok = width_diff <= RECOVERY_WIDTH_TOL
            height_ok = height_diff <= RECOVERY_HEIGHT_TOL
            area_ok = area_diff <= RECOVERY_AREA_TOL

            overlaps_existing = any(
                box_iou(b, existing) > RECOVERY_MAX_IOU for existing in candidates
            )

            if width_ok and height_ok and area_ok and not overlaps_existing:
                candidates.append(b)
                recovered.append(b)

                inspection_history.append(
                    InspectionLog(
                        b.x,
                        b.y,
                        b.w,
                        b.h,
                        "RECOVERED: Failed rectangularity but matched median box size",
                    )
                )

    if debug:
        # Draw accepted first-pass boxes in green
        for b in candidates:
            cv2.rectangle(
                annotated,
                (b.x, b.y),
                (b.x + b.w, b.y + b.h),
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw recovered boxes thicker/yellow-ish
        for b in recovered:
            cv2.rectangle(
                annotated,
                (b.x, b.y),
                (b.x + b.w, b.y + b.h),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.namedWindow("Annotated")
        cv2.setMouseCallback(
            "Annotated",
            mouse_hover_callback,
            param=inspection_history,
        )

        print("\n--- Hover mouse over boxes to debug failures. Press ESC to quit. ---")
        print(f"Accepted first pass: {len(candidates) - len(recovered)}")
        print(f"Recovered second pass: {len(recovered)}")
        print(f"Total candidates: {len(candidates)}\n")

        while True:
            cv2.imshow("Annotated", annotated)

            if cv2.waitKey(10) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

    return sorted(candidates, key=lambda r: (r.x, r.y))


if __name__ == "__main__":
    img = cv2.imread("dataset/uncompressed/22.jpg")
    warped = normalise_img(img)
    find_answer_boxes(warped, debug=True)
