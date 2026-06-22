from dataclasses import dataclass
import cv2
import numpy as np
from warp import normalise_img

MIN_W = 80
MIN_H = 15
MAX_W = 120
MAX_H = 25
MIN_AREA = 2000
MAX_AREA = 2600
LOWER_MARGIN = 480
UPPER_MARGIN = 140

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
            if item.x <= mouse_x <= (item.x + item.w) and item.y <= mouse_y <= (item.y + item.h):
                found_hover = True
                # Only print if it's a NEW reason to prevent terminal spamming
                if last_hovered_reason != item.reason:
                    print(f"Hover Box at ({item.x}, {item.y}) -> Status/Failure: {item.reason}")
                    last_hovered_reason = item.reason
                break
        
        if not found_hover:
            last_hovered_reason = None

def find_answer_boxes(warped, debug=False):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if debug:
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th_clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    inspection_history = []  # Keep track of everything we inspect for debugging

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = w * h
        contour_area = cv2.contourArea(c)
        
        rectangularity = contour_area / float(bbox_area)
        
        # Pre-draw ALL raw bounding boxes in blue so you can see what failed
        if debug:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 1, cv2.LINE_AA)
        # Fail check 1: Positioning
        if y< UPPER_MARGIN or y > LOWER_MARGIN:
            inspection_history.append(InspectionLog(x, y, w, h, f"Failed Positioning: Got y:{y}. ({LOWER_MARGIN}-{UPPER_MARGIN}"))
            continue


        # Fail check 2: Rectangularity
        if rectangularity<  0.7:
            inspection_history.append(InspectionLog(x, y, w, h, f"Failed Rectangularity: Got {rectangularity} < 0.7"))
            continue

        # Fail check 3: Aspect Ratio
        aspect = w / float(h)
        if aspect < 3.0 or aspect > 8.0:
            inspection_history.append(InspectionLog(x, y, w, h, f"Failed Aspect Ratio: {aspect:.2f} (Allowed: 3.0-8.0)"))
            continue

        # Fail check 4: Minimum Width or Height
        if w < MIN_W or h < MIN_H or h > MAX_H or w > MAX_W:
            inspection_history.append(InspectionLog(x, y, w, h, f"Failed Dimensions: {w}x{h} (Min required: {MIN_W}x{MIN_H}) (Max: {MAX_W}x{MAX_H})"))
            continue

        # If it bypassed all continues, it passes!
        candidates.append(CandidateBox(x, y, w, h))
        inspection_history.append(InspectionLog(x, y, w, h, "PASSED (Valid Candidate)"))

    if debug:
        # Overlay green boxes over the verified winners
        for c in candidates:
            cv2.rectangle(annotated, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 255, 0), 1, cv2.LINE_AA)
            
        cv2.namedWindow("Annotated")
        
        # Link mouse tracking callback and explicitly pass our history list to it
        cv2.setMouseCallback("Annotated", mouse_hover_callback, param=inspection_history)
        
        print("\n--- Hover mouse over blue boxes to debug failures. Press ESC to quit. ---\n")
        
        while True:
            cv2.imshow("Annotated", annotated)
            # Break loop on 'ESC' key press
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    return sorted(candidates, key=lambda r: (r.x, r.y))

if __name__ == "__main__":
    img = cv2.imread("dataset/good/52.jpeg")
    warped = normalise_img(img)
    find_answer_boxes(warped, debug=True)
