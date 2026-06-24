import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--bundle-failures",
        action="store_true",
        default=False,
        help="Copy failed detection images into categorized folders.",
    )

    parser.addoption(
        "--failure-bundle-dir",
        action="store",
        default="failure_review",
        help="Directory where failed images are copied.",
    )


def pytest_configure(config):
    config._box_detection_stats = []


@pytest.fixture
def record_detection_stats(request):
    def record(dataset, img_path, expected, count=None, status="ok", reason=""):
        request.config._box_detection_stats.append(
            {
                "dataset": dataset,
                "img_path": str(img_path),
                "expected": expected,
                "count": count,
                "status": status,
                "reason": reason,
            }
        )

    return record


def safe_name(text):
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def categorize_failure(row):
    status = row["status"]

    if status == "read_failed":
        return "read_failed"

    if status == "normalise_failed":
        return "normalise_failed"

    count = row["count"]
    expected = row["expected"]

    if count is None:
        return "unknown_failed"

    diff = count - expected

    if diff == 0:
        return "passed"

    if diff < 0:
        missing = abs(diff)

        if missing <= 3:
            return "near_miss_under"
        if missing <= 9:
            return "under_detected"
        return "severe_under_detected"

    if diff > 0:
        extra = diff

        if extra <= 3:
            return "near_miss_over"
        if extra <= 9:
            return "over_detected"
        return "severe_over_detected"

    return "unknown_failed"


def bundle_failed_images(rows, output_root):
    output_root = Path(output_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "failure_summary.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "filename",
                "category",
                "expected",
                "detected",
                "diff",
                "status",
                "reason",
                "original_path",
            ],
        )

        writer.writeheader()

        for row in rows:
            category = categorize_failure(row)

            if category == "passed":
                continue

            src = Path(row["img_path"])

            dataset_name = safe_name(row["dataset"])
            category_dir = run_dir / dataset_name / category
            category_dir.mkdir(parents=True, exist_ok=True)

            detected = row["count"]
            expected = row["expected"]
            diff = "" if detected is None else detected - expected

            prefix = "none" if detected is None else str(detected)
            dst = category_dir / f"detected_{prefix}__{src.name}"

            if src.exists():
                shutil.copy2(src, dst)

            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "filename": src.name,
                    "category": category,
                    "expected": expected,
                    "detected": detected,
                    "diff": diff,
                    "status": row["status"],
                    "reason": row["reason"],
                    "original_path": str(src),
                }
            )

    return run_dir, csv_path


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    rows = getattr(config, "_box_detection_stats", [])

    if not rows:
        return

    terminalreporter.section("Answer box detection statistics")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    for dataset, items in grouped.items():
        counts = [r["count"] for r in items if r["count"] is not None]
        expected = items[0]["expected"]

        total = len(items)
        passed = sum(1 for r in items if r["count"] == expected)
        failed = total - passed

        terminalreporter.write_line("")
        terminalreporter.write_line(f"[{dataset}]")
        terminalreporter.write_line(f"  total images:     {total}")
        terminalreporter.write_line(f"  passed:           {passed}/{total}")
        terminalreporter.write_line(f"  failed:           {failed}")

        categories = defaultdict(int)
        for row in items:
            category = categorize_failure(row)
            if category != "passed":
                categories[category] += 1

        if categories:
            terminalreporter.write_line("  failure categories:")
            for category, count in sorted(categories.items()):
                terminalreporter.write_line(f"    {category}: {count}")

        if counts:
            errors = [abs(c - expected) for c in counts]

            terminalreporter.write_line(f"  average detected: {mean(counts):.2f}")
            terminalreporter.write_line(f"  median detected:  {median(counts):.2f}")
            terminalreporter.write_line(f"  min detected:     {min(counts)}")
            terminalreporter.write_line(f"  max detected:     {max(counts)}")
            terminalreporter.write_line(f"  avg abs error:    {mean(errors):.2f}")

    if config.getoption("--bundle-failures"):
        output_root = config.getoption("--failure-bundle-dir")
        run_dir, csv_path = bundle_failed_images(rows, output_root)

        terminalreporter.write_line("")
        terminalreporter.write_line(f"Failure images copied to: {run_dir}")
        terminalreporter.write_line(f"Failure CSV written to:   {csv_path}")
