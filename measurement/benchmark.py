"""Five end-to-end inference timings after one warm-up."""
import argparse
import json
import statistics
import time
from pathlib import Path

from inference.measure import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--marker-side-mm", type=float, required=True)
    args = parser.parse_args()
    values = []
    for trial in range(6):
        start = time.perf_counter()
        result = measure(args.image, Path("calibration/camera.npz"), Path("models/best.pt"), args.marker_side_mm, 0, Path("measurement/benchmark_overlay.png"))
        elapsed = time.perf_counter() - start
        if trial:
            values.append(elapsed)
    print(json.dumps({"status": result["status"], "runs_seconds": values, "median_seconds": statistics.median(values), "spread_seconds": [min(values), max(values)]}, indent=2))


if __name__ == "__main__":
    main()
