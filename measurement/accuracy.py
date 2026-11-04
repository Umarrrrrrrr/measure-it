"""Compare system outputs with physical ground truth for the same images."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from inference.measure import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path, help="CSV: image,width_mm,height_mm")
    parser.add_argument("--marker-side-mm", type=float, required=True)
    parser.add_argument("--camera", type=Path, default=Path("calibration/camera.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("measurement/results.csv"))
    args = parser.parse_args()
    with args.ground_truth.open(newline="") as handle:
        inputs = list(csv.DictReader(handle))
    if len(inputs) < 10:
        raise ValueError("At least 10 physically measured instances are required")
    rows = []
    for item in inputs:
        actual = sorted((float(item["width_mm"]), float(item["height_mm"])), reverse=True)
        for mode in ("undistorted", "raw"):
            result = measure(Path(item["image"]), args.camera, args.checkpoint, args.marker_side_mm, 0,
                             Path("measurement/overlays") / f"{Path(item['image']).stem}_{mode}.png", mode == "undistorted")
            row = {"image": item["image"], "mode": mode, "status": result["status"], "actual_width_mm": actual[0], "actual_height_mm": actual[1]}
            if result["status"] == "ok":
                for name, ground_truth in zip(("width", "height"), actual):
                    predicted = result[f"{name}_mm"]
                    row[f"predicted_{name}_mm"] = predicted
                    row[f"absolute_{name}_error_mm"] = abs(predicted - ground_truth)
                    row[f"percentage_{name}_error"] = 100 * abs(predicted - ground_truth) / ground_truth
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mode", "status", "actual_width_mm", "actual_height_mm", "predicted_width_mm", "absolute_width_error_mm", "percentage_width_error", "predicted_height_mm", "absolute_height_error_mm", "percentage_height_error"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for mode in ("undistorted", "raw"):
        subset = [row for row in rows if row["mode"] == mode]
        accepted = [row for row in subset if row["status"] == "ok"]
        summary[mode] = {"accepted": len(accepted), "refused": len(subset) - len(accepted)}
        for name in ("width", "height"):
            values = [row[f"absolute_{name}_error_mm"] for row in accepted]
            percentages = [row[f"percentage_{name}_error"] for row in accepted]
            if values:
                summary[mode][name] = {"mean_absolute_error_mm": statistics.mean(values), "median_absolute_error_mm": statistics.median(values), "worst_absolute_error_mm": max(values), "mean_percentage_error": statistics.mean(percentages)}
    print(json.dumps(summary, indent=2))
    print(f"Per-instance results: {args.output}")


if __name__ == "__main__":
    main()
