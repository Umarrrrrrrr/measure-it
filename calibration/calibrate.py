"""Calibrate a fixed-focus camera from self-captured checkerboard photos."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def calibrate(paths: list[Path], columns: int, rows: int, square_mm: float, output: Path) -> dict:
    if square_mm <= 0:
        raise ValueError("Measured square size must be positive")
    grid = np.zeros((columns * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_mm
    object_points, image_points, used = [], [], []
    image_size = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError(f"Mixed image resolutions: {path}")
        image_size = size
        found, corners = cv2.findChessboardCornersSB(gray, (columns, rows))
        if found:
            object_points.append(grid.copy())
            image_points.append(corners)
            used.append(path.name)
    if len(used) < 20:
        raise ValueError(f"Only {len(used)} usable boards; need at least 20")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors = []
    for obj, corners, rotation, translation in zip(object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        errors.append(float(np.mean(np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1))))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, camera_matrix=matrix, distortion=distortion, image_size=np.array(image_size))
    return {"used_images": used, "rms_px": float(rms), "mean_reprojection_error_px": float(np.mean(errors)),
            "per_view_error_px": dict(zip(used, errors)), "camera_matrix": matrix.tolist(),
            "distortion": distortion.tolist(), "image_size": list(image_size)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, help="Folder of checkerboard photos")
    parser.add_argument("--columns", type=int, default=9, help="Inner corners across")
    parser.add_argument("--rows", type=int, default=7, help="Inner corners down")
    parser.add_argument("--square-mm", type=float, required=True, help="Measured board square width")
    parser.add_argument("--output", type=Path, default=Path("calibration/camera.npz"))
    args = parser.parse_args()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    result = calibrate(paths, args.columns, args.rows, args.square_mm, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
