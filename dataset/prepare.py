"""Undistort raw photos and validate captured masks and fixed split IDs."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def load_camera(path: Path):
    with np.load(path) as params:
        return params["camera_matrix"], params["distortion"], tuple(params["image_size"])


def undistort(image: np.ndarray, camera_path: Path) -> np.ndarray:
    matrix, distortion, size = load_camera(camera_path)
    if (image.shape[1], image.shape[0]) != size:
        raise ValueError(f"Image resolution does not match calibration: expected {size}")
    return cv2.undistort(image, matrix, distortion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--out", type=Path, default=Path("dataset/images"))
    parser.add_argument("--camera", type=Path, default=Path("calibration/camera.npz"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ids = [line.strip() for split in ("train", "val", "test") for line in (Path("dataset/splits") / f"{split}.txt").read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("Split IDs overlap")
    for image_id in ids:
        matches = [p for p in args.raw.glob(f"{image_id}.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(matches) != 1:
            raise ValueError(f"Expected one raw image for {image_id}, found {len(matches)}")
        image = cv2.imread(str(matches[0]))
        if image is None:
            raise ValueError(f"Unreadable image: {matches[0]}")
        cv2.imwrite(str(args.out / f"{image_id}.png"), undistort(image, args.camera))
    print(f"Prepared {len(ids)} undistorted images in {args.out}")


if __name__ == "__main__":
    main()
