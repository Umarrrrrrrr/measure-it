"""Measure a flat object beside a measured ArUco reference on one plane."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset.prepare import undistort
from models.unet import UNet


def reference_corners(image: np.ndarray, marker_id: int) -> np.ndarray | None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return None
    matches = np.where(ids.reshape(-1) == marker_id)[0]
    return corners[int(matches[0])].reshape(4, 2).astype(np.float32) if len(matches) == 1 else None


def predict_mask(image: np.ndarray, model: UNet, device: str, size: int = 256) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    resized = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float()[None].to(device) / 255
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (width, height))
    mask = (probability >= 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask), 0.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    confidence = float(probability[mask != 0].mean()) if mask.any() else 0.0
    return mask, confidence


def dimensions_mm(mask: np.ndarray, corners: np.ndarray, marker_side_mm: float) -> tuple[float, float]:
    if marker_side_mm <= 0:
        raise ValueError("Measured marker side must be positive")
    target = np.array([[0, 0], [marker_side_mm, 0], [marker_side_mm, marker_side_mm], [0, marker_side_mm]], np.float32)
    transform = cv2.getPerspectiveTransform(corners, target)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Object not found")
    points_mm = cv2.perspectiveTransform(max(contours, key=cv2.contourArea).astype(np.float32), transform)
    (_, _), (width, height), _ = cv2.minAreaRect(points_mm)
    return tuple(sorted((float(width), float(height)), reverse=True))


def measure(image_path: Path, camera_path: Path, checkpoint_path: Path, marker_side_mm: float,
            marker_id: int, overlay_path: Path, apply_undistortion: bool = True) -> dict:
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = undistort(raw, camera_path) if apply_undistortion else raw
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mask, confidence = predict_mask(image, model, device, checkpoint.get("size", 256))
    if not mask.any():
        return {"status": "object_not_found", "confidence": 0.0}
    corners = reference_corners(image, marker_id)
    if corners is None:
        return {"status": "reference_not_found", "confidence": confidence}
    width, height = dimensions_mm(mask, corners, marker_side_mm)
    overlay = image.copy()
    overlay[mask != 0] = (0.55 * overlay[mask != 0] + 0.45 * np.array([0, 180, 0])).astype(np.uint8)
    cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.putText(overlay, f"{width:.2f} x {height:.2f} mm | {confidence:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)
    return {"status": "ok", "width_mm": width, "height_mm": height, "confidence": confidence, "overlay": str(overlay_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--camera", type=Path, default=Path("calibration/camera.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--marker-side-mm", type=float, required=True, help="Physically measured black square side")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--overlay", type=Path, default=Path("measurement/overlay.png"))
    parser.add_argument("--raw", action="store_true", help="Skip undistortion for the controlled comparison only")
    args = parser.parse_args()
    print(json.dumps(measure(args.image, args.camera, args.checkpoint, args.marker_side_mm, args.marker_id, args.overlay, not args.raw), indent=2))


if __name__ == "__main__":
    main()
