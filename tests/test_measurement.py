import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from inference.measure import dimensions_mm, measure, reference_corners
from models.unet import UNet


class MeasurementTests(unittest.TestCase):
    def test_reference_and_metric_geometry(self):
        marker = cv2.imread("calibration/printables/aruco_4x4_id0.png")
        image = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        self.assertIsNotNone(reference_corners(image, 0))
        mask = np.zeros((300, 400), np.uint8)
        mask[100:200, 100:300] = 1
        width, height = dimensions_mm(mask, np.array([[10, 10], [50, 10], [50, 50], [10, 50]], np.float32), 20)
        self.assertAlmostEqual(width, 99.5, places=1)
        self.assertAlmostEqual(height, 49.5, places=1)

    def test_missing_object_and_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_path = root / "image.png"
            cv2.imwrite(str(image_path), np.full((256, 256, 3), 255, np.uint8))
            camera_path = root / "camera.npz"
            np.savez(camera_path, camera_matrix=np.array([[200, 0, 128], [0, 200, 128], [0, 0, 1]], float),
                     distortion=np.zeros(5), image_size=np.array([256, 256]))
            checkpoint_path = root / "model.pt"
            torch.save({"state_dict": UNet().state_dict(), "size": 256}, checkpoint_path)
            empty = np.zeros((256, 256), np.uint8)
            with patch("inference.measure.predict_mask", return_value=(empty, 0.0)):
                result = measure(image_path, camera_path, checkpoint_path, 20, 0, root / "overlay.png")
            self.assertEqual(result["status"], "object_not_found")
            full = np.ones((256, 256), np.uint8)
            with patch("inference.measure.predict_mask", return_value=(full, 0.9)):
                result = measure(image_path, camera_path, checkpoint_path, 20, 0, root / "overlay.png")
            self.assertEqual(result["status"], "reference_not_found")

    def test_success_writes_overlay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = cv2.imread("calibration/printables/aruco_4x4_id0.png")
            image = np.full((800, 1000, 3), 255, np.uint8)
            image[100:700, 100:700] = marker
            image_path = root / "image.png"
            cv2.imwrite(str(image_path), image)
            camera_path = root / "camera.npz"
            np.savez(camera_path, camera_matrix=np.array([[800, 0, 500], [0, 800, 400], [0, 0, 1]], float),
                     distortion=np.zeros(5), image_size=np.array([1000, 800]))
            checkpoint_path = root / "model.pt"
            torch.save({"state_dict": UNet().state_dict(), "size": 256}, checkpoint_path)
            mask = np.zeros((800, 1000), np.uint8)
            mask[300:500, 750:950] = 1
            overlay_path = root / "overlay.png"
            with patch("inference.measure.predict_mask", return_value=(mask, 0.9)):
                result = measure(image_path, camera_path, checkpoint_path, 20, 0, overlay_path)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(overlay_path.exists())
            self.assertGreater(result["width_mm"], 0)
            self.assertGreater(result["height_mm"], 0)


if __name__ == "__main__":
    unittest.main()
