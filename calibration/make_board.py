"""Create printable checkerboard and ArUco reference; measure prints before use."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("calibration/printables"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    square_px = 120
    board = np.zeros((8 * square_px, 10 * square_px), np.uint8)
    for row in range(8):
        for col in range(10):
            board[row * square_px:(row + 1) * square_px, col * square_px:(col + 1) * square_px] = 255 if (row + col) % 2 else 0
    cv2.imwrite(str(args.out / "checkerboard_9x7_inner.png"), board)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 600)
    cv2.imwrite(str(args.out / "aruco_4x4_id0.png"), marker)
    print("Print at any suitable size, mount flat, and measure one checker square and the ArUco black square.")


if __name__ == "__main__":
    main()
