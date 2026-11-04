# Measure It

Camera calibration, trained object segmentation, and metric measurement from a flat ArUco reference. This is an **incomplete assessment project** until real self-captured photos, masks, physical measurements, and recorded outputs are added. No accuracy result is claimed yet.

## Current status

The Plan, Objectives, Definition of Done, and fixed split were committed first. Calibration, preparation, model training, evaluation, inference, and measurement scripts are implemented. They have only passed syntax checks; no end-to-end run is possible without the applicant's photos and measurements. The GitHub repository, PR, weights, and Loom are pending.

## Capture setup

- **Object:** pending selection and physical measurement. Use one small, rigid, flat object that stays in the same plane as the marker.
- **Camera:** pending model and resolution. Lock focus, zoom, and resolution before calibration, then keep them fixed.
- **Computer:** pending CPU, GPU, RAM, and OS report from the actual run.
- **Calibration:** print `calibration/printables/checkerboard_9x7_inner.png` on rigid backing. Measure one square with a calliper; take at least 20 varied checkerboard views at the locked camera settings. Keep the full board visible and avoid blur.
- **Reference:** print `calibration/printables/aruco_4x4_id0.png`, mount flat and rigid, and measure the outer black square side. Put it next to the object in the same plane in every measurement image. Its border must be fully visible.
- **Dataset:** photograph at least 70 different poses/backgrounds/light conditions of the object. Name raw images `001.jpg` through `070.jpg` under `dataset/raw/`. Assigning IDs fixes the committed 49 train / 14 validation / 7 test split; do not move them. Use CVAT, Label Studio, or similar to annotate the object's visible silhouette on the undistorted images. Export single-channel masks as `dataset/masks/001.png` through `070.png`, white object and black background. Record the labelling tool.
- **Measurement:** capture at least 10 additional photos with the marker and measure the object's width and height by hand. Record them in a CSV with columns `image,width_mm,height_mm`; use real paths and values. These photos should be new, not from the split.

## Setup and commands

Run from the repository root. Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m calibration.make_board
.venv/bin/python -m calibration.calibrate calibration/photos --square-mm YOUR_MEASURED_SQUARE
.venv/bin/python -m dataset.prepare
# Annotate the prepared images and place binary masks in dataset/masks/.
.venv/bin/python -m models.train --epochs 30
.venv/bin/python -m models.evaluate --checkpoint models/best.pt
.venv/bin/python -m inference.measure measurement/new_photo.jpg --marker-side-mm YOUR_MEASURED_MARKER
.venv/bin/python -m measurement.accuracy measurement/ground_truth.csv --marker-side-mm YOUR_MEASURED_MARKER
.venv/bin/python -m measurement.benchmark measurement/new_photo.jpg --marker-side-mm YOUR_MEASURED_MARKER
```

The inference command prints `object_not_found` or `reference_not_found` when either is missing. On success it prints width, height, confidence, and the overlay path. The larger measured side is called width and the smaller side height, independent of image orientation. The conversion requires the object and marker to be flat and coplanar. A tilted object invalidates this assumption.

## Reproducibility and evidence

Training seed: 42. Library versions are pinned in `requirements.txt`. The training script selects the best checkpoint by validation IoU; `models.evaluate` reads the held-out test list once after training. The weight SHA-256 will be added after real training. Unfilled result sections in `docs/Objectives.md` are intentional and must be replaced with raw outputs from an actual run. Do not report simulated measurements as evidence.

## Recording outline

Record one Loom of at most five minutes. Photograph a new object and marker, run the command, and measure it by hand on camera. Then show `dataset/prepare.py` (`undistort`), `models/unet.py` and `inference/measure.py` (`predict_mask`, `reference_corners`, `dimensions_mm`). Explain the coplanar assumption, actual raw-versus-undistorted error difference, one decision you would reverse, and one thing you did not understand and how you investigated it. Speak from the real repository and results.
