# Measure It — Plan

**Start:** 2026-11-04. **Budget:** 8 active work hours; installs and training waits excluded. Record actual start and elapsed time before submission.

I will measure a small, rigid, flat household object with a phone camera and a rigid rectangular reference beside it on the same plane. The object, camera model, resolution, physical dimensions, and reference dimensions must be recorded from the actual capture setup. I will lock focus, zoom, and resolution for all photos. No downloaded object images or invented measurements will be used.

| Order | Work | Active time |
| --- | --- | ---: |
| 1 | Print a checkerboard of at least 7×9 inner corners, mount it flat, measure a square, take at least 20 views, calibrate with OpenCV, inspect reprojection error | 1 h |
| 2 | Take at least 70 object photos and separate measurement photos; undistort and annotate object masks; freeze the committed 49/14/7 train/validation/test lists before training | 2 h |
| 3 | Fine-tune a small U-Net on the training masks, select a checkpoint by validation IoU only, then report held-out test metrics once | 1.5 h |
| 4 | Build and check one-image inference: undistort, segment, detect reference, convert pixel extents to millimetres, save overlay; refuse missing object/reference | 1.5 h |
| 5 | Measure at least 10 instances with a calliper or ruler; compute per-instance error, summary statistics, and raw-versus-undistorted comparison | 1 h |
| 6 | Run the two objectives, write evidence and limitations, rehearse and record the five-minute Loom | 1 h |

The split lists are fixed in the first commit using capture IDs 001–070. If extra photos are captured, assign them to training only and disclose that choice; never move test IDs after seeing results. The test set is not used for checkpoint selection or thresholds. I will prioritize requirements 1–4 and declare later requirements incomplete if time or physical inputs run out. I am initially skipping a separate condition stress test and advanced refusal threshold; I will add them only after measured accuracy is complete.

**Decision record (provisional):** A small U-Net keeps the training and inference path inspectable. A general prompt-based pretrained segmenter would reduce labelling work, but would not satisfy the requirement to train on our own images. Final costs and any changed decision will be recorded after the run.
