# Objectives

These targets are chosen before collecting results. The actual object and its required tolerance may justify a documented revision before evaluating the test set.

| ID | What is measured | How | Target | Conditions | Exclusions |
| --- | --- | --- | --- | --- | --- |
| MO-1 | Absolute width and height error | Compare each system result with calliper or ruler ground truth for at least 10 new instances; report each value, mean, median, worst, and percentage error | Mean absolute error ≤ 2 mm for each dimension | Locked camera settings, flat object and reference in the same plane, fully visible | Explicitly refused images are counted separately, not silently dropped |
| MO-2 | Inference latency | Time the image-to-mm command five times on this machine; report median and min–max spread | Median ≤ 2 s per image | Same image resolution and model, after one warm-up run | Model loading is included |

## Evidence to fill from actual runs

- Hardware: Intel Core i5-7300U (4 logical CPUs), Intel HD Graphics 620, 15 GiB RAM, Arch Linux. Camera model and resolution: **pending capture**.
- Calibration image count, camera matrix, distortion coefficients, mean and per-view reprojection errors: **pending calibration**.
- Dataset image count, split counts, labelling tool and mask audit: **pending capture**.
- Held-out mask IoU, mAP@0.5, mAP@0.5:0.95, precision and recall: **pending training and test**.
- Per-instance measurement table and mean, median, worst-case errors: **pending physical measurements**.
- Raw versus undistorted error table: **pending physical measurements**.
- Five raw timing outputs and summary: **pending runnable pipeline**.

No metric will be reported without its underlying files and command output.
