# Definition of Done

Evidence is added beside each item when it is completed. An empty box means it has not been verified.

- [ ] At least 20 calibration photos and a measured board square; saved OpenCV parameters and reprojection error. Evidence:
- [ ] At least 70 self-captured object photos and matching masks; locked camera settings documented. Evidence:
- [ ] Fixed train/validation/test lists committed before training; test images never used for tuning. Evidence: first commit in this branch, `dataset/splits/*.txt`.
- [ ] Fine-tuned non-YOLO segmentation model; held-out IoU, mAP@0.5, mAP@0.5:0.95, precision and recall. Evidence:
- [ ] One command measures a new image in mm and saves an overlay; missing object and reference cases checked. Evidence:
- [ ] At least 10 calliper or ruler comparisons with raw outputs and mean, median, worst and percentage error. Evidence:
- [ ] Raw versus undistorted error reported on the same instances. Evidence:
- [ ] MO-1 and MO-2 measured against targets, including any misses. Evidence:
- [ ] README documents setup, commands, hardware, dataset and weight SHA-256. Evidence:
- [ ] All unfinished work and reasons disclosed. Evidence:
- [ ] Loom is at most five minutes and answers K1–K4 against this repository. Evidence:
