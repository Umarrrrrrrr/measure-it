"""Held-out test evaluation; this script never selects a checkpoint."""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.train import MaskDataset
from models.unet import UNet


def average_precision(records, threshold):
    # One ground-truth object and at most one predicted object per image.
    records = sorted(records, key=lambda row: row[0], reverse=True)
    positives = len(records)
    tp = np.array([int(iou >= threshold) for _, iou in records], dtype=float)
    fp = 1 - tp
    recall = np.cumsum(tp) / positives
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1)
    recall = np.r_[0, recall, 1]
    precision = np.r_[0, precision, 0]
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best.pt")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    records, ious, tp, fp, fn = [], [], 0, 0, 0
    with torch.no_grad():
        for image, actual in DataLoader(MaskDataset("test"), batch_size=1):
            probability = torch.sigmoid(model(image.to(device))).cpu()[0, 0].numpy()
            predicted = probability > 0.5
            truth = actual[0, 0].numpy() > 0.5
            intersection = np.logical_and(predicted, truth).sum()
            union = np.logical_or(predicted, truth).sum()
            iou = float(intersection / union) if union else 0.0
            confidence = float(probability[predicted].mean()) if predicted.any() else 0.0
            records.append((confidence, iou))
            ious.append(iou)
            tp += int(intersection)
            fp += int(np.logical_and(predicted, ~truth).sum())
            fn += int(np.logical_and(~predicted, truth).sum())
    aps = [average_precision(records, x) for x in np.arange(0.5, 1, 0.05)]
    print(json.dumps({"test_images": len(records), "mean_mask_iou": float(np.mean(ious)),
                      "mAP@0.5": aps[0], "mAP@0.5:0.95": float(np.mean(aps)),
                      "pixel_precision": tp / (tp + fp) if tp + fp else 0,
                      "pixel_recall": tp / (tp + fn) if tp + fn else 0}, indent=2))


if __name__ == "__main__":
    main()
