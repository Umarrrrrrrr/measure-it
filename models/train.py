"""Train on train split; choose checkpoint by validation IoU only."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.unet import UNet


class MaskDataset(Dataset):
    def __init__(self, split: str, size: int = 256):
        self.ids = (Path("dataset/splits") / f"{split}.txt").read_text().splitlines()
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_path = Path("dataset/images") / f"{image_id}.png"
        mask_path = Path("dataset/masks") / f"{image_id}.png"
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Missing or misaligned image/mask: {image_id}")
        image = cv2.cvtColor(cv2.resize(image, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255, torch.from_numpy((mask > 127).astype(np.float32))[None]


def mean_iou(model, loader, device):
    model.eval()
    values = []
    with torch.no_grad():
        for image, mask in loader:
            predicted = (torch.sigmoid(model(image.to(device))) > 0.5).cpu()
            actual = mask.bool()
            intersection = (predicted & actual).sum(dim=(1, 2, 3)).float()
            union = (predicted | actual).sum(dim=(1, 2, 3)).float()
            values.extend(((intersection + 1e-7) / (union + 1e-7)).tolist())
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/best.pt"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(MaskDataset("train"), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(MaskDataset("val"), batch_size=args.batch_size, num_workers=0)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    best = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(image), mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        score = mean_iou(model, val_loader, device)
        print(json.dumps({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_iou": score}), flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "seed": args.seed, "size": 256, "val_iou": score}, args.output)
    print(json.dumps({"best_val_iou": best, "checkpoint": str(args.output)}))


if __name__ == "__main__":
    main()
