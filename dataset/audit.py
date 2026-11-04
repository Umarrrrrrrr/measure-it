"""Check that the captured dataset is complete and aligned before training."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def audit(root: Path) -> dict:
    splits = {name: (root / "splits" / f"{name}.txt").read_text().splitlines()
              for name in ("train", "val", "test")}
    expected = {"train": 49, "val": 14, "test": 7}
    for name, ids in splits.items():
        if len(ids) != expected[name]:
            raise ValueError(f"{name} split has {len(ids)} IDs, expected {expected[name]}")
    ids = [image_id for values in splits.values() for image_id in values]
    if len(set(ids)) != len(ids):
        raise ValueError("An image ID occurs in multiple splits")
    resolutions, hashes = set(), set()
    for image_id in ids:
        image_path = root / "images" / f"{image_id}.png"
        mask_path = root / "masks" / f"{image_id}.png"
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Missing or unreadable image/mask: {image_id}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image and mask have different dimensions: {image_id}")
        if not ((mask > 127).any() and (mask <= 127).any()):
            raise ValueError(f"Mask lacks foreground or background: {image_id}")
        resolutions.add((image.shape[1], image.shape[0]))
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate prepared image: {image_id}")
        hashes.add(digest)
    if len(resolutions) != 1:
        raise ValueError(f"Mixed prepared image resolutions: {sorted(resolutions)}")
    return {"counts": {name: len(values) for name, values in splits.items()},
            "total": len(ids), "resolution": list(resolutions.pop())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))


if __name__ == "__main__":
    main()
