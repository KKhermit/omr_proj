from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from omr.utils import ensure_dir, write_jsonl


def _fill_checkmark(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Draw a tick/checkmark (✓) inside the box."""
    w, h = x2 - x1, y2 - y1
    jx, jy = random.randint(-3, 3), random.randint(-3, 3)
    # three points: left, dip (bottom-center-ish), top-right
    p1 = (x1 + int(w * 0.12) + jx, y1 + int(h * 0.55) + jy)
    p2 = (x1 + int(w * 0.38) + jx, y1 + int(h * 0.80) + jy)
    p3 = (x1 + int(w * 0.88) + jx, y1 + int(h * 0.20) + jy)
    color = random.randint(10, 80)
    thickness = random.randint(2, 4)
    cv2.line(img, p1, p2, color, thickness)
    cv2.line(img, p2, p3, color, thickness)


def _fill_x_mark(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Draw an X mark inside the box."""
    margin = random.randint(5, 10)
    jx, jy = random.randint(-3, 3), random.randint(-3, 3)
    color = random.randint(10, 80)
    thickness = random.randint(2, 4)
    cv2.line(img, (x1 + margin + jx, y1 + margin + jy), (x2 - margin + jx, y2 - margin + jy), color, thickness)
    cv2.line(img, (x2 - margin + jx, y1 + margin + jy), (x1 + margin + jx, y2 - margin + jy), color, thickness)


def _fill_circle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Draw a circle/oval inside the box."""
    side = x2 - x1
    cx = (x1 + x2) // 2 + random.randint(-3, 3)
    cy = (y1 + y2) // 2 + random.randint(-3, 3)
    rx = int(side * random.uniform(0.25, 0.38))
    ry = int(side * random.uniform(0.25, 0.38))
    color = random.randint(10, 80)
    thickness = random.randint(2, 3)
    cv2.ellipse(img, (cx, cy), (rx, ry), random.uniform(0, 30), 0, 360, color, thickness)


def _fill_heavy(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Original style: dense random dots and scribble lines."""
    for _ in range(random.randint(18, 40)):
        rx = random.randint(x1 + 4, x2 - 4)
        ry = random.randint(y1 + 4, y2 - 4)
        cv2.circle(img, (rx, ry), random.randint(2, 5), random.randint(20, 100), -1)
    for _ in range(random.randint(5, 12)):
        p1 = (random.randint(x1 + 2, x2 - 2), random.randint(y1 + 2, y2 - 2))
        p2 = (random.randint(x1 + 2, x2 - 2), random.randint(y1 + 2, y2 - 2))
        cv2.line(img, p1, p2, random.randint(30, 110), random.randint(1, 2))


# weights: checkmark and X are most common in practice
_FILL_STYLES = [_fill_checkmark, _fill_x_mark, _fill_circle, _fill_heavy]
_FILL_WEIGHTS = [0.35, 0.30, 0.15, 0.20]


def draw_checkbox(label: int, size: int = 128) -> np.ndarray:
    img = np.full((size, size), random.randint(235, 255), dtype=np.uint8)
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    x1 = random.randint(28, 40)
    y1 = random.randint(28, 40)
    side = random.randint(42, 54)
    x2 = x1 + side
    y2 = y1 + side
    thickness = random.randint(2, 4)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=random.randint(10, 70), thickness=thickness)

    if label == 1:
        style_fn = random.choices(_FILL_STYLES, weights=_FILL_WEIGHTS, k=1)[0]
        style_fn(img, x1, y1, x2, y2)
    else:
        for _ in range(random.randint(0, 4)):
            rx = random.randint(0, size - 1)
            ry = random.randint(0, size - 1)
            cv2.circle(img, (rx, ry), random.randint(1, 2), random.randint(120, 170), -1)

    angle = random.uniform(-8, 8)
    center = (size // 2, size // 2)
    m = cv2.getRotationMatrix2D(center, angle, random.uniform(0.95, 1.05))
    img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=255)

    shift = random.randint(-5, 5)
    img = np.roll(img, shift, axis=random.choice([0, 1]))

    if random.random() < 0.6:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.1, 1.0))
    if random.random() < 0.3:
        img = cv2.equalizeHist(img)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight OMR binary classification synthetic dataset")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = ensure_dir(args.output)
    train_dir = ensure_dir(root / "train")
    val_dir = ensure_dir(root / "val")
    train_rows = []
    val_rows = []

    for split_dir, count, rows, split_name in [
        (train_dir, args.train, train_rows, "train"),
        (val_dir, args.val, val_rows, "val"),
    ]:
        for idx in range(count):
            label = 1 if random.random() < 0.5 else 0
            img = draw_checkbox(label)
            file_path = split_dir / f"{split_name}_{idx:05d}_{label}.png"
            cv2.imwrite(str(file_path), img)
            rows.append(
                {
                    "image": str(file_path),
                    "label": label,
                    "split": split_name,
                    "scan": "synthetic",
                    "question": idx + 1,
                    "choice": "A",
                    "box_id": f"SYN_{split_name}_{idx:05d}",
                    "bbox": [0, 0, img.shape[1], img.shape[0]],
                    "crop_bbox": [0, 0, img.shape[1], img.shape[0]],
                    "source": "synthetic",
                }
            )

    write_jsonl(train_rows, root / "train.jsonl")
    write_jsonl(val_rows, root / "val.jsonl")
    write_jsonl(train_rows + val_rows, root / "all.jsonl")
    print(f"Synthetic dataset has been generated: {root}")
    print(f"train={len(train_rows)} | val={len(val_rows)}")


if __name__ == "__main__":
    main()
