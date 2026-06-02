from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import yaml


LOGGER = logging.getLogger("omr")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("write_csv 收到空数据，无法推断表头。")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def is_pdf_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in PDF_SUFFIXES


def list_inputs(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    files = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES | PDF_SUFFIXES]
    return files


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalized_entropy(prob: float) -> float:
    eps = 1e-8
    p = float(np.clip(prob, eps, 1.0 - eps))
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(entropy / np.log(2.0))


def box_xyxy_to_xywh(box: list[int] | tuple[int, int, int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return int(x1), int(y1), int(x2), int(y2)


def draw_labeled_boxes(
    image: np.ndarray,
    items: list[dict[str, Any]],
    font_scale: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    canvas = image.copy()
    for item in items:
        x1, y1, x2, y2 = map(int, item["bbox"])
        uncertain = bool(item.get("uncertain", False))
        pred = item.get("pred", "unknown")
        if uncertain:
            color = (0, 255, 255)
        elif pred == "filled":
            color = (0, 180, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = item.get("box_id", "box")
        prob = item.get("prob_fill")
        score_text = f"{label} {prob:.2f}" if prob is not None else label
        cv2.putText(
            canvas,
            score_text,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


__all__ = [
    "LOGGER",
    "setup_logging",
    "set_seed",
    "ensure_dir",
    "load_yaml",
    "save_yaml",
    "load_json",
    "save_json",
    "read_jsonl",
    "write_jsonl",
    "read_csv",
    "write_csv",
    "is_image_file",
    "is_pdf_file",
    "list_inputs",
    "imread",
    "sigmoid_np",
    "normalized_entropy",
    "box_xyxy_to_xywh",
    "clamp_box",
    "draw_labeled_boxes",
]
