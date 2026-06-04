from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from omr.utils import clamp_box, save_json


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_list(self) -> list[int]:
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]


def _contour_candidates(binary: np.ndarray, cfg: dict[str, Any]) -> list[Box]:
    h, w = binary.shape[:2]
    page_area = h * w
    min_area_ratio = float(cfg.get("min_area_ratio", 0.00002))
    max_area_ratio = float(cfg.get("max_area_ratio", 0.005))
    aspect_tol = float(cfg.get("aspect_ratio_tol", 0.35))
    min_fill_ratio = float(cfg.get("min_contour_fill_ratio", 0.10))
    max_fill_ratio = float(cfg.get("max_contour_fill_ratio", 0.95))

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < page_area * min_area_ratio or area > page_area * max_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw <= 3 or bh <= 3:
            continue
        aspect = bw / float(bh)
        if abs(aspect - 1.0) > aspect_tol:
            continue
        rect_area = bw * bh
        fill_ratio = area / max(rect_area, 1)
        if not (min_fill_ratio <= fill_ratio <= max_fill_ratio):
            continue
        boxes.append(Box(x, y, x + bw, y + bh))
    return boxes


def non_max_suppression(boxes: list[Box], iou_threshold: float = 0.3) -> list[Box]:
    if not boxes:
        return []
    arr = np.array([b.to_list() for b in boxes], dtype=np.float32)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    scores = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_rest = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        union = area_i + area_rest - inter + 1e-6
        iou = inter / union
        remain = np.where(iou <= iou_threshold)[0]
        order = order[remain + 1]
    return [boxes[i] for i in keep]


def detect_checkbox_contours(binary: np.ndarray, cfg: dict[str, Any]) -> list[Box]:
    boxes = _contour_candidates(binary, cfg)
    boxes = non_max_suppression(boxes, iou_threshold=float(cfg.get("nms_iou", 0.3)))
    return sorted(boxes, key=lambda b: (b.cy, b.cx))


def group_rows(boxes: list[Box], row_merge_px: int = 18) -> list[list[Box]]:
    rows: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: (b.cy, b.cx)):
        placed = False
        for row in rows:
            row_cy = np.mean([b.cy for b in row])
            if abs(box.cy - row_cy) <= row_merge_px:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    rows = [sorted(row, key=lambda b: b.cx) for row in rows]
    rows.sort(key=lambda r: np.mean([b.cy for b in r]))
    return rows


def boxes_to_template_map(
    boxes: list[Box],
    page_shape: tuple[int, int],
    valid_counts: list[int],
    choice_labels: list[str],
    row_merge_px: int = 18,
) -> dict[str, Any]:
    rows = group_rows(boxes, row_merge_px=row_merge_px)
    print(f"\nDetected row lengths: {[len(r) for r in rows]}")

    labeled: list[dict[str, Any]] = []
    question_id = 1

    for row_idx, row in enumerate(rows):
        row = sorted(row, key=lambda b: b.cx)
        n = len(row)
        print(f"\nRow {row_idx}: {n} boxes")

        if n not in valid_counts:
            print(f"Skipping row {row_idx} (got {n} boxes, valid counts: {valid_counts})")
            continue

        widths = [b.w for b in row]
        heights = [b.h for b in row]
        mean_w = np.mean(widths)
        mean_h = np.mean(heights)
        valid_geometry = all(
            abs(b.w - mean_w) <= mean_w * 0.5 and abs(b.h - mean_h) <= mean_h * 0.5
            for b in row
        )
        if not valid_geometry:
            print(f"Skipping row {row_idx} due to inconsistent geometry")
            continue

        labels = choice_labels[:n]
        for choice, box in zip(labels, row):
            labeled.append(
                {
                    "box_id": f"Q{question_id:03d}_{choice}",
                    "question": question_id,
                    "choice": choice,
                    "bbox": box.to_list(),
                }
            )
        print(f"Accepted row {row_idx} as Question {question_id} ({n} choices: {labels})")
        question_id += 1

    return {
        "page_width": int(page_shape[1]),
        "page_height": int(page_shape[0]),
        "num_questions": int(question_id - 1),
        "valid_counts": valid_counts,
        "boxes": labeled,
    }


def save_template_map(template_map: dict[str, Any], path: str | Path) -> None:
    save_json(template_map, path)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def map_detected_to_template(
    detected_boxes: list[Box],
    template_map: dict[str, Any],
    match_distance_px: float = 25.0,
) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    used = set()
    template_items = template_map["boxes"]
    for item in template_items:
        tx1, ty1, tx2, ty2 = item["bbox"]
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0
        best_idx = None
        best_dist = 1e9
        for idx, box in enumerate(detected_boxes):
            if idx in used:
                continue
            d = _distance((tcx, tcy), (box.cx, box.cy))
            if d < best_dist:
                best_idx = idx
                best_dist = d
        if best_idx is not None and best_dist <= match_distance_px:
            used.add(best_idx)
            source = "contour"
            detected_bbox: list[int] | None = detected_boxes[best_idx].to_list()
        else:
            source = "template_fallback"
            detected_bbox = None
        row = dict(item)
        # bbox always stays as the template position — used for cropping so that
        # marks drawn around or over the box don't distort the crop region.
        row["detected_bbox"] = detected_bbox
        row["source"] = source
        assigned.append(row)
    return assigned


def extract_crops_from_template(
    aligned_bgr: np.ndarray,
    template_map: dict[str, Any],
    detector_cfg: dict[str, Any],
    binary: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if binary is None:
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    detected = detect_checkbox_contours(binary, detector_cfg)
    mapped = map_detected_to_template(
        detected,
        template_map,
        match_distance_px=float(detector_cfg.get("match_distance_px", 25.0)),
    )
    h, w = aligned_bgr.shape[:2]
    pad_ratio = float(detector_cfg.get("pad_ratio", 0.35))
    results: list[dict[str, Any]] = []
    for item in mapped:
        x1, y1, x2, y2 = item["bbox"]
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(round(bw * pad_ratio))
        pad_y = int(round(bh * pad_ratio))
        crop_box = clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), w, h)
        cx1, cy1, cx2, cy2 = crop_box
        crop = aligned_bgr[cy1:cy2, cx1:cx2].copy()
        row = dict(item)
        row["crop_bbox"] = [cx1, cy1, cx2, cy2]
        row["crop"] = crop
        results.append(row)
    return results


def save_crops(items: list[dict[str, Any]], output_dir: str | Path, scan_stem: str) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for item in items:
        path = output_dir / f"{scan_stem}__{item['box_id']}.png"
        cv2.imwrite(str(path), item["crop"])
        row = dict(item)
        row.pop("crop", None)
        row["image"] = str(path)
        saved.append(row)
    return saved


def draw_template_boxes(image: np.ndarray, items: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for item in items:
        x1, y1, x2, y2 = map(int, item["bbox"])
        color = (0, 255, 0) if item.get("source") == "contour" else (0, 255, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            item["box_id"],
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


__all__ = [
    "Box",
    "detect_checkbox_contours",
    "group_rows",
    "boxes_to_template_map",
    "save_template_map",
    "extract_crops_from_template",
    "save_crops",
    "draw_template_boxes",
]
