from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import cv2

from omr.detector import extract_crops_from_template
from omr.preprocessing import preprocess_scan_and_template
from omr.utils import ensure_dir, list_inputs, load_json, load_yaml, read_csv, write_jsonl


def load_label_index(path: str | Path | None) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not path:
        return {}
    rows = read_csv(path)
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        scan = Path(row["scan"]).stem
        key = (scan, int(row["question"]), row["choice"])
        index[key] = row
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract checkbox crops from the scanned copy and construct the JSONL list")
    parser.add_argument("--config", required=True)
    parser.add_argument("--scans", required=True, help="Scanned copy directory or single file")
    parser.add_argument("--labels", default=None, help="Optional mark CSV: scan, question, choice, label, [the split]")
    parser.add_argument("--output", default="data/manifests", help="output directory")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    template_map = load_json(cfg["data"]["template_map"])
    label_index = load_label_index(args.labels)
    output_dir = ensure_dir(args.output)
    crops_dir = ensure_dir(output_dir / "images")

    rows: list[dict[str, Any]] = []
    for scan_path in list_inputs(args.scans):
        aligned, binary, _, _ = preprocess_scan_and_template(scan_path, cfg["data"]["template"], cfg["preprocess"])
        items = extract_crops_from_template(aligned, template_map, cfg["detector"], binary=binary)
        scan_stem = Path(scan_path).stem
        scan_crop_dir = ensure_dir(crops_dir / scan_stem)
        for item in items:
            crop_path = scan_crop_dir / f"{item['box_id']}.png"
            cv2.imwrite(str(crop_path), item["crop"])
            key = (scan_stem, int(item["question"]), item["choice"])
            label_row = label_index.get(key)
            if label_row is None:
                label = None
                split = "unlabeled"
            else:
                label = int(label_row["label"])
                split = label_row.get("split", "") or ""
            rows.append(
                {
                    "image": str(crop_path),
                    "label": label,
                    "split": split,
                    "scan": str(scan_path),
                    "question": int(item["question"]),
                    "choice": item["choice"],
                    "box_id": item["box_id"],
                    "bbox": item["bbox"],
                    "crop_bbox": item["crop_bbox"],
                    "source": item.get("source", "unknown"),
                }
            )

    labeled = [r for r in rows if r["label"] is not None]
    unlabeled = [r for r in rows if r["label"] is None]

    if labeled:
        random.seed(int(cfg.get("seed", 42)))
        missing_split = [r for r in labeled if not r["split"]]
        if missing_split:
            random.shuffle(missing_split)
            val_ratio = float(cfg["train"].get("val_ratio", 0.2))
            val_count = int(round(len(missing_split) * val_ratio))
            val_set = set(id(r) for r in missing_split[:val_count])
            for r in missing_split:
                r["split"] = "val" if id(r) in val_set else "train"
        train_rows = [r for r in labeled if r["split"] == "train"]
        val_rows = [r for r in labeled if r["split"] == "val"]
        write_jsonl(train_rows, output_dir / "train.jsonl")
        write_jsonl(val_rows, output_dir / "val.jsonl")
    if unlabeled:
        write_jsonl(unlabeled, output_dir / "unlabeled.jsonl")
    write_jsonl(rows, output_dir / "all.jsonl")

    print(f"total crops: {len(rows)}")
    print(f"labeled: {len(labeled)} | unlabeled: {len(unlabeled)}")
    if labeled:
        print(f"train: {len([r for r in labeled if r['split']=='train'])} | val: {len([r for r in labeled if r['split']=='val'])}")


if __name__ == "__main__":
    main()
