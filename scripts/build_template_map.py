from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from omr.detector import boxes_to_template_map, detect_checkbox_contours, draw_template_boxes, save_template_map
from omr.preprocessing import adaptive_binarize, load_input_as_bgr, to_gray
from omr.utils import ensure_dir, load_yaml


_ALL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _resolve_template_args(
    cfg: dict, variant: str | None, template_override: str | None, output_override: str | None
) -> tuple[str, str, list[int], list[str]]:
    """Return (template_path, output_path, valid_counts, choice_labels)."""
    if variant is not None:
        variants = cfg.get("templates", {})
        if variant not in variants:
            available = list(variants.keys()) or ["(none defined)"]
            raise SystemExit(f"Unknown template variant '{variant}'. Available: {available}")
        entry = variants[variant]
        template_path = template_override or entry["template"]
        output_path = output_override or entry["template_map"]
        if "choices" in entry:
            # single-format variant: exactly one valid count
            choice_labels = list(entry["choices"])
            valid_counts = [len(choice_labels)]
        else:
            # mixed variant: multiple valid counts, labels auto-assigned A-Z
            valid_counts = list(entry["valid_counts"])
            choice_labels = _ALL_LABELS[:max(valid_counts)]
    else:
        template_path = template_override or cfg["data"]["template"]
        output_path = output_override or cfg["data"]["template_map"]
        choice_labels = list(cfg["detector"].get("choices", ["A", "B", "C", "D"]))
        valid_counts = [len(choice_labels)]
    return template_path, output_path, valid_counts, choice_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically generate template_map.json from the template page")
    parser.add_argument("--config", required=True, help="config YAML path")
    parser.add_argument("--variant", default=None, help="named template variant from config.templates (e.g. 2choice, 4choice, 6choice)")
    parser.add_argument("--template", default=None, help="override template PDF/image path")
    parser.add_argument("--output", default=None, help="override output template_map.json path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    template_path, output_path, valid_counts, choice_labels = _resolve_template_args(cfg, args.variant, args.template, args.output)

    print(f"Template:     {template_path}")
    print(f"Valid counts: {valid_counts}")
    print(f"Labels:       {choice_labels}")
    print(f"Output:       {output_path}")

    image = load_input_as_bgr(
        template_path,
        dpi=int(cfg["preprocess"].get("dpi", 300)),
        page=int(cfg["preprocess"].get("page", 1)),
        poppler_path=cfg["preprocess"].get("poppler_path"),
    )
    gray = to_gray(image)
    binary = adaptive_binarize(
        gray,
        block_size=int(cfg["preprocess"].get("adaptive_block_size", 31)),
        c=int(cfg["preprocess"].get("adaptive_c", 15)),
    )
    boxes = detect_checkbox_contours(binary, cfg["detector"])
    template_map = boxes_to_template_map(
        boxes,
        page_shape=image.shape[:2],
        valid_counts=valid_counts,
        choice_labels=choice_labels,
        row_merge_px=int(cfg["detector"].get("row_merge_px", 18)),
    )
    save_template_map(template_map, output_path)

    overlay_items = [{**b, "source": "contour"} for b in template_map["boxes"]]
    overlay = draw_template_boxes(image, overlay_items)
    out_dir = ensure_dir(Path(output_path).parent)
    cv2.imwrite(str(out_dir / f"template_overlay{'_' + args.variant if args.variant else ''}.png"), overlay)
    print(f"template_map saved: {output_path}")
    print(f"Detected number of questions: {template_map['num_questions']}")




if __name__ == "__main__":
    main()
