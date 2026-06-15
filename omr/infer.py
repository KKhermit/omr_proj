from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

from omr.dataset import preprocess_crop_for_inference
from omr.detector import draw_template_boxes, extract_crops_from_template
from omr.export import load_checkpoint_model
from omr.preprocessing import preprocess_scan_and_template
from omr.utils import draw_labeled_boxes, ensure_dir, load_json, load_yaml, normalized_entropy, save_json, write_csv


class BaseBackend:
    def predict(self, batch_tensor: torch.Tensor) -> np.ndarray:  # noqa: ARG002
        raise NotImplementedError


class CheckpointBackend(BaseBackend):
    def __init__(self, model_path: str | Path, cfg: dict[str, Any], device: str = "cpu") -> None:
        self.device = device
        self.model = load_checkpoint_model(model_path, cfg, device=device)

    @torch.no_grad()
    def predict(self, batch_tensor: torch.Tensor) -> np.ndarray:
        logits = self.model(batch_tensor.to(self.device))
        return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)


class TorchScriptBackend(BaseBackend):
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.device = device
        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, batch_tensor: torch.Tensor) -> np.ndarray:
        logits = self.model(batch_tensor.to(self.device))
        return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)


class OnnxBackend(BaseBackend):
    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover
            raise ImportError("to use ONNX need to install onnxruntime。") from exc
        self.ort = ort
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, batch_tensor: torch.Tensor) -> np.ndarray:
        input_arr = batch_tensor.detach().cpu().numpy().astype(np.float32)
        logits = self.session.run([self.output_name], {self.input_name: input_arr})[0].reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))


def load_backend(model_path: str | Path, cfg: dict[str, Any], device: str = "cpu") -> BaseBackend:
    suffix = Path(model_path).suffix.lower()
    if suffix in {".pt", ".pth"}:
        return CheckpointBackend(model_path, cfg, device=device)
    if suffix in {".ts", ".jit"}:
        return TorchScriptBackend(model_path, device=device)
    if suffix == ".onnx":
        return OnnxBackend(model_path)
    raise ValueError(f"Not supported model format: {suffix}")


def _crop_fill_ratio(crop_bgr: np.ndarray, dark_threshold: int = 127) -> float:
    """Fraction of pixels darker than dark_threshold in the crop."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    return float(np.count_nonzero(gray < dark_threshold)) / max(gray.size, 1)


def _batch_predict_crops(
    backend: BaseBackend,
    crop_arrays: list[np.ndarray],
    model_cfg: dict[str, Any],
    batch_size: int = 256,
) -> np.ndarray:
    tensors: list[torch.Tensor] = []
    for crop in crop_arrays:
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = preprocess_crop_for_inference(
            pil,
            img_size=int(model_cfg.get("img_size", 64)),
            mean=float(model_cfg.get("mean", 0.5)),
            std=float(model_cfg.get("std", 0.5)),
        )
        tensors.append(tensor)
    probs: list[np.ndarray] = []
    for i in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[i : i + batch_size], dim=0)
        probs.append(np.asarray(backend.predict(x), dtype=np.float32))
    return np.concatenate(probs) if probs else np.array([], dtype=np.float32)


def _question_summary(rows: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["question"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for question, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda x: x["choice"])
        filled = [item["choice"] for item in ordered if item["prob_fill"] >= threshold and not item["uncertain"]]
        best = max(ordered, key=lambda x: x["prob_fill"])
        any_uncertain = any(item["uncertain"] for item in ordered)
        summary.append(
            {
                "question": question,
                "selected_choices": filled,
                "best_choice": best["choice"],
                "best_prob": float(best["prob_fill"]),
                "needs_review": bool(any_uncertain or len(filled) != 1),
            }
        )
    return summary


def _resolve_infer_paths(cfg: dict[str, Any], variant: str | None) -> tuple[str, str]:
    """Return (template_path, template_map_path) for the given variant (or default)."""
    if variant is not None:
        variants = cfg.get("templates", {})
        if variant not in variants:
            available = list(variants.keys()) or ["(none defined)"]
            raise ValueError(f"Unknown template variant '{variant}'. Available: {available}")
        entry = variants[variant]
        return entry["template"], entry["template_map"]
    return cfg["data"]["template"], cfg["data"]["template_map"]


def infer_single_scan(
    scan_path: str | Path,
    config_path: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    device: str = "cpu",
    variant: str | None = None,
    backend: BaseBackend | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    if backend is None:
        backend = load_backend(model_path, cfg, device=device)
    out_dir = ensure_dir(Path(output_root) / Path(scan_path).stem)

    template_path, template_map_path = _resolve_infer_paths(cfg, variant)

    aligned, _, debug_images, debug_meta = preprocess_scan_and_template(
        scan_path,
        template_path,
        cfg["preprocess"],
    )
    template_map = load_json(template_map_path)
    items = extract_crops_from_template(aligned, template_map, cfg["detector"])

    for name, img in debug_images.items():
        cv2.imwrite(str(out_dir / f"debug_{name}.png"), img)

    overlay = draw_template_boxes(aligned, [{k: v for k, v in item.items() if k != "crop"} for item in items])
    cv2.imwrite(str(out_dir / "debug_template_overlay.png"), overlay)

    crop_dir = ensure_dir(out_dir / "crops")
    review_dir = ensure_dir(out_dir / "review")
    crop_arrays: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for item in items:
        crop_path = crop_dir / f"{Path(scan_path).stem}__{item['box_id']}.png"
        cv2.imwrite(str(crop_path), item["crop"])
        crop_arrays.append(item["crop"])
        meta = {k: v for k, v in item.items() if k != "crop"}
        meta["image"] = str(crop_path)
        meta_rows.append(meta)

    probs = _batch_predict_crops(
        backend,
        crop_arrays,
        cfg["model"],
        batch_size=int(cfg["infer"].get("batch_size", 256)),
    )

    threshold = float(cfg["infer"].get("threshold", 0.5))
    uncertain_margin = float(cfg["infer"].get("uncertain_margin", 0.10))
    min_confidence = float(cfg["infer"].get("min_confidence", 0.55))
    fill_ratio_low = float(cfg["infer"].get("fill_ratio_low", 0.04))
    fill_ratio_high = float(cfg["infer"].get("fill_ratio_high", 0.25))

    fill_ratios = [_crop_fill_ratio(c) for c in crop_arrays]

    results: list[dict[str, Any]] = []
    for row, prob, fill_ratio in zip(meta_rows, probs.tolist(), fill_ratios):
        entropy = normalized_entropy(prob)
        confidence = 1.0 - entropy
        model_uncertain = (abs(prob - threshold) < uncertain_margin) or (confidence < min_confidence)
        # Flag when CNN and pixel density strongly disagree.
        fill_disagrees = (
            (prob < threshold - uncertain_margin and fill_ratio > fill_ratio_high) or
            (prob > threshold + uncertain_margin and fill_ratio < fill_ratio_low)
        )
        uncertain = model_uncertain or fill_disagrees
        pred = "filled" if prob >= threshold else "empty"
        out_row = {
            "scan": str(scan_path),
            "box_id": row["box_id"],
            "question": int(row["question"]),
            "choice": row["choice"],
            "bbox": row["bbox"],
            "crop_bbox": row["crop_bbox"],
            "image": row["image"],
            "prob_fill": float(prob),
            "fill_ratio": round(float(fill_ratio), 4),
            "pred": pred,
            "confidence": float(confidence),
            "uncertain": bool(uncertain),
            "fill_disagrees": bool(fill_disagrees),
        }
        results.append(out_row)
        if uncertain:
            target = review_dir / Path(row["image"]).name
            img = cv2.imread(row["image"], cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(str(target), img)

    question_summary = _question_summary(results, threshold=threshold)

    annotated = draw_labeled_boxes(aligned, results)
    cv2.imwrite(str(out_dir / "annotated.png"), annotated)

    csv_rows = [{k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in results]
    write_csv(csv_rows, out_dir / "results_boxes.csv")
    write_csv(question_summary, out_dir / "results_questions.csv")

    payload = {
        "scan": str(scan_path),
        "debug_meta": debug_meta,
        "boxes": results,
        "questions": question_summary,
    }
    save_json(payload, out_dir / "results.json")
    return payload


__all__ = ["infer_single_scan", "load_backend"]
