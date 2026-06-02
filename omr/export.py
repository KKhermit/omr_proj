from __future__ import annotations

from pathlib import Path

import torch

from omr.model import build_model
from omr.utils import ensure_dir, load_yaml


@torch.no_grad()
def load_checkpoint_model(checkpoint_path: str | Path, cfg: dict, device: str = "cpu") -> torch.nn.Module:
    model = build_model(cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def export_torchscript(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 1, 64, 64),
    device: str = "cpu",
) -> Path:
    ensure_dir(Path(output_path).parent)
    example = torch.randn(*input_shape, device=device)
    traced = torch.jit.trace(model, example, strict=False)
    traced = torch.jit.freeze(traced)
    traced.save(str(output_path))
    return Path(output_path)


@torch.no_grad()
def export_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 1, 64, 64),
    opset: int = 17,
    device: str = "cpu",
) -> Path:
    ensure_dir(Path(output_path).parent)
    example = torch.randn(*input_shape, device=device)
    try:
        torch.onnx.export(
            model,
            (example,),
            str(output_path),
            input_names=["input"],
            output_names=["logit"],
            opset_version=opset,
            dynamo=True,
        )
    except Exception:
        torch.onnx.export(
            model,
            (example,),
            str(output_path),
            input_names=["input"],
            output_names=["logit"],
            opset_version=opset,
            dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
            dynamo=False,
        )
    return Path(output_path)


def export_from_config(config_path: str | Path, checkpoint_path: str | Path, device: str = "cpu") -> dict[str, str]:
    cfg = load_yaml(config_path)
    model = load_checkpoint_model(checkpoint_path, cfg, device=device)
    export_cfg = cfg.get("export", {})
    img_size = int(cfg["model"].get("img_size", 64))
    input_shape = (1, int(cfg["model"].get("in_channels", 1)), img_size, img_size)
    ts_path = export_torchscript(
        model,
        export_cfg.get("torchscript_path", "outputs/exports/model.ts"),
        input_shape=input_shape,
        device=device,
    )
    onnx_path = export_onnx(
        model,
        export_cfg.get("onnx_path", "outputs/exports/model.onnx"),
        input_shape=input_shape,
        opset=int(export_cfg.get("onnx_opset", 17)),
        device=device,
    )
    return {"torchscript": str(ts_path), "onnx": str(onnx_path)}


__all__ = [
    "load_checkpoint_model",
    "export_torchscript",
    "export_onnx",
    "export_from_config",
]
