from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from omr.infer import BaseBackend, infer_single_scan, load_backend
from omr.utils import load_yaml


# ── configuration from environment variables (with sensible defaults) ──────────
#
# Set these before starting the server, e.g.:
#   export OMR_CONFIG=config/config.yaml
#   export OMR_MODEL=outputs/exports/omr_model.onnx
#   uvicorn server.main:app --reload
#
CONFIG_PATH = os.environ.get("OMR_CONFIG", "config/config.yaml")
MODEL_PATH  = os.environ.get("OMR_MODEL",  "outputs/exports/omr_model.onnx")
OUTPUT_ROOT = os.environ.get("OMR_OUTPUT", "outputs/inference")
DEVICE      = os.environ.get("OMR_DEVICE", "cpu")

# Loaded once at startup, shared across all requests.
_cfg: dict[str, Any] = {}
_backend: BaseBackend | None = None


# ── lifespan: runs once when the server starts and once when it stops ──────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _backend
    _cfg = load_yaml(CONFIG_PATH)
    _backend = load_backend(MODEL_PATH, _cfg, device=DEVICE)
    print(f"OMR server ready — model: {MODEL_PATH}  config: {CONFIG_PATH}")
    yield
    # nothing to clean up for ONNX/PyTorch backends


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OMR Inference Server",
    description=(
        "Optical Mark Recognition API. "
        "Upload a scanned exam page and get back which boxes are marked. "
        "Interactive docs available at /docs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — returns ok if the server is running and model is loaded."""
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/variants")
def list_variants():
    """List the template variant names defined in config (e.g. 'mixed', '4choice')."""
    return {"variants": list(_cfg.get("templates", {}).keys())}


@app.post("/infer")
async def infer(
    scan: UploadFile = File(..., description="Scanned exam page — PDF or image (PNG/JPG)"),
    variant: str | None = Form(None, description="Template variant name, e.g. 'mixed' or '4choice'. Leave blank for the default."),
):
    """
    Run OMR inference on a single scanned exam page.

    **How it works:**
    1. Aligns the scan to the template using fiducial markers.
    2. Crops each answer box from the aligned scan.
    3. Runs the CNN/ONNX model on each crop.
    4. Returns per-question results and per-box probabilities.

    **Returns:**
    - `questions`: one entry per question with the selected choice(s) and a
      `needs_review` flag for uncertain marks.
    - `boxes`: detailed probability and confidence for every individual box.
    - `debug_meta`: alignment method used (fiducial / ORB / resize).
    """
    # Save the uploaded file to a temp location so the pipeline can read it.
    suffix = Path(scan.filename or "scan.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(scan.file, tmp)
        tmp_path = tmp.name

    try:
        payload = infer_single_scan(
            scan_path=tmp_path,
            config_path=CONFIG_PATH,
            model_path=MODEL_PATH,
            output_root=OUTPUT_ROOT,
            device=DEVICE,
            variant=variant,
            backend=_backend,   # reuse the model loaded at startup
        )
    except ValueError as exc:
        # e.g. unknown variant name
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Strip file-system paths — the caller only needs the data, not local paths.
    clean_boxes = [
        {k: v for k, v in box.items() if k not in {"image", "scan"}}
        for box in payload["boxes"]
    ]
    return {
        "questions": payload["questions"],
        "boxes": clean_boxes,
        "debug_meta": payload["debug_meta"],
    }
