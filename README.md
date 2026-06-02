# Lightweight Hybrid OMR

A ready-to-run end-to-end OMR (Optical Mark Recognition) project:

- Template page PDF/image reading
- Scanned page alignment, correction, and binarization
- Checkbox contour detection with template fallback
- Crop export and JSONL manifest building
- Lightweight CNN / MobileNetV2 training
- TorchScript / ONNX export
- Inference, confidence estimation, uncertain sample review, CSV/JSON output, and annotated image saving

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
