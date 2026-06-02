# Lightweight Hybrid OMR

一个可直接运行的端到端 OMR 项目：

- 模板页 PDF/图像读取
- 扫描页对齐/校正/二值化
- 复选框 contour 检测 + template fallback
- crop 导出与 JSONL manifest 构建
- 轻量 CNN / MobileNetV2 训练
- TorchScript / ONNX 导出
- 推理、置信度估计、不确定样本复核、CSV/JSON 输出、标注图保存

## 环境安装

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
