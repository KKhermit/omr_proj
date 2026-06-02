from __future__ import annotations

import argparse
from pathlib import Path

from omr.export import export_from_config
from omr.infer import infer_single_scan
from omr.train import train_from_config
from omr.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight OMR project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="train model")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--device", default=None)

    p_export = subparsers.add_parser("export", help="export TorchScript/ONNX")
    p_export.add_argument("--config", required=True)
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--device", default="cpu")

    p_infer = subparsers.add_parser("infer", help="Run inference on a single scan")
    p_infer.add_argument("--config", required=True)
    p_infer.add_argument("--model", required=True)
    p_infer.add_argument("--scan", required=True)
    p_infer.add_argument("--output", default="outputs/inference")
    p_infer.add_argument("--device", default="cpu")
    p_infer.add_argument("--variant", default=None, help="named template variant (e.g. 2choice, 4choice, 6choice)")

    subparsers.add_parser("synth-data", help="Generate synthetic training data")
    p_tmap = subparsers.add_parser("template-map", help="Automatically generate template_map.json")
    p_tmap.add_argument("--variant", default=None, help="named template variant (e.g. 2choice, 4choice, 6choice)")
    subparsers.add_parser("build-dataset", help="Build a real scans data inventory")

    parser.add_argument("--log-level", default="INFO")
    args, unknown = parser.parse_known_args()
    setup_logging(args.log_level)

    if args.command == "train":
        summary = train_from_config(args.config, device=args.device)
        print(summary)
        return

    if args.command == "export":
        exported = export_from_config(args.config, args.checkpoint, device=args.device)
        print(exported)
        return

    if args.command == "infer":
        payload = infer_single_scan(
            args.scan, args.config, args.model, args.output,
            device=args.device, variant=args.variant,
        )
        print(f"Inference complete: {Path(payload['scan']).name}")
        return

    if args.command == "synth-data":
        from scripts.generate_synthetic_dataset import main as synth_main
        import sys
        sys.argv = [sys.argv[0]] + unknown
        synth_main()
        return

    if args.command == "template-map":
        from scripts.build_template_map import main as map_main
        import sys
        # Pass --variant through to the subcommand's own argparse
        variant_args = ["--variant", args.variant] if args.variant else []
        sys.argv = [sys.argv[0]] + variant_args + unknown
        map_main()
        return

    if args.command == "build-dataset":
        from scripts.build_dataset import main as ds_main
        import sys
        sys.argv = [sys.argv[0]] + unknown
        ds_main()
        return


if __name__ == "__main__":
    main()
