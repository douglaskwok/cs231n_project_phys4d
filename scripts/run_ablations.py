#!/usr/bin/env python
"""Run checklist ablations: CNN vs pose-MLP; optional view-count sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs/sphere_bounce_batch/dataset_manifest.json",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--view-sweep", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/ablations/report.json")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "outputs/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if not args.skip_train:
        _run(
            [
                py,
                "scripts/train_param_predictor.py",
                "--manifest",
                str(args.manifest),
                "--data-root",
                str(args.data_root),
                "--epochs",
                str(args.epochs),
                "--model",
                "cnn",
                "--out-dir",
                str(out_dir / "cnn"),
                "--device",
                args.device,
            ]
        )
        _run(
            [
                py,
                "scripts/train_param_predictor.py",
                "--manifest",
                str(args.manifest),
                "--data-root",
                str(args.data_root),
                "--epochs",
                str(args.epochs),
                "--model",
                "mlp",
                "--out-dir",
                str(out_dir / "mlp"),
                "--device",
                args.device,
            ]
        )

    report: dict = {"cnn_vs_mlp": {}}
    for model, sub in (("cnn", "cnn"), ("mlp", "mlp")):
        ckpt = out_dir / sub / f"{'param_predictor' if model == 'cnn' else 'pose_mlp'}_best.pt"
        eval_out = out_dir / f"eval_{model}_test.json"
        _run(
            [
                py,
                "scripts/eval_param_predictor.py",
                "--manifest",
                str(args.manifest),
                "--data-root",
                str(args.data_root),
                "--checkpoint",
                str(ckpt),
                "--model",
                model,
                "--split",
                "test",
                "--out-json",
                str(eval_out),
                "--device",
                args.device,
            ]
        )
        report["cnn_vs_mlp"][model] = json.loads(eval_out.read_text(encoding="utf-8"))["summary"]

    if args.view_sweep:
        report["view_sweep"] = {}
        for k in (4, 6, 10):
            sub = out_dir / f"cnn_k{k}"
            if not args.skip_train:
                _run(
                    [
                        py,
                        "scripts/train_param_predictor.py",
                        "--manifest",
                        str(args.manifest),
                        "--data-root",
                        str(args.data_root),
                        "--epochs",
                        str(max(20, args.epochs // 2)),
                        "--model",
                        "cnn",
                        "--max-views",
                        str(k),
                        "--out-dir",
                        str(sub),
                        "--device",
                        args.device,
                    ]
                )
            ckpt = sub / "param_predictor_best.pt"
            eval_out = out_dir / f"eval_cnn_k{k}_test.json"
            _run(
                [
                    py,
                    "scripts/eval_param_predictor.py",
                    "--manifest",
                    str(args.manifest),
                    "--data-root",
                    str(args.data_root),
                    "--checkpoint",
                    str(ckpt),
                    "--model",
                    "cnn",
                    "--max-views",
                    str(k),
                    "--split",
                    "test",
                    "--out-json",
                    str(eval_out),
                    "--device",
                    args.device,
                ]
            )
            report["view_sweep"][f"K={k}"] = json.loads(eval_out.read_text(encoding="utf-8"))["summary"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
