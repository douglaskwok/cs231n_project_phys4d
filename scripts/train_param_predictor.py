#!/usr/bin/env python
"""Train CNN to predict physics parameters from multi-view video (phases 2–3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.param_ident.dataset import ParamIdDataset, load_manifest  # noqa: E402
from phys4d.param_ident.model import MultiViewParamPredictor  # noqa: E402

PREDICT_NAMES = ["restitution", "mass_kg", "drop_z_m"]


def _param_mse(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = ((pred - target) ** 2).mean(dim=0)
    return {name: float(mse[i]) for i, name in enumerate(PREDICT_NAMES)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {n: 0.0 for n in PREDICT_NAMES}
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        batch_metrics = _param_mse(pred, y)
        for k, v in batch_metrics.items():
            totals[k] += v
        n_batches += 1
    if n_batches == 0:
        return totals
    return {k: v / n_batches for k, v in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs/param_id_dataset/dataset_manifest.json",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Parent of batch_root/ (REPO/outputs locally, /data on Modal)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/param_predictor",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(
            f"Missing {args.manifest}. Run:\n"
            "  python scripts/generate_param_id_batch.py --limit 10\n"
            "  python scripts/build_param_id_splits.py"
        )

    train_scenes = load_manifest(args.manifest, "train", data_root=args.data_root)
    val_scenes = load_manifest(args.manifest, "val", data_root=args.data_root)
    train_ds = ParamIdDataset(train_scenes, augment=True)
    val_ds = ParamIdDataset(val_scenes, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = MultiViewParamPredictor(num_params=len(PREDICT_NAMES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            n_train += 1

        val_metrics = evaluate(model, val_loader, device)
        val_mse = sum(val_metrics.values()) / len(val_metrics)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_train, 1),
            "val_param_mse": val_metrics,
            "val_mean_mse": val_mse,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_mse={val_mse:.5f} {val_metrics}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "predict_names": PREDICT_NAMES,
            "epoch": epoch,
            "val_mean_mse": val_mse,
        }
        torch.save(ckpt, args.out_dir / "param_predictor_last.pt")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(ckpt, args.out_dir / "param_predictor_best.pt")

    with (args.out_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Best val mean MSE: {best_val:.5f}")
    print(f"Checkpoint: {args.out_dir / 'param_predictor_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
