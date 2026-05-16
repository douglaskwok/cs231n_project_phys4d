#!/usr/bin/env python
"""Train visually conditioned dynamics on cross-scene PyBullet batch data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d.object_state import stack_state_vectors, trajectory_to_states, window_states  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402
from phys4d.visual_dynamics.dataset import (  # noqa: E402
    VisualDynamicsDataset,
    discover_batch_scenes,
    split_scenes,
)
from phys4d.visual_dynamics.dynamics_head import VisualDynamicsModel  # noqa: E402


def _rollout_scene(
    model: VisualDynamicsModel,
    scene,
    *,
    train_end_frame: int,
    history: int,
    device: torch.device,
) -> float:
    model.eval()
    traj = load_object_poses_csv(scene.poses_path)
    states = trajectory_to_states(traj)
    dataset = VisualDynamicsDataset([scene], history=history, train_end_frame=train_end_frame)
    errors: list[float] = []
    with torch.no_grad():
        for frame in range(train_end_frame + 1, len(states) - 1):
            hist = window_states(states, frame, history)
            hist_t = torch.from_numpy(stack_state_vectors(hist, include_velocity=True)).float()
            hist_t = hist_t.unsqueeze(0).to(device)
            crop = dataset._load_crop(scene, frame).unsqueeze(0).to(device)
            pred = model.predict_next(hist_t, crop).cpu().numpy()[0]
            target = stack_state_vectors([states[frame + 1]], include_velocity=True)[0]
            errors.append(float(((pred - target) ** 2).mean()))
    return float(sum(errors) / max(len(errors), 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sphere_bounce_batch",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "visual_dynamics",
    )
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-end-frame", type=int, default=59)
    args = parser.parse_args()

    scenes = discover_batch_scenes(args.batch_root)
    if len(scenes) < 2:
        print(
            f"Need >=2 scenes in {args.batch_root}. "
            "Run: python scripts/generate_sphere_bounce_batch.py",
            file=sys.stderr,
        )
        return 1

    train_scenes, val_scenes = split_scenes(scenes)
    train_ds = VisualDynamicsDataset(
        train_scenes, history=args.history, train_end_frame=args.train_end_frame
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisualDynamicsModel(history=args.history).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            hist = batch["state_history"].to(device)
            target = batch["target"].to(device)
            img = batch["image"].to(device)
            pred = model.predict_next(hist, img)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        train_loss = total / max(n, 1)
        val_mse = (
            sum(
                _rollout_scene(
                    model,
                    s,
                    train_end_frame=args.train_end_frame,
                    history=args.history,
                    device=device,
                )
                for s in val_scenes
            )
            / max(len(val_scenes), 1)
        )
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_rollout_mse": val_mse}
        log.append(row)
        print(f"epoch {epoch + 1}: train_loss={train_loss:.6f} val_rollout_mse={val_mse:.6f}")

    ckpt = args.out_dir / "visual_dynamics.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "history": args.history,
            "train_end_frame": args.train_end_frame,
            "train_scenes": [s.scene_id for s in train_scenes],
            "val_scenes": [s.scene_id for s in val_scenes],
        },
        ckpt,
    )
    with (args.out_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Saved checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
