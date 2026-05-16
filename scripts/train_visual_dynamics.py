#!/usr/bin/env python
"""Train visually conditioned object-token dynamics (project.md Phases 3–4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.visual_dynamics.dataset import VisualDynamicsDataset, collate_visual_dynamics  # noqa: E402
from phys4d.visual_dynamics.dynamics import ObjectTokenDynamics  # noqa: E402
from phys4d.visual_dynamics.losses import physics_energy_penalty, state_loss  # noqa: E402
from phys4d.visual_dynamics.rollout import rollout_scene  # noqa: E402
from phys4d.visual_dynamics.scenes import discover_batch_scenes, load_manifest_splits  # noqa: E402


def _eval_rollout_mse(model, scenes, *, cfg: dict, device: torch.device) -> float:
    model.eval()
    errors = []
    tcfg = cfg["temporal"]
    for scene in scenes:
        pred = rollout_scene(
            model,
            scene,
            start_frame=int(tcfg["train_frames"][1]),
            horizon=min(10, int(tcfg["T_future"])),
            history=int(tcfg["history_K"]),
            dt_s=float(tcfg["dt_s"]),
            device=device,
        )
        from phys4d.perception.states import load_states_csv_or_poses
        from phys4d.object_state import stack_state_vectors

        states = load_states_csv_or_poses(
            scene.scene_dir / "object_poses.csv", dt_s=float(tcfg["dt_s"])
        )
        mat = stack_state_vectors(states, include_velocity=True)
        start = int(tcfg["train_frames"][1])
        gt_slice = mat[start : start + pred.shape[0]]
        errors.append(float(((pred - gt_slice) ** 2).mean()))
    return float(sum(errors) / max(len(errors), 1))


def _train_one_step_batch(
    model: ObjectTokenDynamics,
    batch: dict,
    device: torch.device,
    *,
    w_physics: float,
) -> torch.Tensor:
    hist = batch["state_history"].to(device)
    vis = batch["visual_feat"].to(device)
    target = batch["target"].to(device)
    pred = model.predict_next(hist, vis)
    loss = state_loss(pred.squeeze(1), target.squeeze(1))
    if w_physics > 0:
        seq = torch.cat([hist[:, :, -1, :], pred], dim=1)
        loss = loss + w_physics * physics_energy_penalty(seq)
    return loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/visual_dynamics.json")
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Parent of batch_root/ (use /data on Modal)",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/visual_dynamics")
    parser.add_argument("--no-visual", action="store_true", help="States-only ablation baseline")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tcfg = cfg["temporal"]
    tr = cfg["training"]
    batch_root = args.batch_root or (REPO_ROOT / cfg["data"]["batch_root"])

    data_root = args.data_root.resolve()
    if args.manifest and args.manifest.is_file():
        splits = load_manifest_splits(args.manifest, data_root)
        train_scenes = splits["train"]
        val_scenes = splits.get("val", splits.get("test", []))
    else:
        scenes = discover_batch_scenes(batch_root)
        n_val = max(1, len(scenes) // 10)
        val_scenes = scenes[-n_val:]
        train_scenes = scenes[:-n_val]

    train_ds = VisualDynamicsDataset(
        train_scenes,
        history=int(tcfg["history_K"]),
        train_end_frame=int(tcfg["train_frames"][1]),
        dt_s=float(tcfg["dt_s"]),
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(tr["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_visual_dynamics,
    )

    device = torch.device(args.device)
    model = ObjectTokenDynamics(
        history=int(tcfg["history_K"]),
        visual_dim=int(cfg["perception"]["visual_dim"]),
        hidden=int(cfg["dynamics"]["hidden_dim"]),
        use_visual=not args.no_visual,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr["lr"]),
        weight_decay=float(tr["weight_decay"]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    grad_clip = float(tr["grad_clip"])
    w_phys = float(tr["w_physics"])

    # Stage A: one-step teacher forcing
    for epoch in range(int(tr["epochs_stage_a"])):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            loss = _train_one_step_batch(model, batch, device, w_physics=w_phys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += float(loss.item())
            n += 1
        val_mse = _eval_rollout_mse(model, val_scenes, cfg=cfg, device=device)
        row = {"stage": "A", "epoch": epoch + 1, "train_loss": total / max(n, 1), "val_rollout_mse": val_mse}
        log.append(row)
        print(row)

    # Stage B: longer rollout curriculum (noise on history during eval rollout only for now)
    for horizon in tr["rollout_curriculum"]:
        for epoch in range(int(tr["epochs_stage_b"]) // len(tr["rollout_curriculum"])):
            model.train()
            total = 0.0
            n = 0
            for batch in loader:
                loss = _train_one_step_batch(model, batch, device, w_physics=w_phys)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                total += float(loss.item())
                n += 1
            val_mse = _eval_rollout_mse(model, val_scenes, cfg=cfg, device=device)
            row = {
                "stage": "B",
                "horizon": horizon,
                "epoch": epoch + 1,
                "train_loss": total / max(n, 1),
                "val_rollout_mse": val_mse,
            }
            log.append(row)
            print(row)

    ckpt_path = args.out_dir / ("visual_dynamics.pt" if not args.no_visual else "visual_dynamics_states_only.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "history": int(tcfg["history_K"]),
            "visual_dim": int(cfg["perception"]["visual_dim"]),
            "use_visual": not args.no_visual,
            "train_end_frame": int(tcfg["train_frames"][1]),
            "T_future": int(tcfg["T_future"]),
        },
        ckpt_path,
    )
    with (args.out_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Saved {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
