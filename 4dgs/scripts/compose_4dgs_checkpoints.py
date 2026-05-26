#!/usr/bin/env python
"""Compose multiple fudan 4DGS checkpoints by concatenating Gaussian tensors.

This produces a single checkpoint that can be rendered by the normal 4DGS
renderer. It assumes all input checkpoints were trained in the same coordinate
system and with compatible 4DGS config settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _gaussian_count(model_params: Any) -> int:
    if not isinstance(model_params, (tuple, list)) or len(model_params) < 7:
        raise ValueError("Unexpected 4DGS model parameter tuple.")
    xyz = model_params[1]
    if not torch.is_tensor(xyz) or xyz.ndim != 2:
        raise ValueError("Could not find Gaussian xyz tensor at model_params[1].")
    return int(xyz.shape[0])


def _can_concat(values: list[Any], counts: list[int]) -> bool:
    if not all(torch.is_tensor(value) for value in values):
        return False
    if not all(value.ndim >= 1 for value in values):
        return False
    if not all(int(value.shape[0]) == count for value, count in zip(values, counts)):
        return False
    tail = tuple(values[0].shape[1:])
    return all(tuple(value.shape[1:]) == tail for value in values[1:])


def compose_checkpoints(checkpoints: list[Path], output: Path) -> dict[str, Any]:
    if len(checkpoints) < 2:
        raise ValueError("Pass at least two checkpoints to compose.")

    blobs = []
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        blob = torch.load(checkpoint, map_location="cpu")
        if not isinstance(blob, (tuple, list)) or len(blob) != 2:
            raise ValueError(f"Unexpected checkpoint format: {checkpoint}")
        blobs.append(blob)

    params_list = [blob[0] for blob in blobs]
    iterations = [int(blob[1]) for blob in blobs]
    lengths = {len(params) for params in params_list}
    if len(lengths) != 1:
        raise ValueError(f"Checkpoint parameter lengths differ: {sorted(lengths)}")

    counts = [_gaussian_count(params) for params in params_list]
    composed: list[Any] = []
    for idx, values in enumerate(zip(*params_list)):
        values = list(values)
        if _can_concat(values, counts):
            composed.append(torch.cat(values, dim=0))
        else:
            # Non-Gaussian metadata/config state should match across compatible
            # checkpoints. Copy from the first checkpoint for rendering.
            composed.append(values[0])

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save((tuple(composed), min(iterations)), output)
    meta = {
        "output": str(output.resolve()),
        "inputs": [str(path.resolve()) for path in checkpoints],
        "input_iterations": iterations,
        "input_gaussian_counts": counts,
        "num_points": int(sum(counts)),
        "iteration": int(min(iterations)),
    }
    (output.parent / "compose_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = compose_checkpoints([p.resolve() for p in args.checkpoints], args.output.resolve())
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
