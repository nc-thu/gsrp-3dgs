#!/usr/bin/env python
"""Extract exact tile-wide T-stop summaries from the isolated Speedy-Splat tree.

The output deliberately contains no row, chunk, lane, or event trace.  The
CUDA extension replays each tile's depth-ordered KVP list and reports the first
KVP after which every valid pixel has reached T < 1e-4.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["SARD_V11_SUMMARY"] = "1"
os.environ["SARD_SOD"] = "0"
os.environ.pop("SARD_STATS", None)
os.environ.pop("SARD_FIFO_TRACE", None)
os.environ.pop("SARD_T_TRACE", None)
os.environ.pop("SORTFREE_TORCH_DYNAMIC_PRECISION", None)

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel, render
from diff_gaussian_rasterization import get_last_tile_t_summary


TILE_FIELDS = (
    "scene", "allocator", "cam", "camera_name", "tile_id", "tile_x", "tile_y",
    "n_kvp_raw", "n_kvp_retained", "n_kvp_tstopped",
    "producer_cycles_raw", "producer_cycles_retained", "tstop_after_kvp",
    "valid_pixels",
)
FRAME_FIELDS = (
    "scene", "allocator", "cam", "camera_name", "width", "height", "tile_count",
    "n_kvp_raw", "n_kvp_retained", "n_kvp_tstopped",
    "producer_cycles_raw", "producer_cycles_retained", "tstop_tile_count",
)
SUMMARY_ORDER = (
    "n_kvp_raw", "n_kvp_retained", "n_kvp_tstopped",
    "producer_cycles_raw", "producer_cycles_retained", "tstop_after_kvp",
    "valid_pixels",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(root: Path) -> str:
    points = sorted(root.rglob("point_cloud.ply"))
    if not points:
        raise RuntimeError(f"no point_cloud.ply under {root}")
    return sha256_file(points[-1])


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def source_hash(root: Path) -> str:
    rels = (
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu",
        "submodules/diff-gaussian-rasterization/rasterize_points.cu",
        "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py",
    )
    digest = hashlib.sha256()
    for rel in rels:
        digest.update(rel.encode())
        digest.update((root / rel).read_bytes())
    return digest.hexdigest()


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--allocator", choices=("baseline_aabb", "atae_snugbox"), required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--camera_set", choices=("full_test", "quick_train"), default="full_test")
    parser.add_argument("--num_cams", type=int, default=1)
    args, _ = parser.parse_known_args()

    if os.environ.get("SARD_ALLOCATOR_MODE") != args.allocator:
        raise RuntimeError("SARD_ALLOCATOR_MODE does not match --allocator")
    physical_gpu = os.environ.get("SARD_PHYSICAL_GPU")
    if physical_gpu is None or not physical_gpu.isdigit():
        raise RuntimeError("SARD_PHYSICAL_GPU must name the physical GPU")

    sys.argv = [
        "extract_v11_summary", "--model_path", args.model_path,
        "--source_path", args.source_path,
    ]
    p = argparse.ArgumentParser()
    mp = ModelParams(p, sentinel=True)
    pp = PipelineParams(p)
    combined = get_combined_args(p)
    dataset = mp.extract(combined)
    pipeline = pp.extract(combined)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    tests = scene.getTestCameras()
    trains = scene.getTrainCameras()
    cameras = tests if args.camera_set == "full_test" and tests else trains
    if args.camera_set == "quick_train":
        cameras = cameras[: args.num_cams]
    if not cameras:
        raise RuntimeError("no cameras selected")

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    render(cameras[0], gaussians, pipeline, background)  # warm-up

    tile_rows: list[dict] = []
    frame_rows: list[dict] = []
    for cam_id, camera in enumerate(cameras):
        render(camera, gaussians, pipeline, background)
        tensor = get_last_tile_t_summary()
        if tensor is None or tensor.ndim != 2 or tuple(tensor.shape[1:]) != (7,):
            raise RuntimeError(f"invalid v1.1 summary tensor: {None if tensor is None else tuple(tensor.shape)}")
        values = tensor.detach().cpu().tolist()
        tiles_x = (int(camera.image_width) + 15) // 16
        sums = {key: 0 for key in SUMMARY_ORDER[:5]}
        stopped_tiles = 0
        camera_tiles: list[dict] = []
        for tile_id, packed in enumerate(values):
            row = {
                "scene": args.scene, "allocator": args.allocator, "cam": cam_id,
                "camera_name": getattr(camera, "image_name", str(cam_id)),
                "tile_id": tile_id, "tile_x": tile_id % tiles_x,
                "tile_y": tile_id // tiles_x,
            }
            row.update({name: int(value) for name, value in zip(SUMMARY_ORDER, packed)})
            if row["n_kvp_raw"] != row["n_kvp_retained"] + row["n_kvp_tstopped"]:
                raise RuntimeError(f"KVP closure failed: {row}")
            if not 0 <= row["tstop_after_kvp"] <= row["n_kvp_raw"]:
                raise RuntimeError(f"invalid T stop position: {row}")
            if row["producer_cycles_retained"] > row["producer_cycles_raw"]:
                raise RuntimeError(f"producer closure failed: {row}")
            if not 1 <= row["valid_pixels"] <= 256:
                raise RuntimeError(f"invalid valid-pixel count: {row}")
            for name in sums:
                sums[name] += row[name]
            stopped_tiles += row["n_kvp_tstopped"] > 0
            camera_tiles.append(row)
        frame = {
            "scene": args.scene, "allocator": args.allocator, "cam": cam_id,
            "camera_name": getattr(camera, "image_name", str(cam_id)),
            "width": int(camera.image_width), "height": int(camera.image_height),
            "tile_count": len(camera_tiles), **sums, "tstop_tile_count": stopped_tiles,
        }
        tile_rows.extend(camera_tiles)
        frame_rows.append(frame)
        print(
            f"[v11] {args.scene}/{args.allocator} cam={cam_id} "
            f"kvp={sums['n_kvp_raw']} retained={sums['n_kvp_retained']} "
            f"stopped={sums['n_kvp_tstopped']}", flush=True,
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "tiles.csv", TILE_FIELDS, tile_rows)
    write_csv(out / "frames.csv", FRAME_FIELDS, frame_rows)
    root = Path.cwd()
    manifest = {
        "schema": "sard-v11-tile-summary-1",
        "status": "complete",
        "scene": args.scene,
        "allocator": args.allocator,
        "allocator_definition": (
            "original 3sigma axis-aligned tile enumeration"
            if args.allocator == "baseline_aabb"
            else "ATAE-equivalent SNUGBOX conic/opacity tile enumeration"
        ),
        "camera_set": args.camera_set,
        "camera_count": len(frame_rows),
        "tile_size": [16, 16],
        "t_threshold": 1e-4,
        "t_semantics": "stop only future KVP after all valid tile pixels are done",
        "gpu_name": torch.cuda.get_device_name(0),
        "physical_gpu": int(physical_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "speedy_splat_commit": git_head(root),
        "dgr_commit": git_head(root / "submodules/diff-gaussian-rasterization"),
        "instrumentation_sha256": source_hash(root),
        "model_sha256": checkpoint_hash(Path(args.model_path)),
        "model_label": Path(args.model_path).name,
        "source_label": Path(args.source_path).name,
        "tiles_sha256": sha256_file(out / "tiles.csv"),
        "frames_sha256": sha256_file(out / "frames.csv"),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[v11] DONE {out}")


if __name__ == "__main__":
    main()
