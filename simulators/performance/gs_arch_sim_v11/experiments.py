from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from .io import atomic_json, group_cameras, load_tiles, sha256_file, write_csv
from .model import Spec, replay_camera


def run_workloads(workload_root: Path, out: Path, spec: Spec | None = None) -> dict:
    spec = spec or Spec()
    tile_files = sorted(workload_root.glob("*/*/tiles.csv"))
    if not tile_files:
        raise FileNotFoundError(f"no allocator/scene/tiles.csv under {workload_root}")
    camera_rows: list[dict] = []
    inputs: list[dict] = []
    for path in tile_files:
        inputs.append({"path": "/".join(path.parts[-3:]), "sha256": sha256_file(path)})
        for key, camera_tiles in group_cameras(load_tiles(path)):
            scene, allocator, cam, camera_name = key
            result = replay_camera(camera_tiles, spec)
            camera_rows.append({
                "scene": scene, "allocator": allocator, "cam": cam,
                "camera_name": camera_name, **result,
            })

    camera_rows.sort(key=lambda row: (row["allocator"], row["scene"], int(row["cam"])))
    scene_rows = _scene_summary(camera_rows, spec)
    overall_rows = _overall_summary(scene_rows, camera_rows, spec)
    comparison_rows = _comparison_summary(overall_rows)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "camera_results.csv", camera_rows)
    write_csv(out / "scene_summary.csv", scene_rows)
    write_csv(out / "overall_summary.csv", overall_rows)
    write_csv(out / "comparison_2x2.csv", comparison_rows)
    manifest = {
        "schema": "gs-arch-sim-v11-results-1",
        "status": "complete",
        "profile": "continuous_w16_tile_tstop",
        "scene_naming": {
            "canonical_key": "train",
            "display_name": "Train",
            "summary_migration": True,
        },
        "spec": spec.as_dict(),
        "camera_count": len(camera_rows),
        "scene_count": len({row["scene"] for row in camera_rows}),
        "allocator_count": len({row["allocator"] for row in camera_rows}),
        "inputs": inputs,
    }
    atomic_json(out / "manifest.json", manifest)
    manifest["camera_results_sha256"] = sha256_file(out / "camera_results.csv")
    manifest["scene_summary_sha256"] = sha256_file(out / "scene_summary.csv")
    manifest["overall_summary_sha256"] = sha256_file(out / "overall_summary.csv")
    manifest["comparison_2x2_sha256"] = sha256_file(out / "comparison_2x2.csv")
    atomic_json(out / "manifest.json", manifest)
    return manifest


def _scene_summary(cameras: list[dict], spec: Spec) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in cameras:
        groups.setdefault((row["allocator"], row["scene"]), []).append(row)
    output = []
    for (allocator, scene), rows in sorted(groups.items()):
        no_t = fmean(float(row["continuous_w16_no_t_cycles"]) for row in rows)
        with_t = fmean(float(row["continuous_w16_tile_tstop_cycles"]) for row in rows)
        raw = sum(int(row["n_kvp_raw"]) for row in rows)
        stopped = sum(int(row["n_kvp_tstopped"]) for row in rows)
        output.append({
            "allocator": allocator, "scene": scene, "camera_count": len(rows),
            "mean_no_t_cycles": no_t, "mean_with_t_cycles": with_t,
            "no_t_fps": spec.fclk_mhz * 1e6 / no_t,
            "with_t_fps": spec.fclk_mhz * 1e6 / with_t,
            "t_speedup": no_t / with_t,
            "n_kvp_raw": raw,
            "n_kvp_retained": sum(int(row["n_kvp_retained"]) for row in rows),
            "n_kvp_tstopped": stopped,
            "t_stopped_fraction": stopped / raw if raw else 0.0,
        })
    return output


def _overall_summary(scenes: list[dict], cameras: list[dict], spec: Spec) -> list[dict]:
    output = []
    for allocator in sorted({row["allocator"] for row in scenes}):
        selected = [row for row in scenes if row["allocator"] == allocator]
        no_t = fmean(float(row["mean_no_t_cycles"]) for row in selected)
        with_t = fmean(float(row["mean_with_t_cycles"]) for row in selected)
        pooled = [row for row in cameras if row["allocator"] == allocator]
        output.append({
            "allocator": allocator, "scene_count": len(selected),
            "camera_count": len(pooled), "equal_scene_no_t_cycles": no_t,
            "equal_scene_with_t_cycles": with_t,
            "equal_scene_no_t_fps": spec.fclk_mhz * 1e6 / no_t,
            "equal_scene_with_t_fps": spec.fclk_mhz * 1e6 / with_t,
            "equal_scene_t_speedup": no_t / with_t,
            "pooled_camera_no_t_cycles": fmean(float(row["continuous_w16_no_t_cycles"]) for row in pooled),
            "pooled_camera_with_t_cycles": fmean(float(row["continuous_w16_tile_tstop_cycles"]) for row in pooled),
        })
    return output


def _comparison_summary(overall: list[dict]) -> list[dict]:
    by_allocator = {row["allocator"]: row for row in overall}
    baseline = float(by_allocator["baseline_aabb"]["equal_scene_no_t_cycles"])
    output = []
    for allocator in ("baseline_aabb", "atae_snugbox"):
        row = by_allocator[allocator]
        for tstop, cycle_key, fps_key in (
            ("off", "equal_scene_no_t_cycles", "equal_scene_no_t_fps"),
            ("on", "equal_scene_with_t_cycles", "equal_scene_with_t_fps"),
        ):
            cycles = float(row[cycle_key])
            output.append({
                "allocator": allocator, "tile_tstop": tstop,
                "equal_scene_cycles": cycles, "raster_fps": float(row[fps_key]),
                "speedup_vs_baseline_aabb_no_t": baseline / cycles,
            })
    return output
