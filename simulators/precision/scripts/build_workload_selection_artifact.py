#!/usr/bin/env python3
"""Normalize completed camera-level results for public profile selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROFILE20 = "selected_coord18_conic20_opacity8_rgb10"
PROFILE21 = "selected_coord18_conic21_opacity8_rgb10"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalize(
    rows: list[dict[str, str]],
    profile: str,
    *,
    source_profile: str | None = None,
    method: str | None = None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if source_profile is not None and row.get("profile") != source_profile:
            continue
        if method is not None and row.get("method") != method:
            continue
        item: dict[str, object] = {
            "scene": row.get("scene", ""),
            "scene_display": row.get("scene_display", row.get("scene", "")),
            "camera": row.get("camera", ""),
            "profile": profile,
            "profile_bits": 20 if "conic20" in profile else 21,
            "guard_mode": "full_pd_guard",
            "delta_psnr_vs_standard": row.get("delta_psnr_vs_standard", ""),
            "delta_ssim_vs_standard": row.get("delta_ssim_vs_standard", ""),
            "delta_lpips_vs_standard": row.get("delta_lpips_vs_standard", ""),
            "psnr_gt": row.get("psnr_gt", ""),
            "standard_psnr_gt": row.get("standard_psnr_gt", ""),
            "ssim_gt": row.get("ssim_gt", ""),
            "standard_ssim_gt": row.get("standard_ssim_gt", ""),
            "lpips_gt": row.get("lpips_gt", ""),
            "standard_lpips_gt": row.get("standard_lpips_gt", ""),
            "psnr_standard_ref": row.get("psnr_standard_ref", ""),
            "final_non_pd": row.get("guard_final_non_pd", row.get("guard_final_non_psd", "0")),
            "saturations": row.get("saturations", "0"),
            "unexplained_saturations": row.get("unexplained_saturations", row.get("unexplained_saturation", "0")),
            "dangerous_negative_saturations": row.get("dangerous_negative_saturations", "0"),
            "underflows": row.get("underflows", "0"),
            "guard_eigen_floor_triggered": row.get("guard_eigen_floor_triggered", ""),
            "guard_diagonal_floor_triggered": row.get("guard_diagonal_floor_triggered", ""),
            "guard_b_clamp_triggered": row.get("guard_b_clamp_triggered", ""),
        }
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile20", type=Path, required=True)
    parser.add_argument("--profile21", type=Path, required=True)
    parser.add_argument("--profile20-name", default="eigen_floor_psd_1lsb")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows20 = normalize(read_csv(args.profile20), PROFILE20, source_profile=args.profile20_name)
    rows21 = normalize(read_csv(args.profile21), PROFILE21, method="full_gsrp")
    if len(rows20) != 573 or len(rows21) != 573:
        raise SystemExit(f"expected 573 rows per profile, got {len(rows20)} and {len(rows21)}")

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "profile20_camera_results.csv", rows20)
    write_csv(args.output / "profile21_camera_results.csv", rows21)
    write_csv(args.output / "profile_selection_camera_results.csv", rows20 + rows21)
    manifest = {
        "schema": "gsrp-public-workload-selection-input-v1",
        "workload_role": "profile_selection_validation",
        "calibration_role": "candidate_generation",
        "camera_count_per_profile": 573,
        "scene_count": 8,
        "profiles": [PROFILE20, PROFILE21],
        "source_labels": {
            PROFILE20: "completed 20-bit full-PD-guard camera results",
            PROFILE21: "completed 21-bit full-GSRP camera results",
        },
        "source_sha256": {
            "profile20": sha256(args.profile20),
            "profile21": sha256(args.profile21),
        },
        "cuda_rerun": False,
    }
    (args.output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
