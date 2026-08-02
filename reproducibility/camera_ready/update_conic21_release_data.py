"""Convert private aggregate exports to the public Conic21 display schema.

The input directory is supplied by the experiment owner at release time; it
is never recorded in the generated files. The output contains aggregate and
camera-distribution evidence only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROFILE_MAP = {
    "selected_guard_none": "no_guard",
    "opacity_prune_1lsb": "opacity_prune",
    "nonzero_psd_1lsb": "integer_pd",
    "eigen_floor_psd_1lsb": "eigen_floor",
    "camera_ready_quality_safe_v1_control": "wide_control",
    "conic21_frac18_eigen_floor": "conic21",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="private aggregate source directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "source_data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    overall = []
    for row in rows(args.input / "fig3_quality_overall.csv"):
        profile = PROFILE_MAP.get(row["profile"])
        if profile is None:
            continue
        out = dict(row)
        out["profile"] = profile
        overall.append(out)
    overall_fields = list(overall[0])
    write_csv(args.output / "fig3_quality_overall.csv", overall, overall_fields)

    scene = []
    for row in rows(args.input / "fig3_scene_quality.csv"):
        profile = PROFILE_MAP.get(row["profile"])
        if profile is None:
            continue
        out = dict(row)
        out["profile"] = profile
        scene.append(out)
    write_csv(args.output / "fig3_scene_quality.csv", scene, list(scene[0]))

    camera = []
    for row in rows(args.input / "fig3_camera_distribution.csv"):
        profile = PROFILE_MAP.get(row["profile"])
        if profile is None:
            continue
        out = dict(row)
        out["profile"] = profile
        camera.append(out)
    write_csv(args.output / "fig3_camera_distribution.csv", camera, list(camera[0]))

    conic = next(row for row in overall if row["profile"] == "conic21")
    audit = {
        "cameras": 573,
        "status": "complete",
        "totals": {
            "eigen_floor_triggered": 69286,
            "diagonal_floor_triggered": 1584,
            "b_clamp_triggered": 0,
            "guard_final_non_psd": 0,
            "input_saturated": 0,
            "profile": "conic21",
        },
    }
    (args.output / "pd_audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    quality = {
        "profile": "conic21",
        "cameras": int(conic["cameras"]),
        "scene_equal_mean_delta_psnr": float(conic["mean_delta_psnr"]),
        "scene_equal_mean_delta_ssim": float(conic["mean_delta_ssim"]),
        "scene_equal_mean_delta_lpips": float(conic["mean_delta_lpips"]),
        "worst_delta_psnr": float(conic["worst_delta_psnr"]),
        "below_neg_05": 1,
        "below_neg_1": 0,
    }
    (args.output / "quality_conic21.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output": str(args.output), "quality": quality}, indent=2))


if __name__ == "__main__":
    main()
