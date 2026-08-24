"""Cross-scene static-Q calibration helpers."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


SIGNALS = ("mean_x", "mean_y", "conic_a", "conic_b", "conic_c", "opacity", "rgb")
SIGNED = {
    "mean_x": True, "mean_y": True, "conic_a": False, "conic_b": True,
    "conic_c": False, "opacity": False, "rgb": False,
}
BIT_RANGES = {
    "mean_x": range(16, 31), "mean_y": range(16, 31),
    "conic_a": range(16, 31), "conic_b": range(16, 31), "conic_c": range(16, 31),
    "opacity": range(4, 13), "rgb": range(6, 19),
}


def largest_frac(bits: int, signed: bool, minimum: float, maximum: float) -> int | None:
    for frac in range(bits, -1, -1):
        lo_code = -(1 << (bits - 1)) if signed else 0
        hi_code = (1 << (bits - (1 if signed else 0))) - 1
        step = 2.0 ** -frac
        if lo_code * step <= minimum and hi_code * step >= maximum:
            return frac
    return None


def aggregate_camera0_ranges(root: Path) -> dict[str, dict[str, float]]:
    values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in sorted(root.glob("*/ranges.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                values[row["signal"]].append((float(row["minimum"]), float(row["maximum"])))
    missing = set(SIGNALS) - set(values)
    if missing:
        raise ValueError(f"missing calibration ranges: {sorted(missing)}")
    return {
        signal: {
            "minimum": min(item[0] for item in values[signal]),
            "maximum": max(item[1] for item in values[signal]),
            "scenes": len(values[signal]),
        }
        for signal in SIGNALS
    }


def make_input_sweep(root: Path, output: Path) -> dict[str, object]:
    ranges = aggregate_camera0_ranges(root)
    profiles = []
    for signal in SIGNALS:
        item = ranges[signal]
        for bits in BIT_RANGES[signal]:
            frac = largest_frac(bits, SIGNED[signal], item["minimum"], item["maximum"])
            if frac is None:
                continue
            profiles.append({
                "name": f"input_{signal}_b{bits}_f{frac}",
                "stage": "input",
                "swept_signal": signal,
                "inputs": {
                    signal: {
                        "bits": bits,
                        "frac_bits": frac,
                        "signed": SIGNED[signal],
                    }
                },
                "quadratic": {"enabled": False},
                "exp": {"enabled": False},
                "blend": {"enabled": False},
            })
    payload = {
        "schema": "gs-raster-precision-profiles-1",
        "calibration": "eight_scene_camera0",
        "ranges": ranges,
        "profiles": profiles,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def select_unified_inputs(
    profile_files: list[Path],
    profile_definition: Path,
    output: Path,
    min_delta_psnr: float = -0.2,
    min_delta_ssim: float = -0.002,
    min_direct_psnr: float = 50.0,
) -> dict[str, object]:
    definition = json.loads(profile_definition.read_text(encoding="utf-8"))
    profile_map = {item["name"]: item for item in definition["profiles"]}
    rows = []
    for path in profile_files:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    by_profile: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_profile[row["profile"]].append(row)

    selected = {}
    for signal in SIGNALS:
        candidates = []
        for name, group in by_profile.items():
            profile = profile_map.get(name)
            if not profile or profile.get("swept_signal") != signal:
                continue
            candidates.append({
                "name": name,
                "profile": profile,
                "scenes": len(group),
                "mean_delta_psnr": sum(float(row["delta_psnr_vs_standard"]) for row in group) / len(group),
                "worst_delta_psnr": min(float(row["delta_psnr_vs_standard"]) for row in group),
                "mean_delta_ssim": sum(float(row["delta_ssim_vs_standard"]) for row in group) / len(group),
                "worst_direct_psnr": min(float(row["psnr_standard_ref"]) for row in group),
                "saturations": sum(int(row["saturations"]) for row in group),
            })
        passing = [
            item for item in candidates
            if item["mean_delta_psnr"] >= min_delta_psnr
            and item["worst_delta_psnr"] >= -0.5
            and item["mean_delta_ssim"] >= min_delta_ssim
            and item["worst_direct_psnr"] >= min_direct_psnr
            and item["saturations"] == 0
        ]
        if not passing:
            raise ValueError(f"no unified camera-0 input format passes for {signal}")
        passing.sort(key=lambda item: (
            item["profile"]["inputs"][signal]["bits"],
            -item["profile"]["inputs"][signal]["frac_bits"],
        ))
        selected[signal] = passing[0]

    frozen_inputs = {
        signal: selected[signal]["profile"]["inputs"][signal] for signal in SIGNALS
    }
    payload = {
        "schema": "gs-raster-precision-input-profile-1",
        "status": "camera0_calibrated",
        "reference": "standard_renderCUDA",
        "allocator": "atae_snugbox",
        "inputs": frozen_inputs,
        "selection": {
            signal: {key: value for key, value in selected[signal].items() if key != "profile"}
            for signal in SIGNALS
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload

