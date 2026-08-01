#!/usr/bin/env python3
"""Run range collection or staged precision profiles on real CUDA images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from pathlib import Path

import torch
import torchvision
import scene as scene_module
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch import lpips
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim


INPUTS = ("mean_x", "mean_y", "conic_a", "conic_b", "conic_c", "opacity", "rgb")
PREFIX = {
    "mean_x": "MEAN_X", "mean_y": "MEAN_Y", "conic_a": "CONIC_A",
    "conic_b": "CONIC_B", "conic_c": "CONIC_C", "opacity": "OPACITY", "rgb": "RGB",
}
SAT_NAMES = INPUTS + (
    "quad_geometry", "quad_weighted_term", "quad_sum", "power",
    "alpha", "transmittance", "visibility", "rgb_accumulator",
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def decode_double(value: int) -> float:
    if not value:
        return 0.0
    return struct.unpack("<d", struct.pack("<Q", int(value) & ((1 << 64) - 1)))[0]


def metrics(candidate: torch.Tensor, reference: torch.Tensor, gt: torch.Tensor, with_lpips: bool) -> dict[str, float]:
    candidate_psnr = float(psnr(candidate, gt).mean())
    reference_psnr = float(psnr(reference, gt).mean())
    candidate_ssim = float(ssim(candidate, gt).mean())
    reference_ssim = float(ssim(reference, gt).mean())
    candidate_lpips = float(lpips(candidate, gt, net_type="vgg").mean()) if with_lpips else math.nan
    reference_lpips = float(lpips(reference, gt, net_type="vgg").mean()) if with_lpips else math.nan
    return {
        "psnr_gt": candidate_psnr,
        "standard_psnr_gt": reference_psnr,
        "delta_psnr_vs_standard": candidate_psnr - reference_psnr,
        "ssim_gt": candidate_ssim,
        "standard_ssim_gt": reference_ssim,
        "delta_ssim_vs_standard": candidate_ssim - reference_ssim,
        "lpips_gt": candidate_lpips,
        "standard_lpips_gt": reference_lpips,
        "delta_lpips_vs_standard": candidate_lpips - reference_lpips if with_lpips else math.nan,
        "psnr_standard_ref": float(psnr(candidate, reference).mean()),
        "max_abs_error_vs_standard": float((candidate - reference).abs().max()),
        "mean_abs_error_vs_standard": float((candidate - reference).abs().mean()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clear_profile_env() -> None:
    for key in list(os.environ):
        if key.startswith("GSRP_"):
            os.environ.pop(key, None)


def apply_profile(profile: dict[str, object]) -> None:
    clear_profile_env()
    os.environ["GSRP_MODE"] = "7"
    os.environ["GSRP_RANGE_STATS"] = "0"
    inputs = profile.get("inputs", {})
    mask = 0
    for index, signal in enumerate(INPUTS):
        if signal not in inputs:
            continue
        fmt = inputs[signal]
        os.environ[f"GSRP_{PREFIX[signal]}_BITS"] = str(int(fmt["bits"]))
        os.environ[f"GSRP_{PREFIX[signal]}_FRAC"] = str(int(fmt["frac_bits"]))
        mask |= 1 << index
    os.environ["GSRP_ACTIVE_MASK"] = hex(mask)

    quad = profile.get("quadratic", {})
    os.environ["GSRP_QUAD_FIXED"] = "1" if quad.get("enabled") else "0"
    if quad.get("enabled"):
        os.environ["GSRP_QUAD_GEOM_BITS"] = str(int(quad["geometry_bits"]))
        os.environ["GSRP_QUAD_GEOM_FRAC"] = str(int(quad["geometry_frac_bits"]))
        os.environ["GSRP_QUAD_TERM_BITS"] = str(int(quad["term_bits"]))
        os.environ["GSRP_QUAD_TERM_FRAC"] = str(int(quad["term_frac_bits"]))
        os.environ["GSRP_QUAD_SUM_BITS"] = str(int(quad["sum_bits"]))
        os.environ["GSRP_QUAD_SUM_FRAC"] = str(int(quad["sum_frac_bits"]))
        os.environ["GSRP_POWER_BITS"] = str(int(quad["power_bits"]))
        os.environ["GSRP_POWER_FRAC"] = str(int(quad["power_frac_bits"]))

    exp = profile.get("exp", {})
    os.environ["GSRP_EXP_LUT_ENABLE"] = "1" if exp.get("enabled") else "0"
    if exp.get("enabled"):
        os.environ["GSRP_EXP_ADDR_BITS"] = str(int(exp["address_bits"]))
        os.environ["GSRP_EXP_OUT_BITS"] = str(int(exp["output_bits"]))

    blend = profile.get("blend", {})
    os.environ["GSRP_BLEND_FIXED"] = "1" if blend.get("enabled") else "0"
    if blend.get("enabled"):
        enabled = set(blend.get("signals", ("alpha", "t", "visibility", "accumulator")))
        names = ("alpha", "t", "visibility", "accumulator")
        mask = sum(1 << index for index, name in enumerate(names) if name in enabled)
        os.environ["GSRP_BLEND_MASK"] = hex(mask)
        for key, env_name in (
            ("alpha", "ALPHA"), ("t", "T"), ("visibility", "VIS"), ("accumulator", "ACC")
        ):
            if key in blend:
                os.environ[f"GSRP_BLEND_{env_name}_BITS"] = str(int(blend[key]["bits"]))
                os.environ[f"GSRP_BLEND_{env_name}_FRAC"] = str(int(blend[key]["frac_bits"]))


def deep_merge(base: dict[str, object], update: dict[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline_args = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-start", type=int, default=0)
    parser.add_argument("--num-cams", type=int, default=1)
    parser.add_argument("--stage", choices=("range", "evaluate"), required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--with-lpips", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    profiles_path = getattr(args, "profiles", None)
    if args.stage == "evaluate" and not profiles_path:
        raise SystemExit("--profiles is required for evaluate")

    safe_state(args.quiet)
    dataset, pipeline = model.extract(args), pipeline_args.extract(args)
    original_loader = scene_module.cameraList_from_camInfos
    calls = {"n": 0}

    def limited_loader(infos, resolution_scale, model_args):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        end = args.camera_start + args.num_cams
        return original_loader(infos[args.camera_start:end], resolution_scale, model_args)

    scene_module.cameraList_from_camInfos = limited_loader
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = list(scene.getTestCameras())
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["SARD_ALLOCATOR_MODE"] = "atae_snugbox"
    os.environ["SARD_SOD"] = "0"
    os.environ.pop("SARD_PRECISION_QUANT", None)

    clear_profile_env()
    os.environ["GSRP_MODE"] = "0"
    references, gts = [], []
    with torch.no_grad():
        for offset, camera in enumerate(cameras):
            image = torch.clamp(render(camera, gaussians, pipeline, background)["render"], 0, 1).detach()
            references.append(image)
            gts.append(torch.clamp(camera.original_image[:3].cuda(), 0, 1))
            if args.save_images and offset == 0:
                torchvision.utils.save_image(image, args.output / "standard.png")

    clear_profile_env()
    os.environ["GSRP_MODE"] = "5"
    os.environ["GSRP_ACTIVE_MASK"] = "0"
    os.environ["GSRP_RANGE_STATS"] = "1"
    range_rows, contract = [], []
    contract_counters = []
    with torch.no_grad():
        for offset, (camera, reference, gt) in enumerate(zip(cameras, references, gts)):
            image = torch.clamp(render(camera, gaussians, pipeline, background)["render"], 0, 1)
            counters = __import__("diff_gaussian_rasterization").get_last_counters().cpu().tolist()
            contract_counters.append(counters)
            contract.append(metrics(image, reference, gt, False))
            for index, signal in enumerate(INPUTS):
                base = 12 + index * 6
                range_rows.append({
                    "scene": args.scene,
                    "camera": args.camera_start + offset,
                    "signal": signal,
                    "minimum": decode_double(counters[base]),
                    "maximum": decode_double(counters[base + 1]),
                    "operations": int(counters[base + 2]),
                    "positive_values": int(counters[base + 3]),
                    "negative_values": int(counters[base + 4]),
                })
    baseline = {
        "minimum_psnr_standard_ref": min(row["psnr_standard_ref"] for row in contract),
        "maximum_pixel_error": max(row["max_abs_error_vs_standard"] for row in contract),
    }
    baseline["pass"] = baseline["minimum_psnr_standard_ref"] >= 50.0
    (args.output / "baseline_check.json").write_text(json.dumps(baseline, indent=2) + "\n")
    write_csv(args.output / "ranges.csv", range_rows)
    if not baseline["pass"]:
        raise SystemExit("corrected float path failed renderCUDA contract")

    result_path = args.output / "ranges.csv"
    if args.stage == "evaluate":
        payload = json.loads(profiles_path.read_text(encoding="utf-8"))
        defaults = payload.get("defaults", {})
        profiles = [deep_merge(defaults, item) for item in payload["profiles"]]
        rows = []
        for profile in profiles:
            apply_profile(profile)
            with torch.no_grad():
                for offset, (camera, reference, gt, ref_counters) in enumerate(
                    zip(cameras, references, gts, contract_counters)
                ):
                    image = torch.clamp(render(camera, gaussians, pipeline, background)["render"], 0, 1)
                    counters = __import__("diff_gaussian_rasterization").get_last_counters().cpu().tolist()
                    item = {
                        "scene": args.scene,
                        "camera": args.camera_start + offset,
                        "profile": profile["name"],
                        **metrics(image, reference, gt, args.with_lpips),
                    }
                    for index, name in enumerate(SAT_NAMES):
                        item[f"sat_{name}"] = int(counters[index])
                    item["saturations"] = sum(int(counters[index]) for index in range(len(SAT_NAMES)))
                    item["delta_last_contributor_sum"] = int(counters[111]) - int(ref_counters[111])
                    item["delta_t_stop_pixels"] = int(counters[112]) - int(ref_counters[112])
                    rows.append(item)
                    if args.save_images and offset == 0:
                        torchvision.utils.save_image(image, args.output / f"{profile['name']}.png")
        result_path = args.output / "results.csv"
        write_csv(result_path, rows)

    manifest = {
        "schema": "gs-raster-precision-camera-ready-1",
        "status": "complete",
        "scene": args.scene,
        "stage": args.stage,
        "reference": "standard_renderCUDA",
        "allocator": "atae_snugbox",
        "camera_start": args.camera_start,
        "cameras": len(cameras),
        "physical_gpu": os.environ.get("GSRP_PHYSICAL_GPU"),
        "result_sha256": sha256(result_path),
        "baseline_check": baseline,
    }
    if profiles_path:
        manifest["profiles_sha256"] = sha256(profiles_path)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
