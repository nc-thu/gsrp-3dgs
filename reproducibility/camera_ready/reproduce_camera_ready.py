"""Reproduce the camera-ready GSRP aggregate figures and tables.

This script intentionally consumes only sanitized aggregate CSV/JSON files.
It does not download models, access a server, or rerun the private 573-camera
CUDA workload.  The output is therefore an exact aggregate-data reproduction
of the reported quality and hardware summaries, not a new renderer run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "source_data"

COLORS = {
    "no_guard": "#B44C4C",
    "opacity_prune": "#D88A43",
    "integer_pd": "#7A6AA6",
    "eigen_floor": "#315B8A",
    "conic21": "#4C9A91",
    "wide_control": "#777777",
}
PROFILE_LABELS = {
    "no_guard": "No guard",
    "opacity_prune": "Opacity prune",
    "integer_pd": "Integer PD",
    "eigen_floor": "Eigen-floor",
    "conic21": "Conic21",
    "wide_control": "Wide control",
}
PROFILE_ORDER = ["no_guard", "opacity_prune", "integer_pd", "eigen_floor", "conic21", "wide_control"]
SCENE_ORDER = ["lego", "hotdog", "tt107k", "truck", "drjohnson", "playroom", "bicycle", "garden"]
SCENE_LABELS = {
    "lego": "Lego",
    "hotdog": "Hotdog",
    "tt107k": "Train",
    "truck": "Truck",
    "drjohnson": "DrJohnson",
    "playroom": "Playroom",
    "bicycle": "Bicycle",
    "garden": "Garden",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.03, **kwargs)
    plt.close(fig)


def make_fig3(out: Path) -> None:
    overall = {row["profile"]: row for row in read_csv("fig3_quality_overall.csv")}
    scene_rows = read_csv("fig3_scene_quality.csv")
    camera_rows = read_csv("fig3_camera_distribution.csv")
    audit = read_json("pd_audit_summary.json")["totals"]

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.45), constrained_layout=True)
    ax = axes[0, 0]
    values = [float(overall[p]["mean_delta_psnr"]) for p in PROFILE_ORDER]
    bars = ax.bar(range(len(values)), values, color=[COLORS[p] for p in PROFILE_ORDER], width=0.72)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.axhline(-0.2, color="#B44C4C", ls="--", lw=0.9)
    ax.set_xticks(range(len(values)), [PROFILE_LABELS[p] for p in PROFILE_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Mean ΔPSNR (dB)")
    ax.set_title("Scene-equal quality change", loc="left")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value - 0.015, f"{value:.2f}", ha="center", va="top", fontsize=6)
    panel(ax, "a")

    ax = axes[0, 1]
    matrix = np.array(
        [
            [
                float(next(r for r in scene_rows if r["profile"] == p and r["scene"] == scene)["mean_delta_psnr"])
                for scene in SCENE_ORDER
            ]
            for p in PROFILE_ORDER[:5]
        ]
    )
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.9, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(SCENE_ORDER)), [SCENE_LABELS[s] for s in SCENE_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(5), [PROFILE_LABELS[p] for p in PROFILE_ORDER[:5]])
    ax.set_title("Scene-wise ΔPSNR", loc="left")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white" if value < -0.45 else "#222222")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="dB")
    panel(ax, "b")

    ax = axes[1, 0]
    distributions = [[float(r["delta_psnr_vs_standard"]) for r in camera_rows if r["profile"] == p] for p in PROFILE_ORDER[:5]]
    box = ax.boxplot(distributions, patch_artist=True, showfliers=True, flierprops={"markersize": 1.5, "alpha": 0.3})
    for patch, profile in zip(box["boxes"], PROFILE_ORDER[:5]):
        patch.set_facecolor(COLORS[profile])
        patch.set_alpha(0.8)
    ax.axhline(-0.5, color="#B44C4C", ls="--", lw=0.9)
    ax.set_xticks(range(1, 6), [PROFILE_LABELS[p] for p in PROFILE_ORDER[:5]], rotation=25, ha="right")
    ax.set_ylabel("Camera ΔPSNR (dB)")
    ax.set_title("Camera-level tail remains visible", loc="left")
    panel(ax, "c")

    ax = axes[1, 1]
    labels = ["Eigen-floor", "Diagonal floor", "B clamp", "Final non-PD"]
    counts = [audit["eigen_floor_triggered"], audit["diagonal_floor_triggered"], audit["b_clamp_triggered"], audit["guard_final_non_psd"]]
    bars = ax.bar(range(4), counts, color=["#315B8A", "#D88A43", "#7A6AA6", "#548C67"])
    ax.set_xticks(range(4), labels, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Structure audit", loc="left")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, max(count, 1) * 1.02, f"{count:,}", ha="center", va="bottom", fontsize=6)
    ax.set_yscale("symlog", linthresh=1)
    panel(ax, "d")
    fig.suptitle("GSRP quality validation from released aggregate evidence", x=0.01, ha="left", fontsize=10, fontweight="bold")
    save_figure(fig, out, "fig3_quality_validation")


def make_fig4(out: Path) -> None:
    waterfall = read_csv("fig4_area_waterfall.csv")
    budget = {row["component"]: float(row["area"]) for row in read_csv("fig4_area_budget.csv")}
    ppa = {row["variant"]: row for row in read_csv("fig4_ppa.csv")}
    system = read_json("system_overall_summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.45), constrained_layout=True)
    ax = axes[0, 0]
    vals = [float(row["area"]) / 1000 for row in waterfall]
    bars = ax.bar(range(len(vals)), vals, color=["#E7B2AC", "#D88A43", "#9CB8D8", "#315B8A"], width=0.68)
    ax.set_xticks(range(4), ["Generic\nwidth", "Exact\nwidth", "Shared\nx/y", "Final"], fontsize=6)
    ax.set_ylabel("DC cell area (k units)")
    ax.set_title("Area waterfall", loc="left")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 12, f"{value:.1f}", ha="center", fontsize=6)
    panel(ax, "a")

    ax = axes[0, 1]
    fixed = [float(ppa["fixed_081"]["cell_area"]) / 1000, float(ppa["fixed_072"]["cell_area"]) / 1000]
    dense = [float(ppa["fp16_081"]["cell_area"]) / 1000, float(ppa["fp16_072"]["cell_area"]) / 1000]
    x = np.arange(2)
    ax.bar(x - 0.17, fixed, 0.34, label="Fixed point", color="#315B8A")
    ax.bar(x + 0.17, dense, 0.34, label="FP16 baseline", color="#777777")
    ax.set_xticks(x, ["0.81 V", "0.72 V"])
    ax.set_ylabel("DC cell area (k units)")
    ax.set_title("Core area at two voltage corners", loc="left")
    ax.legend(fontsize=6, frameon=False)
    panel(ax, "b")

    ax = axes[1, 0]
    parts = [
        ("Multiplier floor", budget["multiplier_comb_floor"], "#D88A43"),
        ("Sequential", budget["mapped_sequential"], "#4C9A91"),
        ("64 EXP ROMs", budget["mapped_exp_rom_64x"], "#7A6AA6"),
        ("Other comb.", budget["remaining_comb_after_floor_and_rom"], "#9CB8D8"),
    ]
    left = 0.0
    for label, value, color in parts:
        ax.barh([0], [value / 1000], left=left, color=color, label=label, height=0.42)
        left += value / 1000
    ax.axvline(budget["dense_quarter_target"] / 1000, color="#B44C4C", ls="--", lw=1.0, label="Dense / 4 target")
    ax.set_yticks([])
    ax.set_xlabel("DC cell area (k units)")
    ax.set_title("Final fixed-point area budget", loc="left")
    ax.legend(fontsize=5.7, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    panel(ax, "c")

    ax = axes[1, 1]
    ax.axis("off")
    values = [
        ("System area", f"{system['fixed_system_area_mm2']:.2f} / {system['dense_system_area_mm2']:.2f} mm²", "Fixed / FP16"),
        ("System power", f"{system['fixed_system_power_mw']:.2f} / {system['dense_system_power_mw']:.2f} mW", "Fixed / FP16"),
        ("Energy", f"{system['fixed_system_energy_mj_per_frame']:.2f} / {system['dense_system_energy_mj_per_frame']:.2f} mJ", "Fixed / FP16"),
        ("Guard", f"{system['guard_area_mm2']:.4f} mm², {system['guard_power_mw']:.2f} mW", "Included"),
    ]
    ax.set_title("Modeled system summary", loc="left")
    y = 0.82
    for label, value, note in values:
        ax.text(0.04, y, label, fontsize=7, fontweight="bold", color="#315B8A")
        ax.text(0.04, y - 0.11, value, fontsize=8)
        ax.text(0.04, y - 0.20, note, fontsize=6, color="#777777")
        y -= 0.25
    panel(ax, "d")
    fig.suptitle("Hardware value from released aggregate synthesis and system evidence", x=0.01, ha="left", fontsize=10, fontweight="bold")
    save_figure(fig, out, "fig4_hardware_value")


def write_tables(out: Path) -> None:
    quality = read_json("quality_overall.json")["scene_equal"]
    pooled = read_json("quality_overall.json")["pooled"]
    with (out / "table_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "standard", "fixed_point", "delta"])
        writer.writerow(["PSNR (dB)", quality["standard_psnr_mean"], quality["fixed_psnr_mean"], quality["delta_psnr_mean"]])
        writer.writerow(["SSIM", quality["standard_ssim_mean"], quality["fixed_ssim_mean"], quality["delta_ssim_mean"]])
        writer.writerow(["LPIPS", quality["standard_lpips_mean"], quality["fixed_lpips_mean"], quality["delta_lpips_mean"]])
        writer.writerow(["Pooled ΔPSNR median", "", "", pooled["delta_psnr_median"]])
        writer.writerow(["Pooled ΔPSNR P5", "", "", pooled["delta_psnr_p5"]])
        writer.writerow(["Cameras below -0.1/-0.5/-1.0 dB", "", "", f"{pooled['delta_psnr_below_m0p1']}/{pooled['delta_psnr_below_m0p5']}/{pooled['delta_psnr_below_m1p0']}"])

    system = read_json("system_overall_summary.json")
    guard = read_json("guard_ppa_summary.json")
    fixed_core = next(r for r in read_csv("fig4_ppa.csv") if r["variant"] == "fixed_081")
    fp16_core = next(r for r in read_csv("fig4_ppa.csv") if r["variant"] == "fp16_081")
    rows = [
        ("8x8 core area (cell units)", fp16_core["cell_area"], fixed_core["cell_area"]),
        ("Guard area (cell units)", "0", guard["cell_area_units"]),
        ("Guard latency / II (cycles)", "-", f"{guard['latency_cycles']} / {guard['initiation_interval']}"),
        ("System area (mm2)", system["dense_system_area_mm2"], system["fixed_system_area_mm2"]),
        ("System power (mW)", system["dense_system_power_mw"], system["fixed_system_power_mw"]),
        ("Full-system FPS", system["equal_scene_full_system_fps"], system["equal_scene_full_system_fps"]),
        ("Energy (mJ/frame)", system["dense_system_energy_mj_per_frame"], system["fixed_system_energy_mj_per_frame"]),
        ("Efficiency (FPS/W)", system["dense_system_fps_per_w"], system["fixed_system_fps_per_w"]),
    ]
    with (out / "table_hardware.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "fp16_baseline", "gsrp_fixed"])
        writer.writerows(rows)


def write_manifest(out: Path) -> None:
    inputs = {}
    for path in sorted(DATA.iterdir()):
        if path.is_file():
            inputs[path.name] = sha256(path)
    outputs = {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest = {
        "schema": "gsrp-camera-ready-reproduction-v1",
        "repository": "https://github.com/nc-thu/gsrp-3dgs",
        "public_commit": commit,
        "scope": "aggregate-data reproduction of Fig.3, Fig.4, Table quality, and Table hardware",
        "cuda_rerun": "requires user-supplied models and datasets; not performed by this script",
        "inputs": inputs,
        "outputs": outputs,
    }
    (out / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reproduced_camera_ready"))
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_fig3(out)
    make_fig4(out)
    write_tables(out)
    write_manifest(out)
    print(f"Reproduced camera-ready artifacts in {out}")


if __name__ == "__main__":
    main()
