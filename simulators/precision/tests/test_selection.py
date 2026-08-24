from gs_raster_precision_v2.selection import check_heldout_gate, select_calibration_formats
from gs_raster_precision_v2.workload_selection import (
    WorkloadConstraints,
    select_min_area_profile,
)


def test_selects_minimum_and_next_safe_width():
    rows = []
    for signal in ("mean_x", "mean_y", "conic_a", "conic_b", "conic_c", "opacity", "rgb"):
        for bits, delta in ((7, -0.3), (8, -0.1), (9, -0.02)):
            rows.append({
                "signal": signal, "bits": bits, "frac_bits": bits - 2,
                "signed": signal in ("mean_x", "mean_y", "conic_b"),
                "delta_psnr_vs_standard": delta,
                "delta_ssim_vs_standard": -0.001,
                "psnr_standard_ref": 40.0 if bits == 7 else 55.0 + bits,
                "saturations": 0,
            })
    selected = select_calibration_formats(rows)
    assert selected["mean_x"]["minimum"].bits == 8
    assert selected["mean_x"]["safe"].bits == 9


def test_heldout_gate_rejects_saturation():
    result = check_heldout_gate([{
        "delta_psnr_vs_standard": -0.01,
        "delta_ssim_vs_standard": -0.0001,
        "saturations": 1,
    }])
    assert not result["pass"]


def test_workload_selection_rejects_20bit_tail_and_selects_21bit():
    rows = []
    scene_counts = {
        "Lego": 199,
        "Hotdog": 199,
        "Train": 37,
        "Truck": 31,
        "DrJohnson": 32,
        "Playroom": 28,
        "Bicycle": 24,
        "Garden": 23,
    }
    camera = 0
    for scene, count in scene_counts.items():
        for index in range(count):
            camera += 1
            for profile, delta, non_pd in (
                ("selected_coord18_conic20_opacity8_rgb10", -1.88 if camera == 1 else -0.02, 1 if camera == 1 else 0),
                ("selected_coord18_conic21_opacity8_rgb10", -0.89 if camera == 1 else -0.01, 0),
            ):
                rows.append({
                    "profile": profile,
                    "scene": scene,
                    "camera": camera,
                    "delta_psnr_vs_standard": delta,
                    "delta_ssim_vs_standard": -0.0001,
                    "delta_lpips_vs_standard": 0.0001,
                    "final_non_pd": non_pd,
                    "saturations": 0,
                })
    result = select_min_area_profile(
        rows,
        {
            "selected_coord18_conic20_opacity8_rgb10": 197298.0,
            "selected_coord18_conic21_opacity8_rgb10": 197612.0,
        },
        WorkloadConstraints(),
    )
    assert result["selected_profile"] == "selected_coord18_conic21_opacity8_rgb10"
    by_profile = {row["profile"]: row for row in result["profiles"]}
    assert "worst_delta_psnr" in by_profile["selected_coord18_conic20_opacity8_rgb10"]["constraint_failures"]
    assert by_profile["selected_coord18_conic21_opacity8_rgb10"]["candidate_status"] == "pass"


def test_workload_selection_marks_missing_area_software_only():
    rows = [
        {
            "profile": "software_candidate",
            "scene": f"scene_{index % 8}",
            "delta_psnr_vs_standard": -0.01,
            "delta_ssim_vs_standard": -0.0001,
            "delta_lpips_vs_standard": 0.0001,
            "final_non_pd": 0,
            "saturations": 0,
        }
        for index in range(573)
    ]
    result = select_min_area_profile(rows, {}, WorkloadConstraints())
    assert result["selected_profile"] is None
    assert result["profiles"][0]["candidate_status"] == "software_only"
