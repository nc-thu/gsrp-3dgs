from gs_raster_precision_v2.selection import check_heldout_gate, select_calibration_formats


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
