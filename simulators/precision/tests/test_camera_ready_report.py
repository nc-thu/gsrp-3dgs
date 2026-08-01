import csv
import json

import pytest

from gs_raster_precision_v2.camera_ready_report import _profile_summary


def test_profile_summary_applies_quality_metrics():
    rows = [{
        "profile": "p",
        "delta_psnr_vs_standard": "-0.1",
        "psnr_standard_ref": "55",
        "delta_ssim_vs_standard": "-0.001",
        "delta_lpips_vs_standard": "0.002",
        "saturations": "0",
    }, {
        "profile": "p",
        "delta_psnr_vs_standard": "0.02",
        "psnr_standard_ref": "60",
        "delta_ssim_vs_standard": "0.0",
        "delta_lpips_vs_standard": "-0.001",
        "saturations": "0",
    }]
    result = _profile_summary(rows)
    assert result["cameras"] == 2
    assert result["worst_delta_psnr"] == pytest.approx(-0.1)
    assert result["mean_delta_lpips"] == pytest.approx(0.0005)
    assert result["saturations"] == 0


def test_camera_ready_fixture_field_names(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text(
        "scene,camera,profile,delta_psnr_vs_standard,psnr_standard_ref,"
        "delta_ssim_vs_standard,delta_lpips_vs_standard,saturations\n"
        "lego,1,p,-0.01,60,-0.0001,0.0002,0\n",
        encoding="utf-8",
    )
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["profile"] == "p"
    assert json.loads('{"status":"complete"}')["status"] == "complete"
