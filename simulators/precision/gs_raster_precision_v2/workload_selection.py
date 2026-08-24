"""Workload-aware precision-profile selection.

The eight scene camera-0 views generate candidate formats.  The complete
573-camera benchmark workload is then used to select the smallest measured
hardware implementation that satisfies the published quality and structure
constraints.  This module deliberately does not call that workload a test set
or make an unseen-distribution generalization claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean
from typing import Iterable, Mapping


@dataclass(frozen=True)
class WorkloadConstraints:
    camera_count: int = 573
    scene_count: int = 8
    final_non_pd: int = 0
    worst_delta_psnr_db: float = -1.0
    scene_equal_delta_psnr_db: float = -0.2
    scene_equal_delta_ssim: float = -0.002
    scene_equal_delta_lpips: float = 0.005
    max_unexplained_saturation: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_count": self.camera_count,
            "scene_count": self.scene_count,
            "final_non_pd": self.final_non_pd,
            "worst_delta_psnr_db": self.worst_delta_psnr_db,
            "scene_equal_delta_psnr_db": self.scene_equal_delta_psnr_db,
            "scene_equal_delta_ssim": self.scene_equal_delta_ssim,
            "scene_equal_delta_lpips": self.scene_equal_delta_lpips,
            "max_unexplained_saturation": self.max_unexplained_saturation,
        }


def _number(row: Mapping[str, object], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _finite_mean(values: Iterable[float]) -> float:
    valid = [value for value in values if isfinite(value)]
    return fmean(valid) if valid else float("nan")


def _scene_name(row: Mapping[str, object]) -> str:
    return str(row.get("scene_display") or row.get("scene") or "unknown")


def _profile_name(row: Mapping[str, object]) -> str:
    return str(row.get("profile") or "unknown")


def _sum_counter(rows: Iterable[Mapping[str, object]], *names: str) -> int:
    return int(round(sum(_number(row, *names) for row in rows)))


def summarize_workload(
    rows: Iterable[Mapping[str, object]],
    constraints: WorkloadConstraints = WorkloadConstraints(),
) -> list[dict[str, object]]:
    """Summarize camera rows by profile with scene-equal and pooled metrics."""
    rows = [dict(row) for row in rows]
    by_profile: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_profile.setdefault(_profile_name(row), []).append(row)

    summaries: list[dict[str, object]] = []
    for profile, profile_rows in sorted(by_profile.items()):
        by_scene: dict[str, list[dict[str, object]]] = {}
        for row in profile_rows:
            by_scene.setdefault(_scene_name(row), []).append(row)

        scene_psnr = []
        scene_ssim = []
        scene_lpips = []
        for scene_rows in by_scene.values():
            scene_psnr.append(_finite_mean(
                _number(row, "delta_psnr_vs_standard", "delta_psnr")
                for row in scene_rows
            ))
            scene_ssim.append(_finite_mean(
                _number(row, "delta_ssim_vs_standard", "delta_ssim")
                for row in scene_rows
            ))
            scene_lpips.append(_finite_mean(
                _number(row, "delta_lpips_vs_standard", "delta_lpips")
                for row in scene_rows
            ))

        psnr = [_number(row, "delta_psnr_vs_standard", "delta_psnr") for row in profile_rows]
        summary = {
            "profile": profile,
            "camera_count": len(profile_rows),
            "scene_count": len(by_scene),
            "scene_equal_delta_psnr_db": _finite_mean(scene_psnr),
            "scene_equal_delta_ssim": _finite_mean(scene_ssim),
            "scene_equal_delta_lpips": _finite_mean(scene_lpips),
            "pooled_delta_psnr_db": _finite_mean(psnr),
            "pooled_delta_psnr_median": _percentile(psnr, 0.50),
            "pooled_delta_psnr_p5": _percentile(psnr, 0.05),
            "worst_delta_psnr_db": min(psnr) if psnr else float("nan"),
            "cameras_below_minus_0p5_db": sum(value < -0.5 for value in psnr),
            "cameras_below_minus_1_db": sum(value < -1.0 for value in psnr),
            "final_non_pd": _sum_counter(
                profile_rows, "final_non_pd", "guard_final_non_pd", "guard_final_non_psd"
            ),
            "saturations": _sum_counter(
                profile_rows, "saturations", "total_saturations", "unexplained_saturations"
            ),
            "unexplained_saturations": _sum_counter(
                profile_rows, "unexplained_saturations", "unexplained_saturation"
            ),
            "dangerous_negative_saturations": _sum_counter(
                profile_rows, "dangerous_negative_saturations", "negative_saturations"
            ),
            "underflows": _sum_counter(profile_rows, "underflows", "total_underflows"),
        }
        summaries.append(summary)
    return summaries


def _percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(value for value in values if isfinite(value))
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def select_min_area_profile(
    rows: Iterable[Mapping[str, object]],
    area_by_profile: Mapping[str, float],
    constraints: WorkloadConstraints = WorkloadConstraints(),
) -> dict[str, object]:
    """Select the minimum measured-area profile satisfying the workload gate."""
    summaries = summarize_workload(rows, constraints)
    candidates: list[dict[str, object]] = []
    for summary in summaries:
        profile = str(summary["profile"])
        failures: list[str] = []
        if constraints.camera_count > 0 and summary["camera_count"] != constraints.camera_count:
            failures.append("incomplete_workload")
        if constraints.scene_count > 0 and summary["scene_count"] != constraints.scene_count:
            failures.append("incomplete_scene_set")
        if summary["final_non_pd"] != constraints.final_non_pd:
            failures.append("final_non_pd")
        if summary["worst_delta_psnr_db"] < constraints.worst_delta_psnr_db:
            failures.append("worst_delta_psnr")
        if summary["scene_equal_delta_psnr_db"] < constraints.scene_equal_delta_psnr_db:
            failures.append("scene_equal_delta_psnr")
        if summary["scene_equal_delta_ssim"] < constraints.scene_equal_delta_ssim:
            failures.append("scene_equal_delta_ssim")
        if summary["scene_equal_delta_lpips"] > constraints.scene_equal_delta_lpips:
            failures.append("scene_equal_delta_lpips")
        if summary["unexplained_saturations"] > constraints.max_unexplained_saturation:
            failures.append("unexplained_saturation")
        if summary["dangerous_negative_saturations"] > 0:
            failures.append("dangerous_negative_saturation")

        measured_area = area_by_profile.get(profile)
        if measured_area is None:
            status = "software_only" if not failures else "rejected"
            reason = "quality passes but measured area is unavailable" if not failures else "quality/structure gate failed"
        else:
            status = "pass" if not failures else "rejected"
            reason = "minimum measured area among passing workload profiles" if not failures else "quality/structure gate failed"
        candidate = dict(summary)
        candidate.update({
            "candidate_status": status,
            "constraint_failures": failures,
            "measured_area": measured_area,
            "selection_rank": None,
            "selection_reason": reason,
        })
        candidates.append(candidate)

    area_candidates = sorted(
        (candidate for candidate in candidates if candidate["candidate_status"] == "pass"),
        key=lambda candidate: float(candidate["measured_area"]),
    )
    for rank, candidate in enumerate(area_candidates, start=1):
        candidate["selection_rank"] = rank
    selected = area_candidates[0]["profile"] if area_candidates else None
    return {
        "schema": "gsrp-workload-profile-selection-v1",
        "workload_role": "profile_selection_validation",
        "calibration_role": "candidate_generation",
        "generalization_claim": False,
        "constraints": constraints.to_dict(),
        "objective": "minimize_measured_area",
        "selected_profile": selected,
        "profiles": candidates,
    }


def check_workload_gate(
    rows: Iterable[Mapping[str, object]],
    constraints: WorkloadConstraints = WorkloadConstraints(),
) -> dict[str, object]:
    """Return the gate result for one profile without selecting by area."""
    result = select_min_area_profile(rows, {"_gate_only_": 0.0}, constraints)
    summary = result["profiles"][0] if result["profiles"] else {}
    return {
        "camera_count": summary.get("camera_count", 0),
        "scene_count": summary.get("scene_count", 0),
        "mean_delta_psnr": summary.get("scene_equal_delta_psnr_db"),
        "worst_delta_psnr": summary.get("worst_delta_psnr_db"),
        "mean_delta_ssim": summary.get("scene_equal_delta_ssim"),
        "mean_delta_lpips": summary.get("scene_equal_delta_lpips"),
        "final_non_pd": summary.get("final_non_pd", 0),
        "saturations": summary.get("saturations", 0),
        "unexplained_saturations": summary.get("unexplained_saturations", 0),
        "pass": summary.get("candidate_status") == "software_only",
    }
