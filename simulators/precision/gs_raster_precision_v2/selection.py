"""Deterministic fixed-format selection from calibration and held-out rows."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


SIGNALS = ("mean_x", "mean_y", "conic_a", "conic_b", "conic_c", "opacity", "rgb")


@dataclass(frozen=True)
class FormatChoice:
    signal: str
    bits: int
    frac_bits: int
    signed: bool

    @property
    def qname(self) -> str:
        integer_bits = self.bits - self.frac_bits - (1 if self.signed else 0)
        return f"{'Q' if self.signed else 'UQ'}{integer_bits}.{self.frac_bits}<{self.bits}>"

    def to_dict(self) -> dict[str, object]:
        return {
            "signal": self.signal,
            "bits": self.bits,
            "frac_bits": self.frac_bits,
            "signed": self.signed,
            "qname": self.qname,
        }


def select_calibration_formats(
    rows: Iterable[dict[str, object]],
    min_delta_psnr: float = -0.2,
    min_delta_ssim: float = -0.002,
    min_direct_psnr: float = 50.0,
) -> dict[str, dict[str, FormatChoice]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["signal"])].append(row)
    result: dict[str, dict[str, FormatChoice]] = {}
    for signal in SIGNALS:
        passing = [
            row for row in grouped[signal]
            if float(row["delta_psnr_vs_standard"]) >= min_delta_psnr
            and float(row["delta_ssim_vs_standard"]) >= min_delta_ssim
            and float(row["psnr_standard_ref"]) >= min_direct_psnr
            and int(row.get("saturations", 0)) == 0
        ]
        if not passing:
            raise ValueError(f"no passing calibration format for {signal}")
        passing.sort(key=lambda row: (int(row["bits"]), -int(row["frac_bits"])))
        minimum = passing[0]
        wider = [row for row in passing if int(row["bits"]) > int(minimum["bits"])]
        safe = wider[0] if wider else minimum

        def choice(row: dict[str, object]) -> FormatChoice:
            return FormatChoice(
                signal=signal,
                bits=int(row["bits"]),
                frac_bits=int(row["frac_bits"]),
                signed=bool(row["signed"]),
            )

        result[signal] = {"minimum": choice(minimum), "safe": choice(safe)}
    return result


def check_heldout_gate(
    rows: Iterable[dict[str, object]],
    mean_delta_psnr: float = -0.2,
    worst_delta_psnr: float = -0.5,
    mean_delta_ssim: float = -0.002,
) -> dict[str, object]:
    rows = list(rows)
    if not rows:
        raise ValueError("held-out rows are empty")
    psnr = [float(row["delta_psnr_vs_standard"]) for row in rows]
    ssim = [float(row["delta_ssim_vs_standard"]) for row in rows]
    saturation = sum(int(row.get("saturations", 0)) for row in rows)
    return {
        "cameras": len(rows),
        "mean_delta_psnr": sum(psnr) / len(psnr),
        "worst_delta_psnr": min(psnr),
        "mean_delta_ssim": sum(ssim) / len(ssim),
        "saturations": saturation,
        "pass": (
            sum(psnr) / len(psnr) >= mean_delta_psnr
            and min(psnr) >= worst_delta_psnr
            and sum(ssim) / len(ssim) >= mean_delta_ssim
            and saturation == 0
        ),
    }
