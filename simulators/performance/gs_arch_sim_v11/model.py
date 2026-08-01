"""Compact camera-level timing model for the SARD v1.1 target architecture."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Spec:
    tile_size: int = 16
    cores: int = 4
    fclk_mhz: float = 400.0
    kvp_contexts: int = 2
    fifo_rows: int = 32
    w16_cycles_per_kvp: int = 16
    fill_drain_cycles: int = 30

    def as_dict(self) -> dict:
        return asdict(self)


def tile_cycles(row: dict, with_t: bool, spec: Spec) -> int:
    """Latency of one tile under a continuously overlapped RIE/W16 RSPA."""
    suffix = "retained" if with_t else "raw"
    kvp = int(row[f"n_kvp_{suffix}"])
    producer = int(row[f"producer_cycles_{suffix}"])
    if kvp == 0:
        return 0
    return max(producer, spec.w16_cycles_per_kvp * kvp) + spec.fill_drain_cycles


def schedule(rows: list[dict], with_t: bool, spec: Spec) -> tuple[int, list[int]]:
    """Renderer-order tiles go to the earliest-free core; core id breaks ties."""
    loads = [0] * spec.cores
    for row in sorted(rows, key=lambda item: int(item["tile_id"])):
        core = min(range(spec.cores), key=lambda index: (loads[index], index))
        loads[core] += tile_cycles(row, with_t, spec)
    return max(loads, default=0), loads


def replay_camera(rows: list[dict], spec: Spec | None = None) -> dict:
    spec = spec or Spec()
    no_t_cycles, no_t_loads = schedule(rows, False, spec)
    with_t_cycles, with_t_loads = schedule(rows, True, spec)

    def total(key: str) -> int:
        return sum(int(row[key]) for row in rows)

    raw = total("n_kvp_raw")
    retained = total("n_kvp_retained")
    stopped = total("n_kvp_tstopped")
    return {
        "continuous_w16_no_t_cycles": no_t_cycles,
        "continuous_w16_tile_tstop_cycles": with_t_cycles,
        "continuous_w16_no_t_fps": spec.fclk_mhz * 1e6 / max(no_t_cycles, 1),
        "continuous_w16_tile_tstop_fps": spec.fclk_mhz * 1e6 / max(with_t_cycles, 1),
        "t_speedup": no_t_cycles / max(with_t_cycles, 1),
        "n_kvp_raw": raw,
        "n_kvp_retained": retained,
        "n_kvp_tstopped": stopped,
        "t_stopped_fraction": stopped / raw if raw else 0.0,
        "producer_cycles_raw": total("producer_cycles_raw"),
        "producer_cycles_retained": total("producer_cycles_retained"),
        "no_t_core_cycles": no_t_loads,
        "with_t_core_cycles": with_t_loads,
        "no_t_core_imbalance": _imbalance(no_t_loads),
        "with_t_core_imbalance": _imbalance(with_t_loads),
    }


def _imbalance(loads: list[int]) -> float:
    mean = sum(loads) / len(loads) if loads else 0.0
    return (max(loads) - min(loads)) / mean if mean else 0.0
