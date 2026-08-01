from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REQUIRED = {
    "scene", "allocator", "cam", "camera_name", "tile_id",
    "n_kvp_raw", "n_kvp_retained", "n_kvp_tstopped",
    "producer_cycles_raw", "producer_cycles_retained", "tstop_after_kvp",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tiles(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty tile summary: {path}")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"missing fields in {path}: {sorted(missing)}")
    for row in rows:
        validate_tile(row)
    return rows


def validate_tile(row: dict) -> None:
    raw = int(row["n_kvp_raw"])
    retained = int(row["n_kvp_retained"])
    stopped = int(row["n_kvp_tstopped"])
    stop_after = int(row["tstop_after_kvp"])
    if raw != retained + stopped:
        raise ValueError(f"raw/retained/stopped closure failed for tile {row['tile_id']}")
    if not 0 <= stop_after <= raw:
        raise ValueError(f"invalid stop position for tile {row['tile_id']}")
    if int(row["producer_cycles_retained"]) > int(row["producer_cycles_raw"]):
        raise ValueError(f"producer-cycle closure failed for tile {row['tile_id']}")


def group_cameras(rows: list[dict]) -> list[tuple[tuple[str, str, int, str], list[dict]]]:
    grouped: dict[tuple[str, str, int, str], list[dict]] = {}
    for row in rows:
        key = (row["scene"], row["allocator"], int(row["cam"]), row["camera_name"])
        grouped.setdefault(key, []).append(row)
    return sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
