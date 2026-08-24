"""Command-line interface for result validation and reporting."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .calibration import make_input_sweep, select_unified_inputs
from .camera_ready_report import build_camera_ready_report
from .report import build_report
from .workload_selection import WorkloadConstraints, select_min_area_profile


def _read_csv_paths(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(prog="gs-raster-precision-v2")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="build the offline Chinese evidence report")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output", type=Path)
    camera_ready = commands.add_parser(
        "camera-ready-report", help="build the eight-scene camera-ready report"
    )
    camera_ready.add_argument("--results", type=Path, required=True)
    camera_ready.add_argument("--profile", type=Path, required=True)
    camera_ready.add_argument("--output", type=Path)
    make_inputs = commands.add_parser("make-input-sweep")
    make_inputs.add_argument("--ranges", type=Path, required=True)
    make_inputs.add_argument("--output", type=Path, required=True)
    select_inputs = commands.add_parser("select-inputs")
    select_inputs.add_argument("--profiles", type=Path, required=True)
    select_inputs.add_argument("--results", type=Path, nargs="+", required=True)
    select_inputs.add_argument("--output", type=Path, required=True)
    select_workload = commands.add_parser(
        "select-workload-profile",
        help="select the minimum measured-area profile on the complete workload",
    )
    select_workload.add_argument("--results", type=Path, nargs="+", required=True)
    select_workload.add_argument("--costs", type=Path, required=True)
    select_workload.add_argument("--contract", type=Path, required=True)
    select_workload.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "report":
        result = build_report(args.results, args.output)
        print(json.dumps(result["decision"], ensure_ascii=False))
    elif args.command == "camera-ready-report":
        result = build_camera_ready_report(args.results, args.profile, args.output)
        print(json.dumps(result["decision"], ensure_ascii=False))
    elif args.command == "make-input-sweep":
        result = make_input_sweep(args.ranges, args.output)
        print(json.dumps({"profiles": len(result["profiles"])}, ensure_ascii=False))
    elif args.command == "select-inputs":
        result = select_unified_inputs(args.results, args.profiles, args.output)
        print(json.dumps(result["inputs"], ensure_ascii=False))
    elif args.command == "select-workload-profile":
        rows = _read_csv_paths(args.results)
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        constraints = WorkloadConstraints(**contract["constraints"])
        costs = json.loads(args.costs.read_text(encoding="utf-8"))["profiles"]
        result = select_min_area_profile(rows, costs, constraints)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "workload_selection.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        with (args.output / "workload_selection.csv").open("w", newline="", encoding="utf-8") as stream:
            profiles = result["profiles"]
            fields = list(profiles[0].keys()) if profiles else []
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(profiles)
        print(json.dumps({"selected_profile": result["selected_profile"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
