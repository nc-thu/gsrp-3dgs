"""Command-line interface for result validation and reporting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import make_input_sweep, select_unified_inputs
from .camera_ready_report import build_camera_ready_report
from .report import build_report


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


if __name__ == "__main__":
    main()
