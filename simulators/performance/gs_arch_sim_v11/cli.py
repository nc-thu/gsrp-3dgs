from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import run_workloads
from .io import atomic_json, group_cameras, load_tiles
from .model import Spec, replay_camera
from .report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="sard-v11")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract-summary", help="document the server summary adapter")
    extract.add_argument("--show-adapter", action="store_true")

    validate = commands.add_parser("validate-camera")
    validate.add_argument("tiles_csv", type=Path)

    replay = commands.add_parser("replay-camera")
    replay.add_argument("tiles_csv", type=Path)
    replay.add_argument("--camera", type=int, default=0)
    replay.add_argument("--out", type=Path)

    run = commands.add_parser("run-experiments")
    run.add_argument("workload_root", type=Path)
    run.add_argument("--out", type=Path, required=True)

    report = commands.add_parser("report")
    report.add_argument("result_dir", type=Path)
    report.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "extract-summary":
        print("The private renderer extractor is outside this public artifact; provide aggregate CSV inputs instead.")
        return
    if args.command == "validate-camera":
        groups = group_cameras(load_tiles(args.tiles_csv))
        print(json.dumps({"status": "pass", "camera_count": len(groups)}, indent=2))
        return
    if args.command == "replay-camera":
        groups = group_cameras(load_tiles(args.tiles_csv))
        selected = [item for item in groups if item[0][2] == args.camera]
        if len(selected) != 1:
            raise ValueError(f"camera {args.camera} not found uniquely")
        key, tiles = selected[0]
        value = {"scene": key[0], "allocator": key[1], "cam": key[2],
                 "camera_name": key[3], "spec": Spec().as_dict(), **replay_camera(tiles)}
        if args.out:
            atomic_json(args.out, value)
        else:
            print(json.dumps(value, indent=2))
        return
    if args.command == "run-experiments":
        print(json.dumps(run_workloads(args.workload_root, args.out), indent=2))
        return
    build_report(args.result_dir, args.out)


if __name__ == "__main__":
    main()
