"""Build the public release manifest without exposing local paths.

Only the explicitly published tree is hashed.  The manifest describes the
artifact boundary; it is not a record of the private experiment workspace.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT_FILES = {
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
}
ROOT_DIRS = ("docs", "examples", "simulators")
EXCLUDED = (
    "paper/",
    "models/",
    "datasets/",
    "traces/",
    "rtl/",
    "netlists/",
    "dc/",
    "server/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_files(root: Path) -> list[Path]:
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory in ROOT_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        paths.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and "build" not in path.parts
            and "dist" not in path.parts
            and not any(part.endswith(".egg-info") for part in path.parts)
            and path.suffix not in {".pyc", ".pyo"}
        )
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = public_files(root)
    manifest = {
        "schema": "gsrp-public-release-manifest-v2",
        "status": "public",
        "repository": "https://github.com/nc-thu/gsrp-3dgs",
        "release_date": date.today().isoformat(),
        "privacy_boundary": {
            "included": [
                "performance simulator",
                "precision simulator",
                "profiles and tests",
                "aggregate evidence",
                "camera-ready reproduction scripts",
            ],
            "excluded": list(EXCLUDED)
            + ["private models", "dataset copies", "server environments", "complete manuscript"],
        },
        "workload_protocol": {
            "calibration_role": "candidate_generation",
            "workload_role": "profile_selection_validation",
            "camera_count": 573,
            "generalization_claim": False,
            "selection_contract": "simulators/precision/profiles/workload_profile_selection_v1.json",
            "selection_result": "simulators/precision/results/profile_selection_v2/selection/workload_selection.json",
        },
        "files": {
            path.relative_to(root).as_posix(): {"sha256": sha256(path)}
            for path in files
        },
    }
    output = root / "RELEASE_MANIFEST.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(files)} files)")


if __name__ == "__main__":
    main()
