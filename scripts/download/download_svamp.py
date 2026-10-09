"""Download pinned SVAMP files required by the warm-start GRPO study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


REVISION = "78e727689e1c1bebfc4be39c446898e8e10b0518"
BASE_URL = f"https://raw.githubusercontent.com/arkilpatel/SVAMP/{REVISION}"
FILES = {
    "SVAMP.json": "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e",
    "LICENSE": "07ce3a34de7a865345634f9278bc26ddcacc2d0defd348c28e3f4603448b9ba4",
    "fold0_dev.csv": "37a0f26da9d1cc10619f586bc086bb113e894686459dbc1203a798464ff37a7d",
    "fold1_dev.csv": "0fd6b2eddf657a451bdd578b338b341ab5ab3473661b7be545a8f4760668a6b2",
    "fold2_dev.csv": "ea93418576a94ff79f0818d562a8a07866d844b981b3f2769fa62c648e13b12b",
    "fold3_dev.csv": "3618faa3f85db367a2dbbff7c53689d1119e8414bc80b682c15ef101f6d5b722",
    "fold4_dev.csv": "850d7f225870959fbaaed227dc4ef46cda6a0ac1d5cc41e9f1a6257fa7ffe6b7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_url(name: str) -> str:
    if name.startswith("fold"):
        fold = name.removeprefix("fold").removesuffix("_dev.csv")
        return f"{BASE_URL}/data/cv_svamp_augmented/fold{fold}/dev.csv"
    return f"{BASE_URL}/{name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw/svamp")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"revision": REVISION, "base_url": BASE_URL, "files": {}}
    for name, expected_hash in FILES.items():
        destination = output_dir / name
        if not destination.is_file() or sha256(destination) != expected_hash:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            urllib.request.urlretrieve(source_url(name), temporary)
            actual_hash = sha256(temporary)
            if actual_hash != expected_hash:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SVAMP checksum mismatch for {name}: "
                    f"expected={expected_hash}, actual={actual_hash}"
                )
            os.replace(temporary, destination)
        report["files"][name] = {
            "url": source_url(name),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
