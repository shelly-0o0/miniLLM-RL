"""Promote an audited SFT calibration adapter without losing provenance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.gsm8k import sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description="Promote an audited Qwen SFT adapter")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--decision-rationale", required=True)
    return parser.parse_args()


def resolve_inside_root(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"path must stay inside repository: {path}") from error
    return path


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    source = resolve_inside_root(args.source)
    destination = resolve_inside_root(args.destination)
    audit_path = resolve_inside_root(args.audit)
    probe_dir = resolve_inside_root(args.probe_dir)
    if source == destination:
        raise ValueError("source and destination must differ")
    if destination.exists():
        raise FileExistsError(f"promotion destination already exists: {destination}")
    source_model = source / "adapter_model.safetensors"
    source_config = source / "adapter_config.json"
    if not source_model.is_file() or not source_config.is_file():
        raise FileNotFoundError(f"source adapter is incomplete: {source}")
    source_hash = sha256_file(source_model)

    audit = read_json(audit_path)
    audited_after = audit.get("after") or {}
    if audited_after.get("adapter_model_sha256") != source_hash:
        raise RuntimeError("structure audit fingerprint does not match source adapter")

    probe_summary_path = probe_dir / "summary.json"
    probe_runtime_path = probe_dir / "runtime_overrides.json"
    probe_summary = read_json(probe_summary_path)
    probe_runtime = read_json(probe_runtime_path)
    resolved_adapter = probe_runtime.get("resolved_adapter_path")
    if resolved_adapter is None:
        raise RuntimeError("probe did not use an adapter")
    if resolve_inside_root(resolved_adapter) != source:
        raise RuntimeError("probe adapter path does not match source adapter")
    if not probe_summary:
        raise RuntimeError("probe contains no pool summary")

    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale promotion temporary directory: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    provenance = {
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_adapter": str(source.relative_to(ROOT)),
        "destination_adapter": str(destination.relative_to(ROOT)),
        "adapter_model_sha256": source_hash,
        "structure_audit": str(audit_path.relative_to(ROOT)),
        "structure_audit_sha256": sha256_file(audit_path),
        "probe_summary": str(probe_summary_path.relative_to(ROOT)),
        "probe_summary_sha256": sha256_file(probe_summary_path),
        "probe_runtime_overrides": str(probe_runtime_path.relative_to(ROOT)),
        "probe_results": probe_summary,
        "decision_rationale": args.decision_rationale,
    }
    (temporary / "promotion.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if sha256_file(temporary / "adapter_model.safetensors") != source_hash:
        raise RuntimeError("copied adapter fingerprint changed during promotion")
    os.replace(temporary, destination)
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(f"promoted adapter to {destination}")


if __name__ == "__main__":
    main()
