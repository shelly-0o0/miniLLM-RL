"""Collect MiniMind experiment artifacts into one machine-readable index.

The collector never invents defaults for missing experiments: an absent file
is listed under ``missing`` and cannot silently become a zero-valued result.
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([0-9.eE+\-]+).*?"
    r"(?:lr|learning_rate): ([0-9.eE+\-]+)"
)
STATUS_PATTERN = re.compile(r"([A-Za-z0-9_]+)_exit_code=(\d+), elapsed_seconds=(\d+)")


def finite_mean(values):
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else None


def summarize_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    points = [
        {
            "epoch": int(epoch), "epochs": int(epochs),
            "step": int(step), "steps": int(steps),
            "loss": float(loss), "learning_rate": float(lr),
        }
        for epoch, epochs, step, steps, loss, lr in EPOCH_PATTERN.findall(text)
    ]
    statuses = [
        {"name": name, "exit_code": int(code), "elapsed_seconds": int(elapsed)}
        for name, code, elapsed in STATUS_PATTERN.findall(text)
    ]
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "logged_points": len(points),
        "statuses": statuses,
    }
    test_run = re.search(r"Ran (\d+) tests? in [^\n]+\n\n(OK|FAILED[^\n]*)", text)
    if test_run:
        result["unit_tests"] = {
            "count": int(test_run.group(1)),
            "result": test_run.group(2),
        }
    if points:
        losses = [point["loss"] for point in points]
        result.update({
            "first": points[0],
            "last": points[-1],
            "minimum_logged_loss": min(losses),
            "first_100_logged_loss_mean": finite_mean(losses[:100]),
            "last_100_logged_loss_mean": finite_mean(losses[-100:]),
        })
    return result


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description="Index real MiniMind experiment evidence")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="out/run_meta/experiment_evidence.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = root / "out"

    log_names = [
        "pretrain_dense_mini.log", "sft_dense_mini.log",
        "lora_medical_dense.log", "lora_medical_holdout.log",
        "pretrain_moe_mini.log", "pretrain_student_512.log",
        "full_sft_student_512.log", "full_ce_student_512.log",
        "full_dist_student_512.log", "agent_sft_coldstart.log",
        "dpo_holdout_dense.log", "final_unit_tests.log",
    ]
    json_names = [
        "run_meta/sft_mask_audit_128.json",
        "run_meta/lora_medical_split_manifest.json",
        "run_meta/agent_sft_mask_audit_128.json",
        "run_meta/agent_tool_verified_manifest.json",
        "run_meta/agent_tool_challenge_manifest.json",
        "run_meta/agent_sft_coldstart_manifest.json",
        "run_meta/tool_rlvr_execution_audit.json",
        "run_meta/tool_rlvr_challenge_execution_audit.json",
        "run_meta/multiturn_serialization_audit.json",
        "run_meta/moe_routing_audit.json",
        "eval/lora_medical_holdout_metrics_fixed.json",
        "eval/kd_comparison.json",
        "eval/dpo_holdout_metrics.json",
        "flash_sdpa_verification.json",
        "model_flash_sdpa_verification.json",
    ]
    evidence = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository_root": str(root),
        "logs": {}, "json_artifacts": {}, "csv_artifacts": {},
        "pipeline_markers": {}, "checksum_manifests": {}, "missing": [],
    }

    base_commit = out / "run_meta/base_commit.txt"
    if base_commit.exists():
        evidence["base_commit"] = base_commit.read_text(encoding="utf-8").strip()
    else:
        evidence["missing"].append(str(base_commit.relative_to(root)))

    for name in log_names:
        path = out / "logs" / name
        if path.exists():
            evidence["logs"][name] = summarize_log(path)
        else:
            evidence["missing"].append(str(path.relative_to(root)))

    for name in json_names:
        path = out / name
        if path.exists():
            evidence["json_artifacts"][name] = load_json(path)
        else:
            evidence["missing"].append(str(path.relative_to(root)))

    csv_paths = set(out.glob("*.csv"))
    csv_paths.update(out.glob("eval_rlvr/**/*.csv"))
    csv_paths.update(out.glob("metrics/*.csv"))
    for path in sorted(csv_paths):
        evidence["csv_artifacts"][str(path.relative_to(out))] = load_csv(path)

    markers = {
        "kd": ("logs/kd_pipeline.log", "KD pipeline complete"),
        "rlvr_readiness": ("logs/rlvr_readiness.log", "RLVR readiness complete"),
        "dpo": ("logs/dpo_holdout_pipeline.log", "DPO holdout pipeline complete"),
        "rlvr_pilot": ("logs/rlvr_pilot.log", "RLVR pilot complete"),
        "architecture": ("logs/architecture_verification.log", "architecture verification complete"),
        "rlvr_formal": ("logs/rlvr_formal_pipeline.log", "RLVR formal pipeline complete"),
        "final_validation": ("logs/finalize_pipeline.log", "final validation complete"),
    }
    for stage, (relative, marker) in markers.items():
        path = out / relative
        evidence["pipeline_markers"][stage] = bool(
            path.exists() and marker in path.read_text(encoding="utf-8", errors="replace")
        )

    for path in sorted((out / "run_meta").glob("*.sha256")):
        evidence["checksum_manifests"][path.name] = [
            line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]

    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "logs": len(evidence["logs"]),
        "json_artifacts": len(evidence["json_artifacts"]),
        "csv_artifacts": len(evidence["csv_artifacts"]),
        "pipeline_markers": evidence["pipeline_markers"],
        "missing": evidence["missing"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
