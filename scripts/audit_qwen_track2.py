"""Fail-fast data, config, tokenizer and protocol audit for Track 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dataset.gsm8k import read_jsonl, sha256_file
from dataset.qwen_stage2 import QwenAgentSFTDataset, audit_sft_dataset
from scripts.audit_qwen_stage2 import audit_qwen_agent_protocol, package_versions
from trainer.agent_chat import audit_agent_generation_prefixes
from trainer.qwen3_adapter import load_stage2_tokenizer, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Qwen Track 2 readiness")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--output", default="out/run_meta/qwen3_track2_readiness.json"
    )
    return parser.parse_args()


def load_ids(path: Path) -> set[str]:
    return {str(row["id"]) for row in read_jsonl(path)}


def verify_manifest() -> tuple[dict, dict[str, Path]]:
    manifest_path = ROOT / "dataset/manifests/gsm8k_track2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {}
    for section in ("source", "files"):
        for name, expected in manifest[section].items():
            path = ROOT / expected["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            digest = sha256_file(path)
            if rows != expected["rows"] or digest != expected["sha256"]:
                raise RuntimeError(
                    f"Track 2 manifest mismatch for {name}: rows={rows}, sha256={digest}"
                )
            paths[name] = path
    return manifest, paths


def verify_set_relations(paths: dict[str, Path]) -> dict:
    official_train = load_ids(paths["official_train"])
    official_test = load_ids(paths["official_test"])
    a = load_ids(paths["a_source"])
    b = load_ids(paths["b_source"])
    test = load_ids(paths["test_source"])
    a_rl, b_rl = load_ids(paths["a_rl"]), load_ids(paths["b_rl"])
    a_sft, b_sft = load_ids(paths["a_sft"]), load_ids(paths["b_sft"])
    a_compare = load_ids(paths["a_compare_rl"])
    b_compare = load_ids(paths["b_compare_rl"])
    a_probe, b_probe = load_ids(paths["a_probe_rl"]), load_ids(paths["b_probe_rl"])
    if a & b or a & test or b & test:
        raise RuntimeError("Track 2 A/B/test ID leakage detected")
    if a | b != official_train:
        raise RuntimeError("Track 2 A/B does not exactly cover official train")
    if test != official_test:
        raise RuntimeError("Track 2 test does not exactly match official test")
    if a_rl != a or b_rl != b:
        raise RuntimeError("Track 2 RL files do not preserve their entire source pools")
    if not a_sft <= a or not b_sft <= b:
        raise RuntimeError("Track 2 SFT row escaped its source pool")
    if not a_probe <= a or not b_probe <= b or a_probe & b_probe:
        raise RuntimeError("Track 2 probe membership is invalid")
    if not a_compare <= a_sft or b_compare != b_sft:
        raise RuntimeError("Track 2 matched comparison sets are not SFT-verifiable")
    if len(a_compare) != len(b_compare):
        raise RuntimeError("Track 2 A/B comparison budgets are not equal")
    if not a_probe <= a_compare or not b_probe <= b_compare:
        raise RuntimeError("Track 2 probes are outside matched comparison sets")
    return {
        "official_train": len(official_train),
        "official_test": len(official_test),
        "a": len(a),
        "b": len(b),
        "a_b_overlap": len(a & b),
        "a_test_overlap": len(a & test),
        "b_test_overlap": len(b & test),
        "a_sft": len(a_sft),
        "b_sft": len(b_sft),
        "a_compare": len(a_compare),
        "b_compare": len(b_compare),
        "a_probe": len(a_probe),
        "b_probe": len(b_probe),
    }


def verify_configs() -> dict:
    directory = ROOT / "configs/qwen3_4b/track2"
    configs = {
        path.stem: load_yaml_config(path)
        for path in sorted(directory.glob("*.yaml"))
    }
    required = {
        "agent_sft_a", "agent_sft_a_lm_head", "additional_sft_b", "grpo_a", "grpo_b",
        "pre_grpo_probe", "eval_test",
    }
    if set(configs) != required:
        raise RuntimeError(f"unexpected Track 2 config set: {sorted(configs)}")
    base_sft = "out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter"
    start = (
        "out/stage2_track2/qwen3_4b/"
        "agent_sft_a_lm_head_w8_s42_adapter"
    )
    for name in ("additional_sft_b", "grpo_a", "grpo_b", "pre_grpo_probe"):
        if configs[name]["initialization"]["adapter_path"] != start:
            raise RuntimeError(f"{name} does not share Agent-SFT(A) initialization")
    repair = configs["agent_sft_a_lm_head"]
    if repair["initialization"]["adapter_path"] != base_sft:
        raise RuntimeError("lm_head repair must start from formal Agent-SFT(A)")
    if not repair["initialization"].get("expand_target_modules"):
        raise RuntimeError("lm_head repair must explicitly expand adapter targets")
    targets = set(repair["lora"].get("target_modules", []))
    expected_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "lm_head",
    }
    if targets != expected_targets:
        raise RuntimeError(f"unexpected lm_head repair targets: {sorted(targets)}")
    for name in ("additional_sft_b", "grpo_a", "grpo_b", "pre_grpo_probe"):
        downstream_targets = set(configs[name]["lora"].get("target_modules", []))
        if downstream_targets != expected_targets:
            raise RuntimeError(
                f"{name} does not preserve repaired adapter targets: "
                f"{sorted(downstream_targets)}"
            )
    a_rollout = configs["grpo_a"]["rollout"]
    b_rollout = configs["grpo_b"]["rollout"]
    if a_rollout != b_rollout:
        raise RuntimeError("GRPO(A) and GRPO(B) rollout settings differ")
    comparable_training_keys = {
        key for key in configs["grpo_a"]["training"]
        if key != "max_candidate_groups"
    }
    for key in comparable_training_keys:
        if configs["grpo_a"]["training"][key] != configs["grpo_b"]["training"][key]:
            raise RuntimeError(f"GRPO A/B training mismatch: {key}")
    if a_rollout["num_generations"] != 8:
        raise RuntimeError("formal Track 2 GRPO must use G=8")
    sampling = (
        float(a_rollout["temperature"]),
        int(a_rollout["top_k"]),
        float(a_rollout["top_p"]),
    )
    if sampling != (1.0, 0, 1.0):
        raise RuntimeError("formal Track 2 GRPO must match full-softmax log-probs")
    for name in ("grpo_a", "grpo_b"):
        training = configs[name]["training"]
        if training["policy_device"] != training["reference_device"]:
            raise RuntimeError(f"{name} does not use validated single-GPU placement")
    if (
        configs["grpo_a"]["training"]["max_candidate_groups"]
        != configs["grpo_b"]["training"]["max_candidate_groups"]
    ):
        raise RuntimeError("formal Track 2 GRPO A/B budgets must match")
    if float(configs["grpo_a"]["training"].get("max_group_kl_k3", 0)) <= 0:
        raise RuntimeError("formal Track 2 GRPO requires a positive KL safety gate")
    for name in ("grpo_a", "grpo_b"):
        training = configs[name]["training"]
        if training.get("kl_gate_action") != "reject_group":
            raise RuntimeError(f"{name} must reject bounded heavy-tail KL groups")
        if int(training.get("max_kl_rejections", 0)) < 1:
            raise RuntimeError(f"{name} has no bounded KL rejection budget")
        if int(training.get("max_consecutive_kl_rejections", 0)) < 1:
            raise RuntimeError(f"{name} has no consecutive KL rejection bound")
    return {
        "files": sorted(configs),
        "shared_initial_adapter": start,
        "group_size": a_rollout["num_generations"],
        "grpo_a_candidate_groups": configs["grpo_a"]["training"]["max_candidate_groups"],
        "grpo_b_candidate_groups": configs["grpo_b"]["training"]["max_candidate_groups"],
    }


def verify_generation_prefixes(paths, tokenizer, samples: int) -> dict:
    report = {}
    for split in ("a", "b"):
        rows = []
        for index, row in enumerate(read_jsonl(paths[f"{split}_sft"])):
            if index >= samples:
                break
            rows.append(audit_agent_generation_prefixes(
                tokenizer, row["conversations"]
            ))
        if not rows:
            raise RuntimeError(f"Track 2 {split} prefix audit has no rows")
        modes = [
            tuple(turn["open_thinking"] for turn in row["turns"])
            for row in rows
        ]
        unexpected = [index for index, mode in enumerate(modes)
                      if mode != (True, False)]
        if unexpected:
            raise RuntimeError(
                f"Track 2 {split} generation-prefix modes differ at {unexpected[:8]}"
            )
        report[split] = {
            "rows_checked": len(rows),
            "assistant_turns_per_row": rows[0]["assistant_turns"],
            "open_thinking_modes": list(modes[0]),
            "mismatch_count": 0,
        }
    return report


def main():
    args = parse_args()
    local_tokenizer = ROOT / "models/qwen3-4b-base-tokenizer"
    source = args.tokenizer or (
        str(local_tokenizer) if local_tokenizer.is_dir() else "Qwen/Qwen3-4B-Base"
    )
    tokenizer = load_stage2_tokenizer(
        source, padding_side="right", use_im_end_as_eos=True
    )
    manifest, paths = verify_manifest()
    relations = verify_set_relations(paths)
    mask_audits = {}
    for split in ("a", "b"):
        dataset = QwenAgentSFTDataset(
            paths[f"{split}_sft"], tokenizer, max_length=1536
        )
        audit = audit_sft_dataset(dataset, samples=args.samples)
        if audit["zero_supervision_count"]:
            raise RuntimeError(f"Track 2 {split} assistant mask failed: {audit}")
        mask_audits[split] = audit
    report = {
        "status": "PASS",
        "protocol": manifest["protocol"],
        "split_seed": manifest["split_seed"],
        "probe_seed": manifest["probe_seed"],
        "tokenizer_source": source,
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token": tokenizer.eos_token,
        "packages": package_versions(),
        "cuda_available": torch.cuda.is_available(),
        "set_relations": relations,
        "manifest": {
            "path": "dataset/manifests/gsm8k_track2.json",
            "files_verified": len(manifest["source"]) + len(manifest["files"]),
        },
        "assistant_mask": mask_audits,
        "generation_prefix": verify_generation_prefixes(
            paths, tokenizer, args.samples
        ),
        "agent_protocol": audit_qwen_agent_protocol(tokenizer),
        "configs": verify_configs(),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Track 2 readiness audit: PASS ({output})")


if __name__ == "__main__":
    main()
