"""Fail-fast readiness audit for Qwen3 Stage 2 without loading 4B weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dataset.qwen_stage2 import QwenAgentSFTDataset, audit_sft_dataset
from dataset.lm_dataset import AgentRLDataset
from trainer.qwen3_adapter import load_stage2_tokenizer, load_yaml_config
from trainer.rollout_engine import RolloutResult
from trainer.train_agent import calculate_rewards, rollout_single


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Qwen3 Stage 2 readiness")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Local tokenizer directory or Hub model id",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--output",
        default="out/run_meta/qwen3_stage2_readiness.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "PyYAML",
        "safetensors",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def audit_manifest() -> dict:
    manifest_path = ROOT / "dataset/manifests/gsm8k_agent.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for name, expected in manifest["files"].items():
        path = ROOT / expected["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        actual_hash = sha256(path)
        if rows != expected["rows"] or actual_hash != expected["sha256"]:
            raise RuntimeError(
                f"manifest mismatch for {name}: rows={rows}, sha256={actual_hash}"
            )
        files[name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": rows,
            "sha256": actual_hash,
        }
    return {"manifest": str(manifest_path.relative_to(ROOT)), "files": files}


class _DeterministicProtocolEngine:
    """Two-turn fake policy used to exercise Qwen serialization, not quality."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.turn = 0
        self.outputs = [
            '<tool_call>\n{"name":"calculate_math","arguments":{"expression":"48/2"}}\n</tool_call>\n'
            '<tool_call>\n{"name":"calculate_math","arguments":{"expression":"48+24"}}\n</tool_call>',
            "Final answer: 72",
        ]

    def rollout(self, prompt_ids, attention_mask, num_generations, **kwargs):
        text = self.outputs[self.turn]
        self.turn += 1
        completion = self.tokenizer(
            text + self.tokenizer.eos_token,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(prompt_ids.device)
        output = torch.cat([prompt_ids, completion], dim=1)
        width = completion.size(1)
        return RolloutResult(
            output_ids=output,
            completion_ids=completion,
            per_token_logps=torch.zeros((1, width), device=prompt_ids.device),
            completions=[text],
            prompt_lens=torch.tensor([prompt_ids.size(1)], device=prompt_ids.device),
            completion_mask=torch.ones((1, width), dtype=torch.long, device=prompt_ids.device),
        )

    def update_policy(self, model):
        return None


def audit_qwen_agent_protocol(tokenizer) -> dict:
    dataset = AgentRLDataset(
        ROOT / "data/processed/gsm8k_agent/train_rl.jsonl",
        tokenizer,
        max_length=1024,
    )
    sample = dataset[0]
    values = rollout_single(
        _DeterministicProtocolEngine(tokenizer),
        tokenizer,
        [dict(message) for message in sample["messages"]],
        sample["tools"],
        max_turns=3,
        max_new_tokens=128,
        thinking_ratio=0.0,
        temperature=0.7,
        top_k=20,
        top_p=0.95,
        device="cpu",
    )
    completion, _, _, response_ids, response_mask, _, turns, unfinished, trace = values
    reward = calculate_rewards(
        [""],
        [completion],
        [sample["gt"]],
        [sample["tools"]],
        1,
        [sample["required_tools"]],
        device="cpu",
        turn_outputs_batch=[turns],
        unfinished_batch=[unfinished],
        require_tool_call_for_success=True,
        reward_mode="strict",
        return_details=True,
    )
    if not bool(reward.task_success[0]):
        raise RuntimeError("Qwen multi-turn calculator protocol audit failed")
    if 0 not in response_mask or 1 not in response_mask:
        raise RuntimeError("Qwen action/observation mask does not contain both states")
    return {
        "task_success": True,
        "strict_reward": reward.rewards[0].item(),
        "turns": len(turns),
        "response_tokens": len(response_ids),
        "action_tokens": sum(response_mask),
        "observation_tokens": len(response_mask) - sum(response_mask),
        "stop_reason": trace["stop_reason"],
    }


def main():
    args = parse_args()
    local_tokenizer = ROOT / "models/qwen3-4b-base-tokenizer"
    tokenizer_source = args.tokenizer or (
        str(local_tokenizer) if local_tokenizer.is_dir() else "Qwen/Qwen3-4B-Base"
    )
    tokenizer = load_stage2_tokenizer(
        tokenizer_source,
        padding_side="right",
        use_im_end_as_eos=True,
    )
    sft_config = load_yaml_config(ROOT / "configs/qwen3_4b/lora_sft_lm_head.yaml")
    dataset = QwenAgentSFTDataset(
        ROOT / sft_config["data"]["train_path"],
        tokenizer,
        max_length=int(sft_config["data"]["max_length"]),
    )
    mask_audit = audit_sft_dataset(dataset, samples=args.samples)
    if mask_audit["zero_supervision_count"]:
        raise RuntimeError(f"assistant-only mask audit failed: {mask_audit}")

    matrix = load_yaml_config(ROOT / "configs/qwen3_4b/eval_matrix.yaml")
    labels = [run["label"] for run in matrix["runs"]]
    expected_labels = ["base", "sft_only", "pure_grpo", "sft_grpo"]
    if labels != expected_labels:
        raise RuntimeError(f"unexpected comparison matrix: {labels}")
    if not matrix["evaluation"].get("require_all_runs"):
        raise RuntimeError("four-arm evaluation must require every run")
    expected_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "lm_head",
    }
    configs = {
        "sft_repair": sft_config,
        "pure_grpo": load_yaml_config(ROOT / "configs/qwen3_4b/pure_grpo.yaml"),
        "sft_grpo": load_yaml_config(ROOT / "configs/qwen3_4b/sft_grpo.yaml"),
        "probe": load_yaml_config(ROOT / "configs/qwen3_4b/pre_grpo_probe.yaml"),
        "evaluation": matrix,
    }
    for name, config in configs.items():
        targets = set(config["lora"].get("target_modules", []))
        if targets != expected_targets:
            raise RuntimeError(f"{name} has unexpected LoRA targets: {sorted(targets)}")
    repaired_path = "out/stage2/qwen3_4b/sft_lm_head_w8_s42_adapter"
    if configs["sft_grpo"]["initialization"]["adapter_path"] != repaired_path:
        raise RuntimeError("SFT->GRPO does not start from the repaired SFT adapter")
    if configs["probe"]["initialization"]["adapter_path"] != repaired_path:
        raise RuntimeError("SFT reachability probe does not use the repaired adapter")
    if matrix["runs"][1]["adapter_path"] != repaired_path:
        raise RuntimeError("SFT-only evaluation does not use the repaired adapter")
    if not sft_config["initialization"].get("expand_target_modules"):
        raise RuntimeError("SFT repair must explicitly expand adapter targets")
    if float(sft_config["training"].get("structure_token_weight", 1.0)) != 8.0:
        raise RuntimeError("formal SFT protocol repair must use validated weight 8")
    if int(sft_config["training"].get("max_steps", -1)) != 0:
        raise RuntimeError("formal SFT protocol repair must traverse a full epoch")
    for name in ("pure_grpo", "sft_grpo"):
        training = configs[name]["training"]
        if training["policy_device"] != training["reference_device"]:
            raise RuntimeError(
                f"{name} does not use the validated single-GPU placement"
            )
        if training.get("kl_gate_action") != "reject_group":
            raise RuntimeError(f"{name} must reject bounded heavy-tail KL groups")
        if int(training.get("max_kl_rejections", 0)) < 1:
            raise RuntimeError(f"{name} has no bounded KL rejection budget")
        if int(training.get("max_consecutive_kl_rejections", 0)) < 1:
            raise RuntimeError(f"{name} has no consecutive KL rejection bound")
        rollout = configs[name]["rollout"]
        sampling = (
            float(rollout["temperature"]),
            int(rollout["top_k"]),
            float(rollout["top_p"]),
        )
        if sampling != (1.0, 0, 1.0):
            raise RuntimeError(
                f"{name} sampling {sampling} does not match full-softmax log-probs"
            )
    for name, section in (
        ("probe", configs["probe"]["probe"]),
        ("evaluation", configs["evaluation"]["evaluation"]),
    ):
        sampling = (
            float(section["temperature"]),
            int(section["top_k"]),
            float(section["top_p"]),
        )
        if sampling != (1.0, 0, 1.0):
            raise RuntimeError(f"{name} does not match formal full-softmax sampling")

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(
                        properties.total_memory / 1024**3, 2
                    ),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )

    report = {
        "status": "PASS",
        "tokenizer_source": tokenizer_source,
        "tokenizer_class": type(tokenizer).__name__,
        "vocabulary_size": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "packages": package_versions(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "data": audit_manifest(),
        "sft_mask": mask_audit,
        "agent_protocol": audit_qwen_agent_protocol(tokenizer),
        "comparison_matrix": labels,
        "repaired_sft_adapter": repaired_path,
        "lora_targets": sorted(expected_targets),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Stage 2 readiness audit: PASS ({output_path})")


if __name__ == "__main__":
    main()
