"""Audit exact token-prefix preservation across assistant/tool observations."""

import argparse
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trainer.train_agent import TOOLS, extract_observation_suffix


def main():
    parser = argparse.ArgumentParser(description="Audit multi-turn action/logprob token alignment")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--output", default="out/run_meta/multiturn_serialization_audit.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tools = [item for item in TOOLS if item["function"]["name"] == "calculate_math"]
    base = [
        {"role": "system", "content": "你是工具助手"},
        {"role": "user", "content": "计算2+2"},
    ]
    call = '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "2+2"}}\n</tool_call>'
    cases = []
    for opened, body in [
        (False, call),
        (True, "先调用计算器。\n</think>\n\n" + call),
    ]:
        context = tokenizer.apply_chat_template(
            base, tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=opened,
        )
        prompt_ids = tokenizer(context, add_special_tokens=False).input_ids
        action_ids = tokenizer(body, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        messages = base + [
            {"role": "assistant", "content": body},
            {"role": "tool", "content": json.dumps({"result": "4"}, ensure_ascii=False)},
        ]
        observed = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=opened,
        )
        observed_ids = tokenizer(observed, add_special_tokens=False).input_ids
        expected_prefix = prompt_ids + action_ids
        observation_suffix = extract_observation_suffix(
            tokenizer, prompt_ids, body, observed_ids
        )
        mismatch = next(
            (index for index, (left, right) in enumerate(zip(observed_ids, expected_prefix)) if left != right),
            None,
        )
        cases.append({
            "open_thinking": opened,
            "prefix_exact": observed_ids[:len(expected_prefix)] == expected_prefix,
            "first_mismatch": mismatch,
            "prompt_tokens": len(prompt_ids),
            "action_tokens_including_eos": len(action_ids),
            "observation_and_next_prompt_tokens": len(observed_ids) - len(expected_prefix),
            "extracted_suffix_exact": observation_suffix == observed_ids[len(expected_prefix):],
        })

    # Find a real tokenizer example where decode -> encode changes the sampled
    # action ids.  The production ledger must still preserve those exact ids
    # and append only the environment suffix.
    prompt_text = tokenizer.apply_chat_template(
        base, tokenize=False, add_generation_prompt=True,
        tools=tools, open_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    call_ids = tokenizer(call, add_special_tokens=False).input_ids
    non_roundtrip = {"found": False, "exact_sampled_ids_preserved": None}
    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue
        sampled_action_ids = [token_id] + call_ids + [tokenizer.eos_token_id]
        decoded_action = tokenizer.decode(sampled_action_ids, skip_special_tokens=True)
        canonical_action_ids = tokenizer(
            decoded_action, add_special_tokens=False
        ).input_ids + [tokenizer.eos_token_id]
        if canonical_action_ids == sampled_action_ids:
            continue
        observed_text = tokenizer.apply_chat_template(
            base + [
                {"role": "assistant", "content": decoded_action},
                {"role": "tool", "content": json.dumps({"result": "4"}, ensure_ascii=False)},
            ],
            tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=False,
        )
        observed_ids = tokenizer(observed_text, add_special_tokens=False).input_ids
        suffix = extract_observation_suffix(
            tokenizer, prompt_ids, decoded_action, observed_ids
        )
        exact_ledger = prompt_ids + sampled_action_ids + suffix
        non_roundtrip = {
            "found": True,
            "token_id": token_id,
            "sampled_action_tokens": len(sampled_action_ids),
            "canonical_action_tokens": len(canonical_action_ids),
            "exact_sampled_ids_preserved": (
                exact_ledger[len(prompt_ids):len(prompt_ids) + len(sampled_action_ids)]
                == sampled_action_ids
            ),
            "environment_suffix_tokens": len(suffix),
        }
        break

    all_prefixes_exact = all(
        case["prefix_exact"] and case["extracted_suffix_exact"] for case in cases
    )
    report = {
        "cases": cases,
        "all_prefixes_exact": all_prefixes_exact,
        "non_roundtrip_regression": non_roundtrip,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (
        not report["all_prefixes_exact"]
        or not non_roundtrip["found"]
        or not non_roundtrip["exact_sampled_ids_preserved"]
    ):
        raise SystemExit("multi-turn serialization audit failed")


if __name__ == "__main__":
    main()
