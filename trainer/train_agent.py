import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import re
import gc
import json
import math
import random
import signal
import time
import argparse
import warnings
import unicodedata
from dataclasses import dataclass
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import (
    create_rollout_engine, compute_per_token_logps, masked_log_softmax,
    legal_action_blocked_ids,
)
from trainer.policy_optimization import (
    compute_policy_loss,
    distributed_token_mean_scale,
    effective_group_mask,
    group_relative_advantages,
    positive_kl_estimate,
    soft_overlong_penalty,
)
from trainer.experiment_logging import JsonlMetricLogger
from trainer.math_env import safe_calculate as shared_safe_calculate
from trainer.agent_chat import render_agent_chat, resolve_agent_open_thinking

warnings.filterwarnings('ignore')


@dataclass
class VerifiableRewardOutput:
    rewards: torch.Tensor
    task_success: torch.Tensor
    answer_accuracy: torch.Tensor
    format_valid: torch.Tensor
    tool_call_valid: torch.Tensor
    tool_execution_success: torch.Tensor
    required_tool_coverage: torch.Tensor
    tool_evidence_coverage: torch.Tensor
    protocol_progress: torch.Tensor


_CHECKPOINT_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def checkpoint_state_dict(model, dtype_name="float32"):
    """Materialize a portable CPU checkpoint without silently erasing RL updates.

    Small-policy RL commonly uses learning rates around 1e-7--1e-6.  Casting
    every checkpoint to FP16 can round those updates back to the initialization,
    making a trained checkpoint behaviorally identical to Agent-SFT.  RL
    checkpoints therefore default to FP32; lower precision remains an explicit
    storage/serving choice rather than an implicit training decision.
    """

    if dtype_name not in _CHECKPOINT_DTYPES:
        raise ValueError(f"unsupported checkpoint dtype: {dtype_name}")
    dtype = _CHECKPOINT_DTYPES[dtype_name]
    return {
        key: value.detach().to(
            device="cpu", dtype=dtype if value.is_floating_point() else value.dtype
        ).contiguous()
        for key, value in model.state_dict().items()
    }


def save_policy_checkpoint(model, path, dtype_name="float32"):
    """Atomically save one immutable policy snapshot for curve evaluation."""

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    state_dict = checkpoint_state_dict(raw_model, dtype_name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, path)
    del state_dict
    return path

# ================================ 工具与 Reward = Start ================================

def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义 ========
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据 ========
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

def safe_calculate_math(args):
    """Backward-compatible entry point backed by the shared math environment."""

    return shared_safe_calculate(args)


# ======== 模拟执行 ========
MOCK_RESULTS = {
    "calculate_math": safe_calculate_math,
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            value = json.loads(m.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            calls.append(value)
    return calls


def parse_json_tool_candidates(text):
    """Extract JSON tool-shaped objects even when protocol tags are missing.

    These candidates are never executed as environment actions and never count
    toward strict success.  They only expose a dense curriculum signal for a
    base policy that already emits the advertised JSON schema but has not yet
    learned the ``<tool_call>`` wrapper.
    """

    decoder = json.JSONDecoder()
    candidates = []
    cursor = 0
    raw = str(text)
    while cursor < len(raw):
        start = raw.find("{", cursor)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + max(consumed, 1)
        if (
            isinstance(value, dict)
            and isinstance(value.get("name"), str)
            and "arguments" in value
        ):
            candidates.append(value)
    return candidates


def protocol_progress_score(
    turn_answers, valid_names, required_names, ground_truth
):
    """Score bounded, non-authoritative progress toward an executable call.

    The maximum over candidates prevents repetition from increasing reward.
    Strict metrics still require tagged calls that were actually passed through
    the environment; this score cannot make a trajectory successful.
    """

    text = "\n".join(map(str, turn_answers))
    marker_score = 0.125 * int("<tool_call>" in text)
    marker_score += 0.125 * int("</tool_call>" in text)
    best = 0.0
    for candidate in parse_json_tool_candidates(text):
        name = candidate.get("name")
        raw_args = candidate.get("arguments")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = None
        score = 0.25  # JSON has the required top-level keys and name type.
        recognized = name in valid_names
        score += 0.25 * int(recognized)
        checker = CHECK_ARGS.get(name) if recognized else None
        arguments_valid = bool(
            isinstance(raw_args, dict) and checker and checker(raw_args)
        )
        score += 0.25 * int(arguments_valid)
        score += 0.25 * int(name in required_names)
        result = execute_tool(name, raw_args) if arguments_valid else None
        score += 0.25 * int(result is not None)
        if result is not None and ground_truth:
            evidence = validate_gt_in_text(
                json.dumps(result, ensure_ascii=False),
                ground_truth,
                final_answer_only=False,
            )
            score += 0.5 * (len(evidence) / len(ground_truth))
        best = max(best, score)
    # Give a very small gradient to an incomplete JSON skeleton, but keep it
    # below every parseable candidate and independent of repetition count.
    if not best:
        skeleton = 0.1 * int(bool(re.search(r'["\']name["\']\s*:', text)))
        skeleton += 0.1 * int(
            bool(re.search(r'["\']arguments["\']\s*:', text))
        )
        skeleton += 0.1 * int(any(name in text for name in required_names))
        best = min(skeleton, 0.3)
    return marker_score + best

def execute_tool(name, args):
    # Tool calls are sampled model output, so neither field can be trusted to
    # have the schema advertised in the prompt.  In particular, JSON such as
    # {"name": {"tool": "calculate_math"}, ...} must be scored as an invalid
    # call instead of being used as an unhashable dictionary key.
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except:
        return None
    finally:
        try: signal.alarm(0)
        except: pass

# ======== 多轮 Rollout ========
def extract_observation_suffix(tokenizer, observed_context, observed_ids, previous_observation_count=0):
    """Extract environment tokens without re-encoding sampled policy text.

    The decoded assistant action is not guaranteed to round-trip through a
    BPE tokenizer (especially around special-token boundaries), so locating
    the observation by ``tokenize(decoded_action)`` can corrupt the response
    ledger.  Instead, locate the newly-added ``<tool_response>`` in the
    rendered environment context and use fast-tokenizer character offsets to
    find the corresponding token boundary.
    """

    marker = "<tool_response>"
    marker_positions = [
        match.start() for match in re.finditer(re.escape(marker), observed_context)
    ]
    if previous_observation_count >= len(marker_positions):
        raise RuntimeError(
            "The chat template did not render the newly-added tool observation."
        )
    observation_start = marker_positions[previous_observation_count]
    encoded = tokenizer(
        observed_context,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError(
            "The tokenizer must provide offset mappings to isolate tool observations."
        )
    if (
        offsets
        and isinstance(offsets[0], (list, tuple))
        and offsets[0]
        and isinstance(offsets[0][0], (list, tuple))
    ):
        offsets = offsets[0]
    suffix_index = next(
        (index for index, (start, end) in enumerate(offsets) if end > observation_start),
        None,
    )
    if suffix_index is None or suffix_index > len(observed_ids):
        raise RuntimeError("Could not map the rendered tool observation to token ids.")
    return observed_ids[suffix_index:]


def rollout_single(
    rollout_engine,
    tokenizer,
    messages,
    tools,
    max_turns=3,
    max_new_tokens=256,
    thinking_ratio=0.5,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    device="cuda",
    requested_open_thinking=None,
    first_turn_sample=None,
):
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    stop_reason = "max_turns"
    trace = {"turns": [], "stop_reason": stop_reason}
    if requested_open_thinking is None:
        requested_open_thinking = random.random() < thinking_ratio
    else:
        requested_open_thinking = bool(requested_open_thinking)
    actual_context_ids = None
    for turn in range(max_turns):
        open_thinking = resolve_agent_open_thinking(
            tokenizer,
            messages,
            requested_open_thinking=requested_open_thinking,
        )
        context = render_agent_chat(
            tokenizer, messages, tools=tools, tokenize=False,
            add_generation_prompt=True, open_thinking=open_thinking,
        )
        canonical_context_ids = tokenizer(
            context, add_special_tokens=False
        ).input_ids
        if prompt_ids is None:
            prompt_ids = list(canonical_context_ids)
            actual_context_ids = list(prompt_ids)
        elif actual_context_ids != prompt_ids + response_ids:
            raise RuntimeError(
                "Internal multi-turn token ledger is inconsistent with the exact behavior-policy stream."
            )
        inputs = {
            "input_ids": torch.tensor([actual_context_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(actual_context_ids)), dtype=torch.long, device=device),
        }
        if turn == 0 and first_turn_sample is not None:
            new_ids, new_logps, completion_mask = first_turn_sample
            new_ids = list(new_ids)
            new_logps = list(new_logps)
            completion_mask = list(completion_mask)
        else:
            rollout_result = rollout_engine.rollout(
                prompt_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_generations=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            new_ids = rollout_result.completion_ids[0].tolist()
            new_logps = rollout_result.per_token_logps[0].tolist()
            completion_mask = rollout_result.completion_mask[0].tolist()
        if len(new_ids) != len(new_logps): Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        # EOS is a policy action (the stopping decision) and must remain in the
        # action/log-probability stream.  Only right-padding is removed.
        pairs = [(t, lp) for t, lp, keep in zip(new_ids, new_logps, completion_mask) if keep]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        if tokenizer.pad_token_id in new_ids:
            raise RuntimeError(
                "The rollout backend sampled PAD as a policy action. Configure it to suppress PAD; "
                "silently deleting the token would change the behavior-policy trajectory."
            )
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text
        calls = parse_tool_calls(new_text)
        turn_trace = {
            "turn": turn,
            "open_thinking": open_thinking,
            "action": {"text": new_text, "token_ids": new_ids, "tool_calls": calls},
            "observation": [],
            "stop_reason": "final_answer" if not calls else "tool_call",
        }
        if not calls:
            stop_reason = "final_answer"
            trace["turns"].append(turn_trace)
            break
        # A tool turn needs an explicit policy-sampled EOS boundary.  If the
        # response merely hit max_new_tokens, do not synthesize an action that
        # the behavior policy never sampled.
        if not new_ids or new_ids[-1] != tokenizer.eos_token_id:
            unfinished = True
            stop_reason = "max_new_tokens"
            turn_trace["stop_reason"] = stop_reason
            trace["turns"].append(turn_trace)
            break
        unfinished = turn == max_turns - 1
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            turn_trace["observation"].append({
                "tool": name,
                "arguments": raw,
                "result": result_str,
                "executed": result is not None,
            })
            messages.append({"role": "tool", "content": result_str})

        previous_observation_count = context.count("<tool_response>")
        next_open_thinking = resolve_agent_open_thinking(
            tokenizer,
            messages,
            requested_open_thinking=requested_open_thinking,
        )
        observe_context = render_agent_chat(
            tokenizer, messages, tools=tools, tokenize=False,
            add_generation_prompt=not unfinished, open_thinking=next_open_thinking,
        )
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        obs_delta = extract_observation_suffix(
            tokenizer, observe_context, observe_ids, previous_observation_count
        )
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        actual_context_ids = prompt_ids + response_ids
        final_context = observe_context
        if unfinished:
            stop_reason = "max_turns"
            turn_trace["stop_reason"] = stop_reason
        trace["turns"].append(turn_trace)

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    if not trace["turns"] and not all_outputs:
        stop_reason = "empty_generation"
    trace["stop_reason"] = stop_reason
    return (final_output, final_context, prompt_ids, response_ids, response_mask,
            response_old_logps, list(all_outputs), unfinished, trace)

def rollout_batch(
    rollout_engine,
    tokenizer,
    messages_batch,
    tools_batch,
    num_gen,
    max_turns=3,
    max_new_tokens=256,
    thinking_ratio=0.5,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    device="cuda",
):
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    all_traces = []
    for messages, tools in zip(messages_batch, tools_batch):
        # Every trajectory in a GRPO group starts from the same prompt.  Generate
        # that first assistant turn as one GPU batch; only trajectories that
        # actually call a tool need independent continuation turns.  This keeps
        # the exact sampled-token/action-mask ledger while avoiding G serial
        # prefill/decode passes for the common first turn.
        batched_first_turn = num_gen > 1 and thinking_ratio in (0.0, 1.0)
        first_turn_samples = [None] * num_gen
        requested_open_thinking = None
        if batched_first_turn:
            requested_open_thinking = bool(thinking_ratio)
            open_thinking = resolve_agent_open_thinking(
                tokenizer,
                messages,
                requested_open_thinking=requested_open_thinking,
            )
            first_context = render_agent_chat(
                tokenizer,
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=open_thinking,
            )
            first_ids = tokenizer(
                first_context, add_special_tokens=False
            ).input_ids
            first_input = torch.tensor(
                [first_ids], dtype=torch.long, device=device
            )
            first_result = rollout_engine.rollout(
                prompt_ids=first_input,
                attention_mask=torch.ones_like(first_input),
                num_generations=num_gen,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if first_result.completion_ids.size(0) != num_gen:
                raise RuntimeError(
                    "batched rollout returned a different number of generations"
                )
            first_turn_samples = [
                (
                    first_result.completion_ids[index].tolist(),
                    first_result.per_token_logps[index].tolist(),
                    first_result.completion_mask[index].tolist(),
                )
                for index in range(num_gen)
            ]

        for generation_index in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished, trace = rollout_single(
                rollout_engine, tokenizer, msgs_copy, tools, max_turns,
                max_new_tokens, thinking_ratio, temperature, top_k, top_p, device,
                requested_open_thinking=requested_open_thinking,
                first_turn_sample=first_turn_samples[generation_index],
            )
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
            all_traces.append(trace)
    return (all_completions, all_contexts, all_prompt_ids, all_response_ids,
            all_response_masks, all_response_old_logps, all_turn_outputs,
            all_unfinished, all_traces)


def shift_action_mask(full_response_masks):
    """Align an exact rollout action mask with next-token log-probabilities.

    ``rollout_single`` records 1 for every token sampled by the policy and 0
    for prompt/tool-observation tokens.  EOS closes one assistant *turn*, not
    an entire multi-turn trajectory, so later action tokens must not be
    discarded after the first EOS.  The only transformation needed for
    next-token training is the usual one-position shift.
    """

    if full_response_masks.ndim != 2:
        raise ValueError("full_response_masks must have shape [batch, sequence]")
    return full_response_masks[:, 1:]

# ======== Reward 计算 ========
_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _final_answer_region(text):
    """Prefer an explicit/final answer region over intermediate reasoning."""

    raw = unicodedata.normalize("NFKC", str(text))
    answer_tags = re.findall(r"<answer>(.*?)</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    if answer_tags:
        return answer_tags[-1]
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", raw)
    if boxed:
        return boxed[-1]
    marker = re.search(
        r"(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*(.+)$",
        raw, flags=re.IGNORECASE | re.DOTALL,
    )
    if marker:
        return marker.group(1)
    nonempty_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return nonempty_lines[-1] if len(nonempty_lines) > 1 else raw


def validate_gt_in_text(text, gt_list, final_answer_only=True):
    """Return ground-truth items verified in a final answer.

    Numeric answers are compared against complete numeric tokens with a small
    tolerance, so ground truth ``4`` cannot be hacked by emitting ``14``.
    Text answers use Unicode normalization; ASCII words additionally require
    word boundaries.  Task-specific production verifiers should replace this
    generic helper for algebraic equivalence, code tests or structured APIs.
    """

    normalized_text = (
        _final_answer_region(text) if final_answer_only else unicodedata.normalize("NFKC", str(text))
    ).casefold()
    numeric_text = normalized_text.replace(",", "")
    parsed_numbers = [float(value) for value in re.findall(_NUMBER_PATTERN, numeric_text)]
    numeric_ground_truth_count = sum(
        bool(re.fullmatch(_NUMBER_PATTERN, unicodedata.normalize("NFKC", str(value)).strip().replace(",", "")))
        for value in gt_list
    )
    verified = set()
    for ground_truth in gt_list:
        candidate = unicodedata.normalize("NFKC", str(ground_truth)).strip().casefold()
        if not candidate:
            continue
        numeric_candidate = candidate.replace(",", "")
        if re.fullmatch(_NUMBER_PATTERN, numeric_candidate):
            target = float(numeric_candidate)
            candidates = parsed_numbers[-1:] if numeric_ground_truth_count == 1 else parsed_numbers
            if any(math.isclose(target, value, rel_tol=1e-6, abs_tol=1e-6) for value in candidates):
                verified.add(ground_truth)
        elif re.search(r"[a-z0-9_]", candidate):
            # Use ASCII token boundaries here.  Python's ``\w`` also treats
            # CJK characters as word characters, which incorrectly rejects
            # natural answers such as ``温度20°C`` or ``时间2025-...``.
            # Numeric-only values are handled by the strict numeric branch
            # above, so this still rejects embedded ASCII substrings such as
            # ``cat`` in ``scatter``.
            if re.search(rf"(?<![a-z0-9_]){re.escape(candidate)}(?![a-z0-9_])", normalized_text):
                verified.add(ground_truth)
        elif candidate in normalized_text:
            verified.add(ground_truth)
    return verified

def calculate_rewards(
    prompts,
    completions,
    gt_batch,
    tools_batch,
    num_gen,
    required_tools_batch=None,
    reward_model=None,
    device="cuda",
    turn_outputs_batch=None,
    unfinished_batch=None,
    require_tool_call_for_success=False,
    reward_mode="shaped",
    return_details=False,
):
    """Calculate dense shaping rewards and a separate verifiable task outcome.

    DAPO dynamic sampling must use ``task_success`` rather than the dense reward:
    a formatting bonus must never turn an incorrect answer into a "success".
    Tool execution is replayed here because MiniMind's mock tools are
    deterministic; production environments should instead return signed traces
    from the sandbox that executed the original rollout.
    """

    size = len(completions)
    rewards = torch.zeros(size, device=device)
    task_success = torch.zeros(size, device=device, dtype=torch.bool)
    answer_accuracy = torch.zeros(size, device=device)
    format_valid = torch.zeros(size, device=device)
    tool_call_valid = torch.zeros(size, device=device)
    tool_execution_success = torch.zeros(size, device=device)
    required_tool_coverage = torch.zeros(size, device=device)
    tool_evidence_coverage = torch.zeros(size, device=device)
    protocol_progress = torch.zeros(size, device=device)

    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        gt = gt_batch[sample_idx]
        tools = tools_batch[sample_idx]
        required_names = set(required_tools_batch[sample_idx]) if required_tools_batch is not None else set()
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        calls_required = bool(require_tool_call_for_success and tools)
        open_tags = sum(turn.count('<tool_call>') for turn in turn_answers)
        close_tags = sum(turn.count('</tool_call>') for turn in turn_answers)
        tool_calls = []
        for turn_answer in turn_answers:
            tool_calls.extend(parse_tool_calls(turn_answer))
        tags_balanced = open_tags == close_tags == len(tool_calls)
        tags_valid = bool(tags_balanced and (tool_calls or not calls_required))
        format_valid[idx] = float(tags_valid)
        # A merely balanced count is insufficient: nested/empty/repeated tags
        # can have open_tags == close_tags while only a subset parses as JSON.
        # Penalize every discrepancy against the number of executable call
        # candidates, with a non-zero floor for a required but absent call.
        format_error_count = (
            abs(open_tags - close_tags)
            + abs(open_tags - len(tool_calls))
            + abs(close_tags - len(tool_calls))
        )
        reward += (
            0.25
            if tags_valid
            else -0.5 * max(1, format_error_count)
        )

        valid_call_count = 0
        executed_call_count = 0
        executed_valid_names = set()
        executed_results = []
        for tool_call in tool_calls:
            name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            check = CHECK_ARGS.get(name) if isinstance(name, str) else None
            is_valid = bool(
                isinstance(name, str)
                and isinstance(raw, dict)
                and name in valid_names
                and check
                and check(raw)
            )
            valid_call_count += int(is_valid)
            execution_result = execute_tool(name, raw) if is_valid else None
            executed = execution_result is not None
            executed_call_count += int(executed)
            if executed:
                executed_valid_names.add(name)
                executed_results.append(execution_result)
        if tool_calls:
            tool_call_valid[idx] = valid_call_count / len(tool_calls)
            tool_execution_success[idx] = executed_call_count / len(tool_calls)
        required_coverage = (
            len(required_names & executed_valid_names) / len(required_names)
            if required_names else float(bool(executed_valid_names))
        )
        required_tool_coverage[idx] = required_coverage
        verified_by_tools = validate_gt_in_text(
            "\n".join(json.dumps(result, ensure_ascii=False) for result in executed_results),
            gt,
            final_answer_only=False,
        ) if gt and executed_results else set()
        evidence_coverage = len(verified_by_tools) / len(gt) if gt else 0.0
        tool_evidence_coverage[idx] = evidence_coverage

        progress_score = protocol_progress_score(
            turn_answers, valid_names, required_names, gt
        )
        protocol_progress[idx] = progress_score
        if reward_mode == "shaped" and not tool_calls:
            reward += progress_score

        # The final answer is the text after the last tool-call tag.  It is
        # ignored when max_turns was exhausted because the trajectory is open.
        final_text = "" if unfinished else (answer.rsplit('</tool_call>', 1)[-1] if '</tool_call>' in answer else answer)
        verified = validate_gt_in_text(final_text, gt) if gt else set()
        verified_fraction = len(verified) / len(gt) if gt else 0.0
        is_answer_correct = bool(gt) and len(verified) == len(gt) and not unfinished
        answer_accuracy[idx] = float(is_answer_correct)

        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                think, answer_after_think = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
                reward += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_after_think.strip()
            if gt:
                reward += 2.5 * verified_fraction
            if reward_model is not None:
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                reward += reward_model.get_score(messages, answer)
        else:
            invalid_calls = len(tool_calls) - valid_call_count
            reward += 0.5 * (valid_call_count / len(tool_calls))
            reward += 0.5 * (executed_call_count / len(tool_calls))
            reward -= 0.5 * invalid_calls
            if gt:
                reward += 2.5 * verified_fraction
            if unfinished:
                reward -= 0.5

        # Curriculum rewards must distinguish a complete, grounded tool trace
        # from reward-hacking shortcuts such as calling one easy tool, calling
        # every available tool, or guessing the answer without executable
        # evidence.  Strict RLVR remains binary below; these dense terms are
        # used only when ``reward_mode=shaped``.
        if required_names and reward_mode == "shaped":
            reward += 1.0 * (2.0 * required_coverage - 1.0)
            reward += 1.0 * (2.0 * evidence_coverage - 1.0)
            reward -= 0.25 * len(executed_valid_names - required_names)

        reward -= rep_penalty(final_text if final_text else answer)
        calls_satisfied = (
            required_names.issubset(executed_valid_names)
            if required_names else (not calls_required or bool(tool_calls))
        )
        # A syntactically valid tool call plus a guessed final answer is not a
        # verifiable tool trajectory.  Every ground-truth item must also be
        # supported by the replayed deterministic execution results.
        evidence_satisfied = (
            not calls_required or not gt or len(verified_by_tools) == len(gt)
        )
        calls_correct = (not tool_calls or (valid_call_count == len(tool_calls)
                                             and executed_call_count == len(tool_calls)))
        task_success[idx] = bool(
            is_answer_correct and tags_valid and calls_satisfied
            and calls_correct and evidence_satisfied
        )
        if reward_mode == "strict":
            rewards[idx] = 1.0 if task_success[idx] else -1.0
        elif reward_mode == "shaped":
            # Keep a strictly valid, grounded trajectory above near-miss
            # trajectories. A +/-3 clamp previously collapsed both into the
            # same ceiling and removed the curriculum gradient for repairing
            # malformed duplicate tags.
            reward += 2.0 * float(task_success[idx])
            rewards[idx] = max(min(reward, 6.0), -6.0)
        else:
            raise ValueError(f"unsupported reward_mode={reward_mode!r}")

    output = VerifiableRewardOutput(
        rewards=rewards,
        task_success=task_success,
        answer_accuracy=answer_accuracy,
        format_valid=format_valid,
        tool_call_valid=tool_call_valid,
        tool_execution_success=tool_execution_success,
        required_tool_coverage=required_tool_coverage,
        tool_evidence_coverage=tool_evidence_coverage,
        protocol_progress=protocol_progress,
    )
    return output if return_details else rewards

# ================================ 工具与 Reward = End ================================


def _rollout_group_records(messages_batch, tools_batch, required_tools_batch, gt_batch, prompts, rollout_values, reward_output, group_size):
    """Split a rollout batch into indivisible prompt groups for DAPO sampling."""

    (completions, contexts, prompt_ids, response_ids, response_masks,
     old_logps, turn_outputs, unfinished, traces) = rollout_values
    records = []
    for group_idx in range(len(messages_batch)):
        start, end = group_idx * group_size, (group_idx + 1) * group_size
        records.append({
            "message": messages_batch[group_idx], "tools": tools_batch[group_idx],
            "required_tools": required_tools_batch[group_idx],
            "gt": gt_batch[group_idx], "prompt": prompts[group_idx],
            "completions": completions[start:end], "contexts": contexts[start:end],
            "prompt_ids": prompt_ids[start:end], "response_ids": response_ids[start:end],
            "response_masks": response_masks[start:end], "old_logps": old_logps[start:end],
            "turn_outputs": turn_outputs[start:end], "unfinished": unfinished[start:end],
            "traces": traces[start:end],
            "rewards": reward_output.rewards[start:end],
            "task_success": reward_output.task_success[start:end],
            "answer_accuracy": reward_output.answer_accuracy[start:end],
            "format_valid": reward_output.format_valid[start:end],
            "tool_call_valid": reward_output.tool_call_valid[start:end],
            "tool_execution_success": reward_output.tool_execution_success[start:end],
            "required_tool_coverage": reward_output.required_tool_coverage[start:end],
            "tool_evidence_coverage": reward_output.tool_evidence_coverage[start:end],
            "protocol_progress": reward_output.protocol_progress[start:end],
        })
    return records


def _merge_rollout_group_records(records):
    """Merge prompt-group records back into the tensors/lists used for loss."""

    def flatten(key):
        return [value for record in records for value in record[key]]

    output = VerifiableRewardOutput(
        rewards=torch.cat([record["rewards"] for record in records]),
        task_success=torch.cat([record["task_success"] for record in records]),
        answer_accuracy=torch.cat([record["answer_accuracy"] for record in records]),
        format_valid=torch.cat([record["format_valid"] for record in records]),
        tool_call_valid=torch.cat([record["tool_call_valid"] for record in records]),
        tool_execution_success=torch.cat([record["tool_execution_success"] for record in records]),
        required_tool_coverage=torch.cat([record["required_tool_coverage"] for record in records]),
        tool_evidence_coverage=torch.cat([record["tool_evidence_coverage"] for record in records]),
        protocol_progress=torch.cat([record["protocol_progress"] for record in records]),
    )
    return (
        [record["message"] for record in records],
        [record["tools"] for record in records],
        [record["required_tools"] for record in records],
        [record["gt"] for record in records],
        [record["prompt"] for record in records],
        flatten("completions"), flatten("contexts"), flatten("prompt_ids"),
        flatten("response_ids"), flatten("response_masks"), flatten("old_logps"),
        flatten("turn_outputs"), flatten("unfinished"), flatten("traces"), output,
    )


def rl_train_epoch(
    epoch,
    loader,
    iters,
    rollout_engine,
    ref_model,
    reward_model=None,
    start_step=0,
    wandb=None,
    use_sglang=False,
    target_updates=None,
    candidate_start=0,
):
    target_updates = target_updates or iters
    if start_step >= target_updates:
        return
    dynamic_buffer = []
    candidate_groups = 0
    accepted_groups = 0
    effective_groups = 0
    candidate_trajectories = 0
    candidate_action_tokens = 0
    candidate_tool_calls = 0
    update_step = start_step
    optimization_micro_step = 0
    epoch_started = time.time()

    budget_stop_reason = "candidate_pool_exhausted"
    for candidate_step, batch in enumerate(loader, start=candidate_start + 1):
        if update_step >= target_updates or (
            args.max_effective_groups > 0 and effective_groups >= args.max_effective_groups
        ):
            budget_stop_reason = "effective_group_target"
            break
        if args.max_candidate_groups > 0 and candidate_groups >= args.max_candidate_groups:
            budget_stop_reason = "max_candidate_groups"
            break
        if args.max_generated_tokens > 0 and candidate_action_tokens >= args.max_generated_tokens:
            budget_stop_reason = "max_generated_tokens"
            break
        candidate_messages = batch['messages']
        candidate_tools = batch['tools']
        candidate_required_tools = batch['required_tools']
        candidate_gt = batch['gt']
        with torch.no_grad():
            rollout_values = rollout_batch(
                rollout_engine, tokenizer, candidate_messages, candidate_tools,
                args.num_generations, max_turns=args.max_turns,
                max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio,
                temperature=args.rollout_temperature,
                top_k=args.rollout_top_k,
                top_p=args.rollout_top_p,
                device=args.device,
            )
        (candidate_completions, _, _, _, candidate_response_masks, _,
         candidate_turn_outputs, candidate_unfinished, _) = rollout_values
        candidate_prompts = [
            render_agent_chat(
                tokenizer, m, tools=t, tokenize=False,
                add_generation_prompt=True, open_thinking=False,
            )
            for m, t in zip(candidate_messages, candidate_tools)
        ]
        reward_output = calculate_rewards(
            candidate_prompts, candidate_completions, candidate_gt, candidate_tools,
            args.num_generations, candidate_required_tools, reward_model, device=args.device,
            turn_outputs_batch=candidate_turn_outputs,
            unfinished_batch=candidate_unfinished,
            require_tool_call_for_success=bool(args.require_tool_call_for_success),
            reward_mode=args.reward_mode,
            return_details=True,
        )

        response_lengths = torch.tensor(
            [sum(mask) for mask in candidate_response_masks], device=args.device
        )
        candidate_trajectories += len(candidate_response_masks)
        candidate_action_tokens += int(response_lengths.sum().item())
        candidate_tool_calls += sum(
            len(parse_tool_calls(turn))
            for trajectory in candidate_turn_outputs for turn in trajectory
        )
        if args.max_generated_tokens > 0 and candidate_action_tokens >= args.max_generated_tokens:
            budget_stop_reason = "max_generated_tokens"
        use_dapo_sampling_recipe = args.loss_type == "dapo" or (
            args.loss_type == "cispo" and args.dynamic_sampling
        )
        if use_dapo_sampling_recipe and args.overlong_cache_len > 0:
            max_response_tokens = args.max_gen_len * args.max_turns
            reward_output.rewards.add_(
                args.overlong_penalty_coef * soft_overlong_penalty(
                    response_lengths, max_response_tokens,
                    min(args.overlong_cache_len, max_response_tokens),
                )
            )

        records = _rollout_group_records(
            candidate_messages, candidate_tools, candidate_required_tools, candidate_gt, candidate_prompts,
            rollout_values, reward_output, args.num_generations,
        )
        candidate_groups += len(records)

        if args.dynamic_sampling:
            group_mask = effective_group_mask(reward_output.task_success, args.num_generations)
            selected = [record for record, keep in zip(records, group_mask.tolist()) if keep]
            dynamic_buffer.extend(selected)
            accepted_groups += len(selected)
            local_ready = torch.tensor(
                int(len(dynamic_buffer) >= args.batch_size), device=args.device, dtype=torch.int32
            )
            if dist.is_initialized():
                dist.all_reduce(local_ready, op=dist.ReduceOp.MIN)
            if not local_ready.item():
                if candidate_step % args.log_interval == 0 and is_main_process():
                    Logger(
                        f"DAPO dynamic sampling: candidate_groups={candidate_groups}, "
                        f"accepted_groups={accepted_groups}, buffered={len(dynamic_buffer)}"
                    )
                continue
            records = dynamic_buffer[:args.batch_size]
            # Do not carry accepted trajectories across a policy update.  Any
            # surplus was generated by the old policy and would make the next
            # nominally on-policy batch stale.
            dynamic_buffer = []
        else:
            accepted_groups += len(records)

        remaining_groups = (
            args.max_effective_groups - effective_groups
            if args.max_effective_groups > 0 else len(records)
        )
        records = records[:max(0, remaining_groups)]
        if not records:
            if budget_stop_reason == "max_generated_tokens":
                break
            continue

        (messages_batch, tools_batch, required_tools_batch, gt_batch, prompts, completions, contexts,
         prompt_ids_batch, response_ids_batch, response_masks_batch,
         response_old_logps_batch, turn_outputs_batch, unfinished_batch, traces_batch,
         reward_output) = _merge_rollout_group_records(records)
        rewards = reward_output.rewards
        effective_groups += len(records)
        update_step += 1

        trajectory_lengths = [len(p) + len(r) for p, r in zip(prompt_ids_batch, response_ids_batch)]
        max_trajectory_len = torch.tensor(
            max(trajectory_lengths, default=0), device=args.device, dtype=torch.long
        )
        if dist.is_initialized():
            dist.all_reduce(max_trajectory_len, op=dist.ReduceOp.MAX)
        if max_trajectory_len.item() > args.max_total_len:
            raise RuntimeError(
                "A rollout trajectory exceeds --max_total_len "
                f"({max_trajectory_len.item()} > {args.max_total_len}). "
                "Post-rollout left truncation would change the conditioning context and invalidate "
                "the stored behavior-policy log-probabilities. Increase --max_total_len or reduce "
                "the tool prompt / --max_turns / --max_gen_len, then restart this run."
            )

        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            prompt_len = next((i for i, value in enumerate(mask) if value == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([
            ids + [tokenizer.pad_token_id] * (max_len - len(ids))
            for ids, _, _, _ in packed_samples
        ], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([
            mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples
        ], device=args.device, dtype=torch.float32)
        rollout_old_per_token_logps = torch.tensor([
            old_logps + [0.0] * ((max_len - 1) - len(old_logps))
            for _, _, _, old_logps in packed_samples
        ], device=args.device, dtype=torch.float32)
        full_mask = (input_ids != tokenizer.pad_token_id).long()

        completion_mask = shift_action_mask(full_response_masks)
        token_counts = completion_mask.sum(dim=1)

        # Reference/behavior log-probs must use the same precision.  Comparing
        # an FP32 reference forward with a BF16 policy forward can create huge
        # k3 tails even before any optimizer update, especially for very
        # unlikely tokens.  We also recompute behavior log-probs on the exact
        # packed multi-turn trajectory.  The rollout-engine values remain a
        # synchronization audit, while the recomputed values give GSPO a unit
        # ratio before the first optimizer step instead of clipping harmless
        # BF16/cache discrepancies.
        was_training = model.training
        model.eval()
        with torch.no_grad(), autocast_ctx:
            old_per_token_logps = compute_per_token_logps(
                model, input_ids, input_ids.size(1) - 1,
                attention_mask=full_mask,
                blocked_token_ids=legal_action_blocked_ids(tokenizer),
            ).float()
            ref_per_token_logps = compute_per_token_logps(
                ref_model, input_ids, input_ids.size(1) - 1,
                attention_mask=full_mask,
                blocked_token_ids=legal_action_blocked_ids(tokenizer),
            ).float()
        model.train(was_training)

        valid_tokens = completion_mask.sum().clamp(min=1)
        rollout_log_ratio = old_per_token_logps - rollout_old_per_token_logps
        rollout_logprob_mae = (
            rollout_log_ratio.abs() * completion_mask
        ).sum() / valid_tokens
        rollout_ratio_mean = (
            rollout_log_ratio.exp() * completion_mask
        ).sum() / valid_tokens
        if rollout_logprob_mae.item() > args.max_rollout_logprob_mae:
            raise RuntimeError(
                "Behavior-policy log-probability mismatch exceeds the safety threshold: "
                f"MAE={rollout_logprob_mae.item():.6f} > "
                f"{args.max_rollout_logprob_mae:.6f}. Audit chat-template alignment, "
                "rollout/training precision and multi-turn observation packing before "
                "continuing; importance ratios are otherwise invalid."
            )
        old_per_token_logps = old_per_token_logps.detach()

        if args.debug_mode and is_main_process() and update_step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] update={update_step}, gt[{i}]={gt_batch[i]!r}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"[DEBUG] gen[{i}][{j}] context:\n{contexts[idx]}")
                    Logger(f"[DEBUG] trainable completion:\n{tokenizer.decode(input_ids[idx, plen:slen].tolist(), skip_special_tokens=False)}")
                    Logger(
                        f"[DEBUG] reward={rewards[idx].item():.4f}, "
                        f"success={bool(reward_output.task_success[idx])}"
                    )

        grouped_rewards = rewards.view(-1, args.num_generations)
        advantages = group_relative_advantages(rewards, args.num_generations)
        policy_output = None
        per_token_logps = None
        aux_loss = torch.tensor(0.0, device=args.device)
        for _policy_epoch in range(args.policy_update_epochs):
            with autocast_ctx:
                res = model(input_ids, attention_mask=full_mask)
                aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                per_token_logps = masked_log_softmax(
                    res.logits[:, :-1, :], blocked_token_ids=legal_action_blocked_ids(tokenizer)
                ).gather(
                    2, input_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                # Storage PAD targets are outside completion_mask.  Keep them
                # finite so masked policy/KL reductions cannot form 0 * -inf.
                per_token_logps = torch.where(
                    input_ids[:, 1:].eq(tokenizer.pad_token_id),
                    torch.zeros_like(per_token_logps), per_token_logps,
                )
                policy_output = compute_policy_loss(
                    loss_type=args.loss_type,
                    current_logps=per_token_logps,
                    old_logps=old_per_token_logps,
                    reference_logps=ref_per_token_logps,
                    advantages=advantages,
                    completion_mask=completion_mask,
                    beta=args.beta,
                    grpo_epsilon=args.epsilon,
                    cispo_epsilon_high=args.epsilon_high,
                    dapo_epsilon_low=args.dapo_epsilon_low,
                    dapo_epsilon_high=args.dapo_epsilon_high,
                    gspo_epsilon_low=args.gspo_epsilon_low,
                    gspo_epsilon_high=args.gspo_epsilon_high,
                )
                token_mean_scale = (
                    distributed_token_mean_scale(completion_mask)
                    if args.loss_type in {"cispo", "dapo"} else 1.0
                )
                loss = (policy_output.loss * token_mean_scale + aux_loss) / args.accumulation_steps
            loss.backward()
            optimization_micro_step += 1
            if optimization_micro_step % args.accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        updates_this_epoch = update_step
        if update_step % args.log_interval == 0 or updates_this_epoch == target_updates:
            action_positions = completion_mask.bool()
            reference_log_ratio = (
                ref_per_token_logps.float() - per_token_logps.float()
            )[action_positions]
            local_k3_values = positive_kl_estimate(
                per_token_logps.float(), ref_per_token_logps.float()
            )[action_positions]
            cost_counters = torch.tensor(
                [candidate_groups, accepted_groups, effective_groups,
                 candidate_trajectories, candidate_action_tokens,
                 candidate_tool_calls],
                device=args.device, dtype=torch.float64,
            )
            if dist.is_initialized():
                dist.all_reduce(cost_counters, op=dist.ReduceOp.SUM)
            metrics = {
                "reward": rewards.mean().item(),
                "task_accuracy": reward_output.task_success.float().mean().item(),
                "answer_accuracy": reward_output.answer_accuracy.mean().item(),
                "format_valid_rate": reward_output.format_valid.mean().item(),
                "tool_call_valid_rate": reward_output.tool_call_valid.mean().item(),
                "tool_execution_success_rate": reward_output.tool_execution_success.mean().item(),
                "required_tool_coverage_rate": reward_output.required_tool_coverage.mean().item(),
                "tool_evidence_coverage_rate": reward_output.tool_evidence_coverage.mean().item(),
                "protocol_progress": reward_output.protocol_progress.mean().item(),
                "kl_k3": policy_output.approx_kl.item(),
                "local_reference_log_ratio_abs_mean": (
                    reference_log_ratio.abs().mean().item()
                    if reference_log_ratio.numel() else 0.0
                ),
                "local_kl_k3_p95": (
                    torch.quantile(local_k3_values, 0.95).item()
                    if local_k3_values.numel() else 0.0
                ),
                "local_kl_k3_max": (
                    local_k3_values.max().item() if local_k3_values.numel() else 0.0
                ),
                "group_reward_std": grouped_rewards.std(dim=1, unbiased=False).mean().item(),
                "zero_variance_group_rate": (grouped_rewards.std(dim=1, unbiased=False) < 1e-6).float().mean().item(),
                "advantages_std": advantages.std().item(),
                "advantages_mean": advantages.mean().item(),
                "policy_loss": policy_output.policy_loss.item(),
                "aux_loss": aux_loss.item(),
                "clip_fraction": policy_output.clip_fraction.item(),
                "ratio_mean": policy_output.ratio_mean.item(),
                "ratio_std": policy_output.ratio_std.item(),
                "rollout_logprob_mae": rollout_logprob_mae.item(),
                "rollout_ratio_mean": rollout_ratio_mean.item(),
                "avg_response_len": token_counts.float().mean().item(),
                "p95_response_len": torch.quantile(token_counts.float(), 0.95).item(),
                "unfinished_rate": sum(bool(value) for value in unfinished_batch) / len(unfinished_batch),
                "dynamic_acceptance_rate": (
                    cost_counters[1] / cost_counters[0].clamp(min=1)
                ).item(),
                "candidate_groups": int(cost_counters[0].item()),
                "accepted_groups": int(cost_counters[1].item()),
                "effective_groups": int(cost_counters[2].item()),
                "candidate_trajectories": int(cost_counters[3].item()),
                "candidate_action_tokens": int(cost_counters[4].item()),
                "generated_tokens": int(cost_counters[4].item()),
                "tool_calls": int(cost_counters[5].item()),
                "optimizer_updates": int(scheduler.last_epoch),
                "wall_time_seconds": time.time() - epoch_started,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }
            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}] candidate={candidate_step}/{iters} "
                f"update={updates_this_epoch}/{target_updates}, "
                f"Algorithm:{args.loss_type.upper()}, Reward:{metrics['reward']:.4f}, "
                f"Acc:{metrics['task_accuracy']:.4f}, KL_k3:{metrics['kl_k3']:.6f}, "
                f"ClipFrac:{metrics['clip_fraction']:.4f}, AvgLen:{metrics['avg_response_len']:.1f}, "
                f"DynamicAccept:{metrics['dynamic_acceptance_rate']:.3f}"
            )
            if is_main_process():
                metric_logger.log(metrics, step=epoch * target_updates + update_step)
            if wandb and is_main_process():
                wandb.log(metrics)

        if is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            if update_step % args.save_interval == 0 or updates_this_epoch == target_updates:
                ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
                save_policy_checkpoint(model, ckp, args.checkpoint_dtype)
            while args.checkpoint_groups and args._next_checkpoint_group <= effective_groups:
                milestone = args._next_checkpoint_group
                ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}_groups{milestone}.pth'
                if not os.path.exists(ckp):
                    save_policy_checkpoint(model, ckp, args.checkpoint_dtype)
                args._next_checkpoint_group_index += 1
                args._next_checkpoint_group = (
                    args.checkpoint_groups[args._next_checkpoint_group_index]
                    if args._next_checkpoint_group_index < len(args.checkpoint_groups) else math.inf
                )
            if args.save_resume and (update_step % args.save_interval == 0 or updates_this_epoch == target_updates):
                lm_checkpoint(
                    lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                    epoch=epoch, step=update_step, wandb=wandb,
                    save_dir=args.checkpoint_dir, scheduler=scheduler,
                    candidate_step=candidate_step,
                )
            model.train()

        if update_step % args.rollout_sync_interval == 0 or updates_this_epoch == target_updates:
            rollout_engine.update_policy(model)
        if updates_this_epoch >= target_updates:
            break

    if update_step >= target_updates or (
        args.max_effective_groups > 0 and effective_groups >= args.max_effective_groups
    ):
        budget_stop_reason = "effective_group_target"
    elif args.max_candidate_groups > 0 and candidate_groups >= args.max_candidate_groups:
        budget_stop_reason = "max_candidate_groups"
    elif args.max_generated_tokens > 0 and candidate_action_tokens >= args.max_generated_tokens:
        budget_stop_reason = "max_generated_tokens"
    if is_main_process():
        metric_logger.log({
            "budget_stop_reason": budget_stop_reason,
            "candidate_groups": candidate_groups,
            "accepted_groups": accepted_groups,
            "effective_groups": effective_groups,
            "candidate_trajectories": candidate_trajectories,
            "candidate_action_tokens": candidate_action_tokens,
            "generated_tokens": candidate_action_tokens,
            "optimizer_updates": int(scheduler.last_epoch),
            "budget_complete": budget_stop_reason in {
                "effective_group_target", "max_candidate_groups", "candidate_pool_exhausted"
            },
        }, step=epoch * max(target_updates, 1) + update_step)
    if optimization_micro_step and optimization_micro_step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return {
        "budget_stop_reason": budget_stop_reason,
        "candidate_groups": candidate_groups,
        "accepted_groups": accepted_groups,
        "effective_groups": effective_groups,
        "candidate_trajectories": candidate_trajectories,
        "candidate_action_tokens": candidate_action_tokens,
        "optimizer_updates": int(scheduler.last_epoch),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="断点恢复文件目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="RL AdamW权重衰减；默认0避免零advantage时策略漂移")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument(
        "--checkpoint_dtype", default="float32",
        choices=sorted(_CHECKPOINT_DTYPES),
        help="RL权重落盘精度；默认float32保留小学习率更新",
    )
    parser.add_argument(
        "--save_resume", default=1, type=int, choices=[0, 1],
        help="是否同时保存包含Adam/scheduler的断点；短程消融可设0以节省磁盘",
    )
    parser.add_argument("--max_updates", type=int, default=0, help="每个epoch最多optimizer更新数；0表示由数据/有效组预算决定")
    parser.add_argument(
        "--max_candidate_groups", type=int, default=0,
        help="每个epoch最多采样的候选prompt组数；0表示不单独限制，用于候选预算匹配",
    )
    parser.add_argument(
        "--max_effective_groups", type=int, default=0,
        help="DAPO有效组预算；0关闭。有效组按混合成功/失败prompt组计数，可重复抽题",
    )
    parser.add_argument(
        "--max_generated_tokens", type=int, default=0,
        help="assistant候选轨迹生成token预算；0关闭，达到后记录budget不足并停止",
    )
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--max_turns", type=int, default=3, help="每条轨迹最大assistant/tool轮数")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer目录")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--rollout_temperature", type=float, default=1.0, help="策略采样温度")
    parser.add_argument("--rollout_top_k", type=int, default=0, help="rollout top-k；诊断协议固定为0")
    parser.add_argument("--rollout_top_p", type=float, default=1.0, help="rollout top-p；诊断协议固定为1")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo", "dapo", "gspo"], help="策略优化目标")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO epsilon_high_IS；实际ratio上界为1+该值")
    parser.add_argument("--dapo_epsilon_low", type=float, default=0.2, help="DAPO非对称下裁剪")
    parser.add_argument("--dapo_epsilon_high", type=float, default=0.28, help="DAPO Clip-Higher上裁剪")
    parser.add_argument("--gspo_epsilon_low", type=float, default=3e-4, help="GSPO序列比率下裁剪")
    parser.add_argument("--gspo_epsilon_high", type=float, default=4e-4, help="GSPO序列比率上裁剪")
    parser.add_argument("--policy_update_epochs", type=int, default=1, help="每批rollout复用的策略更新轮数；算法对比建议2~4")
    parser.add_argument("--rollout_sync_interval", type=int, default=1, help="多少个训练update同步一次rollout策略")
    parser.add_argument("--max_rollout_logprob_mae", type=float, default=0.1, help="行为策略与训练侧重算logprob的最大允许MAE")
    parser.add_argument("--dynamic_sampling", action="store_true", help="启用DAPO二值动态采样；CISPO论文完整recipe也可复用")
    parser.add_argument("--dynamic_sampling_rounds", type=int, default=10, help="每个epoch最多重复候选数据的轮数，用于填满有效batch")
    parser.add_argument("--overlong_cache_len", type=int, default=128, help="DAPO软超长惩罚的线性缓冲token数；0关闭")
    parser.add_argument("--overlong_penalty_coef", type=float, default=1.0, help="DAPO软超长惩罚权重")
    parser.add_argument("--require_tool_call_for_success", type=int, default=1, choices=[0, 1], help="有工具定义时，成功轨迹是否必须调用工具")
    parser.add_argument("--reward_mode", choices=["strict", "shaped"], default="strict", help="strict使用可验证成功±1；shaped用于课程学习")
    parser.add_argument("--metrics_path", type=str, default="../out/metrics/agent_rlvr.jsonl", help="本地可审计JSONL指标")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument(
        '--from_save_dir', default=None, type=str,
        help="初始化权重所在目录；未设置时使用save_dir，便于把输出checkpoint放到独立目录",
    )
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--use_reward_model", type=int, default=0, choices=[0, 1], help="是否叠加主观Reward Model；RLVR对比建议关闭")
    parser.add_argument(
        "--checkpoint_groups", default="0,50,100,200",
        help="按累计有效组保存独立权重，逗号分隔；空字符串关闭",
    )
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()
    if min(args.policy_update_epochs, args.rollout_sync_interval, args.max_turns, args.dynamic_sampling_rounds) < 1:
        parser.error("policy_update_epochs, rollout_sync_interval, max_turns and dynamic_sampling_rounds must be >= 1")
    if args.num_generations < 2 or args.batch_size < 1:
        parser.error("num_generations must be >= 2 and batch_size must be >= 1")
    if not 0.0 <= args.thinking_ratio <= 1.0:
        parser.error("thinking_ratio must be between 0 and 1")
    if args.rollout_temperature <= 0:
        parser.error("rollout_temperature must be > 0")
    if args.epsilon_high < 0:
        parser.error("epsilon_high must be >= 0")
    if args.weight_decay < 0:
        parser.error("weight_decay must be >= 0")
    if args.max_rollout_logprob_mae <= 0:
        parser.error("max_rollout_logprob_mae must be > 0")
    if min(args.max_updates, args.max_candidate_groups, args.max_effective_groups, args.max_generated_tokens) < 0:
        parser.error("max_updates, max_candidate_groups, max_effective_groups and max_generated_tokens must be >= 0")
    if args.rollout_top_k < 0 or not 0 < args.rollout_top_p <= 1:
        parser.error("rollout_top_k must be >= 0 and rollout_top_p must be in (0, 1]")
    try:
        args.checkpoint_groups = sorted({
            int(item.strip()) for item in args.checkpoint_groups.split(",") if item.strip()
        })
    except ValueError:
        parser.error("checkpoint_groups must be a comma-separated list of integers")
    if any(item < 0 for item in args.checkpoint_groups):
        parser.error("checkpoint_groups must be >= 0")
    args._next_checkpoint_group_index = 0
    args._next_checkpoint_group = (
        args.checkpoint_groups[0] if args.checkpoint_groups else math.inf
    )
    if args.reward_mode == "strict" and args.use_reward_model:
        parser.error("strict RLVR cannot mix a subjective reward model; use --use_reward_model 0")
    if args.dynamic_sampling and args.loss_type not in {"dapo", "cispo"}:
        parser.error("--dynamic_sampling is supported by DAPO and the complete CISPO recipe")
    if args.max_effective_groups and not args.dynamic_sampling:
        parser.error("--max_effective_groups requires --dynamic_sampling so effective groups are well-defined")

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    process_rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(args.seed + process_rank)

    os.makedirs(args.save_dir, exist_ok=True)
    metric_logger = JsonlMetricLogger(
        args.metrics_path if is_main_process() else None,
        run_config={
            "algorithm": args.loss_type,
            "data_path": args.data_path,
            "from_weight": args.from_weight,
            "from_save_dir": args.from_save_dir or args.save_dir,
            "seed": args.seed,
            "num_generations": args.num_generations,
            "rollout_temperature": args.rollout_temperature,
            "rollout_top_k": args.rollout_top_k,
            "rollout_top_p": args.rollout_top_p,
            "checkpoint_dtype": args.checkpoint_dtype,
            "policy_update_epochs": args.policy_update_epochs,
            "learning_rate": args.learning_rate,
            "accumulation_steps": args.accumulation_steps,
            "dynamic_sampling": bool(args.dynamic_sampling),
            "dynamic_sampling_rounds": args.dynamic_sampling_rounds,
            "reward_mode": args.reward_mode,
            "max_updates": args.max_updates,
            "max_candidate_groups": args.max_candidate_groups,
            "max_effective_groups": args.max_effective_groups,
            "max_generated_tokens": args.max_generated_tokens,
            "checkpoint_groups": args.checkpoint_groups,
            "max_turns": args.max_turns,
            "max_gen_len": args.max_gen_len,
            "max_total_len": args.max_total_len,
            "max_rollout_logprob_mae": args.max_rollout_logprob_mae,
            "save_resume": bool(args.save_resume),
            "beta": args.beta,
            "weight_decay": args.weight_decay,
            "grpo_epsilon": args.epsilon,
            "cispo_epsilon_high": args.epsilon_high,
            "dapo_epsilon_low": args.dapo_epsilon_low,
            "dapo_epsilon_high": args.dapo_epsilon_high,
            "gspo_epsilon_low": args.gspo_epsilon_low,
            "gspo_epsilon_high": args.gspo_epsilon_high,
            "overlong_cache_len": args.overlong_cache_len,
            "overlong_penalty_coef": args.overlong_penalty_coef,
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "use_moe": bool(args.use_moe),
        },
    )
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir) if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume)

    model, tokenizer = init_model(
        lm_config, args.from_weight,
        tokenizer_path=args.tokenizer_path,
        save_dir=args.from_save_dir or args.save_dir,
        device=args.device,
    )

    ref_model, _ = init_model(
        lm_config, args.from_weight,
        tokenizer_path=args.tokenizer_path,
        save_dir=args.from_save_dir or args.save_dir,
        device=args.device,
    )
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = None
    if args.use_reward_model:
        reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
        Logger(f'Loaded reward model from {args.reward_model_path}')
    else:
        Logger('Reward model disabled: using deterministic RLVR rewards only')
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    def collate_fn(batch): return {
        'messages': [b['messages'] for b in batch],
        'tools': [b['tools'] for b in batch],
        'required_tools': [b['required_tools'] for b in batch],
        'gt': [b['gt'] for b in batch],
    }
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    if args.max_effective_groups:
        updates_per_epoch = math.ceil(args.max_effective_groups / args.batch_size)
    else:
        updates_per_epoch = min(iters, args.max_updates) if args.max_updates else iters
    total_micro_steps = updates_per_epoch * args.policy_update_epochs * args.epochs
    total_optimizer_steps = math.ceil(total_micro_steps / args.accumulation_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step, start_candidate_step = 0, 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        start_candidate_step = ckp_data.get('candidate_step', start_step)

    # Diagnostic snapshots are immutable and keyed by cumulative effective
    # groups, so a later checkpoint can never replace the initial policy.
    if is_main_process() and args.checkpoint_groups and 0 in args.checkpoint_groups:
        moe_suffix = '_moe' if lm_config.use_moe else ''
        initial_path = (
            f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}'
            f'{moe_suffix}_groups0.pth'
        )
        if not os.path.exists(initial_path):
            save_policy_checkpoint(model, initial_path, args.checkpoint_dtype)
        args._next_checkpoint_group_index = args.checkpoint_groups.index(0) + 1
        args._next_checkpoint_group = (
            args.checkpoint_groups[args._next_checkpoint_group_index]
            if args._next_checkpoint_group_index < len(args.checkpoint_groups) else math.inf
        )

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch * max(dist.get_world_size() if dist.is_initialized() else 1, 1) + process_rank)
        indices = torch.randperm(len(train_ds)).tolist()
        resume_epoch = epoch == start_epoch and start_step > 0
        completed_updates = start_step if resume_epoch else 0
        skip = start_candidate_step if resume_epoch else 0
        epoch_indices = list(train_sampler) if train_sampler is not None else indices
        if args.dynamic_sampling:
            # DAPO oversamples candidate prompts but keeps the same number of
            # optimizer updates as the baselines.  New stochastic rollouts are
            # produced on every repeated pass.
            repeat_count = args.dynamic_sampling_rounds
            if args.max_effective_groups:
                # Dynamic sampling may need many more candidate prompts than
                # effective groups.  Reusing the same questions is intentional
                # and is bounded by --max_generated_tokens when provided.
                repeat_count = max(repeat_count, args.max_effective_groups * args.dynamic_sampling_rounds)
            epoch_indices = epoch_indices * repeat_count
        batch_sampler = SkipBatchSampler(epoch_indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        target_updates = updates_per_epoch
        if skip > 0:
            Logger(
                f'Epoch [{epoch+1}/{args.epochs}]: resume after '
                f'{completed_updates} optimizer updates and {skip} candidate batches'
            )
            rl_train_epoch(
                epoch, loader, len(loader) + skip, rollout_engine, ref_model,
                reward_model, completed_updates, wandb,
                use_sglang=(args.rollout_engine == "sglang"),
                target_updates=target_updates, candidate_start=skip,
            )
        else:
            rl_train_epoch(
                epoch, loader, len(loader), rollout_engine, ref_model,
                reward_model, 0, wandb,
                use_sglang=(args.rollout_engine == "sglang"),
                target_updates=target_updates, candidate_start=0,
            )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
