"""Create compositional Tool-Use RLVR data excluded from Agent-SFT labels.

The basic verified tool set is intentionally easy enough for protocol cold
start and saturates after Agent SFT.  This generator combines familiar tools
in unseen task compositions, including dependencies where a later calculator
call consumes an earlier tool result.  Oracle fields are audit evidence only;
``AgentRLDataset`` never exposes them to the policy.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.prepare_tool_rlvr_data import (
    RATES,
    TIMES,
    TRANSLATIONS,
    UNITS,
    WEATHER,
    make_row,
    stable_hash,
)
from trainer.train_agent import execute_tool


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def call(name, arguments):
    spec = {"name": name, "arguments": arguments}
    observation = execute_tool(name, arguments)
    if observation is None:
        raise RuntimeError(f"oracle execution failed: {spec}")
    return spec, observation


def make_verified_row(question, available, required, ground_truth, calls, category):
    oracle_calls, oracle_observations = zip(*calls)
    return make_row(
        question,
        available,
        [str(value) for value in ground_truth],
        category,
        required_tools=required,
        oracle_calls=list(oracle_calls),
        oracle_observations=list(oracle_observations),
    )


def build_candidates(seed, per_category):
    rng = random.Random(seed)
    rows = []
    rate_pairs = list(RATES)
    unit_pairs = list(UNITS)
    cities = list(WEATHER)
    zones = list(TIMES)
    translations = list(TRANSLATIONS)

    # A second call depends on the rate returned by the first call.  The model
    # may solve this across two tool turns or issue both calls if it knows the
    # deterministic sandbox value, but both executions must support the GT.
    for index in range(per_category):
        source, target = rate_pairs[index % len(rate_pairs)]
        amount = 11 + ((index * 37 + seed) % 989)
        rate_call = call("get_exchange_rate", {"from_currency": source, "to_currency": target})
        rate = rate_call[1]["rate"]
        math_call = call("calculate_math", {"expression": f"{amount}*{rate}"})
        rows.append(make_verified_row(
            f"先查询 {source} 到 {target} 的汇率，再用计算器算出 {amount} {source} 对应多少 {target}；最后只报告换算金额。",
            ["get_exchange_rate", "calculate_math", "unit_converter"],
            ["get_exchange_rate", "calculate_math"],
            [math_call[1]["result"]],
            [rate_call, math_call],
            "challenge_exchange_then_math",
        ))

    for index in range(per_category):
        source, target = unit_pairs[index % len(unit_pairs)]
        value = 2 + ((index * 29 + seed) % 498)
        multiplier = 2 + (index % 8)
        unit_call = call("unit_converter", {
            "value": value, "from_unit": source, "to_unit": target,
        })
        converted = unit_call[1]["result"]
        math_call = call("calculate_math", {"expression": f"{converted}*{multiplier}"})
        rows.append(make_verified_row(
            f"把 {value} {source} 换算成 {target}，然后用计算器把换算结果乘以 {multiplier}；给出最终数值。",
            ["unit_converter", "calculate_math", "translate_text"],
            ["unit_converter", "calculate_math"],
            [math_call[1]["result"]],
            [unit_call, math_call],
            "challenge_unit_then_math",
        ))

    # Unseen combinations of two independent tools test selection, JSON
    # formatting and integration without making every prompt sequential.
    for index in range(per_category):
        text_value, language = translations[index % len(translations)]
        a = 20 + ((index * 31 + seed) % 500)
        b = 3 + ((index * 13 + seed) % 90)
        translation_call = call("translate_text", {
            "text": text_value, "target_language": language,
        })
        math_call = call("calculate_math", {"expression": f"{a}+{b}"})
        rows.append(make_verified_row(
            f"完成两项独立任务：把“{text_value}”翻译成 {language}，并用计算器计算 {a}+{b}。最终答案必须同时包含两项结果。",
            ["translate_text", "calculate_math", "get_current_time"],
            ["translate_text", "calculate_math"],
            [translation_call[1]["translated_text"], math_call[1]["result"]],
            [translation_call, math_call],
            "challenge_translation_and_math",
        ))

    for index in range(per_category):
        city = cities[index % len(cities)]
        source, target = rate_pairs[(index * 3) % len(rate_pairs)]
        weather_call = call("get_current_weather", {"location": city})
        rate_call = call("get_exchange_rate", {"from_currency": source, "to_currency": target})
        rows.append(make_verified_row(
            f"同时核对 {city} 的天气温度，以及 {source} 到 {target} 的汇率；请分别调用工具并汇总两个查询值。",
            ["get_current_weather", "get_exchange_rate", "get_current_time"],
            ["get_current_weather", "get_exchange_rate"],
            [weather_call[1]["temperature"], rate_call[1]["rate"]],
            [weather_call, rate_call],
            "challenge_weather_and_exchange",
        ))

    for index in range(per_category):
        zone = zones[index % len(zones)]
        source, target = unit_pairs[(index * 5) % len(unit_pairs)]
        value = 5 + ((index * 19 + seed) % 300)
        time_call = call("get_current_time", {"timezone": zone})
        unit_call = call("unit_converter", {
            "value": value, "from_unit": source, "to_unit": target,
        })
        rows.append(make_verified_row(
            f"查询 {zone} 的完整当前时间，同时把 {value} {source} 换算为 {target}；最后汇总两项结果。",
            ["get_current_time", "unit_converter", "get_current_weather"],
            ["get_current_time", "unit_converter"],
            [time_call[1]["datetime"], unit_call[1]["result"]],
            [time_call, unit_call],
            "challenge_time_and_unit",
        ))

    # Three tools and a dependent final computation exercise the complete
    # Assistant -> Tool -> Observation -> Assistant loop up to max_turns=3.
    for index in range(per_category):
        city = cities[index % len(cities)]
        zone = zones[(index * 7) % len(zones)]
        style = [
            "请严格按查询结果计算",
            "不要凭记忆估计",
            "请在得到两项 Observation 后再计算",
        ][(index // max(len(cities), 1)) % 3]
        weather_call = call("get_current_weather", {"location": city})
        time_call = call("get_current_time", {"timezone": zone})
        temperature = float(weather_call[1]["temperature"].replace("°C", ""))
        hour = int(time_call[1]["datetime"].split()[1].split(":")[0])
        math_call = call("calculate_math", {"expression": f"{temperature}+{hour}"})
        rows.append(make_verified_row(
            f"先查询 {city} 的温度和 {zone} 的当前时间，再用计算器求“温度数值+小时数”；{style}，最终只给出这个和。",
            ["get_current_weather", "get_current_time", "calculate_math"],
            ["get_current_weather", "get_current_time", "calculate_math"],
            [math_call[1]["result"]],
            [weather_call, time_call, math_call],
            "challenge_weather_time_then_math",
        ))

    unique = {}
    for row in rows:
        question = row["conversations"][1]["content"]
        unique.setdefault(stable_hash(question), row)
    return list(unique.values())


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create compositional Tool-Use RLVR challenge data")
    parser.add_argument("--train_output", default="dataset/agent_rl_tool_challenge_train.jsonl")
    parser.add_argument("--eval_output", default="dataset/agent_rl_tool_challenge_eval.jsonl")
    parser.add_argument("--manifest", default="out/run_meta/agent_tool_challenge_manifest.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per_category", type=int, default=80)
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    args = parser.parse_args()
    if args.per_category < 10 or not 0 < args.eval_ratio < 1:
        parser.error("per_category must be >=10 and eval_ratio must be in (0, 1)")

    candidates = build_candidates(args.seed, args.per_category)
    by_category = collections.defaultdict(list)
    for row in candidates:
        by_category[row["rlvr_category"]].append(row)
    train_rows, eval_rows = [], []
    for category, rows in sorted(by_category.items()):
        rows.sort(key=lambda row: stable_hash(row["conversations"][1]["content"] + str(args.seed)))
        eval_count = max(1, round(len(rows) * args.eval_ratio))
        eval_rows.extend(rows[:eval_count])
        train_rows.extend(rows[eval_count:])
    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed + 1).shuffle(eval_rows)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)

    train_questions = {stable_hash(row["conversations"][1]["content"]) for row in train_rows}
    eval_questions = {stable_hash(row["conversations"][1]["content"]) for row in eval_rows}
    if train_questions & eval_questions:
        raise RuntimeError("challenge train/eval question overlap")
    manifest = {
        "generator": os.path.abspath(__file__),
        "seed": args.seed,
        "per_category_requested": args.per_category,
        "eval_ratio": args.eval_ratio,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "train_category_counts": dict(collections.Counter(row["rlvr_category"] for row in train_rows)),
        "eval_category_counts": dict(collections.Counter(row["rlvr_category"] for row in eval_rows)),
        "question_overlap": 0,
        "agent_sft_oracle_rows_used": 0,
        "train_sha256": file_hash(args.train_output),
        "eval_sha256": file_hash(args.eval_output),
        "environment_source": "trainer/train_agent.py::TOOLS/MOCK_RESULTS/CHECK_ARGS",
        "purpose": "non-saturated compositional Tool-Use RLVR; oracle fields are audit-only",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
