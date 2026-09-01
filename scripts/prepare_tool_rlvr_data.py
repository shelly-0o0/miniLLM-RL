"""Build a deterministic, executable Tool-Use RLVR dataset for MiniMind.

The upstream ``agent_rl.jsonl`` contains 20k verifiable calculator examples
and about 20k ordinary conversations without tools or ground truth.  Calling
the latter a Tool-Use benchmark would make strict success identically zero.
This script therefore creates a small controlled benchmark directly against
the deterministic tool sandbox implemented in ``trainer/train_agent.py``.

Every row has (1) at least one available tool, (2) non-empty ground truth,
(3) arguments accepted by the sandbox, and (4) an exact expected execution
result.  Train/eval prompts are generated from disjoint parameter tuples and
the manifest records hashes and category counts for auditability.
"""

import argparse
import collections
import hashlib
import json
import os
import random


TOOLS = {
    "calculate_math": {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    "unit_converter": {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    "get_current_weather": {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    "get_current_time": {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    "get_exchange_rate": {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    "translate_text": {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
}

WEATHER = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIMES = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
RATES = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATIONS = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNITS = {("km", "miles"): 0.621371, ("miles", "km"): 1.60934, ("kg", "pounds"): 2.20462, ("pounds", "kg"): 0.453592, ("meters", "feet"): 3.28084, ("feet", "meters"): 0.3048}


def stable_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_row(
    question, tool_names, ground_truth, category, required_tools=None,
    oracle_calls=None, oracle_observations=None,
):
    tools = [TOOLS[name] for name in tool_names]
    return {
        "conversations": [
            {"role": "system", "content": "你是一个可靠的工具助手。需要外部结果时必须调用提供的工具，并根据 Observation 给出最终答案。", "tools": json.dumps(tools, ensure_ascii=False)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": ""},
        ],
        "gt": [str(value) for value in ground_truth],
        "required_tools": list(required_tools or tool_names[:1]),
        "rlvr_category": category,
        # The RL loader deliberately ignores these fields.  They are retained
        # only so a disjoint, auditable Agent-SFT cold-start set can be built
        # without asking another model to synthesize unverifiable trajectories.
        "oracle_calls": list(oracle_calls or []),
        "oracle_observations": list(oracle_observations or []),
    }


def build_candidates(seed, per_category):
    rng = random.Random(seed)
    rows = []
    contexts = ["请直接给出结果", "我正在核对数据，请帮忙", "请调用合适的工具后回答", "不要猜测，请查询后回答", "请完成下面的任务"]

    # Arithmetic has a large parameter space and checks exact calculator use.
    for index in range(per_category):
        a, b, c = rng.randint(11, 999), rng.randint(2, 99), rng.randint(2, 40)
        expr = f"({a}+{b})*{c}"
        result = str((a + b) * c)
        rows.append(make_row(
            f"{contexts[index % len(contexts)]}：计算 {expr}。",
            ["calculate_math", "get_current_weather"], [result], "calculator",
            oracle_calls=[{"name": "calculate_math", "arguments": {"expression": expr}}],
            oracle_observations=[{"result": result}],
        ))

    unit_pairs = list(UNITS)
    for index in range(per_category):
        source, target = unit_pairs[index % len(unit_pairs)]
        value = 1 + ((index * 17 + seed) % 499)
        result = round(value * UNITS[(source, target)], 4)
        rows.append(make_row(
            f"{contexts[(index + 1) % len(contexts)]}：把 {value} {source} 换算成 {target}。",
            ["unit_converter", "translate_text"], [result], "unit_conversion",
            oracle_calls=[{"name": "unit_converter", "arguments": {"value": value, "from_unit": source, "to_unit": target}}],
            oracle_observations=[{"result": result}],
        ))

    cities = list(WEATHER)
    for index in range(per_category):
        city = cities[index % len(cities)]
        qualifier = ["出门前", "制定行程时", "安排会议前", "准备运动时", "收拾行李前"][(index // len(cities)) % 5]
        answer_style = ["请分两项回答", "请简洁回答", "请以查询结果为准"][(index // (len(cities) * 5)) % 3]
        temperature, condition = WEATHER[city]
        observation = {"city": city, "temperature": temperature, "humidity": "65%", "condition": condition}
        rows.append(make_row(
            f"我在{qualifier}需要确认 {city} 的天气；请查询并告诉我温度和天气状况，{answer_style}。",
            ["get_current_weather", "get_current_time"], [temperature, condition], "weather",
            oracle_calls=[{"name": "get_current_weather", "arguments": {"location": city}}],
            oracle_observations=[observation],
        ))

    zones = list(TIMES)
    for index in range(per_category):
        zone = zones[index % len(zones)]
        purpose = ["跨国会议", "航班确认", "线上答辩", "远程协作", "家人通话"][(index // len(zones)) % 5]
        answer_style = ["请保留完整时间", "请不要估算", "请直接返回查询值", "请注明时区", "请先查询再回答", "请核对日期"][(index // (len(zones) * 5)) % 6]
        rows.append(make_row(
            f"为了安排{purpose}，请查询 {zone} 的当前日期和时间；{answer_style}。",
            ["get_current_time", "get_exchange_rate"], [TIMES[zone]], "time",
            oracle_calls=[{"name": "get_current_time", "arguments": {"timezone": zone}}],
            oracle_observations=[{"datetime": TIMES[zone], "timezone": zone}],
        ))

    rate_pairs = list(RATES)
    for index in range(per_category):
        source, target = rate_pairs[index % len(rate_pairs)]
        purpose = ["旅行预算", "财务核对", "采购估算", "报销准备", "价格比较"][(index // len(rate_pairs)) % 5]
        answer_style = ["请返回汇率数值", "请调用工具核对", "不要使用记忆值", "请给出查询结果"][(index // (len(rate_pairs) * 5)) % 4]
        rate = RATES[(source, target)]
        rows.append(make_row(
            f"我在做{purpose}，请查询 {source} 到 {target} 的汇率；{answer_style}。",
            ["get_exchange_rate", "unit_converter"], [rate], "exchange",
            oracle_calls=[{"name": "get_exchange_rate", "arguments": {"from_currency": source, "to_currency": target}}],
            oracle_observations=[{"from": source, "to": target, "rate": rate}],
        ))

    translation_pairs = list(TRANSLATIONS)
    for index in range(per_category):
        text, language = translation_pairs[index % len(translation_pairs)]
        purpose = ["邮件", "课程作业", "旅行卡片", "产品文案", "会议材料"][(index // len(translation_pairs)) % 5]
        answer_style = ["请保留原意", "请只给译文", "请调用翻译工具", "请核对后回答", "不要音译", "请简洁回答"][(index // (len(translation_pairs) * 5)) % 6]
        translated = TRANSLATIONS[(text, language)]
        rows.append(make_row(
            f"请为我的{purpose}把“{text}”翻译成 {language}；{answer_style}。",
            ["translate_text", "calculate_math"], [translated], "translation",
            oracle_calls=[{"name": "translate_text", "arguments": {"text": text, "target_language": language}}],
            oracle_observations=[{"translated_text": translated}],
        ))

    # Two independent calls in one trajectory test multi-tool execution and a
    # final answer that must integrate both observations.
    for index in range(per_category):
        city = cities[index % len(cities)]
        zone = zones[(index // len(cities)) % len(zones)]
        purpose = ["远程沟通", "出差安排", "线上会议"][(index // (len(cities) * len(zones))) % 3]
        temperature, _ = WEATHER[city]
        rows.append(make_row(
            f"我要为{purpose}核对两项信息：请同时查询 {city} 的天气温度，以及 {zone} 的当前时间。",
            ["get_current_weather", "get_current_time", "translate_text"],
            [temperature, TIMES[zone]], "multi_tool",
            required_tools=["get_current_weather", "get_current_time"],
            oracle_calls=[
                {"name": "get_current_weather", "arguments": {"location": city}},
                {"name": "get_current_time", "arguments": {"timezone": zone}},
            ],
            oracle_observations=[
                {"city": city, "temperature": temperature, "humidity": "65%", "condition": WEATHER[city][1]},
                {"datetime": TIMES[zone], "timezone": zone},
            ],
        ))

    # Template/entity combinations can collide.  Keeping one copy prevents
    # exact-question leakage while retaining balanced executable categories.
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
    parser = argparse.ArgumentParser(description="Create deterministic Tool-Use RLVR train/eval data")
    parser.add_argument("--train_output", default="dataset/agent_rl_tool_verified_train.jsonl")
    parser.add_argument("--eval_output", default="dataset/agent_rl_tool_verified_eval.jsonl")
    parser.add_argument("--manifest", default="out/run_meta/agent_tool_verified_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_category", type=int, default=160)
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
    assert train_questions.isdisjoint(eval_questions)
    assert all(
        row["gt"] and row["conversations"][0].get("tools")
        and len(row["oracle_calls"]) == len(row["oracle_observations"])
        for row in train_rows + eval_rows
    )
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
        "train_sha256": file_hash(args.train_output),
        "eval_sha256": file_hash(args.eval_output),
        "environment_source": "trainer/train_agent.py::TOOLS/MOCK_RESULTS/CHECK_ARGS",
        "limitations": "Deterministic in-process mock tools; not a networked or adversarial production sandbox.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
