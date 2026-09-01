"""Create executable multi-turn cold-start trajectories for Agent RLVR.

Only training splits are consumed.  The generated SFT rows teach the policy
the chat-template/tool protocol before sparse verifiable rewards are applied;
RL evaluation questions remain disjoint and are never converted to labels.
"""

import argparse
import ast
import collections
import hashlib
import json
import os
import random
import re


ARITHMETIC = re.compile(r"(?<![A-Za-z0-9_])[-+]?\s*(?:\([^\n,;，；。]+\)|\d+(?:\.\d+)?)\s*(?:\*\*|[+\-*/^\xd7\xf7])\s*[^,;，；。的同顺请\s]*(?:\s*(?:\*\*|[+\-*/^\xd7\xf7])\s*[^,;，；。的同顺请\s]*)*")
OPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_message(call):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "",
        "tool_calls": json.dumps([call], ensure_ascii=False),
    }


def final_text(values):
    return "工具执行结果：" + "；".join(str(value) for value in values) + "。"


def safe_number(expression):
    tree = ast.parse(
        expression.replace("^", "**").replace("×", "*").replace("÷", "/"),
        mode="eval",
    )

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError(type(node).__name__)

    return float(visit(tree))


def from_verified_tool(row):
    calls = row.get("oracle_calls") or []
    observations = row.get("oracle_observations") or []
    if not calls or len(calls) != len(observations):
        return None
    prefix = [dict(message) for message in row["conversations"][:-1]]
    # Emit all independent calls in one assistant action.  The template groups
    # consecutive tool observations into the following user observation turn.
    prefix.append({
        "role": "assistant", "content": "", "reasoning_content": "",
        "tool_calls": json.dumps(calls, ensure_ascii=False),
    })
    prefix.extend({"role": "tool", "content": json.dumps(obs, ensure_ascii=False)} for obs in observations)
    prefix.append({"role": "assistant", "content": final_text(row["gt"]), "reasoning_content": ""})
    return {"conversations": prefix}


def extract_expressions(question):
    expressions = []
    for match in ARITHMETIC.finditer(question.replace(" ", "")):
        value = match.group(0).strip().rstrip("=?")
        if any(op in value for op in ("+", "-", "*", "/", "^", "×", "÷")):
            expressions.append(value)
    return expressions


def from_math(row):
    answers = [str(value) for value in row.get("gt") or []]
    question = row["conversations"][-2]["content"]
    expressions = extract_expressions(question)
    if len(expressions) != len(answers):
        return None
    try:
        # Oracle tool arguments must themselves reproduce every stored answer;
        # a merely matching regex count is not sufficient evidence.
        if any(abs(safe_number(expr) - float(answer)) > 1e-6 for expr, answer in zip(expressions, answers)):
            return None
    except (ArithmeticError, SyntaxError, TypeError, ValueError):
        return None
    prefix = [dict(message) for message in row["conversations"][:-1]]
    calls = [{"name": "calculate_math", "arguments": {"expression": expr}} for expr in expressions]
    prefix.append({
        "role": "assistant", "content": "", "reasoning_content": "",
        "tool_calls": json.dumps(calls, ensure_ascii=False),
    })
    prefix.extend(
        {"role": "tool", "content": json.dumps({"result": answer}, ensure_ascii=False)}
        for answer in answers
    )
    prefix.append({"role": "assistant", "content": final_text(answers), "reasoning_content": ""})
    return {"conversations": prefix}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def balanced_limit(rows, limit, seed):
    """Select a deterministic category-balanced train-only curriculum subset."""

    rows = list(rows)
    if limit <= 0 or limit >= len(rows):
        return rows
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[row.get("rlvr_category", "uncategorized")].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    categories = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for category in categories:
            if buckets[category] and len(selected) < limit:
                selected.append(buckets[category].pop())
                progressed = True
        if not progressed:
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_train", default="dataset/agent_rl_tool_verified_train.jsonl")
    parser.add_argument("--math_train", default="dataset/agent_rl_math_train.jsonl")
    parser.add_argument(
        "--tool_limit", type=int, default=0,
        help="0使用全部tool训练轨迹；正数按rlvr_category均衡抽取",
    )
    parser.add_argument("--math_limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset/agent_sft_coldstart.jsonl")
    parser.add_argument("--manifest", default="out/run_meta/agent_sft_coldstart_manifest.json")
    args = parser.parse_args()

    raw_tool_rows = balanced_limit(read_jsonl(args.tool_train), args.tool_limit, args.seed)
    tool_category_counts = collections.Counter(
        row.get("rlvr_category", "uncategorized") for row in raw_tool_rows
    )
    tool_rows = [
        converted for row in raw_tool_rows
        if (converted := from_verified_tool(row))
    ]
    math_candidates = [converted for row in read_jsonl(args.math_train) if (converted := from_math(row))]
    rng = random.Random(args.seed)
    rng.shuffle(math_candidates)
    math_rows = math_candidates[:args.math_limit]
    rows = tool_rows + math_rows
    rng.shuffle(rows)
    if not tool_rows or len(math_rows) < args.math_limit:
        raise RuntimeError(f"insufficient oracle rows: tool={len(tool_rows)}, math={len(math_rows)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "output": os.path.abspath(args.output),
        "seed": args.seed,
        "tool_train_rows": len(tool_rows),
        "tool_limit": args.tool_limit,
        "tool_category_counts": dict(sorted(tool_category_counts.items())),
        "math_train_rows": len(math_rows),
        "total_rows": len(rows),
        "eval_rows_used": 0,
        "sha256": sha256(args.output),
        "purpose": "Agent tool-protocol cold start before sparse-reward RLVR",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
