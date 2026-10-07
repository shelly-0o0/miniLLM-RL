"""Shared safe calculator and exact-answer verifier for math RL tasks."""

from __future__ import annotations

import ast
import json
import math
import re
from fractions import Fraction
from typing import Mapping


_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY = {ast.UAdd: lambda value: value, ast.USub: lambda value: -value}
_FINAL_PATTERNS = (
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"(?:final\s+answer|answer|最终答案|答案)\s*[:：=]\s*([^\n]+)", re.I),
    re.compile(r"\\boxed\{([^{}]+)\}"),
)
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?")


def safe_calculate(expression_or_args: str | Mapping[str, object]) -> dict[str, str]:
    expression = (
        expression_or_args.get("expression", "")
        if isinstance(expression_or_args, Mapping)
        else expression_or_args
    )
    expression = (str(expression).strip().replace("^", "**").replace("×", "*")
                  .replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"))
    if not expression or len(expression) > 256:
        raise ValueError("empty or overlong expression")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node, depth=0):
        if depth > 32:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            value = _UNARY[type(node.op)](evaluate(node.operand, depth + 1))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent exceeds safety limit")
            value = _BINARY[type(node.op)](left, right)
        else:
            raise ValueError(f"unsupported arithmetic node: {type(node).__name__}")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or abs(value) > 1e15:
            raise ValueError("non-finite or oversized result")
        return value

    value = evaluate(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"result": str(value)}


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for payload in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        try:
            call = json.loads(payload.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments", {}), dict):
            calls.append(call)
    return calls


def extract_final_answer(text: str) -> str | None:
    """Extract an explicit final answer, falling back to the last number."""

    region = text.rsplit("</tool_call>", 1)[-1]
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(region)
        if matches:
            numbers = _NUMBER.findall(matches[-1])
            return numbers[-1] if numbers else matches[-1].strip().rstrip(".。")
    numbers = _NUMBER.findall(region)
    return numbers[-1] if numbers else None


def canonical_number(value: object) -> Fraction:
    text = str(value).strip().replace(",", "").replace("$", "")
    if text.endswith("%"):
        return Fraction(text[:-1]) / 100
    return Fraction(text)


def verify_answer(prediction: str, gold_answer: object) -> bool:
    candidate = extract_final_answer(prediction)
    if candidate is None:
        return False
    try:
        return canonical_number(candidate) == canonical_number(gold_answer)
    except (ValueError, ZeroDivisionError):
        return candidate.strip().casefold() == str(gold_answer).strip().casefold()
