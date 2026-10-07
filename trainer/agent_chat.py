"""Shared chat-template helpers for MiniMind and Hugging Face agent models.

The Stage 1 and Stage 2 policies must see exactly the same logical trajectory:
system prompt, user question, assistant tool call, tool observation, and final
assistant answer.  This module keeps the serialization boundary explicit and
provides an assistant-only SFT mask without relying on model-specific token
IDs such as ``<|im_start|>assistant``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Iterable


def normalize_agent_messages(
    conversations: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return chat-template messages and the tools embedded in the system row.

    The prepared JSONL intentionally stores ``tools`` on the system message so
    every row is self-contained.  Hugging Face chat templates expect tools as
    a separate argument, so the field is removed from the message copy.  JSON
    encoded tool calls are decoded here as well.
    """

    messages: list[dict[str, Any]] = []
    tools = None
    for raw_message in conversations:
        message = dict(raw_message)
        raw_tools = message.pop("tools", None)
        if message.get("role") == "system" and raw_tools:
            tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, str) and raw_calls:
            message["tool_calls"] = json.loads(raw_calls)
        # Sparse JSON records often contain null optional fields.  Some Jinja
        # templates distinguish an absent field from an explicitly null one.
        message = {key: value for key, value in message.items() if value is not None}
        messages.append(message)
    return messages, tools


def render_agent_chat(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    tokenize: bool = False,
    open_thinking: bool = False,
):
    """Render one agent trajectory with explicit thinking-mode control.

    MiniMind's template consumes ``open_thinking`` while official Qwen3
    templates consume ``enable_thinking``.  ``apply_chat_template`` forwards
    extra keyword arguments to Jinja, so passing both keeps one call site for
    both model families and prevents Qwen3's default thinking mode from
    silently changing the Stage 2 protocol.
    """

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "The tokenizer has no chat_template. Stage 2 requires a template "
            "that serializes tools, tool calls, and tool observations."
        )
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        open_thinking=open_thinking,
        enable_thinking=open_thinking,
    )


def resolve_agent_open_thinking(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    requested_open_thinking: bool,
) -> bool:
    """Align Qwen's generation prompt with its completed SFT serialization.

    Qwen3's tool template has a context-sensitive asymmetry when thinking is
    disabled: ``add_generation_prompt=True`` inserts an empty thinking block,
    while a completed first assistant tool-call turn contains no such block.
    The model would therefore be trained to emit ``<tool_call>`` immediately
    after the assistant header but asked to emit it after an unseen empty
    thinking block at rollout time.

    For a Qwen tool template, omit that prefilled block on the first assistant
    turn (``enable_thinking=True`` in the template means it does not prefill
    anything).  After a tool observation, keep disabled-thinking behavior so
    the final-answer prompt matches the completed SFT final turn.  Explicitly
    requested thinking and non-Qwen templates retain their previous behavior.
    """

    if requested_open_thinking:
        return True
    template = str(getattr(tokenizer, "chat_template", "") or "")
    is_context_sensitive_qwen_tool_template = all(
        marker in template
        for marker in ("<|im_start|>", "<tool_call>", "enable_thinking")
    )
    if not is_context_sensitive_qwen_tool_template:
        return False

    last_user = max(
        (index for index, message in enumerate(messages)
         if message.get("role") == "user"),
        default=-1,
    )
    has_tool_observation = any(
        message.get("role") == "tool"
        for message in messages[last_user + 1 :]
    )
    return not has_tool_observation


def audit_agent_generation_prefixes(
    tokenizer,
    conversations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify that each rollout prompt is a prefix of its SFT trajectory."""

    messages, tools = normalize_agent_messages(conversations)
    full_text = render_agent_chat(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
        open_thinking=False,
    )
    turns = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        preceding = messages[:index]
        open_thinking = resolve_agent_open_thinking(
            tokenizer,
            preceding,
            requested_open_thinking=False,
        )
        prompt = render_agent_chat(
            tokenizer,
            preceding,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            open_thinking=open_thinking,
        )
        if not full_text.startswith(prompt):
            raise ValueError(
                f"assistant turn {len(turns)} inference prompt is not an SFT prefix"
            )
        turns.append({
            "message_index": index,
            "open_thinking": open_thinking,
            "prompt_characters": len(prompt),
        })
    if not turns:
        raise ValueError("trajectory has no assistant turn to audit")
    return {"assistant_turns": len(turns), "turns": turns}


def _as_token_list(value) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized chat sequence")
        value = value[0]
    return list(value)


def _qwen_assistant_only_labels(
    tokenizer,
    full_text: str,
    full_ids: list[int],
    *,
    expected_assistant_messages: int,
) -> list[int]:
    """Build labels from Qwen's explicit role delimiters.

    Qwen3's tool-aware template is intentionally context sensitive: an
    assistant generation prompt contains an empty thinking block, while the
    same historical tool-call turn in a completed conversation does not.  A
    prefix-rendering mask is therefore not valid for tool trajectories.  The
    completed serialization still has unambiguous ``im_start``/``im_end``
    assistant blocks, so use fast-tokenizer character offsets to select only
    their payload and closing end token.
    """

    assistant_header = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        header_start = full_text.find(assistant_header, cursor)
        if header_start < 0:
            break
        payload_start = header_start + len(assistant_header)
        marker_start = full_text.find(end_marker, payload_start)
        if marker_start < 0:
            raise ValueError("unterminated Qwen assistant block in chat template")
        spans.append((payload_start, marker_start + len(end_marker)))
        cursor = marker_start + len(end_marker)
    if not spans:
        raise ValueError("Qwen chat serialization contains no assistant block")
    if len(spans) != expected_assistant_messages:
        raise ValueError(
            "Qwen assistant-block count does not match conversation structure"
        )

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offset_ids = _as_token_list(encoded)
    if offset_ids != full_ids:
        raise ValueError(
            "Qwen offset tokenization differs from apply_chat_template output"
        )
    offsets = encoded.get("offset_mapping")
    if offsets is None or len(offsets) != len(full_ids):
        raise ValueError("Qwen assistant masking requires fast-tokenizer offsets")

    labels = [-100] * len(full_ids)
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        for span_start, span_end in spans:
            overlaps = token_end > span_start and token_start < span_end
            contained = token_start >= span_start and token_end <= span_end
            if overlaps and not contained:
                raise ValueError(
                    "a tokenizer token crosses a Qwen assistant-mask boundary"
                )
            if contained:
                labels[token_index] = full_ids[token_index]
                break
    return labels


def build_assistant_only_example(
    tokenizer,
    conversations: list[dict[str, Any]],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize a full SFT trajectory and supervise assistant spans only.

    For every assistant message, rendering the preceding messages with
    ``add_generation_prompt=True`` locates the first assistant action token.
    Rendering through the assistant message locates its closing boundary.
    Prefix equality is checked instead of assumed; a template whose output
    depends on future messages is rejected because it would make the mask
    ambiguous and could train on user/tool tokens.
    """

    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    messages, tools = normalize_agent_messages(conversations)
    full_ids = _as_token_list(
        render_agent_chat(
            tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            open_thinking=False,
        )
    )
    full_text = render_agent_chat(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
        open_thinking=False,
    )
    labels = [-100] * len(full_ids)
    assistant_messages = sum(
        message.get("role") == "assistant" for message in messages
    )
    prefix_stable = True
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        before_ids = _as_token_list(
            render_agent_chat(
                tokenizer,
                messages[:index],
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                open_thinking=False,
            )
        )
        through_ids = _as_token_list(
            render_agent_chat(
                tokenizer,
                messages[: index + 1],
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                open_thinking=False,
            )
        )
        if full_ids[: len(before_ids)] != before_ids:
            prefix_stable = False
            break
        if full_ids[: len(through_ids)] != through_ids:
            prefix_stable = False
            break
        if len(through_ids) <= len(before_ids):
            raise ValueError("assistant message produced no supervised tokens")
        labels[len(before_ids) : len(through_ids)] = full_ids[
            len(before_ids) : len(through_ids)
        ]

    if not assistant_messages:
        raise ValueError("SFT trajectory contains no assistant message")
    if not prefix_stable:
        labels = _qwen_assistant_only_labels(
            tokenizer,
            full_text,
            full_ids,
            expected_assistant_messages=assistant_messages,
        )
    input_ids = full_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError(
            "SFT truncation removed every assistant token; increase max_length"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
