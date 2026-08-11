from __future__ import annotations

import math
import re

from .schema import LogStep


def make_chunks(steps: list[LogStep], chunk_count: int) -> list[list[LogStep]]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    n = len(steps)
    if n == 0:
        return []
    chunks: list[list[LogStep]] = []
    for i in range(chunk_count):
        start = round(i * n / chunk_count)
        end = round((i + 1) * n / chunk_count)
        chunk = steps[start:end]
        if chunk:
            chunks.append(chunk)
    return chunks


def split_long_steps(steps: list[LogStep], max_chars: int) -> list[LogStep]:
    if max_chars <= 0:
        return steps
    out: list[LogStep] = []
    for step in steps:
        if len(step.content) <= max_chars:
            out.append(step)
            continue
        parts = split_text(step.content, max_chars)
        total = len(parts)
        for idx, part in enumerate(parts, 1):
            out.append(
                LogStep(
                    step=step.step,
                    agent=step.agent,
                    content=f"[fragment {idx}/{total} of original step {step.step}]\n{part}",
                )
            )
    return out


def make_budgeted_chunks(steps: list[LogStep], chunk_count: int, max_chunk_chars: int) -> list[list[LogStep]]:
    base_chunks = make_chunks(steps, chunk_count)
    if max_chunk_chars <= 0:
        return base_chunks
    return split_chunks_by_char_budget(base_chunks, max_chunk_chars)


def make_adaptive_budgeted_chunks(
    steps: list[LogStep],
    *,
    target_chunk_tokens: int,
    target_chunk_steps: int,
    short_step_threshold: int,
    short_token_threshold: int,
    max_chunks: int,
    max_chunk_chars: int,
    chunk_count_basis: str = "tokens",
) -> list[list[LogStep]]:
    if not steps:
        return []

    total_tokens = estimate_steps_tokens(steps)
    chunk_count_basis = (chunk_count_basis or "tokens").strip().lower()
    if chunk_count_basis == "tokens":
        short_trace = total_tokens <= short_token_threshold
    else:
        short_trace = len(steps) <= short_step_threshold and total_tokens <= short_token_threshold

    if short_trace:
        # Short traces must preserve the base method's all-at-once context.
        # Character-budget splitting is only a long-trace safety valve; applying
        # it here can accidentally route compact traces through the chunked path.
        return [steps]

    token_chunks = math.ceil(total_tokens / max(1, target_chunk_tokens))
    step_chunks = math.ceil(len(steps) / max(1, target_chunk_steps))
    if chunk_count_basis == "tokens":
        base_chunks = make_token_budget_chunks(steps, target_chunk_tokens)
        if max_chunks > 0 and len(base_chunks) > max_chunks:
            base_chunks = make_token_balanced_chunks(steps, max_chunks)
        if max_chunk_chars <= 0:
            return base_chunks
        return split_chunks_by_char_budget(base_chunks, max_chunk_chars)
    elif chunk_count_basis == "steps":
        chunk_count = max(2, step_chunks)
    else:
        chunk_count = max(2, token_chunks, step_chunks)
    chunk_count = min(max(1, max_chunks), chunk_count, len(steps))
    base_chunks = make_token_balanced_chunks(steps, chunk_count)

    if max_chunk_chars <= 0:
        return base_chunks
    return split_chunks_by_char_budget(base_chunks, max_chunk_chars)


def make_token_budget_chunks(steps: list[LogStep], target_chunk_tokens: int) -> list[list[LogStep]]:
    if not steps:
        return []
    target_tokens = max(1, target_chunk_tokens)
    chunks: list[list[LogStep]] = []
    current: list[LogStep] = []
    current_tokens = 0

    for step in steps:
        step_tokens = estimate_step_tokens(step)
        if current and current_tokens + step_tokens > target_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(step)
        current_tokens += step_tokens

    if current:
        chunks.append(current)
    return chunks


def make_token_balanced_chunks(steps: list[LogStep], chunk_count: int) -> list[list[LogStep]]:
    if chunk_count <= 1 or len(steps) <= 1:
        return [steps] if steps else []
    if chunk_count >= len(steps):
        return [[step] for step in steps]

    total_tokens = estimate_steps_tokens(steps)
    target_tokens = max(1, math.ceil(total_tokens / chunk_count))
    chunks: list[list[LogStep]] = []
    current: list[LogStep] = []
    current_tokens = 0

    for idx, step in enumerate(steps):
        remaining_steps = len(steps) - idx
        remaining_chunks = chunk_count - len(chunks)
        step_tokens = estimate_step_tokens(step)
        must_leave_steps = remaining_steps <= remaining_chunks
        if current and not must_leave_steps and current_tokens + step_tokens > target_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(step)
        current_tokens += step_tokens

    if current:
        chunks.append(current)
    return chunks


def split_chunks_by_char_budget(chunks: list[list[LogStep]], max_chunk_chars: int) -> list[list[LogStep]]:
    out: list[list[LogStep]] = []
    for chunk in chunks:
        current: list[LogStep] = []
        current_chars = 0
        for step in chunk:
            step_chars = len(step.content) + len(step.agent) + 32
            if current and current_chars + step_chars > max_chunk_chars:
                out.append(current)
                current = []
                current_chars = 0
            current.append(step)
            current_chars += step_chars
        if current:
            out.append(current)
    return out


def estimate_steps_tokens(steps: list[LogStep]) -> int:
    return sum(estimate_step_tokens(step) for step in steps)


def estimate_step_tokens(step: LogStep) -> int:
    return estimate_text_tokens(step.content) + estimate_text_tokens(step.agent) + 8


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\S+", text)))


def split_text(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary
        parts.append(text[start:end])
        start = end
    return parts


def expand_region(steps: list[LogStep], chunk: list[LogStep], overlap_steps: int) -> list[LogStep]:
    if not chunk:
        return []
    start_idx = find_step_index(steps, chunk[0], prefer_first=True)
    end_idx = find_step_index(steps, chunk[-1], prefer_first=False)
    start_idx = max(0, start_idx - overlap_steps)
    end_idx = min(len(steps) - 1, end_idx + overlap_steps)
    return steps[start_idx : end_idx + 1]


def sliding_windows(region: list[LogStep], window_size: int, stride: int) -> list[list[LogStep]]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if len(region) <= window_size:
        return [region]
    windows: list[list[LogStep]] = []
    start = 0
    while start < len(region):
        end = start + window_size
        window = region[start:end]
        if window:
            windows.append(window)
        if end >= len(region):
            break
        start += stride
    return windows


def steps_before_after(all_steps: list[LogStep], window: list[LogStep], context_steps: int) -> tuple[list[LogStep], list[LogStep]]:
    if not window:
        return [], []
    start_idx = find_step_index(all_steps, window[0], prefer_first=True)
    end_idx = find_step_index(all_steps, window[-1], prefer_first=False)
    before = all_steps[max(0, start_idx - context_steps) : start_idx]
    after = all_steps[end_idx + 1 : min(len(all_steps), end_idx + 1 + context_steps)]
    return before, after


def find_step_index(steps: list[LogStep], target: LogStep, prefer_first: bool) -> int:
    indices = [idx for idx, step in enumerate(steps) if step is target]
    if not indices:
        indices = [idx for idx, step in enumerate(steps) if step.step == target.step and step.agent == target.agent and step.content == target.content]
    if not indices:
        indices = [idx for idx, step in enumerate(steps) if step.step == target.step]
    if not indices:
        raise ValueError(f"Step not found in working log: {target.step}")
    return indices[0] if prefer_first else indices[-1]


def render_steps(steps: list[LogStep]) -> str:
    parts: list[str] = []
    for step in steps:
        parts.append(f"Step {step.step}\nAgent: {step.agent}\nContent:\n{step.content}")
    return "\n\n".join(parts)


def summarize_chunk(chunk: list[LogStep]) -> str:
    if not chunk:
        return "None"
    agents = ", ".join(sorted({s.agent for s in chunk}))
    first = compact(chunk[0].content)
    last = compact(chunk[-1].content)
    return (
        f"Steps {chunk[0].step}-{chunk[-1].step}; "
        f"Agents: {agents}; First: {first}; Last: {last}"
    )


def compact(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
