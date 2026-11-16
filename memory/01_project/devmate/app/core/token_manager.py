"""Token counting and context-budget math for DevMate.

All counts use tiktoken's cl100k_base. That is an ESTIMATE for the Groq/OpenRouter models DevMate will call (Llama, Qwen, ...): they use their own tokenizers. The provider's reported usage is ground truth; phase 6 measures the gap.
"""
from collections import defaultdict

import tiktoken

ENCODING_NAME = "cl100k_base"
DEFAULT_MESSAGE_OVERHEAD = 4  # heuristic: role + formatting tokens per message, not an API rule

_encoding = tiktoken.get_encoding(ENCODING_NAME)


def encode_text(text: str) -> list[int]:
    # disallowed_special=(): user text is untrusted; "<|endoftext|>" is just text here
    return _encoding.encode(text, disallowed_special=())


def count_tokens(text: str) -> int:
    return len(encode_text(text))

def _message_tokens(message: dict) -> int:
    # Precomputed count wins (phase 2 stores token_count once, at append).
    if "token_count" in message:
        return message["token_count"]
    return count_tokens(message["content"])


def count_message_tokens(messages: list[dict], per_message_overhead: int = DEFAULT_MESSAGE_OVERHEAD) -> int:
    return sum(_message_tokens(m) + per_message_overhead for m in messages)

def _check_int(name: str, value) -> None:
    # bool is a subclass of int: isinstance(True, int) is True. Reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def context_usage(
    messages: list[dict],
    max_context: int,
    reserved_output: int,
    per_message_overhead: int = DEFAULT_MESSAGE_OVERHEAD,
) -> dict:
    _check_int("max_context", max_context)
    _check_int("reserved_output", reserved_output)
    if reserved_output >= max_context:
        raise ValueError(f"reserved_output ({reserved_output}) must be < max_context ({max_context})")

    by_role: dict[str, int] = defaultdict(int)
    for m in messages:
        by_role[m["role"]] += _message_tokens(m) + per_message_overhead

    input_tokens = sum(by_role.values())
    used = input_tokens + reserved_output          # total allocation, output included
    input_budget = max_context - reserved_output
    return {
        "max_context": max_context,
        "reserved_output": reserved_output,
        "input_tokens": input_tokens,
        "used": used,
        "remaining": max_context - used,           # negative = overflow
        "utilization_pct": round(100 * used / max_context, 1),
        "input_budget": input_budget,
        "input_utilization_pct": round(100 * input_tokens / input_budget, 1),
        "by_role": dict(by_role),
    }

def print_usage_report(usage: dict) -> None:
    w = 18
    print(f"Context Budget: {usage['max_context']} tokens\n")
    for role, n in usage["by_role"].items():
        print(f"{role + ':':<{w}}{n:>6}")
    print(f"{'reserved output:':<{w}}{usage['reserved_output']:>6}")
    print("-" * 26)
    print(f"{'Total used:':<{w}}{usage['used']:>6}")
    print(f"{'Remaining:':<{w}}{usage['remaining']:>6}")
    print(f"{'Utilization:':<{w}}{usage['utilization_pct']:>5}%")
    print(f"{'Input utilization:':<{w}}{usage['input_utilization_pct']:>5}%"
          f"  ({usage['input_tokens']}/{usage['input_budget']})")
    if usage["remaining"] < 0:
        print(f"\nOVERFLOW by {-usage['remaining']} tokens")


if __name__ == "__main__":
    note_example = [
        {"role": "system", "content": "", "token_count": 500},
        {"role": "conversation", "content": "", "token_count": 3500},
        {"role": "query", "content": "", "token_count": 200},
    ]
    print_usage_report(context_usage(note_example, 8000, 1000, per_message_overhead=0))
