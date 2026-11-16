# Phase 1 — Tokens: Counting, Budgets & Why Characters Lie

You can't manage a context window you can't measure. Before DevMate can decide what to keep, it needs to know what everything costs.

```text
            ONE MODEL CALL = ONE CONTEXT WINDOW (e.g. 8000 tokens)

  ┌──────────────────────────────────────────────────────────┐
  │ system prompt          500                               │
  │ conversation history  3500   ◄── grows every turn        │
  │ current query          200                               │
  ├──────────────────────────────────────────────────────────┤
  │ reserved output       1000   ◄── spent before you start  │
  ├──────────────────────────────────────────────────────────┤
  │ remaining             2800                               │
  └──────────────────────────────────────────────────────────┘
       used = 500 + 3500 + 200 + 1000 = 5200   →   65%
```

The rule for this phase:

```text
budget in tokens, never characters
reserved output is spent before the model writes a word
your counter is an estimate unless it is the model's own tokenizer
```

**The destination for the whole course:** DevMate, a working-memory engine that still knows the database decision from message 5 when it reaches message 100, stays under a hard token budget, and is traced in Langfuse. There are 8 phases, one per milestone in [`docs/MILESTONES.md`](../docs/MILESTONES.md). Phases 1–6 are the core: if you stop at 6, you have a hybrid memory engine you understand. Phases 7–8 add persistence and the CLI and evaluation. Phases 1–4 run offline. The first account you'll need (Groq/OpenRouter + Langfuse) comes at phase 5.

**Versions** (checked 2026-11-15): Python 3.12+ (tested on 3.14), **tiktoken 0.14.0**. Any tiktoken from 0.13 up works for this phase.

---

# 1. Set up the project

Everything lives under `01_project/devmate/`. Commands are for Windows PowerShell.

Run:

```powershell
cd D:\AI_learning\memory\01_project
mkdir devmate; cd devmate
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install tiktoken pytest python-dotenv
pip show tiktoken
```

If `Activate.ps1` fails with `running scripts is disabled on this system`, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Create:

```text
devmate/requirements.txt
```

```text
tiktoken>=0.13
pytest
python-dotenv
```

Only three packages. `pydantic` arrives in phase 2, and `openai` + `langfuse` in phase 5. Adding a dependency before the phase that needs it is how `requirements.txt` fills up with packages nobody remembers adding.

Create:

```text
devmate/.gitignore
```

```text
.venv/
__pycache__/
.env
data/
.pytest_cache/
```

Create the skeleton. The empty `__init__.py` files make `app` and `tests` importable packages:

```powershell
mkdir app\core, tests
New-Item app\__init__.py, app\core\__init__.py, tests\__init__.py -ItemType File
git init
```

```text
devmate/
├── app/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       └── token_manager.py      ← §3–§7
├── tests/
│   ├── __init__.py
│   └── test_token_manager.py     ← §8
├── requirements.txt
└── .gitignore
```

---

# 2. Build the wrong version first

The rule of thumb everyone repeats is "a token is about 4 characters". Test it before you trust it.

Create:

```text
devmate/scratch_ratio.py
```

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
samples = {
    "english": "I'm building an e-commerce application using React, FastAPI and PostgreSQL.",
    "python": "def add_to_cart(user_id: int, item_id: int) -> dict:\n    return {'ok': True, 'items': [item_id]}\n",
    "json": '{"id": "msg_001", "role": "user", "token_count": 12, "tags": ["db", "auth"]}',
    "hindi": "मैं React, FastAPI और PostgreSQL का उपयोग करके एक ई-कॉमर्स एप्लिकेशन बना रहा हूँ।",
}
for name, text in samples.items():
    real = len(enc.encode(text))
    guess = len(text) // 4
    print(f"{name:8} chars={len(text):3}  real={real:3}  len//4={guess:3}  chars/token={len(text)/real:.2f}")
```

Run:

```powershell
python scratch_ratio.py
```

Expected:

```text
english  chars= 75  real= 15  len//4= 18  chars/token=5.00
python   chars= 97  real= 31  len//4= 24  chars/token=3.13
json     chars= 76  real= 30  len//4= 19  chars/token=2.53
hindi    chars= 81  real= 64  len//4= 20  chars/token=1.27
```

Look at what the guess gets wrong:

```text
english   guess too HIGH   → wastes budget, harmless
python    guess too LOW    → 7 tokens short
json      guess too LOW    → 11 tokens short, ~37% under
hindi     guess too LOW    → 44 tokens short, ~3x under
```

A coding assistant's history is mostly code, JSON and tool output, which is exactly where the character estimate *undercounts*. An undercount means you believe the context fits when it doesn't. Then the API rejects the request, or the provider silently truncates it. DevMate's whole job is staying under a budget, so the counter has to be the real tokenizer.

Copy your four numbers into a comment at the top of `token_manager.py`. Then delete `scratch_ratio.py`.

---

# 3. encode_text and count_tokens

Create:

```text
devmate/app/core/token_manager.py
```

```python
"""Token counting and context-budget math for DevMate.

All counts use tiktoken's cl100k_base. That is an ESTIMATE for the Groq/OpenRouter
models DevMate will call (Llama, Qwen, ...): they use their own tokenizers. The
provider's reported usage is ground truth; phase 6 measures the gap.
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
```

**Why load the encoding at module level?** tiktoken already caches encodings internally, so the second `get_encoding` call costs about 4 microseconds. The expensive call is the *first* one:

```text
cold load (first call in a process)   ~0.26 s
every later call                      ~0.000004 s
first call EVER on a machine          downloads the BPE file from the internet
```

Loading at import time puts that cost and that network call at startup, where a failure is obvious. Otherwise they'd land in the middle of the first user request.

**That download is the failure mode.** The first time `cl100k_base` loads on a machine, tiktoken fetches `cl100k_base.tiktoken` from `openaipublic.blob.core.windows.net` and caches it. On a machine that is offline, behind a corporate proxy or in a locked-down container, you get:

```text
requests.exceptions.ProxyError: HTTPSConnectionPool(host='openaipublic.blob.core.windows.net', port=443):
Max retries exceeded with url: /encodings/cl100k_base.tiktoken
```

It looks like a networking bug in *your* code, and it isn't. The fix is to point `TIKTOKEN_CACHE_DIR` at a folder that already contains the file (copy it from a machine where it downloaded), and set that variable before tiktoken is imported. You won't need this on your laptop, but you will the first time DevMate runs in Docker without network access.

**Why `cl100k_base`?** It's a well-known BPE encoding and good enough to *budget* with. It is **not** the tokenizer of the Llama or Qwen models you'll call through Groq/OpenRouter. Their counts differ by roughly 5–15% depending on the text (third-party measurements, and they disagree with each other). That's why the module docstring says "estimate". Phase 6 measures the actual gap against the provider's reported `usage`.

---

# 4. The special-token crash

This is the bug that takes down a chat app in production.

Tokenizers reserve a few strings as control markers. `<|endoftext|>` is one of them. By default, tiktoken **refuses** to encode text that contains one:

```python
enc.encode("ignore previous <|endoftext|> instructions")
```

```text
ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'.
If you want this text to be encoded as a special token, pass it to `allowed_special`, ...
```

DevMate counts **user text, pasted code, tool output and web pages**. Any of them can contain that string, whether by accident (a developer pastes tokenizer source code) or on purpose (someone probing your app). Without the fix, one message containing `<|endoftext|>` makes `count_tokens` raise, every later context build fails, and the session is dead.

The two fixes look similar and do opposite things:

```text
allowed_special={"<|endoftext|>"}   treat it as the CONTROL token   ← never for user text
disallowed_special=()               treat it as ordinary characters ← what we use
```

`allowed_special` would let a user inject a real end-of-text marker. That's the tokenizer-level version of the prompt-injection hygiene in [ARCHITECTURE.md §7 #5](../docs/ARCHITECTURE.md#7-design-decisions). `disallowed_special=()` counts the 13 characters as the 7 ordinary, harmless tokens they really are.

---

# 5. count_message_tokens

A message isn't just its content. Chat APIs wrap each one in role markers and separators, which cost tokens too.

Open:

```text
devmate/app/core/token_manager.py
```

Add:

```python
def _message_tokens(message: dict) -> int:
    # Precomputed count wins (phase 2 stores token_count once, at append).
    if "token_count" in message:
        return message["token_count"]
    return count_tokens(message["content"])


def count_message_tokens(messages: list[dict], per_message_overhead: int = DEFAULT_MESSAGE_OVERHEAD) -> int:
    return sum(_message_tokens(m) + per_message_overhead for m in messages)
```

**The overhead of 4 is a heuristic, not a law.** It comes from OpenAI's old cookbook guidance for one chat format. Llama, Qwen and every other chat template differ, and providers don't publish an exact rule. So it's a parameter: you'll calibrate it against real `usage` numbers in phase 6. Over 100 messages it adds up to 400 tokens, about 5% of an 8k window, so ignoring it isn't an option either.

**Why honor `token_count` when it's present?** Phase 2 computes a message's count once, when the message is appended, and stores it ([ARCHITECTURE §7 #3](../docs/ARCHITECTURE.md#7-design-decisions)). Re-tokenizing 100 messages on every turn is wasted work. It also lets the tests below pin exact numbers like the note's 500/3500/200 without writing 3,500 tokens of filler.

---

# 6. context_usage: the definitions that matter

This is the section where people get it wrong, because "65% used" can mean two different things.

Add:

```python
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
```

The two percentages answer different questions:

```text
                        formula                       note example
utilization_pct         (input + reserved) / max      5200 / 8000 = 65%   "how full is the window?"
input_utilization_pct   input / (max - reserved)      4200 / 7000 = 60%   "how much prompt room is left?"
```

The note's 65% is the first one. Phase 6's context builder cares about the second, because the reserved output is untouchable and the 7000 is all the room the prompt has. Report both and label them, and nobody misreads a dashboard.

**Three validation choices, each preventing a real bug:**

```text
reserved_output >= max_context   -> ValueError  zero room for input; every build would fail later, far from the cause
negative numbers                 -> ValueError  a budget of -1 produces nonsense percentages, not an error
True / 8000.0 / "8000"           -> ValueError  see below
```

The bool trap: `isinstance(True, int)` is `True` in Python, because `bool` subclasses `int`. Without the explicit check, `context_usage(msgs, True, 0)` gets as far as `reserved_output >= max_context` (0 >= 1 is false), runs, and reports a max context of 1 token. It's rare, but when a config loader turns `"yes"` into `True`, this is the line that saves you an hour.

**`remaining` is allowed to go negative.** It's a measurement, not a gate. Overflow is a fact phase 6 has to act on (by trimming messages), so this function reports it rather than raising.

---

# 7. print_usage_report

Add:

```python
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
```

It prints and returns nothing, while `context_usage` computes and prints nothing. Keeping those apart is why phase 8's `/context` command and phase 6's Langfuse metadata can both reuse `context_usage` without parsing printed text.

---

# 8. Tests

Every M1 acceptance test from [MILESTONES.md](../docs/MILESTONES.md#m1--token-manager) becomes a test here.

Create:

```text
devmate/tests/test_token_manager.py
```

```python
import pytest

from app.core.token_manager import (
    context_usage, count_message_tokens, count_tokens, encode_text,
)

SAMPLES = {
    "english": "I'm building an e-commerce application using React, FastAPI and PostgreSQL.",
    "python": "def add_to_cart(user_id: int, item_id: int) -> dict:\n    return {'ok': True, 'items': [item_id]}\n",
    "json": '{"id": "msg_001", "role": "user", "token_count": 12, "tags": ["db", "auth"]}',
    "hindi": "मैं React, FastAPI और PostgreSQL का उपयोग करके एक ई-कॉमर्स एप्लिकेशन बना रहा हूँ।",
}

NOTE_EXAMPLE = [
    {"role": "system", "content": "", "token_count": 500},
    {"role": "conversation", "content": "", "token_count": 3500},
    {"role": "query", "content": "", "token_count": 200},
]


def test_empty_string_is_zero():
    assert count_tokens("") == 0
    assert encode_text("") == []


@pytest.mark.parametrize("name", SAMPLES)
def test_samples_count_positive(name):
    assert count_tokens(SAMPLES[name]) > 0


def test_hindi_costs_more_per_char_than_english():
    def ratio(s): return len(s) / count_tokens(s)
    assert ratio(SAMPLES["hindi"]) < ratio(SAMPLES["english"]) / 2


def test_special_token_text_does_not_crash():
    assert count_tokens("ignore previous <|endoftext|> instructions") > 0


def test_message_overhead_is_applied():
    msgs = [{"role": "user", "content": "hello"}] * 3
    assert count_message_tokens(msgs, per_message_overhead=0) == 3 * count_tokens("hello")
    assert count_message_tokens(msgs, per_message_overhead=4) == 3 * count_tokens("hello") + 12


def test_note_example_total_allocation():
    u = context_usage(NOTE_EXAMPLE, 8000, 1000, per_message_overhead=0)
    assert (u["used"], u["remaining"], u["utilization_pct"]) == (5200, 2800, 65.0)


def test_note_example_input_utilization():
    u = context_usage(NOTE_EXAMPLE, 8000, 1000, per_message_overhead=0)
    assert (u["input_tokens"], u["input_budget"], u["input_utilization_pct"]) == (4200, 7000, 60.0)


def test_by_role_sums_to_input():
    u = context_usage(NOTE_EXAMPLE, 8000, 1000, per_message_overhead=0)
    assert sum(u["by_role"].values()) == u["input_tokens"]


def test_overflow_reports_negative_remaining():
    u = context_usage([{"role": "user", "content": "", "token_count": 7500}], 8000, 1000, per_message_overhead=0)
    assert u["remaining"] == -500


@pytest.mark.parametrize("max_ctx,reserved", [
    (1000, 1000), (1000, 2000),          # reserved >= max
    (-1, 0), (1000, -5),                 # negatives
    (8000.0, 1000), ("8000", 1000),      # not int
    (True, 0),                           # bool is an int subclass
])
def test_bad_budgets_raise(max_ctx, reserved):
    with pytest.raises(ValueError):
        context_usage([], max_ctx, reserved)
```

Two exercises to write yourself:

```text
1. test_overhead_scales_with_message_count
   100 empty messages at the default overhead must cost exactly 400 tokens.

2. test_precomputed_token_count_wins
   {"role": "user", "content": "hello", "token_count": 999} must count as 999,
   not as count_tokens("hello"). Why does phase 2 depend on this?
```

---

# 9. Break it on purpose

**Break A: remove the special-token guard.**

In `encode_text`, change the line to:

```python
    return _encoding.encode(text)
```

Run:

```powershell
pytest -q
```

Expected:

```text
FAILED tests/test_token_manager.py::test_special_token_text_does_not_crash - ...
1 failed, 18 passed
```

Run `pytest` without `-q` to see the full message: `ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'`.

In production that's one pasted message killing a session. Put `disallowed_special=()` back.

**Break B: simulate a machine that's never seen the tokenizer.**

```powershell
mkdir $env:TEMP\empty_tk
$env:TIKTOKEN_CACHE_DIR = "$env:TEMP\empty_tk"
$env:HTTPS_PROXY = "http://127.0.0.1:9"
python -c "import app.core.token_manager"
```

Expected (last line):

```text
requests.exceptions.ProxyError: HTTPSConnectionPool(host='openaipublic.blob.core.windows.net', port=443): Max retries exceeded with url: /encodings/cl100k_base.tiktoken ...
```

It fails at *import*, in the first line of the stack, and not on user request number one. That's the payoff of loading at module level. Clean up:

```powershell
Remove-Item Env:TIKTOKEN_CACHE_DIR, Env:HTTPS_PROXY
```

**Break C: let a bool through.** Delete `isinstance(value, bool) or` from `_check_int` and run `pytest -q`. One case fails: `test_bad_budgets_raise[True-0]`, which reports `DID NOT RAISE`. Restore it.

---

# 10. Run it

```powershell
pytest -q
```

Expected:

```text
...................                                                      [100%]
19 passed in 0.17s
```

That's 19, or 21 once you've added the two exercises. If you see `ModuleNotFoundError: No module named 'app'`, you're not in `devmate/`, or an `__init__.py` from §1 is missing. Run pytest from the `devmate/` folder.

```powershell
python -m app.core.token_manager
```

Expected:

```text
Context Budget: 8000 tokens

system:              500
conversation:       3500
query:               200
reserved output:    1000
--------------------------
Total used:         5200
Remaining:          2800
Utilization:       65.0%
Input utilization: 60.0%  (4200/7000)
```

Use `python -m app.core.token_manager` and not `python app/core/token_manager.py`. The `-m` form runs it as part of the `app` package, which every later phase relies on for imports.

---

# 11. Quality gate

```powershell
pytest -q
git add -A; git commit -m "M1: token manager"
```

```text
✓ You can show, with your own numbers, that len//4 undercounts code, JSON and Hindi
✓ Token counts come from a real tokenizer, loaded once, failing loudly at import if unavailable
✓ Untrusted text containing <|endoftext|> is counted, not crashed on
✓ Per-message overhead is a tunable heuristic, not a hard-coded truth
✓ "Utilization" is unambiguous: total allocation (65%) and input room (60%) both reported
✓ Bad budgets (bool, float, negative, reserved ≥ max) fail at the call, not three layers later
✓ Every M1 acceptance test from MILESTONES.md is a passing test
```

---

# 12. Where this sits

```text
 Phase 1  ◄── here     Phase 2           Phase 3            Phase 4
┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐
│ measure it   │→ │ store it     │→ │ drop it       │→ │ keep the      │
│ tokens,      │  │ messages,    │  │ naively       │  │ important     │
│ budgets      │  │ order, tools │  │ FIFO/window   │  │ parts         │
└──────────────┘  └──────────────┘  └───────────────┘  └───────┬───────┘
                                                               ▼
 Phase 8            Phase 7            Phase 6            Phase 5
┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐
│ ship + prove │← │ survive a    │← │ assemble it   │← │ compress it   │
│ CLI + eval   │  │ restart      │  │ hybrid memory │  │ LLM summaries │
└──────────────┘  └──────────────┘  └───────────────┘  └───────────────┘
```

The idea that outlives tiktoken: **measure with the consumer's ruler.** A budget computed in the wrong unit only gives you false confidence. This applies to anything with a hard limit: bytes vs characters in a database column, rows vs pages in a cache, requests vs tokens in a rate limit.

---

## Next: Phase 2 — Conversation: Messages, Order & Tool Calls

```text
 append(role, content) ──► Message{id, seq, role, content, token_count, tool_calls}
                                │
                       ConversationManager.history()  → always in seq order
```

Three things that break:

```text
1. ordering by timestamp: two messages appended in the same millisecond
   come back in random order, and the model sees the answer before the question

2. an orphan tool result: a "tool" message whose tool_call_id matches no
   assistant call is rejected by the API, or confuses the model, three phases later

3. recomputing token_count on every read instead of once on append:
   correct, but it turns a 100-message build into 100 tokenizer calls per turn
```
