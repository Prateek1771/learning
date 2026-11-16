# DevMate — Milestones

Each milestone produces a module and the tests that prove it works. **Don't start the next milestone until the acceptance tests pass.**
M1–M4 run fully offline. The LLM and Langfuse arrive at M5.

| M | Build | Days | Needs keys? |
|---|---|---|---|
| M1 | Token manager | 0.5–1 | no |
| M2 | Models + conversation manager | 1 | no |
| M3 | Basic memory strategies | 1 | no |
| M4 | Priority selection | 1–2 | no |
| M5 | LLM client + summarizer + fact extraction | 2 | Groq/OpenRouter + Langfuse |
| M6 | Context builder (hybrid memory) | 1–2 | yes, for drift logging |
| M7 | SQLite persistence | 1 | no |
| M8 | DevMate CLI + evaluation | 1–2 | yes |

---

## M0 — Setup (~30 min)

```powershell
mkdir devmate; cd devmate
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install tiktoken pydantic python-dotenv pytest
```

`requirements.txt` starts with only these four packages. Add `openai` and `langfuse` at M5, not before.

- [ ] Create the folders from [ARCHITECTURE.md §8](ARCHITECTURE.md#8-folder-layout) as you reach each milestone, not all at once
- [ ] `git init`, and add `.venv/`, `.env` and `data/` to `.gitignore`
- [ ] `python -c "import tiktoken; print(len(tiktoken.get_encoding('cl100k_base').encode('hello world')))"` prints `2`

---

## M1 — Token manager

**Goal:** measure text in tokens and compute how much of the context budget is used.

**Deliverable:** `app/core/token_manager.py`, `tests/test_token_manager.py`

**Tasks**
- `encode_text(text) -> list[int]`. Load the encoding once at module level, not on every call.
- `count_tokens(text) -> int`
- `count_message_tokens(messages, per_message_overhead=4) -> int`. Sum the content tokens plus a per-message overhead for role and formatting. The 4 is a **heuristic**, not an API rule; keep it a parameter.
- `context_usage(messages, max_context, reserved_output) -> dict` with these exact definitions:
  - `input_tokens` = all message tokens
  - `used` = `input_tokens + reserved_output`, the **total context allocation**
  - `remaining` = `max_context − used`
  - `utilization_pct` = `used / max_context`
  - `input_budget` = `max_context − reserved_output`, and `input_utilization_pct` = `input_tokens / input_budget`
  - a per-role breakdown
- `print_usage_report(...)` that prints the report format from note.md
- A docstring stating that `cl100k_base` is an estimate and that the production tokenizer must match the model

**Acceptance tests**
- [ ] `count_tokens("") == 0`
- [ ] Short text, a 2k-word paragraph, Python source, JSON and Hindi/Chinese text all count above 0, and you've written down the chars-per-token ratio for each
- [ ] The note's example gives 500 + 3500 + 200 + 1000 = **5200 used / 2800 remaining / 65%** of 8000. That 65% is total allocation, including the reserved output.
- [ ] The same example gives `input_tokens = 4200`, `input_budget = 7000` and **`input_utilization_pct = 60%`**
- [ ] `reserved_output >= max_context` raises `ValueError`
- [ ] Negative or non-int budgets raise `ValueError`

**Concepts:** BPE tokenization, input vs output tokens, per-message overhead, why character counts lie (compare code vs. Hindi)

---

## M2 — Models + conversation manager

**Goal:** a typed, ordered history of messages.

**Deliverable:** `app/conversation/models.py`, `app/conversation/manager.py`, `tests/test_conversation.py`

**Tasks**
- Pydantic `Message`, `Fact`, `Summary`, `ContextBudget` and `BuiltContext` as defined in [ARCHITECTURE.md §3](ARCHITECTURE.md#3-data-models)
- `ConversationManager(session_id)` with `append(role, content, **meta) -> Message` (assigns `id` and `seq` and computes `token_count`), `history()`, `get(id)`, `to_json()` and `from_json()`
- Tool calls: an assistant message carries `tool_calls=[ToolCall(id, name, arguments)]`. `append("tool", ..., tool_call_id=...)` must match the `id` of a `ToolCall` on an earlier assistant message.

**Acceptance tests**
- [ ] `seq` is strictly increasing, and `history()` returns messages in order
- [ ] `token_count` is set on append and matches `count_tokens(content)`
- [ ] JSON round-trip gives back an identical history
- [ ] An invalid role raises a Pydantic `ValidationError`
- [ ] A tool message with an unknown `tool_call_id` raises an error

**Concepts:** message roles, sessions vs conversations, serialization, tool messages

---

## M3 — Basic memory strategies

**Goal:** see each naive strategy fail in its own way.

**Deliverable:** `app/memory/strategies.py`, `tests/test_strategies.py`, plus a first draft of `eval/dataset.json` (see [EVALUATION.md](EVALUATION.md))

**Tasks**
- A `MemoryStrategy` Protocol with `select(history, budget_tokens, query="")`
- `BufferMemory`: returns everything
- `FIFOMemory(capacity)`: **evicts from the store**. Calling it mutates the history it owns.
- `SlidingWindowMemory(n)`: the last N messages, without mutating anything
- `TokenWindowMemory`: newest-first until the budget is used up, returned in chronological order
- Tool call and tool result pairs are kept or dropped together

**Acceptance tests**
- [ ] No strategy exceeds its limit (count or tokens) on the 100-message dataset
- [ ] Output is always chronological
- [ ] At message 100, the DB fact planted at message 5 is **missing** from the FIFO, Sliding and TokenWindow output. That's the expected failure, so assert it.
- [ ] Buffer exceeds an 8k budget on the dataset (assert overflow)
- [ ] FIFO evicted messages can't be recovered; with SlidingWindow, `history()` still has them

**Concepts:** FIFO eviction vs a windowed view, count-based vs token-based limits, the needle-in-a-haystack failure

---

## M4 — Priority selection

**Goal:** keep the *important* messages, not just the recent ones.

**Deliverable:** `app/memory/selector.py`, `tests/test_selector.py`

**Tasks**
- `importance(msg) -> float`: keyword/regex heuristic (decision verbs, tech names, code blocks score high; greetings and acks score low). Pinned messages score 1.0.
- `recency(msg, latest_seq, half_life=20) -> float`: exponential decay
- `relevance(msg, query) -> float`: Jaccard overlap of lowercased word sets, minus stopwords
- `PriorityStrategy(w_imp=.5, w_rec=.3, w_rel=.2)`: score every message, pick greedily by `score / token_count` within the budget, then sort the picks back into chronological order

**Acceptance tests**
- [ ] The note's example: [PostgreSQL, "Hello!", "How are you?", JWT] with room for 2 keeps **messages 1 and 4**
- [ ] The DB fact from message 5 survives at message 100 with a 2k budget
- [ ] When the query is "what cache are we using?", the Redis message ranks first on relevance
- [ ] With recency-only weights (0, 1, 0) and **equal-token** messages, newer messages are always picked before older ones. This is not the same as TokenWindow: because the selector ranks by `score / tokens`, a short old message can beat a long new one. Write a test that shows this too.

**Concepts:** importance, recency and relevance scoring, greedy knapsack, why heuristic scoring is brittle (write down 3 cases it gets wrong)

---

## M5 — LLM client, summarizer, fact extraction

**Goal:** compress old turns, and pull out structured facts that don't drift.

**Deliverable:** `app/llm.py`, `app/summarization/summarizer.py`, `tests/test_summarizer.py`, `.env.example`

**Setup**
```powershell
pip install openai langfuse
```
Add both to `requirements.txt`.
Create a Groq and/or OpenRouter key and a Langfuse Cloud project (public and secret keys), then fill in `.env` (see [ARCHITECTURE.md §5](ARCHITECTURE.md#5-llm--monitoring)).

**Tasks**
- `llm.py`: `from langfuse.openai import OpenAI` with `base_url` chosen by `LLM_PROVIDER`. `complete(messages, name, metadata, model=None) -> (text, usage)`. `FakeLLM` returns canned text for tests.
- `roll(prev_summary, new_messages) -> Summary`, an **incremental** summary: the old summary plus only the new chunk, never the whole history
- `maybe_roll(history, summary, threshold_tokens)` only fires when messages older than the recent window, and not yet summarized, exceed the threshold
- `extract_facts(messages) -> list[Fact]`. The LLM returns JSON validated by Pydantic. When it's invalid, retry once, then skip.
- Fact upsert: the same key with a new value overwrites the old one ("switched to MySQL")
- Summary overflow: if the summary is over its budget, `compress(summary)`
- Name the generations `summarize`, `extract_facts` and `compress` in Langfuse

**Acceptance tests**
- [ ] FakeLLM: after rolling, `covers_until_seq` is correct and messages are never summarized twice
- [ ] FakeLLM: invalid fact JSON doesn't crash, and the step is skipped and logged
- [ ] Real LLM, run manually (`pytest -m live`): at least 9 of the 10 planted facts are recalled from summary + facts
- [ ] The fact changed at message 60 shows the **new** value
- [ ] The Langfuse dashboard shows the `summarize` and `extract_facts` generations with token cost

**Concepts:** full vs rolling vs incremental summarization, summary drift, structured extraction, cost of summarization

---

## M6 — Context builder (hybrid working memory)

**Goal:** put it all together under a hard token budget.

**Deliverable:** `app/context/builder.py`, `tests/test_builder.py`, `eval/run_eval.py` (first version)

**Tasks**
- `build(session, query, budget, strategy) -> BuiltContext`, assembled in this order: system, facts, summary, recent (from the strategy), query
- Allocator: section caps from `ContextBudget`, unused budget flows to recent, the safety margin is applied
- Final hard check: if over budget, drop the oldest recent message, then repeat. If the query alone doesn't fit, raise.
- History wrapped in delimiters; the system prompt is never taken from history
- After the LLM call, log `estimated_tokens` vs `usage.prompt_tokens` to Langfuse as a `tokenizer_drift` score

**Acceptance tests**
- [ ] Over the 100-message run, `total_tokens ≤ usable input` on **every** turn
- [ ] `tokens_by_section` adds up to `total_tokens`
- [ ] `included_msg_ids` and `dropped_msg_ids` partition the history with no overlap
- [ ] Hybrid keeps all planted facts at message 100 with an 8k budget
- [ ] Tokenizer drift has been measured for your chosen model, and the 10% margin either holds or gets adjusted (write down the number)

**Concepts:** context budget allocation, context engineering, estimated vs actual tokens

---

## M7 — SQLite persistence

**Goal:** memory survives a restart.

**Deliverable:** `app/storage/db.py`, `tests/test_storage.py`

**Tasks**
- Schema from [ARCHITECTURE.md §6](ARCHITECTURE.md#6-storage-sqlite), applied through `PRAGMA user_version` migrations
- `save_message`, `load_session`, `save_summary`, `latest_summary`, `upsert_fact`, `list_facts`
- `ConversationManager` and the stores write through to the DB
- Parameterized queries only (no f-string SQL)
- Run `PRAGMA foreign_keys = ON` on **every** connection. SQLite leaves it off by default.

**Acceptance tests**
- [ ] Build a context, close the DB, reopen it and build again: the `BuiltContext` is **identical**
- [ ] A migration from v1 to v2 (e.g. adding a `facts.confidence` column) runs on an existing DB without losing data
- [ ] Tests use an in-memory DB (`:memory:`) or `tmp_path`
- [ ] Inserting a message for a session that doesn't exist raises `sqlite3.IntegrityError`

**Concepts:** session restoration, schema migrations, the summary/fact lifecycle across restarts

---

## M8 — DevMate CLI + evaluation

**Goal:** a working assistant, plus numbers that prove which strategy wins.

**Deliverable:** `app/main.py`, a final `eval/run_eval.py`, and the results table added to `EVALUATION.md`

**Tasks**
- REPL chat loop with commands:
  - `/stats`: message count, total tokens, summary version, fact count
  - `/context`: section-by-section token bars plus the dropped message ids
  - `/strategy <buffer|fifo|sliding|token|priority|hybrid>`: switch strategies live
  - `/facts`: the current facts table
  - `/trace`: the Langfuse URL for the last turn
  - `/session new|load <id>`
- Configurable budget via `.env` or flags
- Run the full evaluation (see [EVALUATION.md](EVALUATION.md)) and commit the results table
- Optional: a FastAPI wrapper with `POST /chat` and `GET /sessions/{id}/context`

**Acceptance tests**
- [ ] End-to-end chat works with Groq **and** with OpenRouter (switching only `.env`)
- [ ] `/context` output matches the `BuiltContext` sent to the model
- [ ] Killing the CLI mid-session and resuming it keeps the facts and summary
- [ ] The evaluation table is filled in for all 6 strategies, with scores visible in Langfuse

**Concepts:** integration, observability, evidence-based strategy choice
