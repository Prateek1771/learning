# DevMate — Architecture

## 1. Request flow

```
User message
     │
     ▼
ConversationManager.append()          token_count computed ONCE here and stored
     │
     ▼
ContextBuilder.build(session, query, budget)
     ├─ 1. system prompt              fixed, never evicted
     ├─ 2. FactStore.render()          structured key facts
     ├─ 3. SummaryStore.latest()       rolling summary of old turns
     ├─ 4. MemoryStrategy.select()     which raw messages fit what's left
     ├─ 5. current query
     └─ 6. hard check: total ≤ max_context − reserved_output − safety_margin
              else drop the oldest of the selected recent messages until it fits
     │
     ▼
llm.complete(context)  ──► Groq / OpenRouter      traced in Langfuse
     │
     ▼
assistant message appended (+ real usage.prompt_tokens recorded)
     │
     ▼
post-turn, off the hot path:
     Summarizer.maybe_roll()      if unsummarized old messages > threshold
     FactExtractor.update()       new/changed facts from the latest turn
     │
     ▼
Storage (SQLite)
```

**Two corrections to the diagram in note.md:**

- **TokenManager is not a pipeline stage.** It's a utility that every component calls.
- **Summarization runs after the response is sent.** If it sat in the request path, every Nth turn would pay an extra LLM call in latency.

## 2. Components

| Module | Responsibility | Key API |
|---|---|---|
| `core/token_manager.py` | Encode, count, configurable per-message overhead (heuristic, default 4), usage report | `count_tokens(text)`, `count_message_tokens(msgs)`, `context_usage(msgs, max_ctx, reserved)` |
| `conversation/models.py` | Pydantic data models | `Message`, `Fact`, `Summary`, `ContextBudget`, `BuiltContext` |
| `conversation/manager.py` | Append, order and fetch messages; tool messages | `append(role, content, **meta)`, `history()`, `get(id)` |
| `memory/strategies.py` | `MemoryStrategy` Protocol + Buffer, FIFO, SlidingWindow, TokenWindow | `select(history, budget_tokens) -> list[Message]` |
| `memory/selector.py` | Priority strategy: importance × recency × relevance | `PriorityStrategy(weights).select(history, budget, query)` |
| `summarization/summarizer.py` | Rolling summary + structured fact extraction | `roll(prev, chunk) -> Summary`, `extract_facts(chunk) -> list[Fact]` |
| `context/builder.py` | Budget allocator + section assembly + final check | `build(session, query, budget) -> BuiltContext` |
| `llm.py` | Langfuse-wrapped OpenAI client for Groq/OpenRouter; `FakeLLM` for tests | `complete(messages, name, metadata) -> (text, usage)` |
| `storage/db.py` | sqlite3 repository; migrations | `save_message`, `load_session`, `save_summary`, `upsert_fact` |
| `main.py` | CLI chat | `/stats` `/context` `/strategy <name>` `/facts` `/trace` |

### Strategy interface

```python
class MemoryStrategy(Protocol):
    name: str
    def select(self, history: list[Message], budget_tokens: int, query: str = "") -> list[Message]: ...
```

Every strategy returns messages in **chronological order**, even when it picked them by score.

| Strategy | Rule | What it loses |
|---|---|---|
| Buffer | everything | nothing, until it overflows |
| FIFO | store holds at most N; the oldest is **deleted** on overflow | old decisions, permanently |
| SlidingWindow | send the last N; the store keeps everything | old decisions from the prompt, but they can still be recovered |
| TokenWindow | newest-first until `budget_tokens` is used up | same as sliding window, but budgeted in tokens instead of message count |
| Priority | greedy by `score / tokens` within the budget | low-scored but relevant messages |
| Hybrid | facts + summary + TokenWindow over the rest | details the summary dropped |

### Priority score (M4)

```
score = w_imp * importance + w_rec * recency + w_rel * relevance        (defaults 0.5 / 0.3 / 0.2)

importance  0..1  heuristic: decision verbs ("using", "decided", "switch to"), tech keywords,
                  code blocks → high; greetings / thanks / "ok" → low; pinned → 1.0
recency     0..1  exp(-age / half_life)  where age is measured in messages
relevance   0..1  keyword overlap (Jaccard) between message and current query
                  (embeddings are the upgrade path, but not in Project 1)
```

## 3. Data models

```python
Role = Literal["system", "user", "assistant", "tool"]

class ToolCall(BaseModel):
    id: str                      # "call_abc123"
    name: str                    # "run_tests"
    arguments: str               # JSON string, as the OpenAI API returns it

class Message(BaseModel):
    id: str                      # "msg_0001"
    session_id: str
    seq: int                     # strict ordering within a session
    role: Role
    content: str
    created_at: datetime
    token_count: int             # tiktoken estimate, set on append
    importance: float | None = None
    pinned: bool = False
    tool_calls: list[ToolCall] = []   # assistant only: calls the model made
    tool_call_id: str | None = None   # tool only: must match a ToolCall.id from an earlier assistant msg

class Fact(BaseModel):
    key: str                     # "database"
    value: str                   # "PostgreSQL"
    source_msg_id: str
    updated_at: datetime         # newer value overwrites: "we switched to MySQL"

class Summary(BaseModel):
    session_id: str
    version: int
    text: str
    covers_until_seq: int        # every message with seq ≤ this one is already in the summary
    token_count: int

class ContextBudget(BaseModel):
    max_context: int = 8000
    reserved_output: int = 1000
    system: int = 500
    facts: int = 500
    summary: int = 1000
    safety_margin: float = 0.10  # tiktoken ≠ Llama/Qwen tokenizer
    # recent messages get the remainder

class BuiltContext(BaseModel):
    messages: list[dict]                 # OpenAI-format, ready to send
    tokens_by_section: dict[str, int]    # {"system": 412, "facts": 96, ...}
    included_msg_ids: list[str]
    dropped_msg_ids: list[str]
    total_tokens: int                    # estimate
```

## 4. Budget allocation

Example with an 8k context:

```
max_context                          8000
− reserved_output                   −1000
− safety margin (10% of 7000)        −700
= usable input                       6300
   system   ≤ 500
   facts    ≤ 500
   summary  ≤ 1000
   query    actual size
   recent   = whatever remains (≈ 4100+)
```

Budget a section doesn't use **flows to recent messages**. If the query alone doesn't fit, raise an error. Never silently truncate the user's request.

## 5. LLM + monitoring

### Providers: Groq or OpenRouter

Both are OpenAI-compatible, so `llm.py` uses one client and switches providers through `.env`:

| Provider | base_url | Key | Good for |
|---|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | Fast chat and summarization |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | Comparing many models as summarizers |

```env
# .env.example
LLM_PROVIDER=groq            # groq | openrouter
LLM_MODEL=                   # chat model id for that provider
SUMMARY_MODEL=               # cheaper/faster model for summaries + fact extraction
GROQ_API_KEY=
OPENROUTER_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Langfuse Cloud

The client uses Langfuse's OpenAI drop-in, so every call is traced automatically:

```python
from langfuse.openai import OpenAI   # instead of: from openai import OpenAI
client = OpenAI(base_url=BASE_URLS[provider], api_key=key)
client.chat.completions.create(model=..., messages=..., name="chat",
                               metadata={"session_id": ..., "strategy": ...})
```

| What | How |
|---|---|
| One trace per user turn | session_id + `strategy` tag |
| Generation names | `chat`, `summarize`, `extract_facts`, so summarization cost shows up separately |
| Context metadata | `tokens_by_section`, `dropped_msg_ids`, `estimated_tokens` |
| Tokenizer drift | `actual usage.prompt_tokens − estimated_tokens` logged as a score (M6) |
| Eval scores | `fact_retention`, `answer_correct` per strategy (see EVALUATION.md) |

Tests use `FakeLLM`, which needs no network and no keys. Langfuse stays disabled when its keys are missing.

## 6. Storage (SQLite)

```sql
CREATE TABLE sessions  (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE messages  (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                        seq INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                        token_count INTEGER NOT NULL, importance REAL, pinned INTEGER DEFAULT 0,
                        tool_calls TEXT,          -- JSON list of ToolCall (assistant msgs)
                        tool_call_id TEXT, created_at TEXT NOT NULL,
                        UNIQUE(session_id, seq));
CREATE TABLE summaries (session_id TEXT NOT NULL, version INTEGER NOT NULL, text TEXT NOT NULL,
                        covers_until_seq INTEGER NOT NULL, token_count INTEGER NOT NULL,
                        PRIMARY KEY(session_id, version));
CREATE TABLE facts     (session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                        source_msg_id TEXT, updated_at TEXT NOT NULL,
                        PRIMARY KEY(session_id, key));
```

Migrations use `PRAGMA user_version` plus an ordered list of SQL steps. No migration library is needed.
Every connection runs `PRAGMA foreign_keys = ON`. SQLite doesn't enforce the `REFERENCES` constraints without it.

## 7. Design decisions

| # | Decision | Why |
|---|---|---|
| 1 | Facts are stored separately from the prose summary | Re-summarizing a summary causes drift. Structured `key=value` facts stay stable. |
| 2 | FIFO deletes; sliding window is only a view | note.md treats them as the same thing. The difference matters: FIFO data can't be recovered, sliding-window data can be summarized later. |
| 3 | `token_count` is computed once, at append | So each build doesn't re-tokenize the whole history. |
| 4 | A tool call and its result are kept or dropped together | An orphaned tool result confuses the model, and some APIs reject it. |
| 5 | The system prompt never comes from history; history goes inside delimiters | Basic defense against prompt injection through old messages. |
| 6 | Summarization runs post-turn | Keeps latency out of the request path. |
| 7 | Summary overflow: re-compress, then drop its oldest sections; never drop facts | Facts are the cheapest, highest-value tokens. |
| 8 | tiktoken is an **estimate**; the API's `usage` is ground truth | Groq/OpenRouter models (Llama, Qwen, ...) use different tokenizers. A 10% margin covers the drift, and Langfuse measures it. |

## 8. Folder layout

```
devmate/
├── app/
│   ├── main.py
│   ├── llm.py
│   ├── core/token_manager.py
│   ├── conversation/{models.py, manager.py}
│   ├── memory/{strategies.py, selector.py}
│   ├── summarization/summarizer.py
│   ├── context/builder.py
│   └── storage/db.py
├── tests/            test_<module>.py per milestone
├── eval/             dataset.json, run_eval.py
├── data/             devmate.db (gitignored)
├── requirements.txt  M0: tiktoken pydantic python-dotenv pytest · M5 adds: openai langfuse
└── .env.example
```

This is slimmer than the layout in note.md: one file per strategy family instead of one file per class. Split a file when it passes ~300 lines.
