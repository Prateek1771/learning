# DevMate — AI Working Memory & Context Management Engine

Project 1 of the AI learning track. Source brief: [note.md](note.md). This folder is the clean, buildable version of it.

## The problem

A developer chats with an AI coding assistant for 100+ messages:

```
msg 1   "E-commerce app: React, FastAPI, PostgreSQL"
msg 2   "JWT auth, Redis cache"
...     cart API, product API, payments, errors, deployment ...
msg 100 "Now implement checkout. Same architecture as before."
```

Resending all 100 messages every turn is slow and expensive, and eventually overflows the model's context window.
Dropping old messages loses the decisions from msg 1–2 that msg 100 depends on.

**DevMate decides, every turn, what the model gets to see:**

```
SYSTEM PROMPT        never evicted
KEY FACTS            database=PostgreSQL, auth=JWT, cache=Redis ...
SUMMARY              "Product + cart APIs done, payment pending ..."
RECENT MESSAGES      as many as fit the remaining budget
CURRENT QUERY        "Implement checkout"
```

## Goal

Build the **memory layer behind an agent**, not a chatbot. The chat UI is just the harness used to exercise and measure it.

## Stack

| Tool | Role | From |
|---|---|---|
| Python 3.12 | everything | M1 |
| `tiktoken` | token estimate for budgeting | M1 |
| `pydantic` | message / fact / summary models | M2 |
| `sqlite3` (stdlib) | persistence | M7 |
| `openai` SDK → **Groq** or **OpenRouter** | chat, summarization, fact extraction | M5 |
| **Langfuse Cloud** | tracing, cost, eval scores | M5 |
| `python-dotenv`, `pytest` | config, tests | M1 |
| FastAPI | optional HTTP API | M8 (optional) |

M1–M4 run fully offline with no API keys, so you learn the algorithms before any LLM gets involved.

## Docs

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, data models, request flow, budget allocation, SQLite schema, LLM/Langfuse setup, design decisions |
| [MILESTONES.md](MILESTONES.md) | M1–M8: tasks, deliverables, acceptance tests, concepts, estimates |
| [EVALUATION.md](EVALUATION.md) | Benchmark dataset, metrics, strategy comparison harness |
| [note.md](note.md) | Original brief (kept as-is) |

## Definition of done

You can explain **and demo**:

- [ ] Why LLMs have context windows (M1)
- [ ] How tokenization works and why tiktoken is only an estimate for Llama/Qwen models (M1, M6)
- [ ] How context budgets are calculated (M1, M6)
- [ ] Why full conversation history is inefficient (M3 + eval)
- [ ] How FIFO and sliding window differ (M3)
- [ ] How to preserve important facts (M4, M5)
- [ ] Why summaries lose information, and how to measure it (M5 + eval)
- [ ] How to combine summaries, facts and recent messages (M6)
- [ ] How to persist and rebuild working memory after a restart (M7)
- [ ] The strategy comparison table from [EVALUATION.md](EVALUATION.md), with traces in Langfuse (M8)

Estimated effort: 9–12 focused days.

## Out of scope

Vector databases, embeddings, Neo4j and LangGraph. They belong in later projects of the agent-memory curriculum.
