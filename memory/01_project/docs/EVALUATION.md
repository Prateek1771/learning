# DevMate — Evaluation

The project isn't done when the chatbot works. It's done when you can show **which memory strategy keeps the right information, at what cost.**

## 1. Dataset: `eval/dataset.json`

A scripted 100-message DevMate conversation: about 10 planted facts at known positions, with filler chatter in between.

| Msg | Planted fact | key = value |
|---|---|---|
| 5 | "We'll use PostgreSQL 16 for the main DB." | `database = PostgreSQL` |
| 8 | "Frontend is React with Vite." | `frontend = React` |
| 12 | "Auth is JWT, 15-min access tokens." | `auth = JWT` |
| 20 | "Payments go through Stripe." | `payments = Stripe` |
| 30 | "Redis for caching sessions and carts." | `cache = Redis` |
| 41 | "Prices stored as integer cents." | `money_format = integer cents` |
| 52 | "Deploy target is AWS ECS." | `deploy = AWS ECS` |
| 60 | "Actually, switch the DB to MySQL." | `database = MySQL` ← **changed fact** |
| 73 | "All endpoints under /api/v1." | `api_prefix = /api/v1` |
| 85 | "Use structlog for logging." | `logging = structlog` |

Filler: greetings, "thanks", "ok", long code snippets and tool outputs (to burn tokens), and side questions.

```json
{
  "messages": [{"role": "user", "content": "..."}, ...],
  "facts":    [{"msg_index": 5, "key": "database", "value": "PostgreSQL", "superseded_by": 60}, ...],
  "questions":[{"q": "Which database are we using?", "expected": "MySQL"}, ...]
}
```

Start writing it at M3, and grow it as you go.

## 2. Metrics

| Metric | How it's measured | Needs LLM? |
|---|---|---|
| **Fact retention** | % of current planted facts whose value appears in the `BuiltContext` text at message 100 | no |
| **Stale-fact rate** | % of superseded values (PostgreSQL) still presented as current | no |
| **Answer accuracy** | 10 recall questions; the LLM answers from the built context; case-insensitive match against `expected` | yes |
| **Avg context tokens** | mean `total_tokens` across turns 1–100 | no |
| **Peak utilization** | max `total_tokens / usable input` | no |
| **Build latency** | mean ms for `build()` | no |
| **Summarization cost** | total tokens in `summarize` + `extract_facts` generations (from Langfuse) | yes |
| **Tokenizer drift** | mean `(actual − estimated) / actual` of prompt tokens | yes |

## 3. Harness: `eval/run_eval.py`

```
for strategy in [buffer, fifo, sliding, token, priority, hybrid]:
    replay dataset turn by turn (post-turn summarize/extract for hybrid)
    at msg 100: compute retention + stale rate from BuiltContext
    if --live: ask each question, score answer, log to Langfuse
       (trace tags: eval_run=<timestamp>, strategy=<name>;
        scores: fact_retention, answer_correct)
print results table (markdown) → paste into §4
```

- `python eval/run_eval.py` runs offline metrics only (FakeLLM handles hybrid's summaries)
- `python eval/run_eval.py --live` uses the real LLM and Langfuse
- Fixed budget: 8k context, 1k reserved output. Run it once more at 4k to see how each strategy holds up under pressure.

**Reproducibility rules**
1. Every strategy runs on the **same** `dataset.json`. Record the dataset's hash in the results.
2. Every strategy in a comparison uses the **same** budget, the same questions and the same model with temperature 0.
3. Each run writes `eval/results/<timestamp>.json` (config, dataset hash, per-strategy metrics) **and** the Markdown table. The JSON lets you diff a run against the last one after you change an algorithm.

## 4. Results (fill in at M8)

Budget: 8000 · Model: ____ · Provider: ____ · Date: ____

| Strategy | Retention | Stale | Accuracy | Avg tokens | Peak util | Latency | Summ. cost |
|---|---|---|---|---|---|---|---|
| Buffer | | | | | | | — |
| FIFO | | | | | | | — |
| Sliding | | | | | | | — |
| TokenWindow | | | | | | | — |
| Priority | | | | | | | — |
| Hybrid | | | | | | | |

**Expected shape** (if your numbers disagree, find out why):

- Buffer: 100% retention, but it overflows around message 40–60 at 8k
- FIFO, Sliding and TokenWindow: lose everything before about message 70
- Priority: keeps most facts, but may surface stale PostgreSQL alongside MySQL
- Hybrid: highest retention, lowest stale rate, and the only one that pays a summarization cost

## 5. Write-up (one paragraph per question)

1. Which strategy would you ship, and at what budget?
2. Where did summarization lose information? Paste the summary text.
3. How big was the tokenizer drift for your model, and is a 10% margin enough?
4. Which importance heuristic failed, and what would you replace it with?
