# Phase 2 — Conversation: Messages, Order & Tool Calls

Phase 1 measured text. Now DevMate needs something to measure: a history it can trust to come back in the right order, with the right token counts, and with every tool result attached to the call that produced it.

```text
 append(role, content, ...)
        │
        ▼
 ┌───────────────────────────┐     rejects: bad role, orphan tool result,
 │ ConversationManager._add  │ ◄── seq gap, wrong session, tool fields
 └────────────┬──────────────┘     on the wrong role
              ▼
 Message (frozen)
   id=msg_0004  seq=3  role=tool  token_count=8  tool_call_id=call_1
              │
              ▼
 history()  →  always in seq order, always a copy
 to_json()  ⇄  from_json()   (same checks on the way back in)
```

The rule for this phase:

```text
order by a counter you assign, never by the clock
a message is immutable once its tokens are counted
a tool result without its call is corrupt history, so refuse it at the door
```

**Versions** (checked 2026-11-16): **pydantic 2.14.3** on Python 3.12. You're on `uv`, so every command below uses it.

---

# 1. Housekeeping from phase 1

Three things to tidy before adding a dependency.

Run:

```powershell
cd D:\AI_learning\memory\01_project\devmate
Remove-Item scrach_ratio.py, requirements.txt
uv remove pytest
uv add --dev pytest
uv add pydantic
```

**Why delete `requirements.txt`?** With `uv`, `pyproject.toml` and `uv.lock` are the source of truth. A hand-kept `requirements.txt` beside them goes stale the first time you `uv add` something. That's already happening: yours lists `tiktoken>=0.13` while `pyproject.toml` says `>=0.14.0`. If you need one later (say, for a Docker image), generate it with `uv export`. Don't maintain it by hand.

**Why move pytest to `--dev`?** Runtime dependencies ship with the app. Test tools don't. When DevMate gets a Dockerfile in phase 8, this is the difference between a lean image and one with a test framework inside.

Also write the two phase 1 exercise tests if you skipped them. `test_precomputed_token_count_wins` matters in §5 of this phase.

Check:

```powershell
uv run pytest -q
```

Expected: `19 passed` (21 with the exercises).

---

# 2. The concept: what a message has to carry

DevMate's history will be sent to an LLM, sliced by memory strategies, summarized, stored in SQLite and restored. Every one of those steps needs the same answers from a message:

```text
field          question it answers                       who needs it
─────────────  ────────────────────────────────────────  ─────────────────────────
id             which message is this?                    /context, dropped_msg_ids
seq            where does it go in the conversation?     every strategy, phase 3+
role           who said it?                              the LLM API
content        what was said?                            the LLM API
token_count    what does it cost?                        every budget decision
created_at     when, for humans reading logs?            /stats, debugging
tool_calls     which tools did the assistant invoke?     validating tool results
tool_call_id   which call is this tool result for?       the LLM API
pinned         must this never be dropped?               phase 4 priority
importance     how much does it matter (0..1)?           phase 4 priority
```

The four roles:

```text
system      instructions to the model ("You are DevMate...")
user        the developer
assistant   the model, as text, as tool calls, or both
tool        the output of a tool the assistant called
```

**Tool calls are a pair.** An assistant message says "call `run_tests` with `{"path": "tests/"}`, and my id for this call is `call_1`". A later `tool` message says "here's the output for `call_1`". The API expects both halves. A result whose call has been dropped (by a memory strategy, by a bad import) gets the request rejected by most chat APIs, or quietly confuses the model. So the pair has to be validated where messages enter, and kept together wherever messages are dropped, which is phase 3's job.

---

# 3. Build the wrong version first: order by timestamp

Ordering by time sounds obviously correct, so test it before using it.

Create:

```text
devmate/scratch_order.py
```

```python
from datetime import datetime, timezone

stamps = [datetime.now(timezone.utc) for _ in range(10_000)]
print(f"10,000 timestamps, {len(set(stamps))} distinct")
```

Run:

```powershell
uv run python scratch_order.py
```

What I got on Windows with Python 3.12:

```text
10,000 timestamps, 5 distinct
```

Your number will differ, but expect it to be tiny. The Windows clock that `datetime.now()` reads ticks roughly every 15 ms, so everything inside one tick gets the *same* timestamp. Your machine appends a message in microseconds. A burst like "assistant calls a tool → tool returns → assistant answers" can easily land in one tick.

Now see the damage. You'll rerun this at the end of §6 once the manager exists:

```text
100 messages, 1 distinct timestamps
newest 5 by created_at: ['message 1', 'message 2', 'message 3', 'message 4', 'message 5']
newest 5 by seq:        ['message 100', 'message 99', 'message 98', 'message 97', 'message 96']
```

"Newest 5 by time" returned the **oldest** 5. Python's sort is stable, so when every key ties, `reverse=True` leaves the items in their original order and doesn't reverse them. In phase 3, TokenWindow walks history newest-first. Sort by `created_at` and it keeps message 1–5 while believing it kept 96–100. No error, just a model answering questions about the wrong part of the conversation.

The fix is a counter the manager assigns: `seq = 0, 1, 2, ...`. It can't tie, can't drift and doesn't depend on any clock. `created_at` stays, but only for humans.

---

# 4. The models

Create:

```text
devmate/app/conversation/__init__.py      (empty)
devmate/app/conversation/models.py
```

```python
"""Data models for DevMate conversations.

Only what phase 2 needs. Fact and Summary arrive in phase 5,
ContextBudget and BuiltContext in phase 6.
"""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str                      # "call_abc123", generated by the model
    name: str                    # "run_tests"
    arguments: str = "{}"        # JSON string, exactly as the API returns it


class Message(BaseModel):
    # frozen: token_count is computed from content once; editing content later would make it lie.
    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    seq: int = Field(ge=0)       # THE ordering key. created_at is for humans, never for sorting.
    role: Role
    content: str
    created_at: datetime = Field(default_factory=utcnow)
    token_count: int = Field(ge=0)
    importance: float | None = Field(default=None, ge=0, le=1)
    pinned: bool = False
    # tuple, not list: frozen is shallow, a list field could still be .append()-ed in place.
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _tool_fields_match_role(self) -> "Message":
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages can carry tool_calls")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages need a tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages can have a tool_call_id")
        return self
```

Now the reasoning, one decision at a time.

**Why only `Message` and `ToolCall`?** [MILESTONES.md](../MILESTONES.md#m2--models--conversation-manager) lists `Fact`, `Summary`, `ContextBudget` and `BuiltContext` under M2 as well. They're deferred on purpose: a model written before the code that uses it gets designed from guesses. Each one arrives in the phase that needs it.

**`role: Literal[...]`** makes Pydantic reject anything that isn't one of the four roles, at construction time:

```text
1 validation error for Message
role
  Input should be 'system', 'user', 'assistant' or 'tool' [type=literal_error, input_value='admin', input_type=str]
```

A typo like `"asistant"` fails here, on the line that made it. Without the check it would fail inside the LLM API call three phases later.

**`frozen=True`** makes assignment raise:

```text
Instance is frozen [type=frozen_instance, input_value='We use MySQL.', input_type=str]
```

That protects `token_count`. It was computed from `content` at append time. If someone edits `content` afterwards, the count silently lies, and every budget built on it is wrong. To "edit" a message you append a new one, which is also the honest record of what happened in the conversation.

**`tuple`, not `list`, for `tool_calls`.** `frozen` is **shallow**. It blocks `m.tool_calls = [...]` but not `m.tool_calls.append(...)`. I checked this against pydantic 2.14:

```python
class M(BaseModel):
    model_config = ConfigDict(frozen=True)
    calls: list[str] = []

m = M(); m.calls.append("sneaky")
print(m.calls)        # ['sneaky']  ← frozen model, mutated anyway
```

A tuple has no `.append`, so the freeze actually holds all the way down.

**`created_at` is timezone-aware UTC.** A naive `datetime.now()` has no zone, so a session saved on your laptop and restored in a UTC container reads as hours off. It's only for display, but display that's wrong by 5h30m still costs debugging time.

**`importance` and `pinned` exist now but stay unused until phase 4.** They're here because phase 3's strategies and phase 7's SQLite table both need the full shape of a message, and adding columns later means a migration. `Field(ge=0, le=1)` already rejects `importance=7`.

**The `model_validator`** enforces the tool-call rules that a single field can't express, because they relate two fields to the role. It runs after field validation (`mode="after"`), so it sees typed values.

---

# 5. The conversation manager

Create:

```text
devmate/app/conversation/manager.py
```

```python
"""Ordered, validated message history for one session."""
import json
from uuid import uuid4

from app.conversation.models import Message, Role, ToolCall
from app.core.token_manager import count_tokens


def _count(content: str, tool_calls: tuple[ToolCall, ...]) -> int:
    # Tool-call names and arguments are sent to the model too, so they cost tokens.
    return count_tokens(content) + sum(count_tokens(c.name) + count_tokens(c.arguments) for c in tool_calls)


class ConversationManager:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or f"sess_{uuid4().hex[:8]}"
        self._messages: list[Message] = []
        self._by_id: dict[str, Message] = {}
        self._call_ids: set[str] = set()      # every tool-call id an assistant has made

    def append(
        self,
        role: Role,
        content: str,
        *,
        tool_calls: list[ToolCall] | tuple[ToolCall, ...] = (),
        tool_call_id: str | None = None,
        pinned: bool = False,
    ) -> Message:
        tool_calls = tuple(tool_calls)
        seq = len(self._messages)
        msg = Message(
            id=f"msg_{seq + 1:04d}",
            session_id=self.session_id,
            seq=seq,
            role=role,
            content=content,
            token_count=_count(content, tool_calls),   # computed ONCE, here
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            pinned=pinned,
        )
        self._add(msg)
        return msg

    def _add(self, msg: Message) -> None:
        # Single gate for append() and from_json(): every message enters through here.
        if msg.session_id != self.session_id:
            raise ValueError(f"message {msg.id} belongs to session {msg.session_id}")
        if msg.seq != len(self._messages):
            raise ValueError(f"expected seq {len(self._messages)}, got {msg.seq}")
        if msg.role == "tool" and msg.tool_call_id not in self._call_ids:
            raise ValueError(f"orphan tool result: no earlier assistant call has id {msg.tool_call_id!r}")
        self._messages.append(msg)
        self._by_id[msg.id] = msg
        self._call_ids.update(c.id for c in msg.tool_calls)

    def history(self) -> list[Message]:
        return list(self._messages)          # a copy: callers can't reorder or delete ours

    def get(self, msg_id: str) -> Message:
        return self._by_id[msg_id]           # KeyError if unknown

    def __len__(self) -> int:
        return len(self._messages)
```

**`token_count` is computed once, in `append`.** This is [ARCHITECTURE §7 #3](../ARCHITECTURE.md#7-design-decisions) made real. From here on, everything that needs a cost reads `msg.token_count`, and phase 1's `_message_tokens` already prefers a precomputed count. That's why the phase 1 exercise `test_precomputed_token_count_wins` mattered. Without this, phase 6 would re-tokenize the whole history on every turn: 100 messages → 100 tokenizer calls per user message, growing forever.

**Tool-call arguments cost tokens.** An assistant message with empty `content` and one tool call isn't free. The name and the JSON arguments are sent to the model, so `_count` includes them. Leave them out, and an agent that makes lots of tool calls (which is exactly what a coding assistant does) has its history undercounted by hundreds of tokens.

**`_add` is the single gate.** Both `append` and `from_json` (next section) route every message through it. If the orphan check lived only in `append`, a hand-edited or corrupted JSON file could load a tool result with no call, and you'd find out when the API rejects the request. One gate means one place to get right.

**`_call_ids` is a set of every call id seen so far.** A tool result is valid only if its `tool_call_id` was issued by an *earlier* assistant message. Because `_add` runs in `seq` order, "earlier" comes for free.

**Keyword-only arguments (the `*`).** `append("tool", "12 passed", tool_call_id="call_1")` reads clearly. `append("tool", "12 passed", (), "call_1")` would also work without the `*`, and it's unreadable. The `*` forbids the second form.

**`history()` returns a copy.** Callers get `list(self._messages)`. A phase 3 strategy that does `h = conv.history(); h.pop(0)` changes its own list, not the conversation. The `Message` objects inside are shared, but they're frozen, so sharing them is safe. That's another reason §4 froze them.

**IDs are `msg_0001`, derived from `seq`.** They're readable in `/context` output and predictable in tests. They're unique per session, not globally. Phase 7's SQLite table keys on `(session_id, seq)`, so that's enough. If you ever merge sessions, switch to `uuid4`.

---

# 6. Serialization

Add to the class:

```python
    def to_json(self) -> str:
        return json.dumps({
            "session_id": self.session_id,
            "messages": [m.model_dump(mode="json") for m in self._messages],
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "ConversationManager":
        raw = json.loads(data)
        mgr = cls(raw["session_id"])
        for m in raw["messages"]:
            mgr._add(Message.model_validate(m))   # same checks as append
        return mgr
```

And a demo at the bottom of the file:

```python
if __name__ == "__main__":
    conv = ConversationManager("sess_demo")
    conv.append("system", "You are DevMate, an AI coding assistant.")
    conv.append("user", "I'm building an e-commerce app with React, FastAPI and PostgreSQL.")
    conv.append("assistant", "", tool_calls=[ToolCall(id="call_1", name="run_tests", arguments='{"path": "tests/"}')])
    conv.append("tool", "12 passed in 0.4s", tool_call_id="call_1")
    conv.append("assistant", "All 12 tests pass. What next?")
    for m in conv.history():
        extra = f" calls={[c.name for c in m.tool_calls]}" if m.tool_calls else ""
        extra += f" answers={m.tool_call_id}" if m.tool_call_id else ""
        print(f"{m.seq:>2} {m.id} {m.role:<9} {m.token_count:>3} tok  {m.content[:40]!r}{extra}")
```

**`model_dump(mode="json")`, not plain `model_dump()`.** Plain mode returns a real `datetime` object, and `json.dumps` raises `TypeError: Object of type datetime is not JSON serializable`. JSON mode converts it to an ISO string that `model_validate` parses back into an aware `datetime`. The round trip is exact, and the test in §7 proves it.

**`ensure_ascii=False`** keeps Hindi and other non-ASCII text readable in the file. With the default, it's stored as `मैं`.

**`from_json` validates, it doesn't trust.** It rebuilds through `_add`, so a file with a gap in `seq`, a message from another session or an orphan tool result is refused on load. This is the same code path phase 7 uses when it restores a session from SQLite.

Run:

```powershell
uv run python -m app.conversation.manager
```

Expected:

```text
 0 msg_0001 system     10 tok  'You are DevMate, an AI coding assistant.'
 1 msg_0002 user       15 tok  "I'm building an e-commerce app with Reac"
 2 msg_0003 assistant   9 tok  '' calls=['run_tests']
 3 msg_0004 tool        8 tok  '12 passed in 0.4s' answers=call_1
 4 msg_0005 assistant   9 tok  'All 12 tests pass. What next?'
```

Look at line 2: empty content and 9 tokens. That's the tool call's name and arguments being counted.

Now finish the §3 experiment. Replace `scratch_order.py` with:

```python
from app.conversation.manager import ConversationManager

conv = ConversationManager()
for i in range(1, 101):
    conv.append("user" if i % 2 else "assistant", f"message {i}")

h = conv.history()
print(f"100 messages, {len({m.created_at for m in h})} distinct timestamps")
print("newest 5 by created_at:", [m.content for m in sorted(h, key=lambda m: m.created_at, reverse=True)[:5]])
print("newest 5 by seq:       ", [m.content for m in sorted(h, key=lambda m: m.seq, reverse=True)[:5]])
```

```powershell
uv run python scratch_order.py
```

You should see the output from §3, with the oldest five returned as "newest". Then delete the file.

---

# 7. Tests

Every M2 acceptance test, plus one for each rule in §4–§6.

Create:

```text
devmate/tests/test_conversation.py
```

```python
import pytest
from pydantic import ValidationError

from app.conversation.manager import ConversationManager
from app.conversation.models import ToolCall
from app.core.token_manager import count_tokens


@pytest.fixture
def conv():
    c = ConversationManager("sess_test")
    c.append("system", "You are DevMate.")
    c.append("user", "We use PostgreSQL.")
    c.append("assistant", "", tool_calls=[ToolCall(id="call_1", name="run_tests", arguments="{}")])
    c.append("tool", "12 passed", tool_call_id="call_1")
    c.append("assistant", "All green.")
    return c


def test_seq_is_strictly_increasing_and_ordered(conv):
    assert [m.seq for m in conv.history()] == list(range(len(conv)))


def test_ids_are_unique(conv):
    ids = [m.id for m in conv.history()]
    assert len(ids) == len(set(ids))


def test_token_count_set_on_append():
    m = ConversationManager().append("user", "We use PostgreSQL and JWT.")
    assert m.token_count == count_tokens("We use PostgreSQL and JWT.")


def test_tool_call_arguments_cost_tokens():
    call = ToolCall(id="c1", name="search", arguments='{"q": "jwt refresh"}')
    m = ConversationManager().append("assistant", "", tool_calls=[call])
    assert m.token_count == count_tokens("search") + count_tokens('{"q": "jwt refresh"}')


def test_json_round_trip_is_identical(conv):
    restored = ConversationManager.from_json(conv.to_json())
    assert restored.session_id == conv.session_id
    assert restored.history() == conv.history()


def test_invalid_role_raises():
    with pytest.raises(ValidationError):
        ConversationManager().append("admin", "hi")


def test_orphan_tool_result_raises():
    with pytest.raises(ValueError, match="orphan"):
        ConversationManager().append("tool", "result", tool_call_id="call_missing")


def test_tool_message_needs_tool_call_id():
    with pytest.raises(ValidationError):
        ConversationManager().append("tool", "result")


def test_only_assistant_carries_tool_calls():
    with pytest.raises(ValidationError):
        ConversationManager().append("user", "hi", tool_calls=[ToolCall(id="c1", name="x")])


def test_messages_are_frozen(conv):
    with pytest.raises(ValidationError):
        conv.history()[1].content = "We use MySQL."


def test_history_returns_a_copy(conv):
    conv.history().clear()
    assert len(conv) == 5


def test_from_json_rejects_tampered_seq(conv):
    tampered = conv.to_json().replace('"seq": 2', '"seq": 7')
    with pytest.raises(ValueError, match="expected seq"):
        ConversationManager.from_json(tampered)


def test_get_by_id(conv):
    assert conv.get("msg_0002").content == "We use PostgreSQL."
    with pytest.raises(KeyError):
        conv.get("msg_9999")
```

**Why `ValidationError` in some tests and `ValueError` in others?** Pydantic raises `ValidationError` for anything checked by the model (the role, frozen fields, the `model_validator`). The manager raises plain `ValueError` for rules that need the *whole conversation* (orphans, seq gaps), because a single message can't know what came before it. Since `ValidationError` subclasses `ValueError`, a caller that catches `ValueError` gets both.

**The fixture** builds a realistic five-message conversation with a tool pair. Each test gets a fresh copy, so one test's mutations can't leak into another.

Two exercises to write yourself:

```text
1. test_tool_result_before_its_call_is_rejected
   append the tool result first, then the assistant call with that id.
   The first append must raise. Why is "the id exists somewhere" not enough?

2. test_hindi_round_trips
   append Hindi text, round-trip through to_json/from_json, and assert the
   content is identical AND that the raw JSON string contains the Hindi
   characters (not \u escapes).
```

---

# 8. Break it on purpose

**Break A: unfreeze the message.** In `models.py`, change `Message`'s config to `ConfigDict(frozen=False)`, then run:

```powershell
uv run python -c "from app.conversation.manager import ConversationManager; from app.core.token_manager import count_tokens; c=ConversationManager(); m=c.append('user','We use PostgreSQL.'); m.content='We switched to MySQL for the orders service, Redis for carts, and Kafka for events.'; print('stored', m.token_count, 'real', count_tokens(m.content))"
```

Expected:

```text
stored 4 real 18
```

The message now costs 18 tokens and claims 4. Phase 6's budget would trust the 4. `uv run pytest -q` also fails `test_messages_are_frozen` with `DID NOT RAISE`. Restore `frozen=True`.

**Break B: drop the orphan check.** Delete the two-line `if msg.role == "tool" ...` block in `_add`. Then:

```powershell
uv run pytest -q
```

```text
FAILED tests/test_conversation.py::test_orphan_tool_result_raises - Failed: DID NOT RAISE ...
1 failed, 12 passed
```

Without it, the bad message is accepted silently and the error surfaces later as an API rejection on some future turn, far from the line that caused it. Restore it.

**Break C: sort by time.** You already did this in §6: `created_at` with `reverse=True` returned messages 1–5 as the "newest". Keep that output in mind for phase 3.

---

# 9. Run it

```powershell
uv run pytest -q
```

Expected:

```text
................................                                         [100%]
32 passed
```

That's 19 from phase 1 plus 13 here. Add 2 for each set of exercises you've written. If you get `ModuleNotFoundError: No module named 'app.conversation'`, the empty `app/conversation/__init__.py` from §4 is missing.

```powershell
git add -A; git commit -m "M2: conversation models and manager"
```

---

# 10. Quality gate

```text
✓ Messages are ordered by a manager-assigned seq, and you've seen the clock fail at it
✓ A message's token count is computed once and can't be invalidated, because messages are frozen all the way down
✓ Tool-call names and arguments are counted, not treated as free
✓ A tool result is accepted only if an earlier assistant message issued its call id
✓ Bad roles, misplaced tool fields and seq gaps are refused where they enter
✓ history() can't be used to corrupt the conversation
✓ JSON round-trips exactly, and loading re-validates instead of trusting the file
✓ Every M2 acceptance test from MILESTONES.md passes
```

---

# 11. Where this sits

```text
 Phase 1          Phase 2  ◄── here     Phase 3            Phase 4
┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐
│ measure it   │→ │ store it     │→ │ drop it       │→ │ keep the      │
│ tokens,      │  │ messages,    │  │ naively       │  │ important     │
│ budgets      │  │ order, tools │  │ FIFO/window   │  │ parts         │
└──────────────┘  └──────────────┘  └───────────────┘  └───────────────┘
```

The idea that outlives this project: **validate at the boundary, then trust the inside.** `_add` is the only door into the history. Everything past it (strategies, the context builder, storage) can assume ordered, counted, well-paired messages, and doesn't need defensive checks scattered everywhere. The same pattern applies to API request bodies, queue consumers and file imports.

---

## Next: Phase 3 — Basic Memory: Buffer, FIFO, Sliding & Token Windows

```text
 history (100 msgs) ──► strategy.select(history, budget) ──► what the model sees
                          Buffer  │ FIFO │ Sliding │ TokenWindow
```

Three things that break:

```text
1. FIFO deletes from the store, so the database decision from message 5 is gone
   for good; a sliding window only hides it, and it can still be summarized later

2. a window that cuts between an assistant's tool call and its tool result
   sends an orphan to the model, which is the exact thing phase 2 refuses at the door

3. counting a window in messages instead of tokens: 10 short messages and
   10 pasted stack traces are both "10 messages", but one fits and one doesn't
```
