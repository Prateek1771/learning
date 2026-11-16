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
