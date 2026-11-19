"""Unit tests for the complexity scorer and model router.

No network, no model loading, no file I/O — pure Python only.
"""

import pytest

from llm_gateway.router import resolve_model, score_complexity

SAMPLE_CONFIG = {
    "tiers": {
        "fast": {"model": "gpt-3.5-turbo", "description": "Simple queries"},
        "balanced": {"model": "gpt-4o-mini", "description": "Medium complexity"},
        "powerful": {"model": "gpt-4o", "description": "Reasoning, code, long context"},
    },
    "routing": {
        "fast_threshold": 30,
        "balanced_threshold": 70,
        "default_tier": "balanced",
    },
}


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# score_complexity tests
# ---------------------------------------------------------------------------


def test_score_simple_prompt():
    score = score_complexity(_msgs("hello there"))
    assert score <= 30


def test_score_code_keywords():
    score = score_complexity(_msgs("implement a function to sort a list"))
    # 7 words → base 0; "implement" and "function" each add 10 → at least 20
    assert score >= 20


def test_score_reasoning_keywords():
    score = score_complexity(_msgs("explain and analyse the results"))
    # "explain" and "analyse" each add 10 → at least 20
    assert score >= 20


def test_score_long_prompt():
    # 200-word message → base score of min(200//10, 50) = 20, non-trivial
    long_text = " ".join(["word"] * 200)
    score = score_complexity(_msgs(long_text))
    assert score >= 20


def test_score_caps_at_100():
    # Flood with every keyword and many words
    keywords = "implement debug function algorithm sql regex code class refactor script"
    text = " ".join([keywords] * 20)
    score = score_complexity(_msgs(text))
    assert score == 100


def test_score_empty_messages():
    assert score_complexity([]) == 0


def test_score_multiple_messages():
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "explain recursion"},
    ]
    score = score_complexity(msgs)
    # "explain" adds 10 → at least 10
    assert score >= 10


# ---------------------------------------------------------------------------
# resolve_model tests
# ---------------------------------------------------------------------------


def test_resolve_fast_tier():
    # Short, plain prompt → score ≤ 30 → fast
    model = resolve_model(_msgs("hi"), SAMPLE_CONFIG)
    assert model == "gpt-3.5-turbo"


def test_resolve_powerful_tier():
    # Lots of code + reasoning keywords → score > 70 → powerful
    text = "explain and analyse implement debug function algorithm sql regex code class refactor"
    model = resolve_model(_msgs(text), SAMPLE_CONFIG)
    assert model == "gpt-4o"


def test_resolve_balanced_tier():
    # 300 words → base score = min(300//10, 50) = 30; + "explain" = 10 → total 40 → balanced
    text = " ".join(["word"] * 300) + " explain"
    model = resolve_model(_msgs(text), SAMPLE_CONFIG)
    assert model == "gpt-4o-mini"


def test_resolve_missing_config():
    # Empty config → falls back to hardcoded default without raising
    model = resolve_model(_msgs("hello"), {})
    assert isinstance(model, str)
    assert len(model) > 0


def test_resolve_malformed_config():
    # Config missing required keys → falls back gracefully
    model = resolve_model(_msgs("hello"), {"tiers": {}, "routing": {}})
    assert isinstance(model, str)
    assert len(model) > 0
