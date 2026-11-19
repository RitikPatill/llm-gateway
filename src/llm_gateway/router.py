"""Complexity-based model router.

score_complexity() — pure heuristic, no I/O, deterministic.
resolve_model()    — maps score to a tier and returns the model ID string.
"""

REASONING_KEYWORDS = frozenset({
    "think", "explain", "reason", "compare", "analyse", "analyze",
    "evaluate", "argue",
})
CODE_KEYWORDS = frozenset({
    "code", "function", "implement", "debug", "algorithm", "sql", "regex",
    "script", "class", "refactor",
})


def score_complexity(messages: list[dict]) -> int:
    """Return an integer 0–100 representing prompt complexity.

    Heuristic:
      - base score = min(total_word_count // 10, 50)
      - +10 for each distinct reasoning keyword present
      - +10 for each distinct code keyword present
      - capped at 100
    """
    words: list[str] = []
    for m in messages:
        content = m.get("content") or ""
        words.extend(content.lower().split())

    score = min(len(words) // 10, 50)

    word_set = set(words)
    score += 10 * len(REASONING_KEYWORDS & word_set)
    score += 10 * len(CODE_KEYWORDS & word_set)

    return min(score, 100)


def resolve_model(messages: list[dict], routes_config: dict) -> str:
    """Return the model ID string for the given messages and routing config.

    Falls back to the default_tier model on any config access error.
    """
    try:
        routing = routes_config["routing"]
        tiers = routes_config["tiers"]

        score = score_complexity(messages)

        if score <= routing["fast_threshold"]:
            tier = "fast"
        elif score <= routing["balanced_threshold"]:
            tier = "balanced"
        else:
            tier = "powerful"

        return tiers[tier]["model"]
    except Exception:
        # Graceful fallback: try the default_tier, then a hardcoded safe default
        try:
            return routes_config["tiers"][routes_config["routing"]["default_tier"]]["model"]
        except Exception:
            return "gpt-4o-mini"
