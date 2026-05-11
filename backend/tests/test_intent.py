"""
tests/test_intent.py — Task 3.1: Intent classification test suite.

Run with:
    cd backend
    pytest tests/test_intent.py -v

Golden set covers: aggregation, average, count, rank, lookup, employee, trend,
and ambiguous / low-confidence queries.  Each test checks the resolved Intent
object for correctness of op, fn, and col.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ai_client import extract_intent, CONFIDENCE_THRESHOLD

# ── Representative schema ────────────────────────────────────────────────────
SCHEMA = {
    "revenue":       "numeric",
    "sales":         "numeric",
    "production":    "numeric",
    "region":        "categorical",
    "product":       "categorical",
    "employee_name": "employee",
    "date":          "datetime",
    "month":         "datetime",
}

# ── Golden query pairs ────────────────────────────────────────────────────────
# Format: (query, expected_op, expected_fn, expected_col_partial_or_None)
# expected_col_partial: None means "don't care"; otherwise assert col starts with / equals value
GOLDEN = [
    # Aggregation — sum
    ("what is the total revenue?",            "aggregate", "sum",    "revenue"),
    ("sum of sales for this month",           "aggregate", "sum",    "sales"),
    ("grand total of production",             "aggregate", "sum",    "production"),
    ("how much revenue did we make?",         "aggregate", "sum",    "revenue"),
    # Aggregation — mean
    ("show me average revenue by region",     "average",   "mean",   "revenue"),
    ("what is the mean sales?",               "average",   "mean",   "sales"),
    ("avg production per employee",           "average",   "mean",   "production"),
    # Aggregation — min/max
    ("what is the minimum revenue?",          "aggregate", "min",    "revenue"),
    ("maximum sales value",                   "aggregate", "max",    "sales"),
    # Count
    ("how many rows are there?",              "count",     None,     None),
    ("count of employees",                    "count",     None,     None),
    ("number of regions",                     "count",     None,     None),
    # Rank / comparison
    ("who are the top 5 employees?",          "rank",      "top",    None),
    ("bottom 3 regions by revenue",           "rank",      "bottom", "revenue"),
    ("highest revenue region",                "rank",      "top",    "revenue"),
    ("lowest sales month",                    "rank",      "bottom", "sales"),
    # Lookup
    ("show me row 42",                        "lookup",    None,     None),
    ("find production for region north",      "lookup",    None,     "production"),
    # Employee
    ("what is the performance of employee X?","employee",  None,     None),
    ("who is the best worker?",               "rank",      "top",    None),  # 'best' = rank/top (correct routing)
    # Trend
    ("show the revenue trend over time",      "trend",     None,     "revenue"),
    ("how has production changed?",           "trend",     None,     "production"),
    # Ambiguous — should produce op=none or low confidence
    ("tell me about the data",                None,        None,     None),   # op can be 'none'
    ("what do you think of this dataset?",    None,        None,     None),
]


@pytest.mark.parametrize("query,expected_op,expected_fn,expected_col", GOLDEN)
def test_intent_op(query, expected_op, expected_fn, expected_col):
    intent = extract_intent(query, schema=SCHEMA)

    if expected_op is None:
        # Ambiguous — confidence must be below threshold OR op must be 'none'
        assert intent.op == "none" or intent.confidence < CONFIDENCE_THRESHOLD, (
            f"Expected low-confidence/none for '{query}', got op={intent.op} conf={intent.confidence:.2f}"
        )
        return

    assert intent.op == expected_op, (
        f"Query: '{query}'\n  Expected op={expected_op}, got op={intent.op}"
    )

    if expected_fn is not None:
        assert intent.fn == expected_fn, (
            f"Query: '{query}'\n  Expected fn={expected_fn}, got fn={intent.fn}"
        )

    if expected_col is not None:
        assert intent.col is not None, (
            f"Query: '{query}'\n  Expected col containing '{expected_col}', got None"
        )
        assert expected_col.lower() in intent.col.lower(), (
            f"Query: '{query}'\n  Expected col containing '{expected_col}', got col='{intent.col}'"
        )


def test_confidence_above_threshold_for_clear_queries():
    """High-signal queries should exceed the confidence threshold."""
    clear_queries = [
        "total revenue",
        "average sales",
        "top 5 employees by production",
        "minimum revenue",
    ]
    for q in clear_queries:
        intent = extract_intent(q, schema=SCHEMA)
        assert intent.confidence >= CONFIDENCE_THRESHOLD, (
            f"Expected confidence >= {CONFIDENCE_THRESHOLD} for '{q}', got {intent.confidence:.2f}"
        )


def test_confidence_low_for_vague_queries():
    """Vague queries should fall below the confidence threshold."""
    vague_queries = [
        "tell me about the data",
        "what do you think?",
        "interesting",
        "explain",
    ]
    for q in vague_queries:
        intent = extract_intent(q, schema=SCHEMA)
        assert intent.confidence < CONFIDENCE_THRESHOLD or intent.op == "none", (
            f"Expected low confidence for '{q}', got op={intent.op} conf={intent.confidence:.2f}"
        )


def test_schema_column_resolution():
    """The classifier must resolve column names from the schema, not just regex."""
    # 'revenue' is in the schema — should be resolved even with casual phrasing
    intent = extract_intent("give me the total for revenue", schema=SCHEMA)
    assert intent.col == "revenue", f"Expected col=revenue, got col={intent.col}"


def test_no_schema_still_returns_intent():
    """extract_intent must work even when no schema is provided."""
    intent = extract_intent("what is the total revenue?", schema={})
    assert intent.op == "aggregate"
    assert intent.fn == "sum"
    assert intent.col is None  # no schema → can't resolve column
