# AI Analyst — Intent Classification Refactor Plan

> **Focus:** Moving from fragile regex heuristics to reliable semantic intent recognition
> **Core file:** `ai_client.py`
> **Total phases:** 3 | **Total tasks:** 8

---

## Summary

The root cause of missed intent is that the classifier is doing two jobs badly — understanding what the user wants *and* matching it to column names — with no knowledge of the actual loaded dataset. Those two concerns need to be separated.

The current system uses a single giant `if-elif` regex chain in `ai_client.py`. When a query doesn't match a pattern exactly, it silently falls back to a general AI response. That silent fallback is how hallucinations slip through. The fix is a structured two-stage router that knows about the actual dataset schema at classify time.

---

## Phase 1 — Do First: Replace Regex with a Structured Intent Router

**Priority:** Highest ROI. This is the core of the hallucination problem.

### Task 1.1 — Build a Two-Stage Intent Classifier

**File:** `ai_client.py` → `classify_intent()`
**Effort:** ~1 day

Stop using one giant if-elif regex chain. Split into two stages:

1. A **keyword extractor** that pulls operation tokens (`sum`, `average`, `top`, `who`, `count`) and entity tokens (column names from the loaded dataset).
2. A **decision matrix** that maps `(operation, entity_type)` pairs to a computation path.

This decouples vocabulary from logic and makes each path testable independently.

**Before — fragile, order-dependent:**

```python
if re.search(r'\b(sum|total)\b', query, re.I):
    ...
elif re.search(r'\baverage\b', query, re.I):
    ...
```

**After — explicit intent object:**

```python
intent = extract_intent(query, dataset_schema)
# → { op: "aggregate", fn: "sum", col: "revenue", filter: None }
result = DISPATCH[intent.op](df, intent)
```

**Impact:** Fixes the majority of "intent not recognized" cases immediately.

---

### Task 1.2 — Inject Dataset Schema into Every Intent Extraction Call

**File:** `ai_client.py` + `session_manager.py`
**Effort:** ~2–3 hours

The root cause of missed intent is that the classifier doesn't know what columns exist. A user asking "what's the total revenue?" won't match "revenue" unless the classifier knows that column is present. Pass the column list and their inferred types (`numeric`, `categorical`, `date`, `employee`) into the intent extractor so it can resolve ambiguous words against actual column names using fuzzy matching.

```python
# Retrieve schema from session and pass it in
schema = session_manager.get_schema(session_id)
# → { "revenue": "numeric", "name": "employee", "date": "datetime" }
intent = extract_intent(query, schema=schema)
```

**Why this matters:** Right now a user asking "show me total sales by region" might not trigger your Pandas path simply because "region" wasn't in the regex — even though the dataset has a column called `Region`. Resolving tokens against actual column names at classify time fixes this.

---

### Task 1.3 — Add a Low-Confidence Fallback Path with Explicit Logging

**File:** `ai_client.py`
**Effort:** ~2 hours

Right now when intent fails to match, it silently falls back to a general AI response. That's how hallucinations slip through. Instead:

1. Add a **confidence score** to your intent result.
2. If confidence is below threshold, **log the query and the failed match**.
3. Prepend a note to the AI's prompt instructing it to *not compute numbers* — only describe what it sees.

A response that says "I can see a revenue column but I'm not sure what computation you want" is better than a confidently wrong number.

```python
intent = extract_intent(query, schema=schema)

if intent.confidence < CONFIDENCE_THRESHOLD:
    logger.warning(f"Low-confidence intent: {intent} | query: {query}")
    system_prompt += "\n\nIMPORTANT: You are not confident in the user's intent. " \
                     "Do NOT compute or cite any numbers. Describe the data only."
```

---

## Phase 2 — Do Second: Fix Column Detection and JSON Command Parsing

**Priority:** Removes the two other silent failure modes.

### Task 2.1 — Replace Fuzzy Column Detection with Schema-First Matching

**File:** `data_processor.py` → `detect_columns()`
**Effort:** ~3 hours

Column heuristics (Title Case, underscores, fuzzy name matching) fire false positives because they work on strings rather than data shapes. Improve the signal by layering three checks:

1. **Keyword hint** — match against a curated keyword list (`name`, `employee`, `staff`, `agent`).
2. **dtype check** — validate the column's actual Pandas dtype.
3. **Cardinality check** — a numeric column named `staff_count` is not an employee column; cardinality catches that.

```python
EMPLOYEE_KEYWORDS = {"name", "employee", "staff", "agent", "rep", "associate"}

def is_employee_col(col: str, series: pd.Series) -> bool:
    keyword_hit = any(k in col.lower() for k in EMPLOYEE_KEYWORDS)
    is_string   = pd.api.types.is_string_dtype(series)
    cardinality = series.nunique()
    return keyword_hit and is_string and cardinality > 5
```

Apply the same layered approach to numeric, date, and item columns.

---

### Task 2.2 — Make JSON Command Parsing Fault-Tolerant

**File:** `ai_client.py` → `parse_command()`
**Effort:** ~2 hours

The `regex → json.loads()` pattern fails silently whenever the LLM drops a backtick or adds an unexpected key. Fix this with three layers:

1. **Strip all markdown fencing** aggressively before parsing.
2. **Use a Pydantic model** to validate and coerce the parsed object rather than assuming keys exist.
3. If parsing still fails, **log the raw LLM response and return explicit `None`** — never silently drop the command.

```python
from pydantic import BaseModel, ValidationError

class Command(BaseModel):
    op: str
    col: str | None = None
    fn: str | None = None
    filter: str | None = None

def parse_command(llm_output: str) -> Command | None:
    # Strip aggressively before parsing
    raw = re.sub(r'```(?:json)?\s*|\s*```', '', llm_output).strip()
    try:
        return Command.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"Command parse failed: {e}\nRaw: {raw}")
        return None  # explicit None, never silent drop
```

---

### Task 2.3 — Improve the System Prompt to Guide JSON Compliance

**File:** `ai_client.py` → `build_system_prompt()`
**Effort:** ~1 hour

LLMs deviate from JSON format when the system prompt doesn't show a concrete example. Add an explicit JSON schema example with a DO and DON'T pair. Also instruct the model to always include the command block even for conversational questions — your parser handles the no-computation case, so false positives are fine but false negatives cause hallucinations.

**Add to system prompt:**

```
Always respond with a JSON command block before your answer, wrapped in triple backticks.

DO:
```json
{ "op": "aggregate", "fn": "sum", "col": "revenue" }
```

DON'T:
- Skip the JSON block if you think the question is conversational
- Add extra keys not in the schema
- Use natural language inside the JSON block

If no computation is needed, use: {"op": "none"}
```

---

## Phase 3 — Polish: Observability and Testing Harness

**Priority:** Makes the above improvements maintainable and measurable.

### Task 3.1 — Build an Intent Classification Test Suite

**File:** `tests/test_intent.py` (new file)
**Effort:** ~3 hours

Create a pytest suite with ~20 golden query examples paired with expected intent outputs. Run it every time you change the classifier. This turns "I think it's better now" into "intent accuracy: 91% → 97%".

```python
import pytest
from ai_client import extract_intent

SCHEMA = {
    "revenue": "numeric",
    "region": "categorical",
    "employee_name": "employee",
    "date": "datetime",
}

GOLDEN = [
    # Aggregation
    ("what is the total revenue?",          {"op": "aggregate", "fn": "sum",   "col": "revenue"}),
    ("show me average revenue by region",   {"op": "aggregate", "fn": "mean",  "col": "revenue"}),
    ("how many rows are there?",            {"op": "count",     "fn": None,     "col": None}),
    # Ranking
    ("who are the top 5 employees?",        {"op": "rank",      "fn": "top",    "col": "employee_name"}),
    ("bottom 3 regions by revenue",         {"op": "rank",      "fn": "bottom", "col": "revenue"}),
    # Lookup
    ("show me row 42",                      {"op": "lookup",    "fn": None,     "col": None}),
    # Ambiguous — should produce low confidence
    ("tell me about the data",              {"op": "none"}),
    ("what do you think of this dataset?",  {"op": "none"}),
]

@pytest.mark.parametrize("query, expected", GOLDEN)
def test_intent(query, expected):
    intent = extract_intent(query, schema=SCHEMA)
    for key, val in expected.items():
        assert getattr(intent, key, None) == val, f"Failed on: '{query}'"
```

Start with the top 10 queries your client actually types, then expand from there.

---

### Task 3.2 — Add Structured Intent Logging to Every Response

**File:** `app.py` → `/chat` route
**Effort:** ~1 hour

Log the resolved intent object alongside every response — not just errors. This creates a goldmine: you can review real user queries, see which paths fired, and spot patterns in misclassification without waiting for a bug report.

```python
# In your /chat route, after intent classification:
logger.info(json.dumps({
    "event":      "intent_resolved",
    "session_id": session_id,
    "query":      user_query,
    "intent":     intent.model_dump(),
    "confidence": intent.confidence,
    "path":       "pandas" if intent.confidence >= THRESHOLD else "ai_fallback",
}))
```

A single JSON log line per request is enough. No new dependencies needed.

---

## Execution Order Summary

| # | Task | File | Effort | Impact |
|---|------|------|--------|--------|
| 1 | Two-stage intent classifier | `ai_client.py` | 1 day | 🔴 High |
| 2 | Inject dataset schema into classifier | `ai_client.py`, `session_manager.py` | 2–3 hrs | 🔴 High |
| 3 | Low-confidence fallback + logging | `ai_client.py` | 2 hrs | 🟡 Medium |
| 4 | Schema-first column detection | `data_processor.py` | 3 hrs | 🟡 Medium |
| 5 | Fault-tolerant JSON command parsing | `ai_client.py` | 2 hrs | 🟡 Medium |
| 6 | System prompt JSON compliance | `ai_client.py` | 1 hr | 🟡 Medium |
| 7 | Intent test suite (20 golden cases) | `tests/test_intent.py` | 3 hrs | 🟢 Long-term |
| 8 | Structured intent logging | `app.py` | 1 hr | 🟢 Long-term |

**Total estimated effort:** ~3–4 focused days

---

## Key Principle

The goal of Phase 1 is not to make the regex smarter — it's to eliminate regex as the decision-making layer entirely. Regex is fine for token extraction; it's not fine for routing. Once you have an explicit `intent` object with a `confidence` score, every other improvement (column detection, JSON parsing, logging) becomes straightforward because you have a clear contract to test against.
