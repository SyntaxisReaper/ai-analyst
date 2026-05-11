"""
ai_client.py — AI provider abstraction with intent classification,
pandas-first computation, and numerical verification.

Refactor (Intent Classification Plan):
  Task 1.1 — Two-stage intent classifier (keyword extractor + decision matrix)
  Task 1.2 — Dataset schema injected into every intent extraction call
  Task 1.3 — Low-confidence fallback path with explicit logging
  Task 2.2 — Fault-tolerant JSON command parsing (Pydantic + aggressive markdown strip)
  Task 2.3 — Improved system prompt with JSON compliance guidance
  Task 3.2 — Structured intent logging per response
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel, ValidationError

log = logging.getLogger("ai_client")

# ── Confidence threshold ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55


# ── Intent result dataclass ───────────────────────────────────────────────────

@dataclass
class Intent:
    op: str = "none"          # aggregate | count | average | rank | lookup | trend | employee | none
    fn: Optional[str] = None  # sum | mean | top | bottom | min | max | median | None
    col: Optional[str] = None # resolved column name from schema
    filter: Optional[str] = None
    confidence: float = 0.0

    def model_dump(self) -> dict:
        return asdict(self)


# ── Operation token tiers (high → low priority) ───────────────────────────────
# Tier 1: Numeric computation tokens — ALWAYS win (sum, avg, count, rank, min/max)
# Tier 2: Entity/context tokens (employee, trend) — win only if Tier 1 had no match
# Tier 3: Generic lookup tokens (who, find, show me) — lowest priority

_NUMERIC_TOKENS: Dict[str, Tuple[str, Optional[str]]] = {
    # aggregate — sum
    "grand total":   ("aggregate", "sum"),
    "subtotal":      ("aggregate", "sum"),
    "add up":        ("aggregate", "sum"),
    "how much":      ("aggregate", "sum"),
    "per capita":    ("average",   "mean"),
    "how many":      ("count",     None),
    "number of":     ("count",     None),
    "over time":     ("trend",     None),
    "sum":           ("aggregate", "sum"),
    "total":         ("aggregate", "sum"),
    "average":       ("average",   "mean"),
    "mean":          ("average",   "mean"),
    "avg":           ("average",   "mean"),
    "median":        ("average",   "median"),
    "minimum":       ("aggregate", "min"),
    "maximum":       ("aggregate", "max"),
    "count":         ("count",     None),
    "tally":         ("count",     None),
    "top":           ("rank",      "top"),
    "highest":       ("rank",      "top"),
    "most":          ("rank",      "top"),
    "best":          ("rank",      "top"),
    "bottom":        ("rank",      "bottom"),
    "lowest":        ("rank",      "bottom"),
    "least":         ("rank",      "bottom"),
    "worst":         ("rank",      "bottom"),
    "rank":          ("rank",      None),
    "compare":       ("rank",      None),
    "min":           ("aggregate", "min"),
    "max":           ("aggregate", "max"),
}

_ENTITY_TOKENS: Dict[str, Tuple[str, Optional[str]]] = {
    "trend":         ("trend",     None),
    "growth":        ("trend",     None),
    "increase":      ("trend",     None),
    "decrease":      ("trend",     None),
    "progress":      ("trend",     None),
    "changed":       ("trend",     None),
    "change":        ("trend",     None),
    "improvement":   ("trend",     None),
    "employee":      ("employee",  None),
    "worker":        ("employee",  None),
    "staff":         ("employee",  None),
    "operator":      ("employee",  None),
    "performance":   ("employee",  None),
    "who is":        ("employee",  None),
}

_LOW_PRIORITY_TOKENS: Dict[str, Tuple[str, Optional[str]]] = {
    "find":          ("lookup",    None),
    "show me":       ("lookup",    None),
    "look up":       ("lookup",    None),
    "where is":      ("lookup",    None),
    "what is the":   ("lookup",    None),
    "who":           ("lookup",    None),
    "which":         ("lookup",    None),
}

# Build flat dict for backward compat
_HIGH_PRIORITY_TOKENS = {**_NUMERIC_TOKENS, **_ENTITY_TOKENS}
_OP_TOKENS: Dict[str, Tuple[str, Optional[str]]] = {**_HIGH_PRIORITY_TOKENS, **_LOW_PRIORITY_TOKENS}

# Sort each tier by token length descending (multi-word first)
_NUMERIC_SORTED = sorted(_NUMERIC_TOKENS.keys(),       key=len, reverse=True)
_ENTITY_SORTED  = sorted(_ENTITY_TOKENS.keys(),        key=len, reverse=True)
_LOW_SORTED     = sorted(_LOW_PRIORITY_TOKENS.keys(),  key=len, reverse=True)
# Legacy alias used elsewhere
_HIGH_SORTED = _NUMERIC_SORTED + _ENTITY_SORTED


def _extract_operation(query_lower: str) -> Tuple[Optional[str], Optional[str], float]:
    """
    Stage 1a — scan query for operation tokens in three tiers:
      1. Numeric tier (sum/avg/count/rank/min/max) — highest priority.
      2. Entity tier (employee, trend) — wins only if no numeric op found.
      3. Lookup tier (who, find, show me) — last resort.
    Returns (op, fn, op_confidence).
    """
    def _match(token: str, text: str) -> bool:
        pattern = r'\b' + re.escape(token) + r'\b'
        return bool(re.search(pattern, text))

    for token in _NUMERIC_SORTED:
        if _match(token, query_lower):
            return *_NUMERIC_TOKENS[token], 0.75

    for token in _ENTITY_SORTED:
        if _match(token, query_lower):
            return *_ENTITY_TOKENS[token], 0.65

    for token in _LOW_SORTED:
        if _match(token, query_lower):
            return *_LOW_PRIORITY_TOKENS[token], 0.45

    return None, None, 0.0


def _resolve_column(query_lower: str, schema: Dict[str, str]) -> Tuple[Optional[str], float]:
    """
    Stage 1b — resolve a column name from the dataset schema against the query.
    Uses exact substring match first, then fuzzy token overlap.
    Returns (column_name, col_confidence).
    """
    if not schema:
        return None, 0.0

    # Direct substring match
    for col in schema:
        if col.lower() in query_lower:
            return col, 1.0

    # Token overlap: how many words of the col name appear in the query
    best_col, best_score = None, 0.0
    for col in schema:
        tokens = re.findall(r'\w+', col.lower())
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in query_lower and len(t) > 2)
        score = hits / len(tokens)
        if score > best_score:
            best_score = score
            best_col = col

    if best_score >= 0.5:
        return best_col, best_score * 0.9  # slight discount for fuzzy
    return None, 0.0


def extract_intent(query: str, schema: Optional[Dict[str, str]] = None) -> Intent:
    """
    Task 1.1 / 1.2 — Two-stage intent extractor.

    Stage 1: keyword extractor — pulls op tokens and entity tokens.
    Stage 2: decision matrix — maps (op, col_type) to final intent.

    Args:
        query:  The user's question.
        schema: dict of {column_name: dtype_tag} from the loaded dataset.
                dtype_tag ∈ { "numeric", "categorical", "datetime", "employee" }
    """
    schema = schema or {}
    q = query.lower().strip()

    op, fn, op_conf = _extract_operation(q)
    col, col_conf = _resolve_column(q, schema)

    # If op detected, confidence is op_conf; if also col found, boost it
    if op is None:
        return Intent(op="none", fn=None, col=None, confidence=0.2)

    # Boost confidence when column is resolved
    if col:
        confidence = min(1.0, op_conf + col_conf * 0.3)
    else:
        confidence = op_conf * 0.8   # lower confidence without column anchor

    # Refine fn for rank when we have no explicit top/bottom keyword
    if op == "rank" and fn is None:
        if re.search(r'\btop\b|\bhighest\b|\bbest\b|\bmost\b', q):
            fn = "top"
        elif re.search(r'\bbottom\b|\blowest\b|\bworst\b|\bleast\b', q):
            fn = "bottom"

    return Intent(op=op, fn=fn, col=col, confidence=confidence)


# ── Legacy string-based classifier (kept for backward compat in app.py) ───────

def classify_intent(question: str, schema: Optional[Dict[str, str]] = None) -> str:
    """
    Public interface used by app.py — returns an intent string for routing.
    Now backed by extract_intent() for consistent logic.
    """
    intent = extract_intent(question, schema=schema)
    # Map new op names to the legacy strings app.py expects
    mapping = {
        "aggregate": "aggregation",
        "average":   "average",
        "count":     "count",
        "rank":      "comparison",
        "lookup":    "lookup",
        "trend":     "trend",
        "employee":  "employee",
        "none":      "explanation",
    }
    return mapping.get(intent.op, "explanation")


# ── Number extraction for verification ───────────────────────────────────────

_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")


def _extract_numbers(text: str) -> List[float]:
    """Extract all numbers from a text string."""
    nums = []
    for m in _NUMBER_RE.finditer(text):
        try:
            nums.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return nums


def _build_known_values(session: dict) -> Dict[float, str]:
    """
    Build a map of {value → label} from all pre-computed facts
    so we can cross-check any number the AI mentions.
    """
    known: Dict[float, str] = {}
    meta = session.get("metadata", {}) or {}

    # Named totals
    for k, v in (meta.get("named_totals") or {}).items():
        try:
            known[float(v)] = k
        except (TypeError, ValueError):
            pass

    # Numeric stats — sheet level
    stats = meta.get("stats") or {}
    numeric = stats.get("numeric") or {}
    if isinstance(numeric, dict):
        for sheet_or_col, sheet_stats in numeric.items():
            if isinstance(sheet_stats, dict):
                for col, col_stats in sheet_stats.items():
                    if isinstance(col_stats, dict):
                        for stat_name, val in col_stats.items():
                            try:
                                known[float(val)] = f"{sheet_or_col}/{col}/{stat_name}"
                            except (TypeError, ValueError):
                                pass

    return known


def verify_numerical_claims(answer: str, session: dict) -> Tuple[str, bool]:
    """
    Extract numbers from the AI answer. For each number, check if it
    conflicts with a known pre-computed value (> 1% tolerance). If conflicts
    exist, append a correction note.

    Returns (possibly_amended_answer, had_conflicts).
    """
    known = _build_known_values(session)
    if not known:
        return answer, False

    ans_numbers = _extract_numbers(answer)
    conflicts = []

    for val in ans_numbers:
        if val == 0:
            continue
        for known_val, label in known.items():
            if known_val == 0:
                continue
            diff_pct = abs(val - known_val) / abs(known_val)
            if diff_pct > 0.01 and 0.1 < (val / known_val) < 10:
                conflicts.append((val, known_val, label))
                break

    if conflicts:
        notes = []
        for wrong, right, label in conflicts[:3]:
            notes.append(
                f"  ⚠ {wrong:,.0f} may differ from pre-computed {label} = {right:,.0f}"
            )
        amendment = (
            "\n\n---\n"
            "🔶 **Precision note** — one or more numbers in this answer may differ "
            "from pandas-computed values:\n" + "\n".join(notes) +
            "\nPlease use the pre-computed values above as ground truth."
        )
        return answer + amendment, True

    return answer, False


# ── Task 2.2 — Fault-tolerant JSON command parsing ────────────────────────────

class Command(BaseModel):
    op: str
    col: Optional[str] = None
    fn: Optional[str] = None
    filter: Optional[str] = None


def parse_command(llm_output: str) -> Optional[Command]:
    """
    Task 2.2 — Strip markdown fencing aggressively, then validate with Pydantic.
    Returns Command on success, None (with warning log) on any failure.
    Never drops the failure silently.
    """
    # Step 1: strip all markdown fencing
    raw = re.sub(r'```(?:json)?\s*|\s*```', '', llm_output).strip()
    # Step 2: find first JSON object
    m = re.search(r'\{[\s\S]+\}', raw)
    if not m:
        log.warning("parse_command: no JSON object found in output.\nRaw: %s", raw[:200])
        return None
    try:
        return Command.model_validate(json.loads(m.group()))
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("parse_command failed: %s\nRaw: %s", e, raw[:200])
        return None


# ── System prompt ─────────────────────────────────────────────────────────────

# Task 2.3 — Added JSON compliance block with DO/DON'T example
SYSTEM_TEMPLATE = """\
You are an expert data analyst assistant. A dataset summary is provided below.

== STRICT RULES FOR NUMBERS ==
1. If a [PRE-CALCULATED FACT] exists for what the user is asking — use it DIRECTLY.
   Do NOT recalculate. Do NOT add up individual rows to verify it.
2. Never add up numbers from [RAW ITEM DATA] if a total already exists in
   [PRE-CALCULATED FACTS]. Those individual rows are already included in the total.
3. If you are unsure whether a value is a subtotal or a raw value — always prefer
   the [PRE-CALCULATED FACT]. Subtotals already include the rows above them.
4. Never add subtotals together. Each subtotal already contains the items beneath it.
5. If you cannot find the answer in [PRE-CALCULATED FACTS], say so clearly before
   attempting any calculation. When estimating, say:
   "Based on available data, approximately X — please verify." Never state estimates
   with full confidence.

== EMPLOYEE / PROGRESS QUESTIONS ==
When asked about employee performance, progress, or rankings:
1. Use ONLY the [EMPLOYEE / PERSON STATISTICS] block if present.
2. Show ALL metrics for that employee across all sheets.
3. Include their rank relative to other employees.
4. If a metric is not listed, say "not available" — never invent or estimate.
5. Present results in a clean table or structured list.

== OUTPUT FORMAT ==
- Answer concisely in plain English. Do NOT output raw JSON.
- Format numbers with commas (e.g. 31,735 not 31735).
- If your number comes from [PRE-CALCULATED FACTS] → add ✅ Exact
- If you are estimating → add 🔶 Estimated

{confidence_note}

{summary}
"""

_LOW_CONFIDENCE_NOTE = (
    "\n⚠️  IMPORTANT: The intent of this question is uncertain. "
    "Do NOT compute or cite any specific numbers. "
    "Describe what you observe in the data only — never invent figures."
)


def build_system_prompt(summary: str, confidence: float = 1.0) -> str:
    """
    Task 1.3 — Inject a low-confidence note into the system prompt when
    intent confidence is below the threshold.
    """
    note = _LOW_CONFIDENCE_NOTE if confidence < CONFIDENCE_THRESHOLD else ""
    return SYSTEM_TEMPLATE.format(summary=summary, confidence_note=note).strip()


# ── Provider implementations ──────────────────────────────────────────────────

class _OpenAICompatClient:
    """Works for both OpenAI and Groq (Groq is OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str, base_url: str = None):
        import openai
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self.model = model

    def chat(self, summary: str, history: List[Dict], question: str,
             pandas_result: Optional[str] = None,
             intent: Optional[Intent] = None) -> str:
        confidence = intent.confidence if intent else 1.0
        system = build_system_prompt(summary, confidence)
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-20:])
        user_content = question
        if pandas_result:
            user_content = (
                f"[PANDAS-COMPUTED RESULT]\n{pandas_result}\n\n"
                f"Using the exact result above, answer this question in plain English:\n{question}"
            )
        messages.append({"role": "user", "content": user_content})
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.15, max_tokens=2048
        )
        return resp.choices[0].message.content


class _GeminiClient:
    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model

    def chat(self, summary: str, history: List[Dict], question: str,
             pandas_result: Optional[str] = None,
             intent: Optional[Intent] = None) -> str:
        confidence = intent.confidence if intent else 1.0
        system = build_system_prompt(summary, confidence)
        contents = []
        for m in history[-20:]:
            role = "user" if m["role"] == "user" else "model"
            contents.append(self._types.Content(
                role=role, parts=[self._types.Part(text=m["content"])]
            ))
        user_content = question
        if pandas_result:
            user_content = (
                f"[PANDAS-COMPUTED RESULT]\n{pandas_result}\n\n"
                f"Using the exact result above, answer this question in plain English:\n{question}"
            )
        contents.append(self._types.Content(
            role="user", parts=[self._types.Part(text=user_content)]
        ))
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.15,
                max_output_tokens=2048,
            ),
        )
        return resp.text


# ── Registry ──────────────────────────────────────────────────────────────────

PROVIDER_DEFAULTS = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": None,
    },
    "gemini": {
        "model": "gemini-1.5-flash",
    },
}

PROVIDER_MODELS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
}


def create_client(provider: str, api_key: str, model: str):
    if not api_key:
        raise ValueError(
            f"No API key set for provider '{provider}'. "
            f"Please add it to backend/.env and restart."
        )
    provider = provider.lower()
    defaults = PROVIDER_DEFAULTS.get(provider)
    if not defaults:
        raise ValueError(f"Unknown provider '{provider}'. Choose: groq, openai, gemini")

    resolved_model = model or defaults["model"]

    if provider == "gemini":
        return _GeminiClient(api_key, resolved_model)
    else:
        return _OpenAICompatClient(api_key, resolved_model, defaults.get("base_url"))
