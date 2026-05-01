"""
ai_client.py — AI provider abstraction with intent classification,
pandas-first computation, and numerical verification.

Priority legend (from plan):
  P0.3 — classify_intent()
  P1.3 — employee progress prompt section in system prompt
  P2.1 — verify_numerical_claims()
  P2.2 — confidence flagging in system prompt
  P2.3 — structured COMPUTED FACTS block injected into prompt
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

log = logging.getLogger("ai_client")

# ── Intent patterns ───────────────────────────────────────────────────────────

_AGGREGATION_PATTERNS = re.compile(
    r"\b(sum|total|add up|how much|grand total|subtotal)\b", re.I
)
_COUNT_PATTERNS = re.compile(
    r"\b(count|how many|number of|tally)\b", re.I
)
_AVERAGE_PATTERNS = re.compile(
    r"\b(average|mean|avg|per capita)\b", re.I
)
_LOOKUP_PATTERNS = re.compile(
    r"\b(find|who|which|where is|look up|show me|what is the .+? of)\b", re.I
)
_COMPARISON_PATTERNS = re.compile(
    r"\b(highest|lowest|most|least|rank|top|bottom|best|worst|compare)\b", re.I
)
_TREND_PATTERNS = re.compile(
    r"\b(trend|growth|change|over time|increase|decrease|progress|improvement)\b", re.I
)
_EMPLOYEE_PATTERNS = re.compile(
    r"\b(employee|worker|staff|person|operator|performance|progress of|progress for|who is)\b", re.I
)


def classify_intent(question: str) -> str:
    """
    P0.3 — Classify the question into one of these intent categories:
      aggregation | count | average | lookup | comparison | trend | employee | explanation
    """
    q = question.lower()
    if _EMPLOYEE_PATTERNS.search(q):
        return "employee"
    if _AGGREGATION_PATTERNS.search(q):
        return "aggregation"
    if _COUNT_PATTERNS.search(q):
        return "count"
    if _AVERAGE_PATTERNS.search(q):
        return "average"
    if _COMPARISON_PATTERNS.search(q):
        return "comparison"
    if _TREND_PATTERNS.search(q):
        return "trend"
    if _LOOKUP_PATTERNS.search(q):
        return "lookup"
    return "explanation"


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
    P2.1 — Build a map of {value → label} from all pre-computed facts
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
                # nested: {sheet: {col: {sum, mean,...}}}
                for col, col_stats in sheet_stats.items():
                    if isinstance(col_stats, dict):
                        for stat_name, val in col_stats.items():
                            try:
                                known[float(val)] = f"{sheet_or_col}/{col}/{stat_name}"
                            except (TypeError, ValueError):
                                pass
                    else:
                        # flat: {col: {sum, ...}}
                        try:
                            known[float(sheet_stats[col])] = f"{sheet_or_col}/{col}"
                        except (TypeError, ValueError):
                            pass

    return known


def verify_numerical_claims(answer: str, session: dict) -> Tuple[str, bool]:
    """
    P2.1 — Extract numbers from the AI answer. For each number, check if it
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
            # Only flag if values are in same order of magnitude and conflict
            if diff_pct > 0.01 and 0.1 < (val / known_val) < 10:
                conflicts.append((val, known_val, label))
                break  # one conflict per number is enough

    if conflicts:
        notes = []
        for wrong, right, label in conflicts[:3]:   # cap at 3 notes
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


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are an expert data analyst assistant. A dataset summary is provided below.

== CORE RULES ==
- Answer concisely in plain English. Do NOT output raw JSON — always respond in readable text.
- The [COMPUTED FACTS] block below contains values pre-calculated by pandas from the full \
dataset. USE THEM EXACTLY — never re-add or re-estimate a value that is already listed there.
- Format numbers with commas (e.g. 31,735 not 31735).
- If a question cannot be answered from the available data, say so clearly.
- When you are NOT certain of a specific number (no pre-computed value exists), say \
"Based on available data, approximately X — please verify." Do NOT state estimates with \
full confidence.

== EMPLOYEE / PROGRESS QUESTIONS ==
When asked about employee performance, progress, or rankings:
1. Use ONLY the [EMPLOYEE / PERSON STATISTICS] block if present.
2. Show ALL metrics for that employee across all sheets.
3. Include their rank relative to other employees.
4. If a metric is not listed, say "not available" — never invent or estimate.
5. Present results in a clean table or structured list.

== AGGREGATION QUESTIONS ==
When asked for totals, sums, or counts:
1. Check [COMPUTED FACTS] first — if the answer is there, quote it directly.
2. Never manually re-add individual row values listed in the summary.
3. If exact data is unavailable, say so and explain why.

== ANSWER CONFIDENCE LABELS ==
- If your number comes from [COMPUTED FACTS] or [EMPLOYEE STATISTICS] → add ✅ Exact
- If you are estimating → add 🔶 Estimated

{summary}
"""


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
             pandas_result: Optional[str] = None) -> str:
        system = SYSTEM_TEMPLATE.format(summary=summary)
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-20:])
        # Prepend pandas result if available (P0.1)
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
             pandas_result: Optional[str] = None) -> str:
        system = SYSTEM_TEMPLATE.format(summary=summary)
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
