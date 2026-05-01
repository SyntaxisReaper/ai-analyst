import os
import io
import re
import json
import hashlib
import logging
import difflib
from typing import Optional, Tuple
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

from session_manager import SessionManager
from data_processor import DataProcessor
from ai_client import create_client, PROVIDER_MODELS, classify_intent, verify_numerical_claims

load_dotenv()

# ── Logging (P5) ─────────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("app")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, origins="*")

# ── Rate limiting (P4.3) ────────────────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# Configure session persistence via env: SESSION_PERSIST (true/false) and SESSION_DB
_sess_persist = os.getenv("SESSION_PERSIST", "true").lower() == "true"
_sess_redis = os.getenv("SESSION_REDIS_URL", None)
session_manager = SessionManager(persist=_sess_persist, redis_url=_sess_redis)
data_processor = DataProcessor()

# ── Runtime config (mutable — can be changed via /config) ────────────────────
_config = {
    "provider": os.getenv("AI_PROVIDER", "groq").lower(),
    "model": os.getenv("AI_MODEL", ""),
}
API_KEYS = {
    "groq": os.getenv("GROQ_API_KEY", ""),
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "gemini": os.getenv("GEMINI_API_KEY", ""),
}

_ai_client = None


def get_client():
    global _ai_client
    if _ai_client is None:
        p = _config["provider"]
        _ai_client = create_client(p, API_KEYS.get(p, ""), _config["model"])
    return _ai_client


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    p = _config["provider"]
    return jsonify({
        "status": "ok",
        "provider": p,
        "model": _config["model"] or "default",
        "api_key_configured": bool(API_KEYS.get(p)),
        "configured_providers": [k for k, v in API_KEYS.items() if v],
        "active_sessions": session_manager.count(),
        "available_providers": list(PROVIDER_MODELS.keys()),
        "available_models": PROVIDER_MODELS,
    })


@app.route("/config", methods=["GET"])
def get_config():
    p = _config["provider"]
    return jsonify({
        "provider": p,
        "model": _config["model"],
        "configured_providers": [k for k, v in API_KEYS.items() if v],
        "available_models": PROVIDER_MODELS,
    })


@app.route("/config", methods=["POST"])
def set_config():
    global _ai_client
    data = request.get_json(silent=True) or {}
    new_provider = data.get("provider", "").lower().strip()
    new_model = data.get("model", "").strip()

    if new_provider and new_provider not in PROVIDER_MODELS:
        return jsonify({"error": f"Unknown provider '{new_provider}'. Choose: {', '.join(PROVIDER_MODELS)}"}), 400

    if new_provider and not API_KEYS.get(new_provider):
        return jsonify({"error": f"No API key configured for '{new_provider}'. Add {new_provider.upper()}_API_KEY to backend/.env"}), 400

    if new_provider:
        _config["provider"] = new_provider
    if new_model is not None:
        _config["model"] = new_model

    _ai_client = None  # force re-init on next request

    p = _config["provider"]
    print(f"[config] Switched to provider={p}, model={_config['model'] or 'default'}")
    return jsonify({"success": True, "provider": p, "model": _config["model"] or "default"})


@app.route("/session/new", methods=["POST"])
def new_session():
    sid = session_manager.create()
    return jsonify({"session_id": sid})


@app.route("/session/<sid>/info", methods=["GET"])
def session_info(sid):
    if not session_manager.exists(sid):
        return jsonify({"error": "Session not found"}), 404
    s = session_manager.get(sid)
    return jsonify({
        "session_id": sid,
        "file_name": s.get("file_name"),
        "has_file": bool(s.get("summary")),
        "metadata": s.get("metadata", {}),
        "message_count": len(s.get("history", [])),
        "created_at": s.get("created_at"),
        "last_used": s.get("last_used"),
    })


@app.route("/session/<sid>/history/clear", methods=["POST"])
def clear_history(sid):
    if not session_manager.exists(sid):
        return jsonify({"error": "Session not found"}), 404
    session_manager.clear_history(sid)
    return jsonify({"success": True})


@app.route("/session/<sid>", methods=["DELETE"])
def delete_session(sid):
    if not session_manager.exists(sid):
        return jsonify({"error": "Session not found"}), 404
    session_manager.delete(sid)
    return jsonify({"success": True})


@app.route("/upload", methods=["POST"])
def upload():
    sid = request.form.get("session_id")
    if not sid or not session_manager.exists(sid):
        return jsonify({"error": "Invalid or missing session_id"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".xlsx", ".xls", ".csv", ".tsv"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported type '{ext}'. Use .xlsx, .xls, .csv, or .tsv"}), 400

    try:
        file_bytes = f.read()
        # P4.1 — file hash caching: skip re-processing identical files
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        result = data_processor.process_cached(file_bytes, f.filename, file_hash)
        session_manager.set_file(sid, {
            "file_name": f.filename,
            "summary": result["summary"],
            "metadata": result["metadata"],
            "dataframe_json": result.get("dataframe_json"),
            # P0.1 — store full DataFrames per sheet for pandas compute
            "sheets_data": result.get("sheets_data", {}),
            # P0.2 — named totals ground-truth dict
            "named_totals": result["metadata"].get("named_totals", {}),
            # P1.1 — employee stats cross-sheet dict
            "employee_stats": result["metadata"].get("employee_stats", {}),
        })
        log.info("Upload OK sid=%s file=%s rows=%s", sid, f.filename, result["metadata"].get("rows"))
        return jsonify({"success": True, "file_name": f.filename, "metadata": result["metadata"]})
    except Exception as e:
        log.exception("Upload failed sid=%s", sid)
        return jsonify({"error": str(e)}), 500


# ── P3.1: Filter rows and return CSV ──────────────────────────────────────
@app.route("/filter", methods=["POST"])
def filter_data():
    import pandas as pd
    import io as _io
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    query = data.get("query", "").strip()
    if not sid or not session_manager.exists(sid):
        return jsonify({"error": "Invalid or missing session_id"}), 400
    s = session_manager.get(sid)
    df_json = s.get("dataframe_json")
    if not df_json:
        return jsonify({"error": "No DataFrame available. Re-upload the file."}), 400
    try:
        df = pd.read_json(_io.StringIO(df_json), orient="split")
        if query:
            df = df.query(query)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return Response(
            csv_bytes,
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="filtered_{s.get("file_name", "data.csv")}"'},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── P3.2: Chart data ─────────────────────────────────────────────────────────
@app.route("/chart_data", methods=["POST"])
def chart_data():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if not sid or not session_manager.exists(sid):
        return jsonify({"error": "Invalid or missing session_id"}), 400
    s = session_manager.get(sid)
    meta = s.get("metadata", {})
    stats = meta.get("stats", {})
    # Return pre-computed numeric stats suitable for charting
    numeric = stats.get("numeric", {})
    categorical = stats.get("categorical", {})
    return jsonify({"numeric": numeric, "categorical": categorical, "columns": meta.get("column_names", [])})


@app.route("/ask", methods=["POST"])
@limiter.limit("30 per minute")
def ask():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    question = data.get("question", "").strip()

    if not sid or not session_manager.exists(sid):
        return jsonify({"error": "Invalid or missing session_id"}), 400

    s = session_manager.get(sid)
    if not s.get("summary"):
        return jsonify({"error": "No file loaded in this session. Upload a file first."}), 400

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        # ── P0.3: Classify intent ─────────────────────────────────────────────
        intent = classify_intent(question)
        log.info("Intent=%s sid=%s q=%s", intent, sid, question[:60])

        # ── P0.1: Try pandas compute first ───────────────────────────────────
        pandas_result = None
        server_answer = None

        try:
            server_answer = try_answer_from_stats(s, question)
        except Exception:
            server_answer = None

        # For numeric intents, try running actual pandas on stored DataFrames
        if server_answer is None and intent in ("aggregation", "count", "average",
                                                "comparison", "employee"):
            try:
                pandas_result = run_pandas_query(s, question, intent)
            except Exception as exc:
                log.warning("pandas compute failed: %s", exc)
                pandas_result = None

        if server_answer is not None:
            session_manager.add_message(sid, "user", question)
            session_manager.add_message(sid, "assistant", server_answer)
            return jsonify({"answer": server_answer, "source": "server",
                            "intent": intent, "confidence": "exact"})

        # ── Fallback: call AI with optional pandas pre-computed result ────────
        client = get_client()
        answer = client.chat(
            summary=s["summary"],
            history=s["history"],
            question=question,
            pandas_result=pandas_result,
        )

        # ── P2.1: Verify numerical claims ─────────────────────────────────────
        answer, had_conflicts = verify_numerical_claims(answer, s)

        # Try to parse structured JSON output (command protocol) from the assistant
        structured = None
        try:
            import json as _json
            stripped = answer.strip()
            import re as _re
            fence_match = _re.match(r'^```(?:json)?\s*([\s\S]+?)```$', stripped)
            if fence_match:
                stripped = fence_match.group(1).strip()
            if stripped.startswith('{'):
                try:
                    parsed = _json.loads(stripped)
                    if isinstance(parsed, dict) and parsed.get("type") in ("answer", "command"):
                        structured = parsed
                        log.info("Parsed structured command: %s", parsed.get("type"))
                except Exception as e:
                    log.debug("JSON in response but not valid structure: %s", str(e)[:100])
        except Exception as e:
            log.debug("Error during JSON parsing: %s", str(e)[:100])

        session_manager.add_message(sid, "user", question)
        session_manager.add_message(sid, "assistant", answer)
        resp = {
            "answer": answer,
            "source": "model",
            "intent": intent,
            "confidence": "estimated" if (pandas_result is None and not had_conflicts) else "exact",
            "pandas_used": pandas_result is not None,
        }
        if structured:
            resp["structured"] = structured
        if pandas_result:
            resp["pandas_result"] = pandas_result
        return jsonify(resp)
    except Exception as e:
        log.exception("Ask failed sid=%s", sid)
        return jsonify({"error": str(e)}), 500


@app.route("/execute_command", methods=["POST"])
def execute_command():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    cmd = data.get("command")
    if not sid or not session_manager.exists(sid):
        return jsonify({"error": "Invalid or missing session_id"}), 400
    if not isinstance(cmd, dict) or not cmd.get("name"):
        return jsonify({"error": "Invalid command payload"}), 400

    # Basic built-in command handlers
    name = cmd["name"]
    args = cmd.get("args") or {}
    s = session_manager.get(sid)

    # Example: show_columns → return column names from session metadata
    if name == "show_columns":
        cols = s.get("metadata", {}).get("column_names")
        return jsonify({"ok": True, "result": cols})

    # Example: export_sample_csv → attempt to return first N sample rows from metadata 'sample' if present
    if name == "export_sample_csv":
        rows = int(args.get("rows", 10))
        # We don't store raw file bytes in this implementation; return an informative error if not available
        return jsonify({"ok": False, "error": "Export not implemented on server; please request a download in the UI"})

    # Default: echo back the command (safe fallback)
    return jsonify({"ok": True, "result": {"echo": cmd}})


# ── Server metadata endpoints ────────────────────────────────────────────
@app.route("/api_base", methods=["GET"])
def api_base():
    # Return the base URL clients should use to contact this backend
    base = request.host_url.rstrip("/")
    return jsonify({"api_base": base})


@app.route('/config.js')
def serve_config_js():
    """Serve a tiny JS file that injects the backend API base.

    This lets static clients automatically receive the correct backend
    URL when they visit the hosted frontend (same origin).
    """
    # Allow an environment override so the operator can force a fixed
    # backend URL for all clients (useful when serving the frontend from
    # other origins or behind a proxy). Set `AI_BACKEND_URL` in the env.
    base = os.getenv("AI_BACKEND_URL") or request.host_url.rstrip("/")
    js = f"window.AI_BACKEND_URL = '{base}'; localStorage.setItem('AI_BACKEND_URL', '{base}');"
    return Response(js, mimetype="application/javascript")


@app.route("/metrics", methods=["GET"])
def metrics():
    p = _config["provider"]
    return jsonify({
        "status": "ok",
        "provider": p,
        "model": _config["model"] or "default",
        "active_sessions": session_manager.count(),
    })


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the frontend for any non-API route."""
    file_path = os.path.join(FRONTEND_DIR, path)
    if path and os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


def run_pandas_query(session: dict, question: str, intent: str) -> Optional[str]:
    """
    P0.1 — Load stored DataFrames from session and run exact pandas operations
    based on the classified intent. Returns a structured result string, or None.

    This is the "pandas-first" path that gives the AI exact numbers to explain.
    """
    import pandas as pd

    sheets_data = session.get("sheets_data") or {}
    named_totals = session.get("named_totals") or {}
    employee_stats = session.get("employee_stats") or {}
    q_lower = question.lower()

    # ── Employee questions: return pre-computed stats ────────────────────────
    if intent == "employee" and employee_stats:
        # Try to find a specific employee mentioned
        matched_emp = None
        for emp in employee_stats:
            if emp.lower() in q_lower or any(
                part.lower() in q_lower for part in emp.split()
            ):
                matched_emp = emp
                break

        if matched_emp:
            stats = employee_stats[matched_emp]
            lines = [f"Employee: {matched_emp}",
                     f"Overall Rank: #{stats.get('overall_rank', '?')}",
                     f"Overall Total Score: {stats.get('overall_total', 0):,.2f}"]
            for k, v in stats.items():
                if k in ("overall_rank", "overall_total"):
                    continue
                lines.append(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
            return "\n".join(lines)

        # All employees overview
        if any(kw in q_lower for kw in ("all employee", "each employee", "every employee",
                                          "employees", "progress of", "rank")):
            lines = ["All Employee Statistics (sorted by overall rank):"]
            for emp, stats in sorted(employee_stats.items(),
                                     key=lambda x: x[1].get("overall_rank", 999)):
                rank = stats.get("overall_rank", "?")
                total = stats.get("overall_total", 0)
                lines.append(f"\n  #{rank} {emp} — Total Score: {total:,.2f}")
                for k, v in stats.items():
                    if k in ("overall_rank", "overall_total"):
                        continue
                    lines.append(f"    {k}: {v:,.2f}" if isinstance(v, float) else f"    {k}: {v}")
            return "\n".join(lines)

    # ── Named totals: return ground-truth pre-calculated values ─────────────
    if named_totals:
        for label, val in named_totals.items():
            if label.lower() in q_lower or any(
                part.lower() in q_lower for part in label.split() if len(part) > 3
            ):
                return f"{label}: {val:,.4f}" if isinstance(val, float) and val != int(val) \
                    else f"{label}: {val:,.0f}"

    if not sheets_data:
        return None

    # ── Load DataFrames and run pandas operations ────────────────────────────
    # Helper: flatten column names from all sheets
    all_col_names: List[str] = []
    for sheet_name, df_json in sheets_data.items():
        try:
            df = pd.read_json(io.StringIO(df_json), orient="split")
            all_col_names.extend(df.columns.tolist())
        except Exception:
            continue

    # Find best matching column
    def find_col(text: str) -> Optional[str]:
        text_lower = text.lower()
        for c in all_col_names:
            if c.lower() in text_lower:
                return c
        # fuzzy
        matches = difflib.get_close_matches(text_lower, [c.lower() for c in all_col_names], n=1, cutoff=0.7)
        if matches:
            return all_col_names[[c.lower() for c in all_col_names].index(matches[0])]
        return None

    target_col = find_col(q_lower)
    if not target_col:
        return None

    results = []
    for sheet_name, df_json in sheets_data.items():
        try:
            df = pd.read_json(io.StringIO(df_json), orient="split")
            if target_col not in df.columns:
                continue
            series = pd.to_numeric(df[target_col], errors="coerce").dropna()
            if len(series) == 0:
                continue

            if intent == "aggregation":
                results.append(f"Sheet '{sheet_name}' — Sum of {target_col}: {series.sum():,.4f}")
            elif intent == "average":
                results.append(f"Sheet '{sheet_name}' — Average of {target_col}: {series.mean():,.4f}")
            elif intent == "count":
                results.append(f"Sheet '{sheet_name}' — Count of {target_col}: {len(series):,}")
            elif intent == "comparison":
                results.append(
                    f"Sheet '{sheet_name}' — {target_col}: "
                    f"Max={series.max():,.4f}, Min={series.min():,.4f}, "
                    f"Mean={series.mean():,.4f}"
                )
            else:
                results.append(f"Sheet '{sheet_name}' — {target_col}: Sum={series.sum():,.4f}, Mean={series.mean():,.4f}")
        except Exception:
            continue

    return "\n".join(results) if results else None


def try_answer_from_stats(session: dict, question: str):

    """Attempt to answer basic aggregation questions from stored stats.

    Returns a plain-text answer string when applicable, otherwise None.
    This handles simple patterns like 'sum of X', 'average of X', 'min X', 'max X',
    'how many X', 'count X', 'median of X'.
    """
    q = question.strip()
    stats = session.get("metadata", {}).get("stats") or {}
    if not stats:
        return None

    # find candidate column by matching column names in question
    cols = []
    col_names = session.get("metadata", {}).get("column_names") or []
    # normalize col names list
    if isinstance(col_names, dict):
        # excel sheets -> dict of sheet -> cols; flatten
        all_cols = []
        for v in col_names.values():
            all_cols.extend(v)
        col_names = all_cols

    for c in col_names:
        if not c:
            continue
        if c.lower() in q:
            cols.append(c)

    # if multiple columns mentioned, pick first
    col = cols[0] if cols else None
    # ops
    if not col:
        return None

    numeric = stats.get("numeric", {})
    col_stats = numeric.get(col) if isinstance(numeric, dict) else None
    if not col_stats:
        # maybe large-file numeric stats stored top-level under numeric
        col_stats = None

    # mapping of keywords to stat keys
    if "sum" in q or "total" in q:
        if col_stats and "sum" in col_stats:
            return f"Sum of {col}: {col_stats['sum']:,}"
    if "average" in q or "mean" in q:
        if col_stats and "mean" in col_stats:
            return f"Average of {col}: {col_stats['mean']:,}"
    if "median" in q:
        if col_stats and "median" in col_stats:
            return f"Median of {col}: {col_stats['median']:,}"
    if "min" in q or "minimum" in q:
        if col_stats and "min" in col_stats:
            return f"Min of {col}: {col_stats['min']:,}"
    if "max" in q or "maximum" in q:
        if col_stats and "max" in col_stats:
            return f"Max of {col}: {col_stats['max']:,}"
    if "count" in q or "how many" in q or "number of" in q:
        if col_stats and "count" in col_stats:
            return f"Count (non-missing) of {col}: {col_stats['count']:,}"

    # normalize and flatten column names
    col_names = session.get("metadata", {}).get("column_names") or []
    if isinstance(col_names, dict):
        flat_cols = []
        for v in col_names.values():
            flat_cols.extend(v)
        col_names = flat_cols

    # helper: fuzzy match a candidate column name from question text
    def find_best_column(text: str) -> Optional[str]:
        lowered = [str(c) for c in col_names]
        # direct substring match
        for c in col_names:
            if c and c.lower() in text.lower():
                return c
        # fuzzy match using difflib
        candidates = difflib.get_close_matches(text, lowered, n=1, cutoff=0.7)
        if candidates:
            # find original-cased name
            idx = lowered.index(candidates[0])
            return col_names[idx]
        # try token-wise matching: check each word in question against columns
        words = re.findall(r"\w+", text.lower())
        for w in words:
            candidates = difflib.get_close_matches(w, lowered, n=1, cutoff=0.8)
            if candidates:
                idx = lowered.index(candidates[0])
                return col_names[idx]
        return None

    # extract filter clause after 'where' or 'for' or 'in'
    filter_clause = None
    m = re.search(r"\bwhere\b(.+)$", q, flags=re.I)
    if not m:
        m = re.search(r"\bfor\b(.+)$", q, flags=re.I)
    if not m:
        m = re.search(r"\bin\b(\s+\d{4})", q, flags=re.I)
    if m:
        filter_clause = m.group(1).strip()

    # detect intended aggregation
    intent = None
    if re.search(r"\bsum\b|\btotal\b|\badd up\b", q, flags=re.I):
        intent = "sum"
    elif re.search(r"\baverage\b|\bmean\b|\bavg\b", q, flags=re.I):
        intent = "mean"
    elif re.search(r"\bmedian\b", q, flags=re.I):
        intent = "median"
    elif re.search(r"\bmin\b|\bminimum\b|\blowest\b", q, flags=re.I):
        intent = "min"
    elif re.search(r"\bmax\b|\bmaximum\b|\bhighest\b", q, flags=re.I):
        intent = "max"
    elif re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", q, flags=re.I):
        intent = "count"

    # try to find column by scanning for numeric-like intents with a nearby column name
    col = find_best_column(q)
    if not col:
        return None

    numeric = stats.get("numeric", {})
    col_stats = numeric.get(col) if isinstance(numeric, dict) else None

    # If no filter, and we have the stat, return it directly
    if not filter_clause and col_stats:
        if intent == "sum" and "sum" in col_stats:
            return f"Sum of {col}: {col_stats['sum']:,}"
        if intent == "mean" and "mean" in col_stats:
            return f"Average of {col}: {col_stats['mean']:,}"
        if intent == "median" and "median" in col_stats:
            return f"Median of {col}: {col_stats['median']:,}"
        if intent == "min" and "min" in col_stats:
            return f"Min of {col}: {col_stats['min']:,}"
        if intent == "max" and "max" in col_stats:
            return f"Max of {col}: {col_stats['max']:,}"
        if intent == "count" and "count" in col_stats:
            return f"Count (non-missing) of {col}: {col_stats['count']:,}"

    # If filter exists and is a simple equality on a categorical column, try to answer from categorical top counts
    if filter_clause:
        # parse simple filters like "Country = 'US'" or "country is US" or "country: US"
        fm = re.search(r"([\w\s]+?)\s*(?:=|==|is|:|=\s?)\s*'?\"?([\w\-\s]+?)'?\"?$", filter_clause.strip(), flags=re.I)
        if fm:
            fcol_raw, fval_raw = fm.group(1).strip(), fm.group(2).strip()
            fcol = find_best_column(fcol_raw)
            fval = fval_raw
            cat_stats = stats.get("categorical", {})
            fcol_stats = cat_stats.get(fcol) if isinstance(cat_stats, dict) else None
            if fcol_stats and fcol_stats.get("top"):
                # search top list for matching value
                for v, cnt in fcol_stats["top"]:
                    if v.lower() == fval.lower() or difflib.get_close_matches(fval.lower(), [v.lower()], n=1, cutoff=0.8):
                        # if user asked for count of rows where fcol == fval
                        if intent in ("count", None) and ("count" in col_stats or intent is None):
                            # if asking count of filtered rows, return count for category
                            return f"Rows where {fcol} = {v}: {cnt:,} (from top-{len(fcol_stats['top'])} counts)"
                # if asked for aggregation on another numeric column with filter, we cannot compute exact without full data
                if intent in ("sum", "mean", "median", "min", "max"):
                    return f"Cannot compute exact {intent} of {col} with filter '{filter_clause}' on server-side stats. Upload the full dataset or use the assistant to compute (may incur token cost)."

    return None


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    p = _config["provider"]
    print(f"\n[*] Excel AI Agent backend  ->  http://localhost:{port}")
    print(f"[*] Provider : {p.upper()}  |  Model: {_config['model'] or 'default'}")
    print(f"[*] API Key  : {'OK - set' if API_KEYS.get(p) else 'MISSING - set in .env'}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
