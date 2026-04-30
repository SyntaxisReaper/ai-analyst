import os
import logging
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv

from session_manager import SessionManager
from data_processor import DataProcessor
from ai_client import create_client, PROVIDER_MODELS
import re

load_dotenv()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, origins="*")

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
        result = data_processor.process(f.read(), f.filename)
        session_manager.set_file(sid, {
            "file_name": f.filename,
            "summary": result["summary"],
            "metadata": result["metadata"],
        })
        return jsonify({"success": True, "file_name": f.filename, "metadata": result["metadata"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
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
        # First: attempt to answer simple numeric/aggregation questions from pre-computed stats
        server_answer = None
        try:
            server_answer = try_answer_from_stats(s, question)
        except Exception:
            server_answer = None

        if server_answer is not None:
            session_manager.add_message(sid, "user", question)
            session_manager.add_message(sid, "assistant", server_answer)
            return jsonify({"answer": server_answer, "source": "server"})

        # Fallback: call the AI client
        client = get_client()
        answer = client.chat(
            summary=s["summary"],
            history=s["history"],
            question=question,
        )
        # Try to parse structured JSON output (command protocol) from the assistant
        structured = None
        try:
            import json as _json
            parsed = _json.loads(answer)
            # basic validation
            if isinstance(parsed, dict) and parsed.get("type") in ("answer", "command"):
                structured = parsed
        except Exception:
            structured = None
        session_manager.add_message(sid, "user", question)
        session_manager.add_message(sid, "assistant", answer)
        resp = {"answer": answer, "source": "model"}
        if structured:
            resp["structured"] = structured
        return jsonify(resp)
    except Exception as e:
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


def try_answer_from_stats(session: dict, question: str):
    """Attempt to answer basic aggregation questions from stored stats.

    Returns a plain-text answer string when applicable, otherwise None.
    This handles simple patterns like 'sum of X', 'average of X', 'min X', 'max X',
    'how many X', 'count X', 'median of X'.
    """
    q = question.lower()
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

    return None


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    p = _config["provider"]
    print(f"\n[*] Excel AI Agent backend  ->  http://localhost:{port}")
    print(f"[*] Provider : {p.upper()}  |  Model: {_config['model'] or 'default'}")
    print(f"[*] API Key  : {'OK - set' if API_KEYS.get(p) else 'MISSING - set in .env'}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
