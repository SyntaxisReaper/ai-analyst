import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from session_manager import SessionManager
from data_processor import DataProcessor
from ai_client import create_client, PROVIDER_MODELS

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
        client = get_client()
        answer = client.chat(
            summary=s["summary"],
            history=s["history"],
            question=question,
        )
        session_manager.add_message(sid, "user", question)
        session_manager.add_message(sid, "assistant", answer)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Server metadata endpoints ────────────────────────────────────────────
@app.route("/api_base", methods=["GET"])
def api_base():
    # Return the base URL clients should use to contact this backend
    base = request.host_url.rstrip("/")
    return jsonify({"api_base": base})


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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    p = _config["provider"]
    print(f"\n[*] Excel AI Agent backend  ->  http://localhost:{port}")
    print(f"[*] Provider : {p.upper()}  |  Model: {_config['model'] or 'default'}")
    print(f"[*] API Key  : {'OK - set' if API_KEYS.get(p) else 'MISSING - set in .env'}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
