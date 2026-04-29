import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from session_manager import SessionManager
from data_processor import DataProcessor
from ai_client import create_client, PROVIDER_MODELS

load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

session_manager = SessionManager()
data_processor = DataProcessor()

# ── Config ────────────────────────────────────────────────────────────────────
AI_PROVIDER = os.getenv("AI_PROVIDER", "groq").lower()
AI_MODEL = os.getenv("AI_MODEL", "")
API_KEYS = {
    "groq": os.getenv("GROQ_API_KEY", ""),
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "gemini": os.getenv("GEMINI_API_KEY", ""),
}

_ai_client = None


def get_client():
    global _ai_client
    if _ai_client is None:
        _ai_client = create_client(AI_PROVIDER, API_KEYS[AI_PROVIDER], AI_MODEL)
    return _ai_client


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "provider": AI_PROVIDER,
        "model": AI_MODEL or "default",
        "api_key_configured": bool(API_KEYS.get(AI_PROVIDER)),
        "active_sessions": session_manager.count(),
        "available_providers": list(PROVIDER_MODELS.keys()),
        "available_models": PROVIDER_MODELS,
    })


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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    print(f"\n[*] Excel AI Agent backend  ->  http://localhost:{port}")
    print(f"[*] Provider : {AI_PROVIDER.upper()}  |  Model: {AI_MODEL or 'default'}")
    print(f"[*] API Key  : {'OK - set' if API_KEYS.get(AI_PROVIDER) else 'MISSING - set in .env'}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
