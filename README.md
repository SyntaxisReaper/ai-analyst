# Excel AI Agent 🚀

A browser-based AI agent — upload any Excel/CSV file and chat with it in plain English. Fully local frontend, cloud AI backend (Groq / OpenAI / Gemini).

---

## Quick Start

### 1. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure your AI provider
Copy `.env.example` to `.env`:
```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Edit `.env`:
```env
AI_PROVIDER=groq          # groq | openai | gemini
AI_MODEL=                 # leave blank for default
GROQ_API_KEY=gsk_...      # get free key at console.groq.com
```

### Session persistence (optional)

By default the backend will persist sessions to Redis so sessions survive restarts. Configure Redis by adding these values to your `.env` (or environment):

```env
SESSION_PERSIST=true
SESSION_REDIS_URL=redis://localhost:6379/0
# To disable persistence set SESSION_PERSIST=false
```

If you don't want persistence, set `SESSION_PERSIST=false`. The frontend will automatically fetch the backend's base URL when served from the same host so clients connect without manual setup.

### Forcing a single backend URL for all users

If you want every client to always connect to the same backend (for example
`https://ai-analyst-yhex.onrender.com`), set the `AI_BACKEND_URL` environment
variable on the backend server before starting it:

```env
AI_BACKEND_URL=https://ai-analyst-yhex.onrender.com
```

When `AI_BACKEND_URL` is set, visiting the hosted frontend will automatically
inject and persist that URL into each user's `localStorage`, so they won't
need to manually enter or apply the backend URL.

| Provider | Get API Key | Default Model |
|---|---|---|
| **Groq** (recommended) | [console.groq.com](https://console.groq.com) | llama3-8b-8192 |
| **OpenAI** | [platform.openai.com](https://platform.openai.com) | gpt-4o-mini |
| **Gemini** | [aistudio.google.com](https://aistudio.google.com) | gemini-1.5-flash |

### 3. Start the backend
```bash
cd backend
python app.py
```

### 4. Open the frontend
Open `frontend/index.html` directly in your browser — **no server needed**.

---

## Features

| Feature | Details |
|---|---|
| Multi-provider AI | Groq, OpenAI, or Gemini — switch in `.env` |
| Large file support | Files >20k rows are streamed in 10k-row chunks |
| Exact statistics | All numeric stats (sum, mean, min, max) pre-computed by pandas |
| Multi-session | Multiple files open simultaneously — each session is independent |
| Conversation memory | Full chat history sent with every question (last 20 turns) |
| Drag & drop upload | Drop files directly onto the sidebar |
| Markdown rendering | AI responses render tables, code blocks, bold, etc. |

---

## File Structure

```
excel-ai-agent/
├── backend/
│   ├── app.py               ← Flask REST API
│   ├── data_processor.py    ← pandas parsing + chunked large-file support
│   ├── session_manager.py   ← UUID sessions, conversation history
│   ├── ai_client.py         ← Groq / OpenAI / Gemini unified client
│   ├── .env                 ← Your API key (create from .env.example)
│   ├── .env.example         ← Template
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/status` | Health check — provider, model, key status |
| POST | `/session/new` | Create a new session → returns UUID |
| GET | `/session/<id>/info` | Session details |
| POST | `/upload` | Upload file to session (multipart) |
| POST | `/ask` | Ask a question → returns AI answer |
| POST | `/session/<id>/history/clear` | Clear chat history |
| DELETE | `/session/<id>` | Delete session |
