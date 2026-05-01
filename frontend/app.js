/* ────────────────────────────────────────────────────────────────────
   Excel AI Agent — app.js
   All state, API calls, and DOM updates live here.
──────────────────────────────────────────────────────────────────── */

// API base URL — resolved in this priority order:
// 1. User-configured URL saved in localStorage (set via Settings modal)
// 2. localhost:5000 when running locally
// 3. Same origin (when frontend is served by the Flask backend on Render)
// Default backend (fallback when frontend is hosted separately)
const DEFAULT_BACKEND = "https://ai-analyst-yhex.onrender.com";

function getApiBase() {
  // 1. Server-injected global (served from /config.js)
  const injected = (typeof window !== "undefined" && window.AI_BACKEND_URL) ? window.AI_BACKEND_URL : null;
  // 2. User-configured URL saved in localStorage (set via Settings modal)
  const saved = localStorage.getItem("AI_BACKEND_URL") || injected;
  if (saved) return saved.replace(/\/$/, "");  // strip trailing slash
  // 3. Local development
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    return "http://localhost:5000";
  }
  // 4. Fallback to a known backend if the frontend is hosted elsewhere (like Vercel).
  return DEFAULT_BACKEND;
}
let API = getApiBase();


// ── State ─────────────────────────────────────────────────────────────
let sessions = [];        // [{ id, name, meta, msgs }]
let activeId = null;      // currently selected session UUID
let isThinking = false;

// ── Init ──────────────────────────────────────────────────────────────
async function ensureApiBase() {
  console.log('[ensureApiBase] Starting backend URL detection...');
  console.log('[ensureApiBase] Current origin:', window.location.origin);
  console.log('[ensureApiBase] DEFAULT_BACKEND:', DEFAULT_BACKEND);
  
  // 1. If backend injected via /config.js (window.AI_BACKEND_URL), prefer it
  if (typeof window !== 'undefined' && window.AI_BACKEND_URL) {
    const base = window.AI_BACKEND_URL.replace(/\/$/, '');
    console.log('[ensureApiBase] ✓ Found injected window.AI_BACKEND_URL:', base);
    localStorage.setItem('AI_BACKEND_URL', base);
    API = getApiBase();
    console.log('[ensureApiBase] API set to:', API);
    return;
  }
  console.log('[ensureApiBase] ✗ No injected window.AI_BACKEND_URL');

  // 2. Try server endpoint on same origin (useful when frontend is served by backend)
  try {
    console.log('[ensureApiBase] Trying same-origin /api_base...');
    const res = await fetch('/api_base');
    console.log('[ensureApiBase] /api_base response:', res.status);
    if (res.ok) {
      const data = await res.json();
      console.log('[ensureApiBase] ✓ /api_base returned:', data);
      if (data.api_base) {
        localStorage.setItem('AI_BACKEND_URL', data.api_base.replace(/\/$/, ''));
        API = getApiBase();
        console.log('[ensureApiBase] API set to:', API);
        return;
      }
    }
  } catch (e) {
    console.log('[ensureApiBase] ✗ /api_base failed:', e.message);
  }

  // 3. If we reach here, try the known DEFAULT_BACKEND directly and persist if reachable
  // (Render apps may take 10-30s to cold-start, so use a longer timeout)
  console.log('[ensureApiBase] Probing DEFAULT_BACKEND:', DEFAULT_BACKEND + '/status', '(timeout: 15s for Render cold-start)');
  try {
    const probe = (url, timeout = 15000) => Promise.race([
      fetch(url, { method: 'GET', mode: 'cors' }),
      new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), timeout))
    ]);
    const statusRes = await probe(DEFAULT_BACKEND + '/status', 15000).catch(err => {
      console.log('[ensureApiBase] Probe error:', err.message);
      return null;
    });
    if (statusRes && statusRes.ok) {
      console.log('[ensureApiBase] ✓ DEFAULT_BACKEND reachable, status:', statusRes.status);
      localStorage.setItem('AI_BACKEND_URL', DEFAULT_BACKEND.replace(/\/$/, ''));
      API = getApiBase();
      console.log('[ensureApiBase] API set to:', API);
      return;
    } else {
      console.log('[ensureApiBase] ✗ DEFAULT_BACKEND returned status:', statusRes?.status || 'null');
    }
  } catch (e) {
    console.log('[ensureApiBase] ✗ DEFAULT_BACKEND probe error:', e.message);
  }

  // 4. Nothing reachable — leave API as-is (getApiBase will fallback to DEFAULT_BACKEND)
  API = getApiBase();
  console.log('[ensureApiBase] Final API:', API, '(fallback)');
}

window.addEventListener("DOMContentLoaded", async () => {
  // Ensure we persist the server-provided backend URL (if backend is serving the frontend)
  await ensureApiBase();
  loadSessions();
  checkStatus();
  bindEvents();
  await validateSessionsWithBackend();

  // P1.1 — Heartbeat: ping the active session every 10 min to prevent Render cold-start
  setInterval(() => {
    if (activeId) {
      fetch(`${API}/session/${activeId}/info`).catch(() => {});
    }
  }, 10 * 60 * 1000);

  // P1.2 — Re-validate sessions when user switches back to this tab
  window.addEventListener('focus', () => {
    validateSessionsWithBackend().catch(() => {});
  });
});

// ── Persist sessions to localStorage ─────────────────────────────────
function saveSessions() {
  localStorage.setItem("excel_ai_sessions", JSON.stringify(sessions));
}
function loadSessions() {
  try {
    const raw = localStorage.getItem("excel_ai_sessions");
    sessions = raw ? JSON.parse(raw) : [];
  } catch { sessions = []; }
  renderSessionList();
}

// ── Validate saved sessions against the live backend ─────────────────
// When the backend restarts, all in-memory sessions are lost.
// This checks each saved session and drops any that no longer exist.
async function validateSessionsWithBackend() {
  const results = await Promise.all(
    sessions.map(async s => {
      try {
        const res = await fetch(`${API}/session/${s.id}/info`);
        return res.ok ? s : null;  // null = stale/dead session
      } catch {
        return s; // backend unreachable, keep for now
      }
    })
  );

  const valid = results.filter(Boolean);
  const staleCount = sessions.length - valid.length;

  if (staleCount > 0) {
    sessions = valid;
    saveSessions();
    renderSessionList();
    showToast(`Backend restarted — ${staleCount} old session(s) cleared. Please re-upload your file.`, "error");
  }

  // Always ensure at least one session exists
  if (sessions.length === 0) {
    // createSession() sets activeId internally via switchSession()
    await createSession();
  } else {
    // Restore last active session — prefer previously active one if still valid
    const lastValid = valid[valid.length - 1];
    activeId = lastValid.id;
    renderSessionList();
    renderChatView();
  }
}


// ── API helpers ───────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  let json;
  try {
    json = await res.json();
  } catch (parseErr) {
    // Backend returned non-JSON (HTML error page, proxy error, etc.)
    const text = await res.text().catch(() => '');
    console.error('[api] Non-JSON response from', method, path, '→', res.status, text.slice(0, 200));
    throw new Error(`Server error (${res.status}): ${res.statusText || 'unexpected response'}`);
  }
  // If the response has an error field, throw it immediately
  if (json.error) {
    console.error('[api]', method, path, '→ error:', json.error);
    throw new Error(json.error);
  }
  return json;
}

async function checkStatus() {
  try {
    console.log('[checkStatus] Querying API:', API);
    const s = await api("GET", "/status");
    const dot = document.getElementById("provider-dot");
    const name = document.getElementById("provider-name");
    const model = document.getElementById("provider-model");
    if (s.api_key_configured) {
      dot.className = "provider-dot";
      name.textContent = s.provider.toUpperCase();
      model.textContent = s.model === "default" ? "default model" : s.model;
      console.log('[checkStatus] ✓ Backend connected:', s.provider, s.model);
    } else {
      dot.className = "provider-dot error";
      name.textContent = s.provider.toUpperCase() + " — No API Key";
      model.textContent = "Set key in backend/.env";
      console.log('[checkStatus] ⚠ Backend connected but missing API key');
    }
  } catch (e) {
    console.error('[checkStatus] ✗ Backend unreachable:', e.message);
    const dot = document.getElementById("provider-dot");
    dot.className = "provider-dot error";
    document.getElementById("provider-name").textContent = "Backend offline";
    document.getElementById("provider-model").textContent = "Start python app.py";
    console.log('[checkStatus] Attempted API base:', API);
  }
}

// ── Session management ────────────────────────────────────────────────
async function createSession() {
  try {
    const res = await api("POST", "/session/new");
    const s = { id: res.session_id, name: null, meta: null, msgs: [] };
    sessions.push(s);
    saveSessions();
    renderSessionList();
    switchSession(res.session_id);
  } catch {
    showToast("Could not reach backend. Is app.py running?", "error");
  }
}

function switchSession(id) {
  activeId = id;
  renderSessionList();
  renderChatView();
}

function getActive() {
  return sessions.find(s => s.id === activeId);
}

async function deleteSession(id, e) {
  e && e.stopPropagation();
  try { await api("DELETE", `/session/${id}`); } catch { /* backend may be down */ }
  sessions = sessions.filter(s => s.id !== id);
  saveSessions();
  if (activeId === id) {
    activeId = sessions.length ? sessions[sessions.length - 1].id : null;
  }
  renderSessionList();
  renderChatView();
}

// ── Dynamic suggestion chips (P2.2) ─────────────────────────────────
function buildSuggestions(meta) {
  const chips = ['Summarise the dataset', 'Show the top 5 rows', 'Any missing values?'];
  // Add numeric-specific chips
  const numCols = meta.stats?.numeric ? Object.keys(meta.stats.numeric) : [];
  if (numCols.length > 0) chips.push(`What is the average ${numCols[0]}?`);
  if (numCols.length > 1) chips.push(`What is the total ${numCols[1]}?`);
  // Sheet-specific chip
  if (meta.sheet_count > 1) chips.push(`Compare the ${meta.sheet_count} sheets`);
  // Column overview
  chips.push('What are the column names?');
  return chips.slice(0, 6);
}

// ── Render session list (with rename on double-click, P2.3) ─────────
function renderSessionList() {
  const list = document.getElementById("sessions-list");
  list.innerHTML = "";
  if (sessions.length === 0) {
    list.innerHTML = `<div style="text-align:center;color:var(--text3);font-size:12px;padding:20px 0;">No sessions yet.<br>Click ＋ to start.</div>`;
    return;
  }
  sessions.forEach(s => {
    const el = document.createElement('div');
    el.className = 'session-card' + (s.id === activeId ? ' active' : '');
    el.innerHTML = `
      <span class="session-icon">${s.name ? '📊' : '🆕'}</span>
      <div class="session-meta">
        <div class="session-name ${s.name ? '' : 'empty'}">${s.label || s.name || 'Empty session'}</div>
        <div class="session-msgs">${s.msgs.length ? Math.floor(s.msgs.length / 2) + ' exchanges' : 'No messages'}</div>
      </div>
      <span class="session-del" title="Delete session">✕</span>`;
    el.onclick = () => switchSession(s.id);
    el.querySelector('.session-del').onclick = (e) => deleteSession(s.id, e);
    // P2.3 — double-click the name to rename
    el.querySelector('.session-name').ondblclick = (e) => {
      e.stopPropagation();
      const nameEl = e.currentTarget;
      const oldLabel = s.label || s.name || '';
      const inp = document.createElement('input');
      inp.className = 'session-rename-input';
      inp.value = oldLabel;
      nameEl.replaceWith(inp);
      inp.focus();
      inp.select();
      const commit = () => {
        s.label = inp.value.trim() || oldLabel;
        saveSessions();
        renderSessionList();
      };
      inp.onblur = commit;
      inp.onkeydown = (ev) => {
        if (ev.key === 'Enter') inp.blur();
        if (ev.key === 'Escape') { inp.value = oldLabel; inp.blur(); }
      };
    };
    list.appendChild(el);
  });
}

// ── Render chat view ──────────────────────────────────────────────────
function renderChatView() {
  const emptyState = document.getElementById("empty-state");
  const chatView   = document.getElementById("chat-view");
  const s = getActive();

  if (!s) {
    emptyState.style.display = "flex";
    chatView.style.display = "none";
    return;
  }
  emptyState.style.display = "none";
  chatView.style.display = "flex";

  // Header
  document.getElementById("header-filename").textContent =
    s.name || "No file loaded — upload one below";
  document.getElementById("header-meta").textContent =
    s.meta ? `${s.meta.rows?.toLocaleString()} rows · ${s.meta.columns} columns${s.meta.sheet_count > 1 ? ` · ${s.meta.sheet_count} sheets` : ""}` : "";

  // File chip
  const chipWrap = document.getElementById("file-chip-wrap");
  const sheetInfo = s.meta?.sheet_count > 1
    ? ` · ${s.meta.sheet_count} sheets: ${s.meta.sheets?.join(", ")}`
    : "";
  chipWrap.innerHTML = s.name
    ? `<div class="file-chip">📄 ${s.name}${sheetInfo}${s.meta?.is_large ? " · large file (chunked)" : ""}</div>`
    : "";

  // Welcome / suggestions — dynamic chips from file metadata (P2.2)
  const welcome = document.getElementById('chat-welcome');
  const suggWrap = document.getElementById('suggestions-wrap');
  if (s.msgs.length === 0) {
    welcome.style.display = 'block';
    document.getElementById('welcome-msg').textContent =
      s.name ? `Ask anything about ${s.name}` : 'Upload a file on the left to begin.';
    if (s.name && s.meta) {
      suggWrap.style.display = 'flex';
      suggWrap.innerHTML = '';
      const chips = buildSuggestions(s.meta);
      chips.forEach(txt => {
        const btn = document.createElement('button');
        btn.className = 'suggestion-chip';
        btn.textContent = txt;
        btn.onclick = () => sendSuggestion(btn);
        suggWrap.appendChild(btn);
      });
    } else {
      suggWrap.style.display = 'none';
    }
  } else {
    welcome.style.display = 'none';
  }

  // Re-render messages (keep existing bubbles to avoid flicker)
  const area = document.getElementById("chat-area");
  // Remove old messages but keep welcome
  area.querySelectorAll(".message").forEach(m => m.remove());
  area.querySelector(".typing-indicator")?.remove();

  s.msgs.forEach(m => appendBubble(m.role, m.content, false, {
    confidence: m.confidence,
    pandasResult: m.pandasResult,
    intent: m.intent
  }));
  area.scrollTop = area.scrollHeight;
}

// ── Upload ────────────────────────────────────────────────────────────
async function uploadFile(file) {
  let s = getActive();
  // Auto-create session if none exists
  if (!s) {
    console.log('[uploadFile] No active session, creating one automatically...');
    try {
      const res = await api("POST", "/session/new");
      const newSession = { id: res.session_id, name: null, meta: null, msgs: [] };
      sessions.push(newSession);
      saveSessions();
      switchSession(res.session_id);
      s = getActive();
    } catch (err) {
      return showToast("Failed to create session: " + err.message, "error");
    }
  }

  const bar = document.getElementById("uploading-bar");
  const zone = document.getElementById("upload-zone");
  bar.classList.add("active");
  zone.style.pointerEvents = "none";

  const fd = new FormData();
  fd.append("file", file);
  fd.append("session_id", s.id);

  try {
    console.log('[uploadFile] Uploading to session:', s.id, 'file:', file.name);
    const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    s.name = data.file_name;
    s.meta = data.metadata;
    s.msgs = [];
    saveSessions();
    renderSessionList();
    renderChatView();
    showToast(`✓ ${data.file_name} loaded (${data.metadata.rows?.toLocaleString()} rows)`, "success");
    // P3.3 — auto-show employee panel if employee data detected
    if (data.metadata.employee_stats && Object.keys(data.metadata.employee_stats).length > 0) {
      renderEmployeePanel(data.metadata.employee_stats);
    } else {
      hideEmployeePanel();
    }
  } catch (err) {
    // P1.3 — Auto-recover if the session expired while the page was open
    if (err.message && err.message.includes('Invalid or missing session_id')) {
      console.warn('[uploadFile] Stale session on upload, recovering...');
      await recoverStaleSession();
      showToast('Session refreshed — please try uploading again.', 'error');
    } else {
      showToast('Upload failed: ' + err.message, 'error');
    }
  } finally {
    bar.classList.remove("active");
    zone.style.pointerEvents = "";
  }
}

// ── Stale session recovery ───────────────────────────────────────────
// Called when the backend says the session doesn't exist (e.g. after a
// Render cold-start wipes in-memory sessions). Drops the dead session,
// creates a fresh one, and asks the user to re-upload their file.
async function recoverStaleSession() {
  const deadId = activeId;
  // Remove dead session from local state
  sessions = sessions.filter(s => s.id !== deadId);
  saveSessions();
  try {
    await createSession(); // creates + switches to a new session
    showToast('⚠️ Session expired (backend restarted). Please re-upload your file.', 'error');
  } catch {
    showToast('Backend unreachable. Please refresh the page.', 'error');
  }
}

// ── Send question ─────────────────────────────────────────────────────
async function sendQuestion(question) {
  const s = getActive();
  if (!s || isThinking) return;
  if (!s.name) return showToast("Upload a file first!", "error");
  question = question.trim();
  if (!question) return;

  console.log('[sendQuestion] Active session:', s.id, 'File:', s.name);

  isThinking = true;
  document.getElementById("send-btn").disabled = true;
  document.getElementById("question-input").value = "";
  autoResize(document.getElementById("question-input"));

  // Add user bubble
  s.msgs.push({ role: "user", content: question });
  saveSessions();
  appendBubble("user", question);
  document.getElementById("chat-welcome").style.display = "none";

  // Typing indicator
  const area = document.getElementById("chat-area");
  const typing = document.createElement("div");
  typing.className = "typing-indicator message";
  typing.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>`;
  area.appendChild(typing);
  area.scrollTop = area.scrollHeight;

  try {
    console.log('[sendQuestion] Sending /ask with session_id:', s.id, 'question:', question.substring(0, 50));
    const res = await api("POST", "/ask", { session_id: s.id, question });
    typing.remove();

    const assistantContent = (res.structured && res.structured.content) ? res.structured.content : res.answer;
    const command = (res.structured && res.structured.command) ? res.structured.command : null;

    // P3.2 — confidence indicator
    const confidence = res.confidence || (res.source === 'server' ? 'exact' : 'estimated');
    const pandasResult = res.pandas_result || null;
    const intent = res.intent || null;

    s.msgs.push({ role: "assistant", content: assistantContent, command, confidence, pandasResult, intent });
    saveSessions();
    const msgEl = appendBubble("assistant", assistantContent, true, { confidence, pandasResult, intent });
    if (command) {
      // add a command action button to the assistant message
      const actionsWrap = document.createElement('div');
      actionsWrap.style.marginTop = '8px';
      const btn = document.createElement('button');
      btn.className = 'msg-action-btn';
      btn.textContent = `Run: ${command.name}`;
      btn.onclick = () => runCommand(command, s.id, btn);
      actionsWrap.appendChild(btn);
      msgEl.querySelector('.msg-bubble').appendChild(actionsWrap);
    }
    renderSessionList();
  } catch (err) {
    typing.remove();
    const errMsg = err.message || 'Unknown error';
    // Auto-recover stale sessions (backend restarted and wiped in-memory sessions)
    if (errMsg.includes('Invalid or missing session_id')) {
      console.warn('[sendQuestion] Stale session detected, recovering...');
      await recoverStaleSession();
    } else {
      const displayMsg = (errMsg.includes('JSON') || errMsg.includes('undefined') || errMsg.includes('null'))
        ? 'Backend error or invalid response. Please try again.'
        : errMsg;
      appendBubble('assistant', '⚠️ **Error:** ' + displayMsg);
      console.error('[sendQuestion] Error:', err);
    }
  } finally {
    isThinking = false;
    document.getElementById("send-btn").disabled = false;
    document.getElementById("question-input").focus();
  }
}

function sendSuggestion(btn) {
  sendQuestion(btn.textContent);
}

// ── Render a chat bubble ──────────────────────────────────────────────
function appendBubble(role, content, scroll = true, meta = {}) {
  const area = document.getElementById("chat-area");
  const isUser = role === "user";

  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const avatar = `<div class="msg-avatar">${isUser ? "👤" : "🤖"}</div>`;

  // support content objects or plain strings; if object, prefer its `content` prop
  let contentText = content;
  if (typeof content === 'object' && content !== null) {
    contentText = content.content || JSON.stringify(content);
  }

  const html = marked.parse(contentText);

  // P3.2 — Confidence badge
  let badgeHtml = '';
  if (!isUser && meta.confidence) {
    const isExact = meta.confidence === 'exact';
    badgeHtml = `<span class="confidence-badge ${isExact ? 'exact' : 'estimated'}">${isExact ? '✅ Exact' : '🔶 Estimated'}</span>`;
  }

  // P3.1 — Show working toggle (only if pandas result available)
  let workingHtml = '';
  if (!isUser && meta.pandasResult) {
    const escapedResult = escapeAttr(meta.pandasResult);
    workingHtml = `
      <div class="show-working-wrap">
        <button class="show-working-btn" onclick="toggleWorking(this)">📊 Show calculation</button>
        <pre class="working-detail" style="display:none">${meta.pandasResult.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</pre>
      </div>`;
  }

  const actions = isUser ? "" : `
    <div class="msg-footer">
      ${badgeHtml}
      <div class="msg-actions">
        <button class="msg-action-btn" onclick="copyMsg(this)" data-text="${escapeAttr(content)}">📋 Copy</button>
      </div>
    </div>
    ${workingHtml}`;

  msg.innerHTML = `
    ${avatar}
    <div>
      <div class="msg-bubble">${html}</div>
      ${actions}
    </div>`;

  if (isUser) {
    // Swap avatar to right
    const [av, body] = msg.children;
    msg.appendChild(av);
  }
  area.appendChild(msg);
  if (scroll) area.scrollTop = area.scrollHeight;
  return msg;
}

function toggleWorking(btn) {
  const detail = btn.nextElementSibling;
  const showing = detail.style.display !== 'none';
  detail.style.display = showing ? 'none' : 'block';
  btn.textContent = showing ? '📊 Show calculation' : '📊 Hide calculation';
}

function escapeAttr(s) {
  // Safely coerce non-strings (objects, null, undefined) before escaping
  const str = (typeof s === 'string') ? s : (s == null ? '' : JSON.stringify(s));
  return str.replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/\n/g, "\\n");
}

function copyMsg(btn) {
  const raw = btn.dataset.text.replace(/\\n/g, "\n");
  navigator.clipboard.writeText(raw).then(() => {
    btn.textContent = "✓ Copied";
    setTimeout(() => (btn.textContent = "📋 Copy"), 1800);
  });
}

// Execute a structured command by calling the backend `/execute_command` endpoint
async function runCommand(command, sessionId, btn) {
  btn.disabled = true;
  btn.textContent = 'Running...';
  try {
    const res = await api('POST', '/execute_command', { session_id: sessionId, command });
    if (res.error) throw new Error(res.error || 'Command failed');
    // Show result in the chat as an assistant message
    const s = getActive();
    const out = typeof res.result !== 'undefined' ? JSON.stringify(res.result, null, 2) : JSON.stringify(res);
    s.msgs.push({ role: 'assistant', content: `Command result:\n\n${out}` });
    saveSessions();
    appendBubble('assistant', `Command result:\n\n${out}`);
  } catch (err) {
    appendBubble('assistant', `⚠️ Error running command: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = `Run: ${command.name}`;
  }
}

// ── Toast (stacked) ──────────────────────────────────────────────────
let _toastOffset = 0;
function showToast(msg, type = 'info') {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  // Stack toasts vertically above each other
  _toastOffset += 1;
  t.style.bottom = (24 + (_toastOffset - 1) * 56) + 'px';
  document.body.appendChild(t);
  setTimeout(() => {
    t.remove();
    _toastOffset = Math.max(0, _toastOffset - 1);
  }, 3500);
}

// ── Bind events ───────────────────────────────────────────────────────
function bindEvents() {
  // New session
  document.getElementById("new-session-btn").onclick = createSession;

  // Settings modal
  bindSettingsEvents();

  // Clear chat history
  document.getElementById("clear-history-btn").onclick = async () => {
    const s = getActive();
    if (!s) return;
    try {
      await api("POST", `/session/${s.id}/history/clear`);
      s.msgs = [];
      saveSessions();
      renderChatView();
      showToast("Chat history cleared", "success");
    } catch { showToast("Failed to clear history", "error"); }
  };

  // Close session
  document.getElementById("close-session-btn").onclick = () => {
    if (activeId) deleteSession(activeId);
  };

  // Stats panel (P3.4)
  document.getElementById("stats-panel-btn").onclick = openStatsPanel;
  document.getElementById("stats-modal-close").onclick = () => {
    document.getElementById("stats-modal").style.display = "none";
  };
  document.getElementById("stats-modal").addEventListener("click", (e) => {
    if (e.target.id === "stats-modal") document.getElementById("stats-modal").style.display = "none";
  });
  // Filter download (P3.1)
  document.getElementById("filter-download-btn").onclick = downloadFilteredCSV;

  // Send button
  document.getElementById("send-btn").onclick = () => {
    sendQuestion(document.getElementById("question-input").value);
  };

  // Enter to send
  const input = document.getElementById("question-input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion(input.value);
    }
  });
  input.addEventListener("input", () => autoResize(input));

  // Upload zone
  const zone = document.getElementById("upload-zone");
  const fileInput = document.getElementById("file-input");

  zone.onclick = () => fileInput.click();
  fileInput.onchange = () => fileInput.files[0] && uploadFile(fileInput.files[0]);

  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });

  // Quick Win C — Ctrl+K focuses the input
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      document.getElementById('question-input').focus();
    }
  });

  // Scroll-to-bottom button (P2.4)
  const chatArea = document.getElementById('chat-area');
  const scrollBtn = document.getElementById('scroll-bottom-btn');
  chatArea.addEventListener('scroll', () => {
    const distFromBottom = chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight;
    scrollBtn.classList.toggle('visible', distFromBottom > 200);
  });
  scrollBtn.onclick = () => { chatArea.scrollTop = chatArea.scrollHeight; };

  // Quick Win F — Theme toggle (both mobile + desktop buttons)
  const applyTheme = (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    const icon = theme === 'dark' ? '☀️' : '🌙';
    document.querySelectorAll('#theme-toggle-btn, #theme-toggle-btn-desktop').forEach(b => { if (b) b.textContent = icon; });
  };
  const savedTheme = localStorage.getItem('theme') || 'dark';
  applyTheme(savedTheme);
  document.querySelectorAll('#theme-toggle-btn, #theme-toggle-btn-desktop').forEach(btn => {
    if (btn) btn.onclick = () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      applyTheme(current === 'dark' ? 'light' : 'dark');
    };
  });

  // Sidebar toggle (mobile, P2.5)
  document.getElementById('sidebar-toggle-btn')?.addEventListener('click', () => {
    document.querySelector('.sidebar').classList.toggle('open');
  });
}

// ── Auto-resize textarea ──────────────────────────────────────────────
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

// ── Settings Modal ────────────────────────────────────────────────────
let _settingsData = null;   // cached from /config

async function openSettings() {
  const modal = document.getElementById("settings-modal");

  // Pre-fill backend URL from localStorage — always show the modal
  const savedUrl = localStorage.getItem("AI_BACKEND_URL") || "";
  document.getElementById("setting-backend-url").value = savedUrl;

  // Clear provider/model dropdowns until we know backend is reachable
  document.getElementById("setting-provider").innerHTML = "<option>— set backend URL first —</option>";
  document.getElementById("setting-model").innerHTML = "";
  document.getElementById("provider-key-status").className = "provider-key-status";

  modal.style.display = "flex";

  // Try to load live config from backend (may fail if URL not set yet)
  try {
    _settingsData = await api("GET", "/config");
    populateSettingsModal(_settingsData);
  } catch {
    // Backend not reachable — that's OK, user just needs to set the URL first
    document.getElementById("provider-key-status").className = "provider-key-status missing";
    document.getElementById("provider-key-status").textContent =
      "Enter your backend URL above and click Apply to connect.";
    document.getElementById("provider-key-status").style.display = "block";
  }
}

function closeSettings() {
  document.getElementById("settings-modal").style.display = "none";
}

function populateSettingsModal(data) {
  const providerSel = document.getElementById("setting-provider");
  const modelSel    = document.getElementById("setting-model");

  // Populate providers
  providerSel.innerHTML = "";
  Object.keys(data.available_models).forEach(p => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p.charAt(0).toUpperCase() + p.slice(1);
    if (p === data.provider) opt.selected = true;
    providerSel.appendChild(opt);
  });

  // Populate models for current provider
  updateModelOptions(data.provider, data.model, data);

  // Show key status for current provider
  updateKeyStatus(data.provider, data.configured_providers);

  // On provider change — refresh model list + key status
  providerSel.onchange = () => {
    const p = providerSel.value;
    updateModelOptions(p, null, data);
    updateKeyStatus(p, data.configured_providers);
  };
}

function updateModelOptions(provider, currentModel, data) {
  const modelSel = document.getElementById("setting-model");
  const models = data.available_models[provider] || [];
  modelSel.innerHTML = "";
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === currentModel) opt.selected = true;
    modelSel.appendChild(opt);
  });
}

function updateKeyStatus(provider, configuredProviders) {
  const el = document.getElementById("provider-key-status");
  const hasKey = configuredProviders.includes(provider);
  el.className = "provider-key-status " + (hasKey ? "ok" : "missing");
  el.textContent = hasKey
    ? "API key configured"
    : `No API key set — add ${provider.toUpperCase()}_API_KEY to backend/.env`;

  document.getElementById("modal-save-btn").disabled = !hasKey;
}

async function saveSettings() {
  const backendUrl = document.getElementById("setting-backend-url").value.trim().replace(/\/$/, "");
  const saveBtn    = document.getElementById("modal-save-btn");

  // Save backend URL and refresh API variable first
  if (backendUrl) {
    localStorage.setItem("AI_BACKEND_URL", backendUrl);
  } else {
    localStorage.removeItem("AI_BACKEND_URL");
  }
  API = getApiBase();

  saveBtn.disabled = true;
  saveBtn.textContent = "Connecting...";

  // If dropdowns have no real selection yet (backend wasn't reachable before),
  // just fetch config to confirm connection and populate them
  try {
    _settingsData = await api("GET", "/config");
    populateSettingsModal(_settingsData);
    saveBtn.textContent = "Apply";
    saveBtn.disabled = false;

    // Now apply provider/model if user already selected them
    const provider = document.getElementById("setting-provider").value;
    const model    = document.getElementById("setting-model").value;
    if (provider && !provider.includes("—")) {
      const res = await api("POST", "/config", { provider, model });
      if (res.error) throw new Error(res.error);
      document.getElementById("provider-name").textContent = res.provider.toUpperCase();
      document.getElementById("provider-model").textContent = res.model;
      document.getElementById("provider-dot").className = "provider-dot";
    }

    checkStatus();
    closeSettings();
    showToast(`Connected to ${API}`, "success");
  } catch (err) {
    showToast("Could not reach backend: " + err.message, "error");
    saveBtn.disabled = false;
    saveBtn.textContent = "Apply";
  }
}


// Wire up modal buttons (called after DOM ready via bindEvents)
function bindSettingsEvents() {
  document.getElementById("settings-btn").onclick = openSettings;
  document.getElementById("modal-close-btn").onclick = closeSettings;
  document.getElementById("modal-cancel-btn").onclick = closeSettings;
  document.getElementById("modal-save-btn").onclick = saveSettings;
  // Click overlay to close
  document.getElementById("settings-modal").addEventListener("click", (e) => {
    if (e.target.id === "settings-modal") closeSettings();
  });
}

// ── P3.4 + P3.2 + P3.3: Stats Panel ─────────────────────────────────
let _chartInstances = [];

async function openStatsPanel() {
  const s = getActive();
  if (!s || !s.meta) return showToast('Upload a file first!', 'error');
  document.getElementById('stats-modal').style.display = 'flex';

  // P3.3 — Sheet selector
  const sheetWrap = document.getElementById('sheet-selector-wrap');
  const sheetSel = document.getElementById('sheet-selector');
  if (s.meta.sheet_count > 1 && s.meta.sheets) {
    sheetWrap.style.display = 'block';
    sheetSel.innerHTML = '';
    s.meta.sheets.forEach(sh => {
      const opt = document.createElement('option');
      opt.value = sh; opt.textContent = sh;
      sheetSel.appendChild(opt);
    });
    sheetSel.onchange = () => renderStats(s.meta);
  } else {
    sheetWrap.style.display = 'none';
  }
  renderStats(s.meta);
}

function renderStats(meta) {
  // Destroy previous Chart.js instances
  _chartInstances.forEach(c => c.destroy());
  _chartInstances = [];

  const body = document.getElementById('stats-panel-body');
  const stats = meta.stats || {};
  const numeric = stats.numeric || {};
  const categorical = stats.categorical || {};
  let html = '';

  // Numeric columns table
  const numKeys = Object.keys(numeric);
  if (numKeys.length) {
    html += `<h4 style="font-size:13px;font-weight:600;color:var(--text2);margin:16px 0 10px;">Numeric Columns</h4>
    <table class="stats-table">
      <thead><tr><th>Column</th><th>Count</th><th>Min</th><th>Max</th><th>Mean</th><th>Sum</th></tr></thead><tbody>`;
    numKeys.forEach(col => {
      const s = numeric[col];
      html += `<tr>
        <td><strong>${col}</strong></td>
        <td>${(s.count||0).toLocaleString()}</td>
        <td>${(s.min??'—')}</td>
        <td>${(s.max??'—')}</td>
        <td>${typeof s.mean === 'number' ? s.mean.toFixed(2) : '—'}</td>
        <td>${typeof s.sum === 'number' ? s.sum.toLocaleString() : '—'}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
    // P3.2 — bar chart of means
    html += `<canvas id="chart-numeric" height="90" style="margin:14px 0;"></canvas>`;
  }

  // Categorical columns
  const catKeys = Object.keys(categorical);
  if (catKeys.length) {
    html += `<h4 style="font-size:13px;font-weight:600;color:var(--text2);margin:16px 0 10px;">Categorical Columns</h4>`;
    catKeys.forEach(col => {
      const cs = categorical[col];
      html += `<div style="margin-bottom:12px;">
        <div style="font-size:12px;font-weight:600;margin-bottom:4px;">${col} <span style="color:var(--text3);font-weight:400;">(${(cs.unique||0).toLocaleString()} unique)</span></div>
        <canvas id="chart-cat-${col.replace(/\W/g,'_')}" height="70"></canvas>
      </div>`;
    });
  }

  body.innerHTML = html;

  // Render numeric means chart
  if (numKeys.length && document.getElementById('chart-numeric')) {
    _chartInstances.push(new Chart(document.getElementById('chart-numeric'), {
      type: 'bar',
      data: {
        labels: numKeys,
        datasets: [{ label: 'Mean', data: numKeys.map(k => numeric[k].mean ?? 0),
          backgroundColor: 'rgba(124,110,247,0.7)', borderRadius: 5 }]
      },
      options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#8b92b0' } }, y: { ticks: { color: '#8b92b0' } } } }
    }));
  }

  // Render categorical top-value charts
  catKeys.forEach(col => {
    const cs = categorical[col];
    const canvasId = `chart-cat-${col.replace(/\W/g,'_')}`;
    const canvas = document.getElementById(canvasId);
    if (!canvas || !cs.top) return;
    const labels = cs.top.map(([v]) => v);
    const data   = cs.top.map(([,c]) => c);
    _chartInstances.push(new Chart(canvas, {
      type: 'bar',
      data: { labels, datasets: [{ label: col, data, backgroundColor: 'rgba(93,184,245,0.7)', borderRadius: 4 }] },
      options: { indexAxis: 'y', plugins: { legend: { display: false } },
        scales: { x: { ticks: { color: '#8b92b0' } }, y: { ticks: { color: '#8b92b0', font: { size: 10 } } } } }
    }));
  });
}

// ── P3.1: Filter and download CSV ─────────────────────────────────────
async function downloadFilteredCSV() {
  const s = getActive();
  if (!s) return;
  const query = document.getElementById('filter-query-input').value.trim();
  const btn = document.getElementById('filter-download-btn');
  btn.disabled = true;
  btn.textContent = 'Preparing…';
  try {
    const res = await fetch(`${API}/filter`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: s.id, query }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(err.error || 'Filter failed');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `filtered_${s.name || 'data'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('✓ CSV downloaded', 'success');
  } catch (err) {
    showToast('Download failed: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '⬇ Download CSV';
  }
}

// ── P3.3: Employee Progress Dashboard Card ────────────────────────────
function renderEmployeePanel(employeeStats) {
  // Remove any existing panel
  document.getElementById('employee-panel')?.remove();

  if (!employeeStats || Object.keys(employeeStats).length === 0) return;

  // Sort by overall rank
  const sorted = Object.entries(employeeStats).sort(
    ([, a], [, b]) => (a.overall_rank || 999) - (b.overall_rank || 999)
  );

  const rankColors = ['#f7c948', '#c0c0c0', '#cd7f32', '#7c6ef7', '#5db8f5', '#6cf0b0'];

  let rows = '';
  sorted.forEach(([name, stats], idx) => {
    const rank = stats.overall_rank || idx + 1;
    const total = (stats.overall_total || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });
    const rankColor = rankColors[Math.min(rank - 1, rankColors.length - 1)];

    // Collect per-sheet metrics (exclude meta fields)
    const metrics = Object.entries(stats)
      .filter(([k]) => !['overall_rank', 'overall_total'].includes(k))
      .slice(0, 4)  // show up to 4 metrics
      .map(([k, v]) => {
        const label = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const val = typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : v;
        return `<div class="emp-metric"><span class="emp-metric-label">${label}</span><span class="emp-metric-val">${val}</span></div>`;
      }).join('');

    rows += `
      <div class="emp-row">
        <div class="emp-rank" style="color:${rankColor}">#${rank}</div>
        <div class="emp-info">
          <div class="emp-name">${name}</div>
          <div class="emp-total">Total Score: ${total}</div>
          <div class="emp-metrics">${metrics}</div>
        </div>
      </div>`;
  });

  const panel = document.createElement('div');
  panel.id = 'employee-panel';
  panel.className = 'employee-panel';
  panel.innerHTML = `
    <div class="employee-panel-header" onclick="toggleEmployeePanel(this)">
      <span>👥 Employee Progress</span>
      <span class="emp-count">${sorted.length} people</span>
      <span class="emp-toggle-icon">▼</span>
    </div>
    <div class="employee-panel-body">
      ${rows}
    </div>`;

  // Insert before the chat area in the right panel
  const chatView = document.getElementById('chat-view');
  if (chatView) {
    chatView.insertBefore(panel, chatView.firstChild);
  }
}

function hideEmployeePanel() {
  document.getElementById('employee-panel')?.remove();
}

function toggleEmployeePanel(header) {
  const body = header.nextElementSibling;
  const icon = header.querySelector('.emp-toggle-icon');
  const isOpen = body.style.display !== 'none';
  body.style.display = isOpen ? 'none' : 'block';
  icon.textContent = isOpen ? '▶' : '▼';
}
