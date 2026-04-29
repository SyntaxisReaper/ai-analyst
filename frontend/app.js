/* ────────────────────────────────────────────────────────────────────
   Excel AI Agent — app.js
   All state, API calls, and DOM updates live here.
──────────────────────────────────────────────────────────────────── */

const API = "http://localhost:5000";

// ── State ─────────────────────────────────────────────────────────────
let sessions = [];        // [{ id, name, meta, msgs }]
let activeId = null;      // currently selected session UUID
let isThinking = false;

// ── Init ──────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  loadSessions();
  checkStatus();
  bindEvents();
  await validateSessionsWithBackend();
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
    if (staleCount > 0) {
      showToast(`Backend restarted — ${staleCount} old session(s) cleared. Please re-upload your file.`, "error");
    }
  }

  // Always ensure at least one session exists
  if (sessions.length === 0) {
    await createSession();
  } else {
    activeId = sessions[sessions.length - 1].id;
    renderSessionList();
    renderChatView();
  }
}


// ── API helpers ───────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  return res.json();
}

async function checkStatus() {
  try {
    const s = await api("GET", "/status");
    const dot = document.getElementById("provider-dot");
    const name = document.getElementById("provider-name");
    const model = document.getElementById("provider-model");
    if (s.api_key_configured) {
      dot.className = "provider-dot";
      name.textContent = s.provider.toUpperCase();
      model.textContent = s.model === "default" ? "default model" : s.model;
    } else {
      dot.className = "provider-dot error";
      name.textContent = s.provider.toUpperCase() + " — No API Key";
      model.textContent = "Set key in backend/.env";
    }
  } catch {
    const dot = document.getElementById("provider-dot");
    dot.className = "provider-dot error";
    document.getElementById("provider-name").textContent = "Backend offline";
    document.getElementById("provider-model").textContent = "Start python app.py";
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

// ── Render session list ───────────────────────────────────────────────
function renderSessionList() {
  const list = document.getElementById("sessions-list");
  list.innerHTML = "";
  if (sessions.length === 0) {
    list.innerHTML = `<div style="text-align:center;color:var(--text3);font-size:12px;padding:20px 0;">No sessions yet.<br>Click ＋ to start.</div>`;
    return;
  }
  sessions.forEach(s => {
    const el = document.createElement("div");
    el.className = "session-card" + (s.id === activeId ? " active" : "");
    el.innerHTML = `
      <span class="session-icon">${s.name ? "📊" : "🆕"}</span>
      <div class="session-meta">
        <div class="session-name ${s.name ? "" : "empty"}">${s.name || "Empty session"}</div>
        <div class="session-msgs">${s.msgs.length ? s.msgs.length / 2 + " exchanges" : "No messages"}</div>
      </div>
      <span class="session-del" title="Delete session">✕</span>`;
    el.onclick = () => switchSession(s.id);
    el.querySelector(".session-del").onclick = (e) => deleteSession(s.id, e);
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

  // Welcome / suggestions
  const welcome = document.getElementById("chat-welcome");
  const suggWrap = document.getElementById("suggestions-wrap");
  if (s.msgs.length === 0) {
    welcome.style.display = "block";
    document.getElementById("welcome-msg").textContent =
      s.name ? `Ask anything about ${s.name}` : "Upload a file on the left to begin.";
    suggWrap.style.display = s.name ? "flex" : "none";
  } else {
    welcome.style.display = "none";
  }

  // Re-render messages (keep existing bubbles to avoid flicker)
  const area = document.getElementById("chat-area");
  // Remove old messages but keep welcome
  area.querySelectorAll(".message").forEach(m => m.remove());
  area.querySelector(".typing-indicator")?.remove();

  s.msgs.forEach(m => appendBubble(m.role, m.content, false));
  area.scrollTop = area.scrollHeight;
}

// ── Upload ────────────────────────────────────────────────────────────
async function uploadFile(file) {
  const s = getActive();
  if (!s) return showToast("Create a session first (＋ button)", "error");

  const bar = document.getElementById("uploading-bar");
  const zone = document.getElementById("upload-zone");
  bar.classList.add("active");
  zone.style.pointerEvents = "none";

  const fd = new FormData();
  fd.append("file", file);
  fd.append("session_id", s.id);

  try {
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
  } catch (err) {
    showToast("Upload failed: " + err.message, "error");
  } finally {
    bar.classList.remove("active");
    zone.style.pointerEvents = "";
  }
}

// ── Send question ─────────────────────────────────────────────────────
async function sendQuestion(question) {
  const s = getActive();
  if (!s || isThinking) return;
  if (!s.name) return showToast("Upload a file first!", "error");
  question = question.trim();
  if (!question) return;

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
    const res = await api("POST", "/ask", { session_id: s.id, question });
    typing.remove();
    if (res.error) throw new Error(res.error);

    s.msgs.push({ role: "assistant", content: res.answer });
    saveSessions();
    appendBubble("assistant", res.answer);
    renderSessionList();
  } catch (err) {
    typing.remove();
    appendBubble("assistant", "⚠️ **Error:** " + err.message);
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
function appendBubble(role, content, scroll = true) {
  const area = document.getElementById("chat-area");
  const isUser = role === "user";

  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const avatar = `<div class="msg-avatar">${isUser ? "👤" : "🤖"}</div>`;
  const html = marked.parse(content);
  const actions = isUser ? "" : `
    <div class="msg-actions">
      <button class="msg-action-btn" onclick="copyMsg(this)" data-text="${escapeAttr(content)}">📋 Copy</button>
    </div>`;

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
}

function escapeAttr(s) {
  return s.replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/\n/g, "\\n");
}

function copyMsg(btn) {
  const raw = btn.dataset.text.replace(/\\n/g, "\n");
  navigator.clipboard.writeText(raw).then(() => {
    btn.textContent = "✓ Copied";
    setTimeout(() => (btn.textContent = "📋 Copy"), 1800);
  });
}

// ── Toast ─────────────────────────────────────────────────────────────
function showToast(msg, type = "info") {
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
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
  modal.style.display = "flex";

  try {
    _settingsData = await api("GET", "/config");
    populateSettingsModal(_settingsData);
  } catch {
    showToast("Could not load settings — is backend running?", "error");
    modal.style.display = "none";
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
  const provider = document.getElementById("setting-provider").value;
  const model    = document.getElementById("setting-model").value;
  const saveBtn  = document.getElementById("modal-save-btn");

  saveBtn.disabled = true;
  saveBtn.textContent = "Applying...";

  try {
    const res = await api("POST", "/config", { provider, model });
    if (res.error) throw new Error(res.error);

    // Update provider badge
    document.getElementById("provider-name").textContent = res.provider.toUpperCase();
    document.getElementById("provider-model").textContent = res.model;
    document.getElementById("provider-dot").className = "provider-dot";

    closeSettings();
    showToast(`Switched to ${res.provider.toUpperCase()} / ${res.model}`, "success");
  } catch (err) {
    showToast("Failed: " + err.message, "error");
  } finally {
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
