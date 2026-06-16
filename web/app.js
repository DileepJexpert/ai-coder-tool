// watchdog web UI client.
//
// Streams Server-Sent Events from POST /api/message and renders each agent
// event as a message bubble, tool card, or confirm prompt. No build step.

const messagesEl = document.getElementById("messages");
const composer   = document.getElementById("composer");
const input      = document.getElementById("input");
const sendBtn    = document.getElementById("send");
const resetBtn   = document.getElementById("reset");
const metaEl     = document.getElementById("meta");
const hintEl     = document.getElementById("hint");

let busy = false;

// Tools we expand inline by default because the args themselves are the point.
const EXPAND_BY_DEFAULT = new Set([
  "edit_file", "multi_edit", "write_file", "run_command", "run_tests", "generate_image",
]);

// ---------------- boot ----------------

fetch("/api/info").then(r => r.json()).then(info => {
  metaEl.textContent =
    `model: ${info.model} · vision: ${info.vision_model} · host: ${info.host}` +
    (info.yolo ? " · YOLO" : "");
}).catch(() => {
  metaEl.textContent = "offline";
});

// Configure marked once
if (window.marked) {
  marked.setOptions({
    breaks: true,
    gfm: true,
  });
}

// ---------------- composer ----------------

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 220) + "px";
}
input.addEventListener("input", autosize);
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

composer.addEventListener("submit", e => {
  e.preventDefault();
  if (busy) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  autosize();
  sendUserMessage(text);
});

resetBtn.addEventListener("click", async () => {
  if (busy) return;
  const r = await fetch("/api/reset", { method: "POST" });
  if (r.ok) {
    messagesEl.innerHTML = "";
    setHint("history cleared.");
  }
});

function setHint(text, cls = "") {
  hintEl.textContent = text;
  hintEl.className = "hint " + cls;
}

function setBusy(b) {
  busy = b;
  sendBtn.disabled = b;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

// ---------------- DOM helpers ----------------

function fromHTML(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function escapeHTML(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

function highlightAll(root) {
  if (!window.hljs) return;
  root.querySelectorAll("pre code").forEach(b => {
    try { hljs.highlightElement(b); } catch (_) { /* best-effort */ }
  });
}

function renderMarkdown(text) {
  if (!window.marked) return escapeHTML(text);
  try { return marked.parse(text); }
  catch { return escapeHTML(text); }
}

// ---------------- message renderers ----------------

function appendUser(text) {
  const node = fromHTML(`
    <div class="row user">
      <div class="avatar">you</div>
      <div class="bubble">
        <div class="who">you</div>
        <div class="md"></div>
      </div>
    </div>`);
  node.querySelector(".md").innerHTML = renderMarkdown(text);
  highlightAll(node);
  messagesEl.appendChild(node);
  scrollToBottom();
}

function appendAssistant(text) {
  removeThinking();
  const node = fromHTML(`
    <div class="row agent">
      <div class="avatar">wd</div>
      <div class="bubble">
        <div class="who">watchdog</div>
        <div class="md"></div>
      </div>
    </div>`);
  node.querySelector(".md").innerHTML = renderMarkdown(text || "*(no reply)*");
  highlightAll(node);
  messagesEl.appendChild(node);
  scrollToBottom();
}

function summarizeArgs(name, args) {
  if (!args) return "";
  switch (name) {
    case "edit_file":
    case "multi_edit":
    case "write_file":
    case "read_file":
    case "list_dir":
    case "analyze_image":
      return args.path || "";
    case "search_code":  return JSON.stringify(args.query || "");
    case "grep":         return `${JSON.stringify(args.pattern || "")} in ${args.path || "."}`;
    case "glob":         return args.pattern || "";
    case "run_command":  return args.command || "";
    case "run_tests":    return args.command || "(auto-detected)";
    case "git":          return args.subcommand || "";
    case "run_subtask":  return args.goal || "";
    case "generate_image": {
      const p = (args.prompt || "").trim();
      return p.length > 80 ? p.slice(0, 80) + "…" : p;
    }
    default:
      try { return JSON.stringify(args); } catch { return ""; }
  }
}

function renderDiff(oldStr, newStr) {
  // Naive line-level diff: old as deletions, new as additions. Good enough
  // for the small snippets edit_file works with.
  const o = String(oldStr || "").split("\n");
  const n = String(newStr || "").split("\n");
  let out = "";
  for (const l of o) out += `<span class="del">- ${escapeHTML(l)}</span>`;
  for (const l of n) out += `<span class="add">+ ${escapeHTML(l)}</span>`;
  return out || `<span class="hd">(no change)</span>`;
}

function renderToolBody(name, args) {
  if (name === "edit_file") {
    return `
      <div class="kv">
        <span class="k">path</span><span class="v">${escapeHTML(args.path || "")}</span>
      </div>
      <div class="diff">${renderDiff(args.old, args.new)}</div>`;
  }
  if (name === "multi_edit") {
    const edits = Array.isArray(args.edits) ? args.edits : [];
    let html = `<div class="kv">
      <span class="k">path</span><span class="v">${escapeHTML(args.path || "")}</span>
      <span class="k">edits</span><span class="v">${edits.length}</span>
    </div>`;
    edits.slice(0, 5).forEach((e, i) => {
      html += `<div class="diff">
        <span class="hd">--- edit #${i + 1} ---</span>
        ${renderDiff(e.old, e.new)}
      </div>`;
    });
    if (edits.length > 5) {
      html += `<div class="kv"><span class="k">…</span><span class="v">${edits.length - 5} more edits not shown</span></div>`;
    }
    return html;
  }
  if (name === "generate_image") {
    const w = args.width || 512;
    const h = args.height || 512;
    return `
      <div class="kv">
        <span class="k">prompt</span><span class="v">${escapeHTML(args.prompt || "")}</span>
        ${args.filename ? `<span class="k">filename</span><span class="v">${escapeHTML(args.filename)}</span>` : ""}
        <span class="k">size</span><span class="v">${w} × ${h}px</span>
        ${args.steps ? `<span class="k">steps</span><span class="v">${args.steps}</span>` : ""}
      </div>`;
  }
  if (name === "write_file") {
    const content = String(args.content || "");
    const lines = content.split("\n");
    const preview = lines.slice(0, 30).join("\n") + (lines.length > 30 ? "\n…" : "");
    return `
      <div class="kv">
        <span class="k">path</span><span class="v">${escapeHTML(args.path || "")}</span>
        <span class="k">size</span><span class="v">${content.length} chars · ${lines.length} lines</span>
      </div>
      <pre>${escapeHTML(preview)}</pre>`;
  }
  // generic: list all args as kv
  const entries = Object.entries(args || {});
  if (!entries.length) return `<div class="kv"><span class="k">(no args)</span><span class="v"></span></div>`;
  const kvs = entries.map(([k, v]) => {
    let shown;
    if (typeof v === "string") {
      shown = v.length > 1200 ? v.slice(0, 1200) + " …" : v;
    } else {
      try { shown = JSON.stringify(v, null, 2); } catch { shown = String(v); }
      if (shown.length > 1200) shown = shown.slice(0, 1200) + " …";
    }
    return `<span class="k">${escapeHTML(k)}</span><span class="v">${escapeHTML(shown)}</span>`;
  }).join("");
  return `<div class="kv">${kvs}</div>`;
}

function appendToolCall(name, args) {
  removeThinking();
  const collapsed = !EXPAND_BY_DEFAULT.has(name);
  const summary = summarizeArgs(name, args);
  const node = fromHTML(`
    <div class="tool ${collapsed ? "collapsed" : ""}" data-name="${escapeHTML(name)}" data-pending="1">
      <div class="tool-head">
        <span class="chev">▾</span>
        <span class="name">${escapeHTML(name)}</span>
        <span class="summary">${escapeHTML(summary)}</span>
      </div>
      <div class="tool-body"></div>
    </div>`);
  node.querySelector(".tool-body").innerHTML = renderToolBody(name, args);
  node.querySelector(".tool-head").addEventListener("click", () => node.classList.toggle("collapsed"));
  messagesEl.appendChild(node);
  scrollToBottom();
  return node;
}

function findPendingTool(name) {
  const all = messagesEl.querySelectorAll(`.tool[data-pending="1"]`);
  // Walk back; match same name, else first pending of any name.
  for (let i = all.length - 1; i >= 0; i--) {
    if (all[i].getAttribute("data-name") === name) return all[i];
  }
  return all[all.length - 1] || null;
}

function appendToolResult(name, result, isError) {
  const card = findPendingTool(name);
  const text = String(result == null ? "" : result);
  const lines = text.split("\n");
  const preview = lines.slice(0, 60).join("\n");
  const more    = lines.length > 60 ? `\n…(${lines.length - 60} more lines)` : "";

  const node = fromHTML(`<div class="tool-result${isError ? " error" : ""}"></div>`);
  node.textContent = (preview + more) || "(empty)";

  if (card) {
    card.removeAttribute("data-pending");
    card.querySelector(".tool-body").appendChild(node);
    // Auto-expand if the tool errored so the user sees it
    if (isError) card.classList.remove("collapsed");
    // Inline-render generated images right inside the tool card.
    if (name === "generate_image" && !isError) {
      const m = text.match(/saved to (\S+)/);
      if (m) {
        const src = "/" + m[1].replace(/\\/g, "/").replace(/^\/+/, "");
        const img = fromHTML(`<img class="tool-image" alt="generated" loading="lazy">`);
        img.src = src;
        card.querySelector(".tool-body").appendChild(img);
        card.classList.remove("collapsed");
      }
    }
  } else {
    messagesEl.appendChild(node);
  }
  scrollToBottom();
}

function appendConfirm(payload) {
  removeThinking();
  // Attach the Allow/Deny buttons to the pending tool card created by tool_call.
  // (The agent emits tool_call BEFORE perms.ask, so the card already exists.)
  let card = findPendingTool(payload.name);
  if (!card) {
    // Fallback: render a standalone confirm with the args.
    card = appendToolCall(payload.name, payload.args);
  }
  card.classList.add("awaiting");
  card.classList.remove("collapsed");

  const actions = fromHTML(`
    <div class="confirm-actions">
      <div class="confirm-label">Approve this action?</div>
      <button class="allow" type="button">Allow</button>
      <button class="deny"  type="button">Deny</button>
    </div>`);
  actions.querySelector(".allow").addEventListener("click",
    () => resolveConfirm(card, actions, payload.id, true));
  actions.querySelector(".deny").addEventListener("click",
    () => resolveConfirm(card, actions, payload.id, false));
  card.querySelector(".tool-body").appendChild(actions);
  scrollToBottom();
}

async function resolveConfirm(card, actions, id, allow) {
  actions.innerHTML =
    `<div class="confirm-label resolved">${allow ? "✓ allowed" : "✗ denied"}</div>`;
  card.classList.remove("awaiting");
  try {
    await fetch("/api/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, allow }),
    });
  } catch (e) {
    appendError(`failed to send decision: ${e.message}`);
  }
}

function appendSubtaskMarker(text) {
  const node = fromHTML(`<div class="subtask-marker"></div>`);
  node.textContent = text;
  messagesEl.appendChild(node);
  scrollToBottom();
}

function appendError(msg) {
  removeThinking();
  const node = fromHTML(`<div class="error-box"></div>`);
  node.textContent = msg;
  messagesEl.appendChild(node);
  scrollToBottom();
}

function appendThinking() {
  removeThinking();
  const node = fromHTML(`
    <div class="row agent thinking-row">
      <div class="avatar">wd</div>
      <div class="bubble thinking">
        thinking
        <span class="dots"><span></span><span></span><span></span></span>
      </div>
    </div>`);
  messagesEl.appendChild(node);
  scrollToBottom();
}

function removeThinking() {
  document.querySelectorAll(".thinking-row").forEach(n => n.remove());
}

// ---------------- SSE stream parsing ----------------

async function sendUserMessage(text) {
  appendUser(text);
  appendThinking();
  setBusy(true);
  setHint("watchdog is working…", "busy");

  let resp;
  try {
    resp = await fetch("/api/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${body || resp.statusText}`);
    }
  } catch (e) {
    appendError(`request failed: ${e.message}`);
    setBusy(false);
    setHint("error.", "error");
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (e) {
      appendError(`stream broke: ${e.message}`);
      break;
    }
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      handleEventBlock(block);
    }
  }

  setBusy(false);
  setHint("Ready.");
  removeThinking();
  input.focus();
}

function handleEventBlock(block) {
  let type = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue; // SSE comment / blank
    if (line.startsWith("event: ")) type = line.slice(7).trim();
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  let payload;
  try { payload = data ? JSON.parse(data) : {}; }
  catch { payload = { raw: data }; }

  switch (type) {
    case "user":          /* already rendered locally */ break;
    case "assistant":     appendAssistant(payload.text || ""); break;
    case "tool_call":     appendToolCall(payload.name, payload.args || {}); break;
    case "tool_result":   appendToolResult(payload.name, payload.result || "", !!payload.is_error); break;
    case "confirm":       appendConfirm(payload); break;
    case "subtask_start": appendSubtaskMarker("⇢ subtask: " + (payload.goal || "")); break;
    case "subtask_end":   appendSubtaskMarker("⇠ subtask done"); break;
    case "error":         appendError(payload.message || ""); break;
    case "round_cap":     appendError(`[${payload.label}] hit round limit (${payload.max_rounds}) — stopping.`); break;
    case "done":          break;
    default:              /* unknown — ignore */ break;
  }
}

// focus on load
input.focus();
