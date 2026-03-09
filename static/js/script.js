// ============================================================
// CEUB IA Assistant — script.js v2.0
// Melhorias:
//   ✅ Histórico de conversa enviado ao backend
//   ✅ Tratamento de erros específicos (429, 401, 500, rede)
//   ✅ Renderização de markdown básico (negrito, bullets)
//   ✅ Suggestions funcionais no estado vazio
//   ✅ Badge de intenção classificada pelo LLM
//   ✅ Ctrl+Enter envia; Enter simples quebra linha
//   ✅ Auto-resize do textarea
// ============================================================

/** @type {Array<{role: string, content: string}>} */
let conversationHistory = [];

const SUGGESTIONS = [
  "Qual é a frequência mínima exigida?",
  "Como acesso o Espaço Aluno?",
  "Quais são as menções para aprovação?",
  "Como solicito a carteirinha estudantil?",
  "Onde fica o campus Asa Norte?",
];

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  renderEmptyState();
  setupTextareaAutoResize();
});

// ---------------------------------------------------------------------------
// Estado vazio com suggestions
// ---------------------------------------------------------------------------

function renderEmptyState() {
  const container = document.getElementById("messagesContainer");
  const suggestionsHTML = SUGGESTIONS.map(
    (s) => `<button class="suggestion-btn" onclick="useSuggestion('${s}')">${s}</button>`
  ).join("");

  container.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">📚</div>
      <div class="empty-text">Olá! Sou o CEUB IA Assistant</div>
      <p style="font-size:13px; color:var(--text-secondary); margin-top:4px;">
        Pergunte sobre regras acadêmicas, sistemas ou o campus.
      </p>
      <div class="suggestions" style="max-width:400px; margin-top:20px;">
        ${suggestionsHTML}
      </div>
    </div>
  `;
}

function useSuggestion(text) {
  const input = document.getElementById("messageInput");
  input.value = text;
  input.focus();
  sendMessage();
}

// ---------------------------------------------------------------------------
// Auto-resize do textarea
// ---------------------------------------------------------------------------

function setupTextareaAutoResize() {
  const textarea = document.getElementById("messageInput");
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
  });
}

// ---------------------------------------------------------------------------
// Envio de mensagem
// ---------------------------------------------------------------------------

function handleKeyPress(event) {
  // Ctrl+Enter envia; Enter puro quebra linha (comportamento natural)
  if (event.key === "Enter" && !event.shiftKey && !event.ctrlKey) {
    event.preventDefault();
    sendMessage();
  }
}

async function sendMessage() {
  const input = document.getElementById("messageInput");
  const sendBtn = document.querySelector(".send-btn");
  const text = input.value.trim();

  if (!text) return;

  // Limpa empty state se for primeira mensagem
  clearEmptyState();

  // Exibe mensagem do usuário
  addMessage("user", text);

  // Adiciona ao histórico local
  conversationHistory.push({ role: "user", content: text });

  // Reseta input
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;

  // Loading
  const loadingId = addLoading();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        // Envia histórico SEM a mensagem atual (já foi adicionada acima)
        history: conversationHistory.slice(0, -1),
      }),
    });

    removeLoading(loadingId);

    if (!response.ok) {
      const err = await parseError(response);
      addErrorMessage(err);
      // Remove a mensagem do histórico se falhou
      conversationHistory.pop();
      return;
    }

    const data = await response.json();

    // Adiciona resposta ao histórico
    conversationHistory.push({ role: "assistant", content: data.response });

    // Limita histórico a 20 trocas (40 entradas) para não crescer infinito
    if (conversationHistory.length > 40) {
      conversationHistory = conversationHistory.slice(-40);
    }

    addMessage("bot", data.response, data.source, data.intent);

  } catch (error) {
    removeLoading(loadingId);
    conversationHistory.pop();

    if (error instanceof TypeError && error.message.includes("fetch")) {
      addErrorMessage("Sem conexão com o servidor. Verifique se o main.py está rodando na porta 8000.");
    } else {
      addErrorMessage("Erro inesperado. Tente novamente.");
    }
    console.error("[CEUB IA]", error);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

async function parseError(response) {
  try {
    const data = await response.json();
    const detail = data?.detail || "Erro desconhecido.";
    if (response.status === 429) return "⏳ " + detail;
    if (response.status === 401) return "🔑 " + detail;
    if (response.status === 500) return "⚙️ " + detail;
    return detail;
  } catch {
    return `Erro HTTP ${response.status}. Tente novamente.`;
  }
}

// ---------------------------------------------------------------------------
// Renderização de mensagens
// ---------------------------------------------------------------------------

function clearEmptyState() {
  const container = document.getElementById("messagesContainer");
  if (container.querySelector(".empty-state")) {
    container.innerHTML = "";
  }
}

function addMessage(type, text, source = null, intent = null) {
  const container = document.getElementById("messagesContainer");
  const div = document.createElement("div");
  div.className = `message ${type}`;

  const avatar = type === "user" ? "👤" : "🤖";
  const renderedText = renderMarkdown(text);

  const intentBadge =
    intent && intent !== "geral"
      ? `<span class="intent-badge">${intentLabel(intent)}</span>`
      : "";

  div.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div>
      <div class="message-content">
        ${renderedText}
        ${source ? `<div class="source-badge">${source}</div>` : ""}
        ${intentBadge}
      </div>
      <div class="message-time">${formatTime()}</div>
    </div>
  `;

  container.appendChild(div);
  scrollToBottom();
}

function addErrorMessage(detail) {
  const container = document.getElementById("messagesContainer");
  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = `
    <div class="message-avatar">⚠️</div>
    <div>
      <div class="message-content" style="border-color: var(--error);">
        <span style="color: var(--error);">${escapeHtml(detail)}</span>
      </div>
      <div class="message-time">${formatTime()}</div>
    </div>
  `;
  container.appendChild(div);
  scrollToBottom();
}

function addLoading() {
  const id = "loading-" + Date.now();
  const container = document.getElementById("messagesContainer");
  const div = document.createElement("div");
  div.id = id;
  div.className = "message bot";
  div.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-content">
      <div class="loading">
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
      </div>
    </div>
  `;
  container.appendChild(div);
  scrollToBottom();
  return id;
}

function removeLoading(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ---------------------------------------------------------------------------
// Utilitários
// ---------------------------------------------------------------------------

/** Renderiza markdown básico: **negrito**, bullets com •, quebras de linha */
function renderMarkdown(text) {
  return escapeHtml(text)
    // **negrito**
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    // • item ou - item → lista
    .replace(/^[•\-]\s(.+)$/gm, "<li>$1</li>")
    // Agrupa <li> em <ul>
    .replace(/(<li>.*<\/li>(\n|$))+/g, (match) => `<ul style="margin: 6px 0 6px 16px;">${match}</ul>`)
    // Quebras de linha
    .replace(/\n/g, "<br>");
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatTime() {
  return new Date().toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function scrollToBottom() {
  const container = document.getElementById("messagesContainer");
  container.scrollTop = container.scrollHeight;
}

function intentLabel(intent) {
  const map = {
    regimento: "📋 Regimento",
    faq: "⚡ FAQ",
    guia: "📖 Guia",
  };
  return map[intent] || intent;
}

// Nova conversa
function newChat() {
  conversationHistory = [];
  renderEmptyState();
}