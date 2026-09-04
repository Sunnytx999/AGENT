const elements = {
  messages: document.querySelector("#messages"),
  form: document.querySelector("#messageForm"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendButton"),
  newSession: document.querySelector("#newSessionButton"),
  workingDirectory: document.querySelector("#workingDirectoryInput"),
  plan: document.querySelector("#planButton"),
  approve: document.querySelector("#approveButton"),
  reject: document.querySelector("#rejectButton"),
  exitPlan: document.querySelector("#exitPlanButton"),
  mode: document.querySelector("#modeBadge"),
  sessionId: document.querySelector("#sessionId"),
  currentTask: document.querySelector("#currentTask"),
  planFile: document.querySelector("#planFile"),
  planPreview: document.querySelector("#planPreview"),
};

let sessionId = null;
let busy = false;
let latestState = null;

function appendMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "你" : role === "error" ? "错误" : "Agent";
  const content = document.createElement("div");
  content.textContent = text;
  article.append(label, content);
  elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function setBusy(value) {
  busy = value;
  elements.input.disabled = value;
  elements.send.disabled = value || !sessionId;
  elements.newSession.disabled = value;
  updateControls(latestState);
}

function updateControls(state) {
  if (!state) return;
  const hasPlan = Boolean(state.pending_plan);
  elements.mode.textContent = busy ? "处理中…" : state.plan_mode ? "PLAN" : state.execution_authorized ? "PLAN APPROVED" : "NORMAL";
  elements.mode.className = `badge ${state.plan_mode ? "plan" : ""}`;
  elements.sessionId.textContent = state.session_id;
  elements.currentTask.textContent = state.current_task || "—";
  elements.planFile.textContent = state.plan_file;
  elements.planPreview.textContent = state.pending_plan || "尚未生成计划。";
  elements.plan.disabled = busy || state.plan_mode;
  elements.approve.disabled = busy || !hasPlan;
  elements.reject.disabled = busy || !hasPlan;
  elements.exitPlan.disabled = busy || !state.plan_mode;
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败：${response.status}`);
  return data;
}

async function createSession() {
  setBusy(true);
  try {
    const workingDirectory = elements.workingDirectory.value.trim();
    const state = await requestJson("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ working_directory: workingDirectory }),
    });
    sessionId = state.session_id;
    latestState = state;
    elements.workingDirectory.value = state.working_directory;
    elements.messages.replaceChildren();
    appendMessage("assistant", "会话已创建。你可以直接聊天，或先进入 Plan Mode。");
    updateControls(state);
    elements.input.focus();
  } catch (error) {
    appendMessage("error", error.message);
  } finally {
    setBusy(false);
  }
}

async function runAction(path) {
  if (!sessionId || busy) return;
  setBusy(true);
  try {
    const state = await requestJson(path, {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    });
    latestState = state;
    updateControls(state);
    if (state.response) appendMessage("assistant", state.response);
  } catch (error) {
    appendMessage("error", error.message);
  } finally {
    setBusy(false);
  }
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = elements.input.value.trim();
  if (!message || !sessionId || busy) return;
  appendMessage("user", message);
  elements.input.value = "";
  setBusy(true);
  try {
    const state = await requestJson("/api/messages", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    });
    latestState = state;
    updateControls(state);
    appendMessage("assistant", state.response || "请求已完成。");
  } catch (error) {
    appendMessage("error", error.message);
  } finally {
    setBusy(false);
    elements.input.focus();
  }
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

elements.newSession.addEventListener("click", createSession);
elements.plan.addEventListener("click", () => runAction("/api/plan"));
elements.approve.addEventListener("click", () => runAction("/api/approve"));
elements.reject.addEventListener("click", () => runAction("/api/reject"));
elements.exitPlan.addEventListener("click", () => runAction("/api/exit-plan"));

createSession();
