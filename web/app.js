/*
 * Multi-Agent Experiment Board — dashboard SPA.
 *
 * Reads data/session_summary.json (built by scripts/build_dashboard_data.py)
 * and streams live updates over /ws. No frameworks; no build step.
 */

const DATA_URL = "data/session_summary.json";
const AUTO_REFRESH_MS = 5000;
const WS_RECONNECT_MS = 2000;

// ---------- Metric families for Architectures tab ----------

const METRIC_FAMILIES = [
  {
    id: "edubench_scenario",
    label: "EduBench · Scenario",
    keys: ["edubench_iftc", "edubench_rtc", "edubench_crsc", "edubench_sei", "edubench_scenario_adaptation"],
  },
  {
    id: "edubench_factual",
    label: "EduBench · Factual/Reasoning",
    keys: ["edubench_bfa", "edubench_dka", "edubench_rpr", "edubench_eicp", "edubench_factual_reasoning_accuracy"],
  },
  {
    id: "edubench_pedagogy",
    label: "EduBench · Pedagogy",
    keys: ["edubench_csi", "edubench_mgp", "edubench_pas", "edubench_hots", "edubench_pedagogical_application"],
  },
  {
    id: "tutoreval",
    label: "TutorEval",
    keys: ["tutoreval_keypoint_hit_rate", "tutoreval_correctness", "tutoreval_completeness", "tutoreval_relevance", "tutoreval_keypoint_recall", "tutoreval_chapter_grounding"],
  },
  {
    id: "cost",
    label: "Cost · Latency · Tokens",
    keys: ["latency_ms", "api_time_ms", "total_tokens", "llm_call_count", "complexity_units"],
    lowerIsBetter: true,
  },
];

const METRIC_GUIDE = [
  { key: "score", label: "composite", digits: 3, direction: "higher is better", description: "Aggregate score used for leaderboard ranking. Combines available quality metrics weighted across EduBench 12d, TutorEval key-point, and cost-efficiency signals." },
  { key: "edubench_12d_mean", label: "edubench_12d_mean", digits: 3, direction: "higher is better", description: "Mean of all 12 EduBench rubric dimensions (Scenario × 4, Factual/Reasoning × 4, Pedagogical × 4). NaN-skipped: rows without a reference score do not drag this down." },
  { key: "edubench_scenario_adaptation", label: "scenario_adaptation", digits: 3, direction: "higher is better", description: "Mean over IFTC / RTC / CRSC / SEI — whether the answer follows task constraints, responds with appropriate tone, stays relevant, and weaves scenario elements." },
  { key: "edubench_factual_reasoning_accuracy", label: "factual_reasoning", digits: 3, direction: "higher is better", description: "Mean over BFA / DKA / RPR / EICP — basic factual accuracy, domain knowledge application, reasoning rigor, and error-identification/correction precision." },
  { key: "edubench_pedagogical_application", label: "pedagogical", digits: 3, direction: "higher is better", description: "Mean over CSI / MGP / PAS / HOTS — clarity/simplicity, motivational guidance, personalized adaptation, higher-order thinking scaffolding." },
  { key: "tutoreval_keypoint_hit_rate", label: "keypoint_hit_rate", digits: 3, direction: "higher is better", description: "TutorEval per-keypoint hit rate. Soft curve: 1.0 at ≥70% token overlap, linear ramp 25%→70%, 0 below 25%.", code_preview: "def keypoint_token_alignment(answer: str, key_points: list[str]) -> tuple[float, float]:\n    ans_tokens = set(tokenize(answer))\n    hits = [len(set(tokenize(kp)) & ans_tokens) / max(1, len(tokenize(kp))) for kp in key_points]\n    hit_rate = sum(1 for h in hits if h >= 0.70) / len(key_points) if key_points else 0.0\n    return hit_rate, sum(hits) / len(hits)" },
  { key: "tutoreval_correctness", label: "correctness", digits: 3, direction: "higher is better", description: "TutorEval correctness = 0.7·keypoint_hit_rate + 0.3·target_match against the synthesized gold (join of key points)." },
  { key: "tutoreval_completeness", label: "completeness", digits: 3, direction: "higher is better", description: "TutorEval completeness = 0.8·keypoint_hit_rate + 0.2·keypoint_recall; rewards covering the full set of required points." },
  { key: "tutoreval_relevance", label: "relevance", digits: 3, direction: "higher is better", description: "Closed-book: 0.5·question_match + 0.5·keypoint_hit_rate. Open-book: 0.5·question_match + 0.3·context_overlap + 0.2·keypoint_hit_rate." },
  { key: "tutoreval_chapter_grounding", label: "chapter_grounding", digits: 3, direction: "higher is better", description: "Evidence-grounded overlap with the retrieved chapter. NaN (skipped) for rows whose key points are not known to live in-chapter." },
  { key: "rubric_coverage", label: "rubric_coverage", digits: 3, direction: "higher is better", description: "Tightened: item counts if (40-char prefix present) OR (≥55% item-token overlap AND ≥1 rare ≥4-char non-stopword hit).", code_preview: "def rubric_coverage(answer: str, rubric: list[str]) -> float:\n    if not rubric or not answer: return 0.0\n    ans_toks = tokenize(answer)\n    hits = 0\n    for item in rubric:\n        item_toks = tokenize(item)\n        prefix_ok = len(item) >= 40 and answer.startswith(item[:40])\n        overlap = len(set(ans_toks) & set(item_toks)) / max(1, len(item_toks))\n        rare_ok = any(len(w) >= 4 and w not in STOPWORDS for w in item_toks)\n        if prefix_ok or (overlap >= 0.55 and rare_ok):\n            hits += 1\n    return hits / len(rubric)" },
  { key: "edu_json_compliance", label: "edu_json", digits: 3, direction: "higher is better", description: "EduBench JSON schema compliance: fraction of required keys (score, Scoring_Details, Personalized Feedback) that are present in a parseable JSON object." },
  { key: "edu_score_alignment", label: "edu_align", digits: 3, direction: "higher is better", description: "|predicted_score - reference_score|/100 mapped to [0..1]. NaN-skipped when no reference score is available on a row." },
  { key: "token_f1", label: "token_f1", digits: 3, direction: "higher is better", description: "Token-level overlap F1 between predicted answer text and reference text (gold or synthesized).", code_preview: "def token_f1(prediction: str, gold: str) -> float:\n    p_toks = tokenize(prediction)\n    g_toks = tokenize(gold)\n    overlap = len(set(p_toks) & set(g_toks))\n    precision = overlap / len(p_toks) if p_toks else 0.0\n    recall = overlap / len(g_toks) if g_toks else 0.0\n    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0" },
  { key: "exact_match", label: "exact_match", digits: 3, direction: "higher is better", description: "Strict exact-match ratio against normalized gold answer text when available." },
  { key: "grounded_overlap", label: "grounded_overlap", digits: 3, direction: "higher is better", description: "Answer tokens covered by retrieved chunk tokens; proxy for groundedness in RAG pipelines." },
  { key: "latency_ms", label: "latency_ms", digits: 1, direction: "lower is better", description: "End-to-end pipeline latency per example in milliseconds." },
  { key: "api_time_ms", label: "api_time_ms", digits: 1, direction: "lower is better", description: "Time spent in LLM API calls per example." },
  { key: "llm_call_count", label: "llm_call_count", digits: 1, direction: "context-dependent", description: "Average number of LLM calls per example; lower is cheaper, higher may indicate deeper reasoning." },
  { key: "total_tokens", label: "total_tokens", digits: 1, direction: "lower is better", description: "Total tokens across prompt and completion per example." },
  { key: "complexity_units", label: "complexity_units", digits: 1, direction: "lower is better", description: "Composite workload indicator combining token cost, retrieval work, and orchestration overhead." },
];

// ---------- Runtime state ----------

let rawData = { overview: {}, sessions: [], dataset_cards: [] };
let activeSessionKey = null;
let activeTab = "overview";
let activeFamily = METRIC_FAMILIES[0].id;
let sessionStatusFilter = "all";
let sortKey = "score";
let sortDir = "desc";
let searchQuery = "";
let exampleStatusFilter = "all";
let exampleSortKey = "example_id";
let exampleSortDir = "asc";

let refreshTimer = null;
let loading = false;
let socket = null;
let socketReconnectTimer = null;
let socketConnected = false;
const metricGuideExpanded = new Set();

let previousSessionStates = new Map();

// ---------- Architecture diagrams ----------

// Publication-quality architecture diagrams.
// Each node: { id, title, subtitle, kind, stage, x, y, w, h, icon }
// kinds: io | agent | orchestrator | retrieval | memory | decision | aggregator
// Node coordinate system: 960 x 400 canvas with stage bands.

const STAGE_BANDS = [
  { id: "input", label: "INPUT",        x: 0,   w: 160, fill: "var(--accent-yellow-soft)" },
  { id: "plan",  label: "CONTEXT ASSEMBLY", x: 160, w: 260, fill: "var(--accent-violet-soft)" },
  { id: "gen",   label: "GENERATION",   x: 420, w: 220, fill: "var(--accent-cyan-soft)" },
  { id: "verify",label: "VERIFICATION", x: 640, w: 180, fill: "var(--accent-pink-soft)" },
  { id: "out",   label: "OUTPUT",       x: 820, w: 140, fill: "var(--accent-lime-soft)" },
];

const NODE_ICONS = {
  io:           `<path d="M7 7h10v10H7z" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M10 10l4 2-4 2z" fill="currentColor"/>`,
  agent:        `<circle cx="12" cy="9" r="3" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M6 18c0-3 2.5-5 6-5s6 2 6 5" fill="none" stroke="currentColor" stroke-width="1.8"/>`,
  orchestrator: `<circle cx="12" cy="12" r="3" fill="currentColor"/><circle cx="5" cy="5" r="2" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="19" cy="5" r="2" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="5" cy="19" r="2" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="19" cy="19" r="2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M7 7l3 3M17 7l-3 3M7 17l3-3M17 17l-3-3" stroke="currentColor" stroke-width="1.4"/>`,
  retrieval:    `<rect x="4" y="5" width="16" height="3" rx="1" fill="currentColor"/><rect x="4" y="10.5" width="16" height="3" rx="1" fill="currentColor" opacity="0.7"/><rect x="4" y="16" width="16" height="3" rx="1" fill="currentColor" opacity="0.4"/>`,
  decision:     `<path d="M12 3l9 9-9 9-9-9z" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M9 12l2 2 4-4" stroke="currentColor" stroke-width="1.8" fill="none"/>`,
  aggregator:   `<path d="M4 5h16l-6 7v5l-4 2v-7z" fill="none" stroke="currentColor" stroke-width="1.8"/>`,
  memory:       `<ellipse cx="12" cy="6" rx="7" ry="2.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M5 6v12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V6" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5" fill="none" stroke="currentColor" stroke-width="1.4"/>`,
  critic:       `<path d="M5 4l4 4M5 4l0 16M5 20l14 0M9 8l10-4 0 16-10 0z" fill="none" stroke="currentColor" stroke-width="1.6"/>`,
};

const NODE_STYLE = {
  io:           { fill: "var(--accent-yellow)", stroke: "var(--ink)" },
  agent:        { fill: "var(--accent-cyan)", stroke: "var(--ink)" },
  orchestrator: { fill: "var(--accent-pink)", stroke: "var(--ink)" },
  retrieval:    { fill: "var(--accent-violet)", stroke: "var(--ink)" },
  decision:     { fill: "#ffd93d", stroke: "var(--ink)" },
  aggregator:   { fill: "var(--accent-violet)", stroke: "var(--ink)" },
  memory:       { fill: "var(--card-alt)", stroke: "var(--ink)" },
  critic:       { fill: "var(--accent-cyan)", stroke: "var(--ink)" },
};

// Canonical node sizes: 150 x 70
const NODE_W = 150;
const NODE_H = 72;

const ARCHITECTURE_DIAGRAMS = {
  single_agent_no_rag: {
    name: "Single Agent (No RAG)",
    description: "Baseline: direct zero-shot generation without retrieval or orchestration.",
    nodes: [
      { id: "input",  kind: "io",     title: "Question",    subtitle: "prompt + metadata", icon: "io",     x: 25,  y: 165 },
      { id: "tutor",  kind: "agent",  title: "Tutor Agent", subtitle: "Qwen3.5-4B · budget 512", icon: "agent",  x: 460, y: 165 },
      { id: "output", kind: "io",     title: "Answer",      subtitle: "plain text",        icon: "io",     x: 830, y: 165 },
    ],
    edges: [
      { from: "input",  to: "tutor",  label: "question" },
      { from: "tutor",  to: "output", label: "response" },
    ],
    activeBands: ["input", "gen", "out"],
  },

  classical_rag: {
    name: "Classical RAG",
    description: "Retrieval → context packing → single-shot generation. Retrieval is done once, upfront.",
    nodes: [
      { id: "input",   kind: "io",        title: "Question",      subtitle: "prompt + metadata", icon: "io",        x: 25,  y: 165 },
      { id: "retriev", kind: "retrieval", title: "Hybrid Retriever", subtitle: "BM25 + Dense (k=8)", icon: "retrieval", x: 185, y: 80 },
      { id: "corpus",  kind: "memory",    title: "Corpus Index",   subtitle: "16.9k chunks",    icon: "memory",    x: 185, y: 250 },
      { id: "tutor",   kind: "agent",     title: "Tutor Agent",    subtitle: "Qwen3.5-4B + ctx",  icon: "agent",     x: 460, y: 165 },
      { id: "critic",  kind: "critic",    title: "Critic Agent",   subtitle: "self-verify · opt.", icon: "critic",    x: 665, y: 165 },
      { id: "output",  kind: "io",        title: "Answer",         subtitle: "cited response",   icon: "io",        x: 830, y: 165 },
    ],
    edges: [
      { from: "input",   to: "retriev", label: "query" },
      { from: "corpus",  to: "retriev", label: "", dashed: true },
      { from: "retriev", to: "tutor",   label: "k=8 chunks" },
      { from: "input",   to: "tutor",   label: "prompt" },
      { from: "tutor",   to: "critic",  label: "draft", dashed: true },
      { from: "critic",  to: "output",  label: "verified" },
      { from: "tutor",   to: "output",  label: "", dashed: true, bend: -60 },
    ],
    activeBands: ["input", "plan", "gen", "verify", "out"],
  },

  hybrid_fast: {
    name: "Hybrid Fast",
    description: "Router decides: short factual queries → direct tutor. Complex/explanatory → RAG path.",
    nodes: [
      { id: "input",   kind: "io",         title: "Question",     subtitle: "prompt + metadata", icon: "io",        x: 25,  y: 165 },
      { id: "router",  kind: "decision",   title: "Router",       subtitle: "regime classifier", icon: "decision",  x: 180, y: 165 },
      { id: "fast",    kind: "agent",      title: "Fast Path",    subtitle: "Qwen3.5-4B (no ctx)", icon: "agent",    x: 380, y: 75 },
      { id: "retriev", kind: "retrieval",  title: "RAG Retriever", subtitle: "BM25+Dense · k=5",  icon: "retrieval", x: 380, y: 250 },
      { id: "tutor",   kind: "agent",      title: "Tutor + RAG",  subtitle: "Qwen3.5-4B + ctx",  icon: "agent",     x: 580, y: 250 },
      { id: "merge",   kind: "aggregator", title: "Merge",        subtitle: "path collapse",     icon: "aggregator",x: 745, y: 165 },
      { id: "output",  kind: "io",         title: "Answer",       subtitle: "path-tagged",       icon: "io",        x: 830, y: 165 },
    ],
    edges: [
      { from: "input",   to: "router",  label: "question" },
      { from: "router",  to: "fast",    label: "simple",  color: "var(--accent-cyan)" },
      { from: "router",  to: "retriev", label: "complex", color: "var(--accent-violet)" },
      { from: "retriev", to: "tutor",   label: "chunks" },
      { from: "fast",    to: "merge",   label: "draft" },
      { from: "tutor",   to: "merge",   label: "grounded" },
      { from: "merge",   to: "output",  label: "" },
    ],
    activeBands: ["input", "plan", "gen", "verify", "out"],
  },

  non_rag_multi_agent: {
    name: "Multi-Agent (No RAG)",
    description: "Three specialists run in parallel → context fusion → tutor drafts → critic revises.",
    nodes: [
      { id: "input",     kind: "io",           title: "Question",         subtitle: "prompt + metadata", icon: "io",           x: 25,  y: 165 },
      { id: "planner",   kind: "orchestrator", title: "Planner",          subtitle: "decomposes task",   icon: "orchestrator", x: 180, y: 55 },
      { id: "diagnoser", kind: "agent",        title: "Diagnoser",        subtitle: "student state",     icon: "agent",        x: 180, y: 165 },
      { id: "rubric",    kind: "agent",        title: "Rubric Agent",     subtitle: "scoring criteria",  icon: "agent",        x: 180, y: 275 },
      { id: "fuse",      kind: "aggregator",   title: "Context Fusion",   subtitle: "merge traces",      icon: "aggregator",   x: 350, y: 165 },
      { id: "tutor",     kind: "agent",        title: "Tutor Agent",      subtitle: "Qwen3.5-4B · draft",icon: "agent",        x: 500, y: 165 },
      { id: "critic",    kind: "critic",       title: "Critic Agent",     subtitle: "rewrite · opt.",    icon: "critic",       x: 665, y: 165 },
      { id: "output",    kind: "io",           title: "Answer",           subtitle: "verified + rubric", icon: "io",           x: 830, y: 165 },
    ],
    edges: [
      { from: "input",     to: "planner",   label: "", dashed: true },
      { from: "input",     to: "diagnoser", label: "" },
      { from: "input",     to: "rubric",    label: "", dashed: true },
      { from: "planner",   to: "fuse",      label: "plan" },
      { from: "diagnoser", to: "fuse",      label: "state" },
      { from: "rubric",    to: "fuse",      label: "rubric" },
      { from: "fuse",      to: "tutor",     label: "ctx" },
      { from: "tutor",     to: "critic",    label: "draft" },
      { from: "critic",    to: "output",    label: "final" },
    ],
    parallel: ["planner", "diagnoser", "rubric"],
    activeBands: ["input", "plan", "gen", "verify", "out"],
  },

  agentic_rag: {
    name: "Agentic RAG",
    description: "Planner-driven retrieval: planner queries guide retrieval, diagnoser informs tutor, critic verifies.",
    nodes: [
      { id: "input",     kind: "io",           title: "Question",    subtitle: "prompt + metadata", icon: "io",           x: 25,  y: 165 },
      { id: "planner",   kind: "orchestrator", title: "Planner",     subtitle: "→ search queries",  icon: "orchestrator", x: 180, y: 55 },
      { id: "diagnoser", kind: "agent",        title: "Diagnoser",   subtitle: "student state",     icon: "agent",        x: 180, y: 275 },
      { id: "retriev",   kind: "retrieval",    title: "Retriever",   subtitle: "planner-guided k=8",icon: "retrieval",    x: 350, y: 165 },
      { id: "corpus",    kind: "memory",       title: "Corpus",      subtitle: "16.9k chunks",      icon: "memory",       x: 350, y: 300 },
      { id: "tutor",     kind: "agent",        title: "Tutor Agent", subtitle: "Qwen3.5-4B + ctx",  icon: "agent",        x: 510, y: 165 },
      { id: "critic",    kind: "critic",       title: "Critic Agent",subtitle: "rewrite",           icon: "critic",       x: 665, y: 165 },
      { id: "output",    kind: "io",           title: "Answer",      subtitle: "cited + verified",  icon: "io",           x: 830, y: 165 },
    ],
    edges: [
      { from: "input",     to: "planner",   label: "" },
      { from: "input",     to: "diagnoser", label: "" },
      { from: "planner",   to: "retriev",   label: "queries" },
      { from: "corpus",    to: "retriev",   label: "", dashed: true },
      { from: "diagnoser", to: "tutor",     label: "state", dashed: true },
      { from: "retriev",   to: "tutor",     label: "chunks" },
      { from: "tutor",     to: "critic",    label: "draft" },
      { from: "critic",    to: "output",    label: "final" },
    ],
    parallel: ["planner", "diagnoser"],
    activeBands: ["input", "plan", "gen", "verify", "out"],
  },
};

// ---------- Toast system ----------

function showToast(message, type = "info") {
  const container = $("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = "toastOut 200ms ease forwards";
    setTimeout(() => toast.remove(), 200);
  }, 4000);
}

function checkSessionStateChanges(newSessions) {
  for (const s of newSessions) {
    const prev = previousSessionStates.get(s.session_key);
    if (prev && prev.status !== s.status) {
      const msg = `${esc(s.architecture)} on ${esc(s.dataset)} → ${esc(s.status)}`;
      if (s.status === "finished") showToast(msg, "success");
      else if (s.status === "failed") showToast(msg, "error");
      else if (s.status === "running" && prev.status !== "running") showToast(msg, "info");
    }
    previousSessionStates.set(s.session_key, { status: s.status });
  }
}

// ---------- Sparkline ----------

function drawSparkline(values, width = 40, height = 16) {
  if (!values || values.length < 2) return `<svg class="sparkline" viewBox="0 0 ${width} ${height}"><path class="sparkline-path" d="M0,${height/2} L${width},${height/2}"/></svg>`;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / (values.length - 1);
  const points = values.map((v, i) => {
    const x = i * step;
    const y = height - ((v - min) / range) * (height - 2) - 1;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `<svg class="sparkline" viewBox="0 0 ${width} ${height}"><polyline class="sparkline-path" points="${points.join(" ")}" fill="none"/></svg>`;
}

// ---------- Helpers ----------

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined) return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(digits);
}

function fmtWithN(value, nValue, digits = 3) {
  const val = fmtNumber(value, digits);
  if (nValue === undefined || nValue === null) return val;
  return `${val} <span class="n-dim">(n=${Math.round(Number(nValue) || 0)})</span>`;
}

function clampPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n));
}

function statusClass(status) {
  if (status === "finished") return "status-finished";
  if (status === "failed") return "status-failed";
  return "status-running";
}

function readHashState() {
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const tab = params.get("tab");
  if (tab) activeTab = tab;
  const session = params.get("session");
  if (session) activeSessionKey = session;
  const family = params.get("family");
  if (family) activeFamily = family;
  const status = params.get("status");
  if (status) sessionStatusFilter = status;
}

function writeHashState() {
  const params = new URLSearchParams();
  params.set("tab", activeTab);
  if (activeSessionKey) params.set("session", activeSessionKey);
  if (activeTab === "architectures") params.set("family", activeFamily);
  if (sessionStatusFilter !== "all") params.set("status", sessionStatusFilter);
  const next = "#" + params.toString();
  if (window.location.hash !== next) {
    history.replaceState(null, "", next);
  }
}

// ---------- Tab switching ----------

function setActiveTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".nav-item").forEach((el) => {
    const on = el.dataset.tab === tab;
    el.classList.toggle("is-active", on);
    el.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((el) => {
    const on = el.dataset.panel === tab;
    el.classList.toggle("is-active", on);
    el.hidden = !on;
  });
  const titles = {
    overview: "Run Health",
    architectures: "Architectures",
    sessions: "Sessions",
    deepdive: "Deep Dive",
    metrics: "Metric Guide",
  };
  $("stageTitle").textContent = titles[tab] || "Multi-Agent";
  writeHashState();
  render();
}

// ---------- Renderers ----------

function renderOverview() {
  const ov = rawData.overview || {};
  const tiles = [
    ["Total", ov.total_sessions ?? 0, null],
    ["Finished", ov.finished_sessions ?? 0, "ok"],
    ["Running", ov.running_sessions ?? 0, "info"],
    ["Failed", ov.failed_sessions ?? 0, "bad"],
    ["Overall", ov.overall_progress_pct !== undefined ? `${fmtNumber(ov.overall_progress_pct, 1)}%` : "-", null],
    ["Examples", ov.overall_total_examples ? `${ov.overall_completed_examples || 0}/${ov.overall_total_examples}` : "-", null],
  ];
  const grid = $("overviewGrid");
  grid.innerHTML = tiles.map(([label, value]) => `
    <article class="stat-tile">
      <p class="stat-label">${esc(label)}</p>
      <p class="stat-value">${value ?? "-"}</p>
    </article>
  `).join("");

  const pct = clampPct(ov.overall_progress_pct);
  $("overallProgressFill").style.width = `${pct}%`;
  $("overallProgressFill").parentElement.setAttribute("aria-valuenow", pct.toFixed(1));
  $("overallProgressLabel").textContent = ov.overall_total_examples
    ? `overall: ${ov.overall_completed_examples || 0}/${ov.overall_total_examples} (${pct.toFixed(1)}%)`
    : "overall: no totals yet";

  renderLeaderboard();
  renderDatasetCards();
  renderLiveHealth();
  renderSnapshotTime();
  renderArchDiagrams();
}

function renderLeaderboard() {
  const rows = (rawData.overview && rawData.overview.architecture_leaderboard) || [];
  const el = $("rankGrid");
  if (!rows.length) {
    el.innerHTML = `<article class="rank-tile"><p class="rank-head">No finished runs yet</p></article>`;
    return;
  }
  el.innerHTML = rows.map((row, idx) => `
    <article class="rank-tile">
      <p class="rank-head">#${idx + 1} ${esc(row.architecture)}</p>
      <p class="rank-meta">avg ${fmtNumber(row.avg_score)} · best ${fmtNumber(row.best_score)}</p>
      <p class="rank-meta">runs: ${esc(row.runs)}</p>
    </article>
  `).join("");
}

function renderDatasetCards() {
  const cards = rawData.dataset_cards || [];
  const el = $("datasetCards");
  if (!cards.length) {
    el.innerHTML = `<article class="dataset-tile"><p class="dataset-name">No datasets yet</p></article>`;
    return;
  }
  el.innerHTML = cards.map((card) => {
    const best = card.best_architecture || "pending";
    const score = card.best_score ?? null;
    const avg = card.average_score ?? null;
    const avgP = card.average_progress_pct ?? null;
    const b = card.status_breakdown || {};
    return `
      <article class="dataset-tile">
        <p class="dataset-name">${esc(card.dataset)}</p>
        <p class="dataset-best">best: ${esc(best)}</p>
        <p class="dataset-meta">score ${fmtNumber(score)} · avg ${fmtNumber(avg)}</p>
        <p class="dataset-meta">progress ${avgP !== null ? fmtNumber(avgP, 1) + "%" : "-"}</p>
        <p class="dataset-meta">f=${b.finished ?? 0} r=${b.running ?? 0} x=${b.failed ?? 0}</p>
      </article>
    `;
  }).join("");
}

function renderLiveHealth() {
  const el = $("liveIndicator");
  const textEl = $("liveIndicatorText");
  if (!el) return;
  el.classList.remove("live-fresh", "live-warm", "live-stale", "live-unknown");
  const epoch = Number(rawData.overview?.generated_epoch || 0);
  const ws = socketConnected ? "ws:on" : "ws:off";
  const setText = (t) => { if (textEl) textEl.textContent = t; };
  if (!Number.isFinite(epoch) || epoch <= 0) {
    setText(`${ws} · unknown`);
    el.classList.add("live-unknown");
    return;
  }
  const age = Math.max(0, Math.floor(Date.now() / 1000) - epoch);
  setText(`${ws} · ${age}s old`);
  if (age <= 30 && socketConnected) el.classList.add("live-fresh");
  else if (age <= 120) el.classList.add("live-warm");
  else el.classList.add("live-stale");
}

function renderSnapshotTime() {
  const ov = rawData.overview || {};
  $("snapshotTime").textContent = `snapshot: ${ov.generated_at || "--"} · websocket 1s stream`;
}

// ---------- Architectures tab ----------

function renderArchitectures() {
  const chipRow = $("familyChips");
  chipRow.innerHTML = METRIC_FAMILIES.map((fam) =>
    `<button class="chip ${fam.id === activeFamily ? "is-active" : ""}" data-family="${esc(fam.id)}">${esc(fam.label)}</button>`
  ).join("");
  chipRow.querySelectorAll(".chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeFamily = btn.dataset.family;
      writeHashState();
      renderArchitectures();
    });
  });

  const family = METRIC_FAMILIES.find((f) => f.id === activeFamily) || METRIC_FAMILIES[0];
  const lower = !!family.lowerIsBetter;
  const sessions = rawData.sessions || [];
  const byDataset = new Map();
  sessions.forEach((s) => {
    if (!byDataset.has(s.dataset)) byDataset.set(s.dataset, []);
    byDataset.get(s.dataset).push(s);
  });

  const blocks = [];
  for (const [dataset, rows] of byDataset) {
    // for each family key, show a bar chart over architectures
    const keyBlocks = family.keys.map((key) => {
      const vals = rows.map((s) => ({
        architecture: s.architecture,
        v: (s.metric_tiles || {})[key],
        n: (s.metric_tiles || {})[`${key}_n`],
      })).filter((r) => r.v !== undefined && r.v !== null && Number.isFinite(Number(r.v)));
      if (!vals.length) {
        return `<p class="bar-label" style="margin:2px 0">${esc(key)}: no data</p>`;
      }
      const max = Math.max(...vals.map((r) => Number(r.v) || 0));
      const min = Math.min(...vals.map((r) => Number(r.v) || 0));
      const best = lower ? min : max;
      const worst = lower ? max : min;
      const digits = /_ms$|_tokens$|_count$|units|per_second/.test(key) ? 1 : 3;
      const scale = max > 0 ? max : 1;
      return `
        <div class="bar-chart">
          <p class="bar-label" style="font-weight:600;color:var(--text);">${esc(key)}</p>
          ${vals.sort((a, b) => lower ? Number(a.v) - Number(b.v) : Number(b.v) - Number(a.v)).map((r) => {
            const vNum = Number(r.v);
            const w = Math.max(2, Math.min(100, (vNum / scale) * 100));
            const cls = vNum === best ? "is-best" : (vNum === worst ? "is-weak" : "");
            return `
              <div class="bar-row">
                <span class="bar-label">${esc(r.architecture)}</span>
                <span class="bar-track"><span class="bar-fill ${cls}" style="width:${w}%"></span></span>
                <span class="bar-value">${fmtNumber(vNum, digits)}${r.n ? ` <span class="n-dim">·n=${Math.round(r.n)}</span>` : ""}</span>
              </div>
            `;
          }).join("")}
        </div>
      `;
    }).join("");

    blocks.push(`
      <section class="arch-block">
        <h3><span>${esc(dataset)}</span> <span class="dataset-chip">${esc(family.label)}</span></h3>
        ${keyBlocks}
      </section>
    `);
  }

  $("archCompare").innerHTML = blocks.join("") || `<p class="detail-empty">No sessions yet</p>`;
}

// ---------- Sessions table ----------

function sessionComparable(session, key) {
  if (key === "progress_ratio") return Number(session.progress_ratio || 0);
  if (key === "records" || key === "score") return Number(session[key] || 0);
  if (key === "dataset" || key === "architecture" || key === "status") return session[key] || "";
  return Number((session.metric_tiles || {})[key] || session.summary?.[key] || 0);
}

function filteredSortedSessions() {
  const q = searchQuery.trim().toLowerCase();
  let rows = (rawData.sessions || []).slice();
  if (sessionStatusFilter !== "all") {
    rows = rows.filter((s) => s.status === sessionStatusFilter);
  }
  if (q) {
    rows = rows.filter((s) =>
      (s.dataset || "").toLowerCase().includes(q)
      || (s.architecture || "").toLowerCase().includes(q)
      || (s.session_key || "").toLowerCase().includes(q)
      || (s.status || "").toLowerCase().includes(q)
    );
  }
  rows.sort((a, b) => {
    const av = sessionComparable(a, sortKey);
    const bv = sessionComparable(b, sortKey);
    if (typeof av === "string" || typeof bv === "string") {
      return sortDir === "desc" ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
    }
    return sortDir === "desc" ? bv - av : av - bv;
  });
  return rows;
}

function renderSessionsTable() {
  const rows = filteredSortedSessions();
  if (!activeSessionKey && rows.length) activeSessionKey = rows[0].session_key;

  const tbody = $("sessionsBody");
  tbody.innerHTML = rows.map((s) => {
    const tiles = s.metric_tiles || {};
    const active = s.session_key === activeSessionKey ? "row-active" : "";
    const pct = clampPct((s.progress && s.progress.pct) || (s.progress_ratio * 100) || 0);
    const progressText = s.progress ? `${s.progress.completed}/${s.progress.total} (${s.progress.pct_text || pct.toFixed(1) + "%"})` : "-";
    const runningCls = s.status === "running" ? "is-running" : "";
    const spark = s.progress_history ? drawSparkline(s.progress_history) : "";
    return `
      <tr class="${active}" data-session="${esc(s.session_key)}">
        <td>${esc(s.dataset)}</td>
        <td>${esc(s.architecture)}</td>
        <td><span class="status-pill ${statusClass(s.status)} ${s.status === "running" ? "pulse" : ""}">${esc(s.status)}</span></td>
        <td class="num">${s.records ?? 0}</td>
        <td class="num">
          <div class="progress-cell">
            <div class="progress-track"><span class="progress-fill ${runningCls}" style="width:${pct}%"></span></div>
            <span class="progress-text">${esc(progressText)}</span>
            ${spark ? `<span class="sparkline-wrap">${spark}</span>` : ""}
          </div>
        </td>
        <td class="num"><span class="score-chip">${fmtNumber(s.score)}</span></td>
        <td class="num">${fmtNumber(tiles.token_f1)}</td>
        <td class="num">${fmtNumber(tiles.edubench_12d_mean)}</td>
        <td class="num">${fmtNumber(tiles.tutoreval_keypoint_hit_rate)}</td>
        <td class="num">${fmtNumber(tiles.rubric_coverage)}</td>
        <td class="num">${fmtNumber(tiles.edu_score_alignment)}</td>
        <td class="num">${fmtNumber(tiles.latency_ms, 1)}</td>
        <td class="num">${fmtNumber(tiles.total_tokens, 1)}</td>
      </tr>
    `;
  }).join("") || `<tr><td colspan="13" style="text-align:center;color:var(--text-dim);padding:24px;">No sessions match the current filter.</td></tr>`;

  tbody.querySelectorAll("tr[data-session]").forEach((tr) => {
    tr.addEventListener("click", () => {
      activeSessionKey = tr.dataset.session;
      writeHashState();
      renderSessionsTable();
      renderDeepDive();
      renderMetricGuide();
    });
  });

  document.querySelectorAll(".sessions-table th[data-sort]").forEach((th) => {
    th.onclick = () => {
      const k = th.dataset.sort;
      if (sortKey === k) {
        sortDir = sortDir === "desc" ? "asc" : "desc";
      } else {
        sortKey = k;
        sortDir = "desc";
      }
      renderSessionsTable();
    };
  });

  document.querySelectorAll("#statusChips .chip").forEach((chip) => {
    chip.classList.toggle("is-active", chip.dataset.status === sessionStatusFilter);
    chip.onclick = () => {
      sessionStatusFilter = chip.dataset.status;
      writeHashState();
      renderSessionsTable();
    };
  });
}

// ---------- Deep Dive ----------

function activeSession() {
  return (rawData.sessions || []).find((s) => s.session_key === activeSessionKey) || null;
}

function renderDeepDive() {
  const session = activeSession();
  const el = $("sessionDetail");
  if (!session) {
    el.innerHTML = `<p class="detail-empty">Select a session row (in Sessions) to inspect its timeline, metric breakdown, and log tail.</p>`;
    return;
  }

  const tiles = session.metric_tiles || {};
  const miniTiles = [
    ["composite", session.score, 3, null],
    ["EB 12d", tiles.edubench_12d_mean, 3, tiles.edubench_12d_mean_n],
    ["EB scenario", tiles.edubench_scenario_adaptation, 3, tiles.edubench_scenario_adaptation_n],
    ["EB factual", tiles.edubench_factual_reasoning_accuracy, 3, tiles.edubench_factual_reasoning_accuracy_n],
    ["EB pedagogy", tiles.edubench_pedagogical_application, 3, tiles.edubench_pedagogical_application_n],
    ["TE hit_rate", tiles.tutoreval_keypoint_hit_rate, 3, tiles.tutoreval_keypoint_hit_rate_n],
    ["TE correctness", tiles.tutoreval_correctness, 3, tiles.tutoreval_correctness_n],
    ["TE completeness", tiles.tutoreval_completeness, 3, tiles.tutoreval_completeness_n],
    ["TE relevance", tiles.tutoreval_relevance, 3, tiles.tutoreval_relevance_n],
    ["TE chapter_g", tiles.tutoreval_chapter_grounding, 3, tiles.tutoreval_chapter_grounding_n],
    ["token_f1", tiles.token_f1, 3, tiles.token_f1_n],
    ["rubric_cov", tiles.rubric_coverage, 3, tiles.rubric_coverage_n],
    ["edu_json", tiles.edu_json_compliance, 3, tiles.edu_json_compliance_n],
    ["edu_align", tiles.edu_score_alignment, 3, tiles.edu_score_alignment_n],
    ["grounded", tiles.grounded_overlap, 3, tiles.grounded_overlap_n],
    ["latency_ms", tiles.latency_ms, 1, null],
    ["api_ms", tiles.api_time_ms, 1, null],
    ["call_count", tiles.llm_call_count, 1, null],
    ["tokens", tiles.total_tokens, 1, null],
    ["complex_u", tiles.complexity_units, 1, null],
  ];
  const metricSnapshot = miniTiles.map(([lbl, val, dig, n]) => {
    const v = val === undefined || val === null ? "-" : fmtNumber(val, dig);
    const nSpan = n !== null && n !== undefined ? `<span class="n">n=${Math.round(Number(n) || 0)}</span>` : "";
    return `<div class="metric-mini-tile"><span class="lbl">${esc(lbl)}</span><strong>${v}</strong>${nSpan}</div>`;
  }).join("");

  const progress = session.progress ? `${session.progress.completed}/${session.progress.total} (${session.progress.pct_text || ""})` : "-";
  const profileText = JSON.stringify(session.supervision_profile || {}, null, 2);
  const timeline = (session.log_timeline || []).join("\n") || "-";
  const logTail = (session.log_tail || []).join("\n") || "-";
  const errors = (session.log_errors || []).join("\n") || "none";

  el.innerHTML = `
    <div class="detail-grid">
      <article class="detail-block">
        <p class="detail-title">session</p>
        <p class="detail-mono">${esc(session.session_key)}\nstatus=${esc(session.status)} (${esc(session.status_reason || "")})\nprogress=${esc(progress)}\nstarted=${esc(session.started_at || "-")}\nended=${esc(session.ended_at || "-")}\nduration_s=${esc(session.duration_s ?? "-")}\nthinking_budget=${esc(session.thinking_budget ?? "-")}</p>
      </article>
      <article class="detail-block">
        <p class="detail-title">models</p>
        <p class="detail-mono">text=${esc(session.models?.text || "-")}\nvision=${esc(session.models?.vision || "-")}</p>
      </article>
      <article class="detail-block detail-block-wide">
        <p class="detail-title">metric snapshot</p>
        <div class="metric-mini-grid">${metricSnapshot}</div>
      </article>
      <article class="detail-block detail-block-wide">
        <p class="detail-title">supervision profile</p>
        <div class="scrollable-mono-short scrollable-mono"><pre class="detail-mono" style="margin:0">${esc(profileText)}</pre></div>
      </article>
      <article class="detail-block detail-block-wide">
        <p class="detail-title">timeline</p>
        <div class="scrollable-mono"><pre class="detail-mono" style="margin:0">${esc(timeline)}</pre></div>
      </article>
      <article class="detail-block detail-block-wide">
        <p class="detail-title">log tail · errors</p>
        <div class="scrollable-mono"><pre class="detail-mono" style="margin:0">${esc(logTail)}\n\nerrors:\n${esc(errors)}</pre></div>
      </article>
    </div>
  `;
}

// ---------- Metric Guide ----------

function renderMetricGuide() {
  const session = activeSession();
  const panel = $("metricGuidePanel");
  panel.innerHTML = METRIC_GUIDE.map((entry) => {
    const expanded = metricGuideExpanded.has(entry.key);
    const tiles = session ? session.metric_tiles || {} : {};
    const value = entry.key === "score" ? session?.score : tiles[entry.key];
    const n = tiles[`${entry.key}_n`];
    const valText = value === undefined || value === null ? "-" : fmtNumber(value, entry.digits || 3);
    const nText = n !== undefined && n !== null ? ` <span class="n-dim">(n=${Math.round(Number(n) || 0)})</span>` : "";
    return `
      <article class="metric-guide-item">
        <div class="metric-guide-head">
          <p class="metric-guide-name">${esc(entry.label)}</p>
          <button type="button" class="metric-guide-toggle" data-metric-key="${esc(entry.key)}">${expanded ? "hide" : "explain"}</button>
        </div>
        <p class="metric-guide-value">current: ${valText}${nText} · ${esc(entry.direction)}</p>
        <div class="metric-guide-body${expanded ? "" : " is-hidden"}">
          <p>${esc(entry.description)}</p>
          ${entry.code_preview ? `<pre class="code-preview">${esc(entry.code_preview)}</pre>` : ""}
        </div>
      </article>
    `;
  }).join("");

  panel.querySelectorAll(".metric-guide-toggle").forEach((btn) => {
    btn.onclick = () => {
      const k = btn.dataset.metricKey;
      if (!k) return;
      if (metricGuideExpanded.has(k)) metricGuideExpanded.delete(k);
      else metricGuideExpanded.add(k);
      renderMetricGuide();
    };
  });
}

// ---------- Master render ----------

function render() {
  renderOverview();
  if (activeTab === "architectures") renderArchitectures();
  if (activeTab === "sessions") renderSessionsTable();
  if (activeTab === "deepdive") { renderDeepDive(); renderPerExampleInspector(); }
  if (activeTab === "metrics") renderMetricGuide();
}

// ---------- Architecture diagrams ----------

function renderArchDiagrams() {
  const el = $("archDiagrams");
  if (!el) return;
  const known = new Set((rawData.sessions || []).map((s) => s.architecture).filter(Boolean));
  const diagrams = Object.entries(ARCHITECTURE_DIAGRAMS)
    .filter(([key]) => known.has(key) || known.size === 0);

  const canvasW = 960;
  const canvasH = 400;

  const renderNode = (n) => {
    const style = NODE_STYLE[n.kind] || NODE_STYLE.agent;
    const icon = NODE_ICONS[n.icon || n.kind] || NODE_ICONS.agent;
    const cornerR = n.kind === "decision" ? 0 : 8;
    const shape = n.kind === "decision"
      ? `<polygon points="${NODE_W/2},0 ${NODE_W},${NODE_H/2} ${NODE_W/2},${NODE_H} 0,${NODE_H/2}" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2" filter="url(#nodeShadow)"/>`
      : n.kind === "memory"
      ? `<path d="M0,12 Q0,2 ${NODE_W/2},2 Q${NODE_W},2 ${NODE_W},12 L${NODE_W},${NODE_H-12} Q${NODE_W},${NODE_H-2} ${NODE_W/2},${NODE_H-2} Q0,${NODE_H-2} 0,${NODE_H-12} Z" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2" filter="url(#nodeShadow)"/>
         <ellipse cx="${NODE_W/2}" cy="12" rx="${NODE_W/2}" ry="6" fill="none" stroke="${style.stroke}" stroke-width="1.5"/>`
      : `<rect x="0" y="0" width="${NODE_W}" height="${NODE_H}" rx="${cornerR}" ry="${cornerR}" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2" filter="url(#nodeShadow)"/>`;
    const titleColor = ["orchestrator", "critic"].includes(n.kind) ? "#fff" : "var(--ink)";
    const subColor = ["orchestrator", "critic"].includes(n.kind) ? "#fff" : "var(--ink-soft)";
    return `
      <g class="arch-node" transform="translate(${n.x},${n.y})" data-node-id="${esc(n.id)}">
        ${shape}
        <g transform="translate(10, 16)" style="color: ${titleColor}">
          <svg viewBox="0 0 24 24" width="22" height="22" x="0" y="0">${icon}</svg>
        </g>
        <text x="40" y="28" class="arch-node-title" fill="${titleColor}">${esc(n.title)}</text>
        <text x="40" y="48" class="arch-node-sub" fill="${subColor}">${esc(n.subtitle || "")}</text>
      </g>
    `;
  };

  const renderEdge = (e, arch) => {
    const from = arch.nodes.find((n) => n.id === e.from);
    const to = arch.nodes.find((n) => n.id === e.to);
    if (!from || !to) return "";

    const fromCx = from.x + NODE_W / 2;
    const fromCy = from.y + NODE_H / 2;
    const toCx = to.x + NODE_W / 2;
    const toCy = to.y + NODE_H / 2;

    // Determine exit/entry points on node edges
    let x1, y1, x2, y2;
    const dx = toCx - fromCx;
    const dy = toCy - fromCy;

    if (Math.abs(dx) > Math.abs(dy)) {
      x1 = dx > 0 ? from.x + NODE_W : from.x;
      y1 = fromCy;
      x2 = dx > 0 ? to.x : to.x + NODE_W;
      y2 = toCy;
    } else {
      x1 = fromCx;
      y1 = dy > 0 ? from.y + NODE_H : from.y;
      x2 = toCx;
      y2 = dy > 0 ? to.y : to.y + NODE_H;
    }

    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const cx1 = Math.abs(dx) > Math.abs(dy) ? midX : x1;
    const cy1 = Math.abs(dx) > Math.abs(dy) ? y1 : midY;
    const cx2 = Math.abs(dx) > Math.abs(dy) ? midX : x2;
    const cy2 = Math.abs(dx) > Math.abs(dy) ? y2 : midY;

    const dash = e.dashed ? 'stroke-dasharray="6 4"' : "";
    const color = e.color || "var(--ink)";
    const labelBg = e.label ? `
      <g transform="translate(${midX}, ${midY - 4})">
        <rect x="${-(e.label.length * 3.2 + 6)}" y="-10" width="${e.label.length * 6.4 + 12}" height="16" rx="3" fill="var(--card)" stroke="${color}" stroke-width="1.2"/>
        <text x="0" y="2" text-anchor="middle" class="arch-edge-label" fill="var(--ink)">${esc(e.label)}</text>
      </g>
    ` : "";

    return `
      <g class="arch-edge-group">
        <path d="M${x1},${y1} C${cx1},${cy1} ${cx2},${cy2} ${x2},${y2}" fill="none" stroke="${color}" stroke-width="2" ${dash} marker-end="url(#arrowHead-${color.replace(/[^a-z]/gi,'')})" class="${known.size > 0 ? 'edge-animated' : ''}"/>
        ${labelBg}
      </g>
    `;
  };

  const renderStageBands = (activeBands) => {
    return STAGE_BANDS.map((band) => {
      const isActive = !activeBands || activeBands.includes(band.id);
      if (!isActive) return "";
      return `
        <g class="stage-band">
          <rect x="${band.x}" y="0" width="${band.w}" height="${canvasH}" fill="${band.fill}" opacity="0.4"/>
          <rect x="${band.x}" y="0" width="${band.w}" height="24" fill="${band.fill}"/>
          <text x="${band.x + band.w / 2}" y="16" text-anchor="middle" class="stage-band-label">${esc(band.label)}</text>
        </g>
      `;
    }).join("");
  };

  const renderParallelBracket = (parallelIds, arch) => {
    if (!parallelIds || parallelIds.length < 2) return "";
    const parallelNodes = parallelIds.map((id) => arch.nodes.find((n) => n.id === id)).filter(Boolean);
    if (parallelNodes.length < 2) return "";
    const minY = Math.min(...parallelNodes.map((n) => n.y));
    const maxY = Math.max(...parallelNodes.map((n) => n.y + NODE_H));
    const x = Math.min(...parallelNodes.map((n) => n.x)) - 14;
    return `
      <g class="parallel-bracket">
        <path d="M${x},${minY} L${x-6},${minY} L${x-6},${maxY} L${x},${maxY}" fill="none" stroke="var(--accent-pink)" stroke-width="2.5"/>
        <text x="${x - 10}" y="${(minY + maxY) / 2 + 3}" text-anchor="end" class="parallel-label">‖ parallel</text>
      </g>
    `;
  };

  const defs = `
    <defs>
      <filter id="nodeShadow" x="-10%" y="-10%" width="120%" height="120%">
        <feDropShadow dx="2" dy="2" stdDeviation="0" flood-color="var(--ink)" flood-opacity="1"/>
      </filter>
      <marker id="arrowHead-varink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--ink)"/></marker>
      <marker id="arrowHead-varaccentcyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--accent-cyan)"/></marker>
      <marker id="arrowHead-varaccentviolet" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--accent-violet)"/></marker>
      <pattern id="gridPattern" width="20" height="20" patternUnits="userSpaceOnUse">
        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="var(--ink)" stroke-width="0.3" opacity="0.12"/>
      </pattern>
    </defs>
  `;

  const legend = `
    <div class="arch-legend">
      <span class="legend-item"><span class="legend-swatch" style="background:var(--accent-yellow)"></span>I/O</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--accent-cyan)"></span>Agent</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--accent-pink)"></span>Orchestrator</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--accent-violet)"></span>Retrieval</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--card-alt);border:1.5px solid var(--ink)"></span>Memory</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#ffd93d;transform:rotate(45deg)"></span>Decision</span>
      <span class="legend-item legend-edge"><svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke="var(--ink)" stroke-width="2"/></svg>Direct</span>
      <span class="legend-item legend-edge"><svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke="var(--ink)" stroke-width="2" stroke-dasharray="4 3"/></svg>Optional</span>
    </div>
  `;

  el.innerHTML = `
    ${legend}
    ${diagrams.map(([key, arch]) => `
      <div class="arch-diagram-card arch-diagram-wide">
        <div class="arch-diagram-header">
          <h3>${esc(arch.name)}</h3>
          <p class="arch-desc">${esc(arch.description || "")}</p>
        </div>
        <svg viewBox="0 0 ${canvasW} ${canvasH}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
          ${defs}
          <rect x="0" y="0" width="${canvasW}" height="${canvasH}" fill="var(--card)"/>
          <rect x="0" y="0" width="${canvasW}" height="${canvasH}" fill="url(#gridPattern)"/>
          ${renderStageBands(arch.activeBands)}
          ${renderParallelBracket(arch.parallel, arch)}
          <g class="arch-edges">${arch.edges.map((e) => renderEdge(e, arch)).join("")}</g>
          <g class="arch-nodes">${arch.nodes.map(renderNode).join("")}</g>
        </svg>
      </div>
    `).join("")}
  `;
}

// ---------- Per-example inspector ----------

function renderPerExampleInspector() {
  const panel = $("perExamplePanel");
  const tbody = $("perExampleBody");
  if (!panel || !tbody) return;
  const session = activeSession();
  const examples = session?.example_records || [];
  if (!session || examples.length === 0) {
    panel.style.display = "none";
    return;
  }
  panel.style.display = "";
  let rows = examples.slice();
  if (exampleStatusFilter !== "all") {
    rows = rows.filter((r) => (r.success ? "success" : "failed") === exampleStatusFilter);
  }
  rows.sort((a, b) => {
    const av = a[exampleSortKey] || "";
    const bv = b[exampleSortKey] || "";
    if (typeof av === "string") return exampleSortDir === "desc" ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
    return exampleSortDir === "desc" ? Number(bv) - Number(av) : Number(av) - Number(bv);
  });
  tbody.innerHTML = rows.slice(0, 50).map((r) => {
    const status = r.success ? "success" : "failed";
    return `<tr>
      <td><code>${esc(String(r.example_id || "").slice(0, 16))}</code></td>
      <td><span class="status-pill status-${status}">${esc(status)}</span></td>
      <td class="num">${fmtNumber(r.metrics?.token_f1)}</td>
      <td class="num">${fmtNumber(r.metrics?.exact_match)}</td>
      <td class="num">${fmtNumber(r.metrics?.rubric_coverage)}</td>
      <td>${esc(String(r.question || "").slice(0, 60))}${r.question && r.question.length > 60 ? "…" : ""}</td>
      <td>${esc(String(r.answer || "").slice(0, 60))}${r.answer && r.answer.length > 60 ? "…" : ""}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="7" style="text-align:center;padding:16px;">No examples match filter.</td></tr>`;
}

// ---------- Data layer ----------

function applyData(data) {
  const prevSessions = (rawData.sessions || []).map((s) => ({ key: s.session_key, status: s.status }));
  rawData = data || rawData;
  checkSessionStateChanges(rawData.sessions || []);
  render();
  renderPerExampleInspector();
}

async function loadData() {
  if (loading) return;
  loading = true;
  try {
    const r = await fetch(`${DATA_URL}?t=${Date.now()}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    applyData(await r.json());
  } catch (err) {
    $("overviewGrid").innerHTML = `<article class="stat-tile"><p class="stat-label">Error</p><p class="stat-value">!</p><p class="stat-foot">${esc(String(err))}</p></article>`;
    renderLiveHealth();
  } finally {
    loading = false;
  }
}

function wsURL() {
  const u = new URL("ws", window.location.href);
  u.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return u.toString();
}

function connectSocket() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
  try {
    socket = new WebSocket(wsURL());
  } catch (_) {
    socketConnected = false;
    renderLiveHealth();
    scheduleReconnect();
    return;
  }
  socket.addEventListener("open", () => {
    socketConnected = true;
    renderLiveHealth();
    stopAutoRefresh();
  });
  socket.addEventListener("message", (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      if (payload?.type === "session_summary" && payload.data) applyData(payload.data);
    } catch (_) {}
  });
  socket.addEventListener("close", () => {
    socketConnected = false;
    renderLiveHealth();
    startAutoRefresh();
    scheduleReconnect();
  });
  socket.addEventListener("error", () => { try { socket.close(); } catch (_) {} });
}

function scheduleReconnect() {
  if (socketReconnectTimer) return;
  socketReconnectTimer = setTimeout(() => { socketReconnectTimer = null; connectSocket(); }, WS_RECONNECT_MS);
}

function startAutoRefresh() {
  if (refreshTimer || socketConnected) return;
  refreshTimer = setInterval(() => { if (!socketConnected && document.visibilityState === "visible") loadData(); }, AUTO_REFRESH_MS);
}
function stopAutoRefresh() { if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; } }

// ---------- Boot ----------

function applyTheme(next) {
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("ma_theme", next); } catch (_) {}
}

document.addEventListener("DOMContentLoaded", () => {
  // theme
  let theme = "light";
  try { theme = localStorage.getItem("ma_theme") || "light"; } catch (_) {}
  applyTheme(theme);
  $("themeToggle").addEventListener("click", () => {
    const now = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(now);
  });

  // nav
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });
  readHashState();
  setActiveTab(activeTab);

  // search
  $("searchBox").addEventListener("input", (ev) => {
    searchQuery = ev.target.value || "";
    if (activeTab === "sessions") renderSessionsTable();
  });

  // example status chips (deep dive)
  document.querySelectorAll("#exampleStatusChips .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      exampleStatusFilter = chip.dataset.exstatus || "all";
      renderPerExampleInspector();
    });
  });

  $("refreshButton").addEventListener("click", loadData);

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      connectSocket();
      if (!socketConnected) loadData();
    }
  });

  loadData();
  startAutoRefresh();
  connectSocket();
  setInterval(renderLiveHealth, 2000);
});
