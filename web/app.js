const DATA_URL = "data/session_summary.json";
const AUTO_REFRESH_MS = 60000;
const WS_RECONNECT_MS = 2000;

const overviewGrid = document.getElementById("overviewGrid");
const rankGrid = document.getElementById("rankGrid");
const datasetCards = document.getElementById("datasetCards");
const sessionsBody = document.getElementById("sessionsBody");
const sessionDetail = document.getElementById("sessionDetail");
const refreshButton = document.getElementById("refreshButton");
const snapshotTime = document.getElementById("snapshotTime");
const liveIndicator = document.getElementById("liveIndicator");
const overallProgressLabel = document.getElementById("overallProgressLabel");
const overallProgressFill = document.getElementById("overallProgressFill");
const metricGuidePanel = document.getElementById("metricGuidePanel");

const METRIC_GUIDE = [
  {
    key: "score",
    label: "composite",
    digits: 3,
    direction: "higher is better",
    description: "Aggregate score used for run ranking; combines available quality metrics for the current dataset.",
  },
  {
    key: "token_f1",
    label: "token_f1",
    digits: 3,
    direction: "higher is better",
    description: "Token-level overlap F1 between predicted answer text and reference text.",
  },
  {
    key: "tutoreval_keypoint_recall",
    label: "keypt_recall",
    digits: 3,
    direction: "higher is better",
    description: "TutorEval key-point recall; measures how many required tutoring points are covered.",
  },
  {
    key: "exact_match",
    label: "exact_match",
    digits: 3,
    direction: "higher is better",
    description: "Strict exact-match ratio against normalized gold answer text when available.",
  },
  {
    key: "rubric_coverage",
    label: "rubric_coverage",
    digits: 3,
    direction: "higher is better",
    description: "Fraction of rubric items addressed by the model response.",
  },
  {
    key: "edu_json_compliance",
    label: "edu_json",
    digits: 3,
    direction: "higher is better",
    description: "EduBench format-compliance score for schema/JSON validity and required fields.",
  },
  {
    key: "edu_score_alignment",
    label: "edu_align",
    digits: 3,
    direction: "higher is better",
    description: "Alignment between predicted scoring behavior and EduBench consensus reference scoring.",
  },
  {
    key: "grounded_overlap",
    label: "grounded_overlap",
    digits: 3,
    direction: "higher is better",
    description: "Answer overlap with retrieved evidence; higher values suggest better grounding.",
  },
  {
    key: "latency_ms",
    label: "latency_ms",
    digits: 1,
    direction: "lower is better",
    description: "End-to-end pipeline latency per example in milliseconds.",
  },
  {
    key: "api_time_ms",
    label: "api_time_ms",
    digits: 1,
    direction: "lower is better",
    description: "Estimated time spent in model API calls (subset of end-to-end latency).",
  },
  {
    key: "llm_call_count",
    label: "llm_call_count",
    digits: 1,
    direction: "context-dependent",
    description: "Average number of model calls; lower is cheaper, higher may indicate deeper reasoning.",
  },
  {
    key: "total_tokens",
    label: "total_tokens",
    digits: 1,
    direction: "lower is better",
    description: "Total tokens consumed across prompt and completion.",
  },
  {
    key: "complexity_units",
    label: "complexity_units",
    digits: 1,
    direction: "lower is better",
    description: "Composite workload indicator combining token cost, retrieval work, and orchestration overhead.",
  },
  {
    key: "complexity_per_second",
    label: "complexity_per_second",
    digits: 1,
    direction: "context-dependent",
    description: "Throughput-like ratio of complexity processed per second.",
  },
];

let activeSessionKey = null;
let refreshTimer = null;
let loading = false;
let latestOverview = {};
let socket = null;
let socketReconnectTimer = null;
let socketConnected = false;
const metricGuideExpanded = new Set();

function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function statusClass(status) {
  if (status === "finished") return "status-finished";
  if (status === "failed") return "status-failed";
  return "status-running";
}

function clampPct(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return 0;
  return Math.max(0, Math.min(100, numeric));
}

function renderOverview(overview) {
  overviewGrid.innerHTML = "";
  const tiles = [
    ["Total", overview.total_sessions],
    ["Finished", overview.finished_sessions],
    ["Running", overview.running_sessions],
    ["Failed", overview.failed_sessions],
    ["Overall", overview.overall_progress_pct === undefined ? "-" : `${fmtNumber(overview.overall_progress_pct, 1)}%`],
    [
      "Examples",
      overview.overall_total_examples
        ? `${overview.overall_completed_examples || 0}/${overview.overall_total_examples}`
        : "-",
    ],
  ];

  tiles.forEach(([label, value]) => {
    const tile = document.createElement("article");
    tile.className = "stat-tile";
    tile.innerHTML = `
      <p class="stat-label">${label}</p>
      <p class="stat-value">${value ?? "-"}</p>
    `;
    overviewGrid.appendChild(tile);
  });
}

function renderOverallProgress(overview) {
  const completed = Number(overview?.overall_completed_examples || 0);
  const total = Number(overview?.overall_total_examples || 0);
  const pct = clampPct(overview?.overall_progress_pct || 0);
  overallProgressLabel.textContent = total > 0
    ? `overall progress: ${completed}/${total} (${pct.toFixed(1)}%)`
    : "overall progress: no active totals";
  overallProgressFill.style.width = `${pct}%`;
  if (overallProgressFill.parentElement) {
    overallProgressFill.parentElement.setAttribute("aria-valuenow", pct.toFixed(1));
  }
}

function renderLeaderboard(rows) {
  rankGrid.innerHTML = "";
  if (!rows || rows.length === 0) {
    rankGrid.innerHTML = `<article class="rank-tile"><p class="rank-head">No finished runs yet</p></article>`;
    return;
  }

  rows.forEach((row, index) => {
    const node = document.createElement("article");
    node.className = "rank-tile";
    node.innerHTML = `
      <p class="rank-head">#${index + 1} ${esc(row.architecture)}</p>
      <p class="rank-meta">avg ${fmtNumber(row.avg_score)} / best ${fmtNumber(row.best_score)}</p>
      <p class="dataset-meta">runs: ${esc(row.runs)}</p>
    `;
    rankGrid.appendChild(node);
  });
}

function renderDatasetCards(cards) {
  datasetCards.innerHTML = "";
  cards.forEach((card) => {
    const node = document.createElement("article");
    node.className = "dataset-tile";
    const best = card.best_architecture || "pending";
    const score = card.best_score === null || card.best_score === undefined ? "-" : fmtNumber(card.best_score);
    const avg = card.average_score === null || card.average_score === undefined ? "-" : fmtNumber(card.average_score);
    const avgProgress = card.average_progress_pct === null || card.average_progress_pct === undefined
      ? "-"
      : `${fmtNumber(card.average_progress_pct, 1)}%`;
    const breakdown = card.status_breakdown || {};
    node.innerHTML = `
      <h3 class="dataset-name">${esc(card.dataset)}</h3>
      <p class="dataset-best">best: ${esc(best)}</p>
      <p class="dataset-meta">score: ${score}</p>
      <p class="dataset-meta">avg: ${avg}</p>
      <p class="dataset-meta">avg progress: ${avgProgress}</p>
      <p class="dataset-meta">finished: ${card.finished_sessions}/${card.total_sessions}</p>
      <p class="dataset-meta">run state: f=${breakdown.finished ?? 0}, r=${breakdown.running ?? 0}, x=${breakdown.failed ?? 0}</p>
    `;
    datasetCards.appendChild(node);
  });
}

function metricLine(label, value, digits = 3) {
  return `${label}: ${value === null || value === undefined ? "-" : fmtNumber(value, digits)}`;
}

function progressLabel(progress) {
  if (!progress || progress.total === undefined || progress.total === 0) {
    return "-";
  }
  return `${progress.completed}/${progress.total} (${progress.pct_text || "-"})`;
}

function progressPct(progress) {
  if (!progress) {
    return 0;
  }
  if (progress.pct !== undefined && progress.pct !== null) {
    return clampPct(progress.pct);
  }
  if (progress.pct_text) {
    return clampPct(String(progress.pct_text).replace("%", ""));
  }
  return 0;
}

function snapshotLabel(generatedAt) {
  return `snapshot: ${generatedAt || "unknown"} • websocket 1s stream`;
}

function setLiveClass(name) {
  if (!liveIndicator) {
    return;
  }
  liveIndicator.classList.remove("live-fresh", "live-warm", "live-stale", "live-unknown");
  liveIndicator.classList.add(name);
}

function renderLiveHealth(overview) {
  if (!liveIndicator) {
    return;
  }

  const epoch = Number(overview?.generated_epoch || 0);
  const socketText = socketConnected ? "ws:on" : "ws:off";
  if (!Number.isFinite(epoch) || epoch <= 0) {
    liveIndicator.textContent = `${socketText} • live: unknown`;
    setLiveClass("live-unknown");
    return;
  }

  const nowEpoch = Math.floor(Date.now() / 1000);
  const ageSec = Math.max(0, nowEpoch - epoch);
  liveIndicator.textContent = `${socketText} • data ${ageSec}s old`;

  if (ageSec <= 30 && socketConnected) {
    setLiveClass("live-fresh");
  } else if (ageSec <= 120) {
    setLiveClass("live-warm");
  } else {
    setLiveClass("live-stale");
  }
}

function renderSessionDetail(session) {
  if (!session) {
    sessionDetail.innerHTML = `<p class="detail-empty">Select a session row to inspect timeline, metrics, and log trace.</p>`;
    return;
  }

  const profileText = JSON.stringify(session.supervision_profile || {}, null, 2);
  const timeline = (session.log_timeline || []).join("\n") || "-";
  const logTail = (session.log_tail || []).join("\n") || "-";
  const errors = (session.log_errors || []).join("\n") || "none";
  const progress = progressLabel(session.progress);
  const modelText = `text=${session.models?.text || "-"}\nvision=${session.models?.vision || "-"}`;
  const metrics = session.metric_tiles || {};

  const metricSnapshot = [
    ["composite", session.score, 3],
    ["token_f1", metrics.token_f1, 3],
    ["keypt_recall", metrics.tutoreval_keypoint_recall, 3],
    ["exact_match", metrics.exact_match, 3],
    ["rubric_cov", metrics.rubric_coverage, 3],
    ["edu_json", metrics.edu_json_compliance, 3],
    ["edu_align", metrics.edu_score_alignment, 3],
    ["grounded_lap", metrics.grounded_overlap, 3],
    ["latency_ms", metrics.latency_ms, 1],
    ["api_time_ms", metrics.api_time_ms, 1],
    ["call_count", metrics.llm_call_count, 1],
    ["tokens", metrics.total_tokens, 1],
    ["compl_units", metrics.complexity_units, 1],
    ["compl_per_sec", metrics.complexity_per_second, 1],
  ].map(([lbl, val, dig]) => `<div class="metric-mini-tile"><span class="lbl">${esc(lbl)}</span><strong>${val === null || val === undefined ? "-" : fmtNumber(val, dig)}</strong></div>`).join("");

  sessionDetail.innerHTML = `
    <div class="detail-grid">
      <article class="detail-block">
        <p class="detail-title">session</p>
        <p class="detail-mono">${esc(session.session_key)}\nstatus=${esc(session.status)}\nprogress=${esc(progress)}\nstarted=${esc(session.started_at || "-")}\nended=${esc(session.ended_at || "-")}\nduration_s=${esc(session.duration_s ?? "-")}\nthinking_budget=${esc(session.thinking_budget ?? "-")}</p>
      </article>
      <article class="detail-block">
        <p class="detail-title">models</p>
        <p class="detail-mono">${esc(modelText)}</p>
      </article>
      <article class="detail-block detail-block-wide">
        <p class="detail-title">metric snapshot</p>
        <div class="metric-mini-grid">
           ${metricSnapshot}
        </div>
      </article>
      <article class="detail-block detail-block-wide">
        <p class="detail-title">supervision profile</p>
        <p class="detail-mono scrollable-mono-short">${esc(profileText)}</p>
      </article>
      <article class="detail-block detail-block-wide">
        <p class="detail-title">timeline</p>
        <div class="scrollable-mono detail-mono">${esc(timeline)}</div>
      </article>
      <article class="detail-block detail-block-wide">
        <p class="detail-title">log tail / errors</p>
        <div class="scrollable-mono detail-mono">${esc(logTail)}\n\nerrors:\n${esc(errors)}</div>
      </article>
    </div>
  `;
}

function metricValueForSession(session, key) {
  if (!session) {
    return null;
  }
  const metrics = session.metric_tiles || {};
  if (key === "score") {
    return session.score;
  }
  return metrics[key];
}

function renderMetricGuide(session) {
  if (!metricGuidePanel) {
    return;
  }

  metricGuidePanel.innerHTML = "";
  METRIC_GUIDE.forEach((entry) => {
    const expanded = metricGuideExpanded.has(entry.key);
    const value = metricValueForSession(session, entry.key);
    const valueText = value === null || value === undefined ? "-" : fmtNumber(value, entry.digits || 3);

    const node = document.createElement("article");
    node.className = "metric-guide-item";
    node.innerHTML = `
      <div class="metric-guide-head">
        <p class="metric-guide-name">${esc(entry.label)}</p>
        <button type="button" class="metric-guide-toggle" data-metric-key="${esc(entry.key)}">${expanded ? "Hide" : "Show"} explanation</button>
      </div>
      <p class="metric-guide-value">current: ${esc(valueText)} • ${esc(entry.direction)}</p>
      <div class="metric-guide-body${expanded ? "" : " is-hidden"}">
        <p>${esc(entry.description)}</p>
      </div>
    `;
    metricGuidePanel.appendChild(node);
  });

  metricGuidePanel.querySelectorAll(".metric-guide-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.getAttribute("data-metric-key");
      if (!key) {
        return;
      }
      if (metricGuideExpanded.has(key)) {
        metricGuideExpanded.delete(key);
      } else {
        metricGuideExpanded.add(key);
      }
      renderMetricGuide(session);
    });
  });
}

function renderSessions(sessions) {
  sessionsBody.innerHTML = "";
  if (!activeSessionKey && sessions.length > 0) {
    activeSessionKey = sessions[0].session_key;
  }

  sessions.forEach((session) => {
    const tr = document.createElement("tr");
    if (session.session_key === activeSessionKey) {
      tr.classList.add("row-active");
    }
    const tokenF1 = session.summary?.token_f1 ?? null;
    const keypointRecall = session.summary?.tutoreval_keypoint_recall ?? null;
    const rubric = session.summary?.rubric_coverage ?? null;
    const eduAlign = session.summary?.edu_score_alignment ?? null;
    const latencyMs = session.summary?.latency_ms ?? null;
    const apiTimeMs = session.summary?.api_time_ms ?? null;
    const complexityUnits = session.summary?.complexity_units ?? null;
    const totalTokens = session.summary?.total_tokens ?? null;
    const progress = progressLabel(session.progress);
    const progressValue = progressPct(session.progress);
    tr.innerHTML = `
      <td data-label="Dataset">${esc(session.dataset)}</td>
      <td data-label="Architecture">${esc(session.architecture)}</td>
      <td data-label="Status"><span class="status-pill ${statusClass(session.status)}">${session.status}</span></td>
      <td data-label="Processed">${session.records ?? "-"}</td>
      <td data-label="Progress">
        <div class="progress-cell">
          <div class="progress-track"><span class="progress-fill" style="width:${progressValue}%"></span></div>
          <span class="progress-text">${esc(progress)}</span>
        </div>
      </td>
      <td data-label="Composite"><span class="score-chip">${fmtNumber(session.score)}</span></td>
      <td data-label="Token F1">${fmtNumber(tokenF1)}</td>
      <td data-label="KeyPt R">${fmtNumber(keypointRecall)}</td>
      <td data-label="Rubric">${fmtNumber(rubric)}</td>
      <td data-label="Edu Align">${fmtNumber(eduAlign)}</td>
      <td data-label="Latency (ms)">${fmtNumber(latencyMs, 1)}</td>
      <td data-label="API (ms)">${fmtNumber(apiTimeMs, 1)}</td>
      <td data-label="Complexity">${fmtNumber(complexityUnits, 1)}</td>
      <td data-label="Tokens">${fmtNumber(totalTokens, 1)}</td>
    `;
    tr.addEventListener("click", () => {
      activeSessionKey = session.session_key;
      renderSessions(sessions);
    });
    sessionsBody.appendChild(tr);
  });

  const selected = sessions.find((session) => session.session_key === activeSessionKey) || sessions[0] || null;
  renderSessionDetail(selected);
  renderMetricGuide(selected);
}

function applyData(data) {
  const overview = data?.overview || {};
  latestOverview = overview;
  renderOverview(overview);
  renderOverallProgress(overview);
  renderLeaderboard(overview?.architecture_leaderboard || []);
  renderDatasetCards(data?.dataset_cards || []);
  renderSessions(data?.sessions || []);
  snapshotTime.textContent = snapshotLabel(overview?.generated_at);
  renderLiveHealth(overview);
}

function websocketUrl() {
  const url = new URL("ws", window.location.href);
  url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function stopAutoRefresh() {
  if (!refreshTimer) {
    return;
  }
  clearInterval(refreshTimer);
  refreshTimer = null;
}

function scheduleSocketReconnect() {
  if (socketReconnectTimer) {
    return;
  }
  socketReconnectTimer = setTimeout(() => {
    socketReconnectTimer = null;
    connectSocket();
  }, WS_RECONNECT_MS);
}

function connectSocket() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  try {
    socket = new WebSocket(websocketUrl());
  } catch (_error) {
    socketConnected = false;
    renderLiveHealth(latestOverview);
    startAutoRefresh();
    scheduleSocketReconnect();
    return;
  }

  socket.addEventListener("open", () => {
    socketConnected = true;
    renderLiveHealth(latestOverview);
    stopAutoRefresh();
  });

  socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload?.type === "session_summary" && payload.data) {
        applyData(payload.data);
      }
    } catch (_error) {
      // Ignore malformed websocket payloads and keep the stream alive.
    }
  });

  socket.addEventListener("close", () => {
    socketConnected = false;
    renderLiveHealth(latestOverview);
    startAutoRefresh();
    scheduleSocketReconnect();
  });

  socket.addEventListener("error", () => {
    if (socket && socket.readyState < WebSocket.CLOSING) {
      socket.close();
    }
  });
}

async function loadData() {
  if (loading) {
    return;
  }
  loading = true;
  refreshButton.disabled = true;
  try {
    const response = await fetch(`${DATA_URL}?t=${Date.now()}`, {
      cache: "no-store",
      headers: {
        "Cache-Control": "no-cache",
        Pragma: "no-cache",
      },
    });
    if (!response.ok) {
      throw new Error(`Failed to load data: ${response.status}`);
    }
    const data = await response.json();
    applyData(data);
  } catch (error) {
    overviewGrid.innerHTML = `<article class="stat-tile"><p class="stat-label">Error</p><p class="stat-value">!</p></article>`;
    rankGrid.innerHTML = "";
    datasetCards.innerHTML = `<article class="dataset-tile"><p>${String(error)}</p></article>`;
    sessionsBody.innerHTML = "";
    sessionDetail.innerHTML = `<p class="detail-empty">${esc(String(error))}</p>`;
    snapshotTime.textContent = "snapshot: failed";
    if (liveIndicator) {
      liveIndicator.textContent = "ws:off • fetch failed";
      setLiveClass("live-stale");
    }
    overallProgressLabel.textContent = "overall progress: unavailable";
    overallProgressFill.style.width = "0%";
    renderMetricGuide(null);
  } finally {
    refreshButton.disabled = false;
    loading = false;
  }
}

function startAutoRefresh() {
  if (refreshTimer || socketConnected) {
    return;
  }
  refreshTimer = setInterval(() => {
    if (document.visibilityState === "visible" && !socketConnected) {
      loadData();
    }
  }, AUTO_REFRESH_MS);
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    connectSocket();
    if (!socketConnected) {
      loadData();
    }
  }
});

window.addEventListener("focus", () => {
  connectSocket();
  if (!socketConnected) {
    loadData();
  }
});

refreshButton.addEventListener("click", loadData);
loadData();
startAutoRefresh();
connectSocket();
