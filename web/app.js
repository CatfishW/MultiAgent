const DATA_URL = "data/session_summary.json";

const overviewGrid = document.getElementById("overviewGrid");
const rankGrid = document.getElementById("rankGrid");
const datasetCards = document.getElementById("datasetCards");
const sessionsBody = document.getElementById("sessionsBody");
const sessionDetail = document.getElementById("sessionDetail");
const refreshButton = document.getElementById("refreshButton");
const snapshotTime = document.getElementById("snapshotTime");

let activeSessionKey = null;

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

function renderOverview(overview) {
  overviewGrid.innerHTML = "";
  const tiles = [
    ["Total", overview.total_sessions],
    ["Finished", overview.finished_sessions],
    ["Running", overview.running_sessions],
    ["Failed", overview.failed_sessions],
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
    const breakdown = card.status_breakdown || {};
    node.innerHTML = `
      <h3 class="dataset-name">${esc(card.dataset)}</h3>
      <p class="dataset-best">best: ${esc(best)}</p>
      <p class="dataset-meta">score: ${score}</p>
      <p class="dataset-meta">avg: ${avg}</p>
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
    metricLine("composite", session.score),
    metricLine("token_f1", metrics.token_f1),
    metricLine("exact_match", metrics.exact_match),
    metricLine("rubric_coverage", metrics.rubric_coverage),
    metricLine("edu_json", metrics.edu_json_compliance),
    metricLine("edu_align", metrics.edu_score_alignment),
    metricLine("grounded_overlap", metrics.grounded_overlap),
    metricLine("latency_ms", metrics.latency_ms, 1),
  ].join("\n");

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
      <article class="detail-block">
        <p class="detail-title">metric snapshot</p>
        <p class="detail-mono">${esc(metricSnapshot)}</p>
      </article>
      <article class="detail-block">
        <p class="detail-title">supervision profile</p>
        <p class="detail-mono">${esc(profileText)}</p>
      </article>
      <article class="detail-block">
        <p class="detail-title">timeline</p>
        <p class="detail-mono">${esc(timeline)}</p>
      </article>
      <article class="detail-block">
        <p class="detail-title">log tail / errors</p>
        <p class="detail-mono">${esc(logTail)}\n\nerrors:\n${esc(errors)}</p>
      </article>
    </div>
  `;
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
    const rubric = session.summary?.rubric_coverage ?? null;
    const eduAlign = session.summary?.edu_score_alignment ?? null;
    const latencyMs = session.summary?.latency_ms ?? null;
    const progress = progressLabel(session.progress);
    tr.innerHTML = `
      <td>${esc(session.dataset)}</td>
      <td>${esc(session.architecture)}</td>
      <td><span class="status-pill ${statusClass(session.status)}">${session.status}</span></td>
      <td>${session.records ?? "-"}</td>
      <td>${esc(progress)}</td>
      <td><span class="score-chip">${fmtNumber(session.score)}</span></td>
      <td>${fmtNumber(tokenF1)}</td>
      <td>${fmtNumber(rubric)}</td>
      <td>${fmtNumber(eduAlign)}</td>
      <td>${fmtNumber(latencyMs, 1)}</td>
    `;
    tr.addEventListener("click", () => {
      activeSessionKey = session.session_key;
      renderSessions(sessions);
    });
    sessionsBody.appendChild(tr);
  });

  const selected = sessions.find((session) => session.session_key === activeSessionKey) || sessions[0] || null;
  renderSessionDetail(selected);
}

async function loadData() {
  try {
    const response = await fetch(`${DATA_URL}?t=${Date.now()}`);
    if (!response.ok) {
      throw new Error(`Failed to load data: ${response.status}`);
    }
    const data = await response.json();
    renderOverview(data.overview || {});
    renderLeaderboard(data.overview?.architecture_leaderboard || []);
    renderDatasetCards(data.dataset_cards || []);
    renderSessions(data.sessions || []);
    snapshotTime.textContent = `snapshot: ${data.overview?.generated_at || "unknown"}`;
  } catch (error) {
    overviewGrid.innerHTML = `<article class="stat-tile"><p class="stat-label">Error</p><p class="stat-value">!</p></article>`;
    rankGrid.innerHTML = "";
    datasetCards.innerHTML = `<article class="dataset-tile"><p>${String(error)}</p></article>`;
    sessionsBody.innerHTML = "";
    sessionDetail.innerHTML = `<p class="detail-empty">${esc(String(error))}</p>`;
    snapshotTime.textContent = "snapshot: failed";
  }
}

refreshButton.addEventListener("click", loadData);
loadData();
