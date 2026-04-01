const DATA_URL = "data/session_summary.json";

const overviewGrid = document.getElementById("overviewGrid");
const datasetCards = document.getElementById("datasetCards");
const sessionsBody = document.getElementById("sessionsBody");
const refreshButton = document.getElementById("refreshButton");

function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
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

function renderDatasetCards(cards) {
  datasetCards.innerHTML = "";
  cards.forEach((card) => {
    const node = document.createElement("article");
    node.className = "dataset-tile";
    const best = card.best_architecture || "pending";
    const score = card.best_score === null || card.best_score === undefined ? "-" : fmtNumber(card.best_score);
    node.innerHTML = `
      <h3 class="dataset-name">${card.dataset}</h3>
      <p class="dataset-best">best: ${best}</p>
      <p class="dataset-meta">score: ${score}</p>
      <p class="dataset-meta">finished: ${card.finished_sessions}/${card.total_sessions}</p>
    `;
    datasetCards.appendChild(node);
  });
}

function renderSessions(sessions) {
  sessionsBody.innerHTML = "";
  sessions.forEach((session) => {
    const tr = document.createElement("tr");
    const tokenF1 = session.summary?.token_f1 ?? null;
    const latencyMs = session.summary?.latency_ms ?? null;
    tr.innerHTML = `
      <td>${session.dataset}</td>
      <td>${session.architecture}</td>
      <td><span class="status-pill ${statusClass(session.status)}">${session.status}</span></td>
      <td>${session.records ?? "-"}</td>
      <td>${fmtNumber(tokenF1)}</td>
      <td>${fmtNumber(latencyMs, 1)}</td>
    `;
    sessionsBody.appendChild(tr);
  });
}

async function loadData() {
  try {
    const response = await fetch(`${DATA_URL}?t=${Date.now()}`);
    if (!response.ok) {
      throw new Error(`Failed to load data: ${response.status}`);
    }
    const data = await response.json();
    renderOverview(data.overview || {});
    renderDatasetCards(data.dataset_cards || []);
    renderSessions(data.sessions || []);
  } catch (error) {
    overviewGrid.innerHTML = `<article class="stat-tile"><p class="stat-label">Error</p><p class="stat-value">!</p></article>`;
    datasetCards.innerHTML = `<article class="dataset-tile"><p>${String(error)}</p></article>`;
    sessionsBody.innerHTML = "";
  }
}

refreshButton.addEventListener("click", loadData);
loadData();
