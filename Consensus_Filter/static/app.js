const form = document.getElementById("query-form");
const statusNode = document.getElementById("status");
const submitButton = document.getElementById("submit-button");
const winnerCard = document.getElementById("winner-card");
const rankedCandidates = document.getElementById("ranked-candidates");
const rawJson = document.getElementById("raw-json");
const PROVIDER_COLORS = {
  gemini: "#0f766e",
  ollama: "#b45309"
};

function initializeDefaults() {
  document.getElementById("top_k").value = window.APP_CONFIG.defaultTopK;
  document.getElementById("gemini_model").value = window.APP_CONFIG.defaultGeminiModel;
  document.getElementById("ollama_model").value = window.APP_CONFIG.defaultOllamaModel;
  document.getElementById("ollama_url").value = window.APP_CONFIG.defaultOllamaUrl;
  document.getElementById("query").value = "Find three top places for lunch and the budget is $15";
}

function setStatus(message, mode = "idle") {
  statusNode.textContent = message;
  statusNode.classList.remove("is-error", "is-success");
  if (mode === "error") {
    statusNode.classList.add("is-error");
  }
  if (mode === "success") {
    statusNode.classList.add("is-success");
  }
}

function formatValue(value, prefix = "", fallback = "Unknown") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return fallback;
  }
  return `${prefix}${value}`;
}

function renderWinner(result) {
  const best = result.angel_filtered;
  winnerCard.className = "winner-card";
  winnerCard.innerHTML = `
    <div class="provider-chip">Winner from ${best.source_model}</div>
    <h3>${best.name}</h3>
    <div class="winner-grid">
      <div class="winner-metric"><span>Price</span><strong>${formatValue(best.price, "$")}</strong></div>
      <div class="winner-metric"><span>Distance</span><strong>${formatValue(best.distance_miles, "", "Unknown")} mi</strong></div>
      <div class="winner-metric"><span>Rating</span><strong>${formatValue(best.rating)}</strong></div>
    </div>
    <div class="winner-metrics">
      <div class="winner-metric"><span>Origin Distance</span><strong>${best.origin_distance}</strong></div>
      <div class="winner-metric"><span>Score</span><strong>${best.score}</strong></div>
      <div class="winner-metric"><span>Query Distance</span><strong>${best.query_distance}</strong></div>
      <div class="winner-metric"><span>Consensus Size</span><strong>${best.consensus_size}</strong></div>
    </div>
    <p class="winner-note">${best.notes || "No model notes provided."}</p>
  `;
}

function renderRankedCandidates(result) {
  const winnerName = result.angel_filtered.name;
  rankedCandidates.innerHTML = result.ranked_candidates.map((candidate) => `
    <article class="candidate-card ${candidate.name === winnerName ? "is-best" : ""}">
      <div class="provider-chip">${candidate.source_model}</div>
      <h3>${candidate.name}</h3>
      <div class="candidate-grid">
        <div><span class="candidate-meta">Price</span><strong>${formatValue(candidate.price, "$")}</strong></div>
        <div><span class="candidate-meta">Distance</span><strong>${formatValue(candidate.distance_miles)} mi</strong></div>
        <div><span class="candidate-meta">Rating</span><strong>${formatValue(candidate.rating)}</strong></div>
      </div>
      <div class="candidate-metrics">
        <div class="candidate-metric"><span class="candidate-meta">Origin</span><strong>${candidate.origin_distance}</strong></div>
        <div class="candidate-metric"><span class="candidate-meta">Query</span><strong>${candidate.query_distance}</strong></div>
        <div class="candidate-metric"><span class="candidate-meta">Cluster</span><strong>${candidate.cluster_distance}</strong></div>
      </div>
      <p class="candidate-meta">Consensus: ${candidate.consensus_members.join(", ")}</p>
      <p class="note">${candidate.notes || "No notes supplied."}</p>
    </article>
  `).join("");
}

function getProviderColor(providerName, isWinner) {
  if (isWinner) {
    return "#dc2626";
  }
  return PROVIDER_COLORS[providerName] || "#2563eb";
}

function buildDisplayPoints(ranked, winnerName) {
  const maxAxisValue = Math.max(
    0.08,
    ...ranked.flatMap((candidate) => candidate.point_3d.map((value) => Math.abs(value)))
  );

  return ranked.map((candidate, index) => {
    const [priceGap, distanceGap, ratingGap] = candidate.point_3d;
    const normalized = [priceGap / maxAxisValue, distanceGap / maxAxisValue, ratingGap / maxAxisValue];
    const angle = ((Math.PI * 2) / Math.max(ranked.length, 1)) * index;
    const radialLift = candidate.name === winnerName ? 0.9 : 1.15 + (index * 0.12);
    const spreadX = Math.cos(angle) * 0.45 * radialLift;
    const spreadY = Math.sin(angle) * 0.45 * radialLift;
    const spreadZ = ((index % 3) - 1) * 0.18;

    return {
      ...candidate,
      displayPoint: [
        Number((normalized[0] * 2.4 + spreadX).toFixed(4)),
        Number((normalized[1] * 2.4 + spreadY).toFixed(4)),
        Number((normalized[2] * 2.4 + spreadZ).toFixed(4))
      ]
    };
  });
}

function renderPlot(result) {
  const ranked = result.ranked_candidates;
  const winnerName = result.angel_filtered.name;
  const displayCandidates = buildDisplayPoints(ranked, winnerName);
  const pointLabels = displayCandidates.map((candidate, index) => (
    candidate.name === winnerName ? `Winner: ${candidate.name}` : `#${index + 1}`
  ));

  const connectionTraces = displayCandidates.map((candidate) => {
    const color = getProviderColor(candidate.source_model, candidate.name === winnerName);
    return {
      x: [0, candidate.displayPoint[0]],
      y: [0, candidate.displayPoint[1]],
      z: [0, candidate.displayPoint[2]],
      mode: "lines",
      type: "scatter3d",
      line: {
        color,
        width: candidate.name === winnerName ? 8 : 5,
        dash: candidate.name === winnerName ? "solid" : "dot"
      },
      opacity: candidate.name === winnerName ? 0.9 : 0.5,
      hoverinfo: "skip",
      showlegend: false
    };
  });

  const trace = {
    x: displayCandidates.map((candidate) => candidate.displayPoint[0]),
    y: displayCandidates.map((candidate) => candidate.displayPoint[1]),
    z: displayCandidates.map((candidate) => candidate.displayPoint[2]),
    text: pointLabels,
    mode: "markers+text",
    type: "scatter3d",
    textposition: "top center",
    textfont: {
      size: 11,
      color: "#1b1d1f"
    },
    customdata: displayCandidates.map((candidate) => [
      candidate.name,
      candidate.point_3d[0],
      candidate.point_3d[1],
      candidate.point_3d[2],
      candidate.origin_distance,
      candidate.source_model
    ]),
    marker: {
      size: displayCandidates.map((candidate) => candidate.name === winnerName ? 11 : 8),
      color: displayCandidates.map((candidate) => getProviderColor(candidate.source_model, candidate.name === winnerName)),
      opacity: 0.95,
      line: { color: "#ffffff", width: 1.6 },
      symbol: displayCandidates.map((candidate) => candidate.name === winnerName ? "diamond" : "circle")
    },
    hovertemplate:
      "<b>%{customdata[0]}</b><br>Provider: %{customdata[5]}<br>Actual price gap: %{customdata[1]}<br>Actual distance gap: %{customdata[2]}<br>Actual rating gap: %{customdata[3]}<br>Origin distance: %{customdata[4]}<extra></extra>",
    showlegend: false
  };

  const originTrace = {
    x: [0],
    y: [0],
    z: [0],
    text: ["User Query Origin"],
    mode: "markers+text",
    type: "scatter3d",
    textposition: "bottom center",
    marker: {
      size: 12,
      color: "#1b1d1f",
      symbol: "diamond"
    },
    textfont: {
      size: 12,
      color: "#1b1d1f"
    },
    hovertemplate: "<b>User Query Origin</b><extra></extra>"
  };

  const legendTraces = [
    { name: "Gemini", color: PROVIDER_COLORS.gemini },
    { name: "Ollama", color: PROVIDER_COLORS.ollama },
    { name: "Winner", color: "#dc2626" }
  ].map((entry) => ({
    x: [null],
    y: [null],
    z: [null],
    mode: "markers",
    type: "scatter3d",
    marker: {
      size: 8,
      color: entry.color
    },
    name: entry.name,
    hoverinfo: "skip",
    showlegend: true
  }));

  Plotly.newPlot("plot", [...connectionTraces, trace, originTrace, ...legendTraces], {
    margin: { l: 0, r: 0, b: 0, t: 44 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: { title: "Price Gap", backgroundcolor: "rgba(255,255,255,0.18)", gridcolor: "rgba(27,29,31,0.1)", zerolinecolor: "rgba(27,29,31,0.35)" },
      yaxis: { title: "Distance Gap", backgroundcolor: "rgba(255,255,255,0.18)", gridcolor: "rgba(27,29,31,0.1)", zerolinecolor: "rgba(27,29,31,0.35)" },
      zaxis: { title: "Rating Gap", backgroundcolor: "rgba(255,255,255,0.18)", gridcolor: "rgba(27,29,31,0.1)", zerolinecolor: "rgba(27,29,31,0.35)" },
      camera: { eye: { x: 1.55, y: 1.45, z: 1.2 } },
      aspectmode: "cube"
    },
    legend: {
      orientation: "h",
      x: 0,
      y: 0.98,
      yanchor: "top",
      xanchor: "left",
      bgcolor: "rgba(255,255,255,0.72)",
      bordercolor: "rgba(27,29,31,0.1)",
      borderwidth: 1
    },
    annotations: [
      {
        x: 0,
        y: 1.08,
        xref: "paper",
        yref: "paper",
        xanchor: "left",
        yanchor: "bottom",
        text: "Display positions are expanded for readability. Hover shows actual gap values.",
        showarrow: false,
        font: { size: 12, color: "#586169" }
      }
    ],
    showlegend: true
  }, { responsive: true, displayModeBar: false });
}

async function handleSubmit(event) {
  event.preventDefault();
  submitButton.disabled = true;
  setStatus("Querying Gemini and Ollama, then scoring the consensus cluster...");

  const payload = {
    query: document.getElementById("query").value,
    top_k: document.getElementById("top_k").value,
    gemini_model: document.getElementById("gemini_model").value,
    ollama_model: document.getElementById("ollama_model").value,
    ollama_url: document.getElementById("ollama_url").value
  };

  try {
    const response = await fetch("/api/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Request failed");
    }

    renderWinner(result);
    renderRankedCandidates(result);
    renderPlot(result);
    rawJson.textContent = JSON.stringify(result, null, 2);
    setStatus(`Consensus built from ${result.reference_models.join(" + ")}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
}

initializeDefaults();
form.addEventListener("submit", handleSubmit);