/**
 * NewsSentinel - Modern Intelligence Platform Client Engine
 * Stage 4: Live ML Model Inference, Probability Telemetry & Interactive Analytics
 */

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initMobileSidebar();
  initAnalyzerInput();
  initSampleLoaders();
  initThemeSwitcher();
  initHistoryController();
});

/* ==========================================================================
   Live UTC Clock in Header
   ========================================================================== */
function initClock() {
  const clockEl = document.getElementById("header-utc-clock");
  if (!clockEl) return;

  function update() {
    const now = new Date();
    const utcString = now.toUTCString().replace("GMT", "UTC");
    clockEl.textContent = utcString;
  }
  update();
  setInterval(update, 1000);
}

/* ==========================================================================
   Mobile Sidebar Navigation Toggle
   ========================================================================== */
function initMobileSidebar() {
  const toggleBtn = document.getElementById("mobile-menu-toggle");
  const sidebar = document.getElementById("app-sidebar");

  if (!toggleBtn || !sidebar) return;

  toggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    sidebar.classList.toggle("mobile-open");
  });

  document.addEventListener("click", (e) => {
    if (window.innerWidth <= 768 && sidebar.classList.contains("mobile-open")) {
      if (!sidebar.contains(e.target) && e.target !== toggleBtn) {
        sidebar.classList.remove("mobile-open");
      }
    }
  });
}

/* ==========================================================================
   Analyzer Input Telemetry & Live API Integration
   ========================================================================== */
function initAnalyzerInput() {
  const headlineInput = document.getElementById("news-headline");
  const textarea = document.getElementById("news-input");
  const charCountEl = document.getElementById("char-count");
  const wordCountEl = document.getElementById("word-count");
  const readingTimeEl = document.getElementById("reading-time");
  const analyzeBtn = document.getElementById("btn-analyze-news");
  const clearBtn = document.getElementById("btn-clear-input");
  const resultContainer = document.getElementById("analysis-result-container");

  if (!textarea) return;

  // Live typing metrics calculation across headline + article text
  function updateTelemetry() {
    const headline = headlineInput ? headlineInput.value : "";
    const text = textarea.value;
    const combined = `${headline} ${text}`.trim();
    const chars = combined.length;

    const words = combined ? (combined.match(/\b[a-zA-Z0-9_\'-]+\b/g) || []).length : 0;
    const readSec = Math.max(1, Math.ceil((words / 200) * 60));

    if (charCountEl) charCountEl.textContent = chars.toLocaleString();
    if (wordCountEl) wordCountEl.textContent = words.toLocaleString();
    if (readingTimeEl) {
      readingTimeEl.textContent = words > 0 ? (readSec < 60 ? `${readSec}s` : `${(readSec / 60).toFixed(1)}m`) : "0s";
    }
  }

  textarea.addEventListener("input", updateTelemetry);
  if (headlineInput) {
    headlineInput.addEventListener("input", updateTelemetry);
  }
  updateTelemetry();

  // Clear workspace
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (headlineInput) headlineInput.value = "";
      textarea.value = "";
      updateTelemetry();
      if (resultContainer) {
        resultContainer.innerHTML = "";
        resultContainer.style.display = "none";
      }
      showToast("Input workspace cleared", "info");
    });
  }

  // Submit for machine learning prediction
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      const headline = headlineInput ? headlineInput.value.trim() : "";
      const text = textarea.value.trim();
      const combined = `${headline} ${text}`.trim();

      if (!combined) {
        showToast("Please enter a headline or news article text to analyze.", "error");
        if (headlineInput) headlineInput.focus();
        else textarea.focus();
        return;
      }

      if (combined.length < 10) {
        showToast("Input is too short (minimum 10 characters required).", "error");
        return;
      }

      // UI Loading State
      setButtonLoading(analyzeBtn, true);

      try {
        const response = await fetch("/api/analyze", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ headline, text })
        });

        const resData = await response.json();

        if (!response.ok || !resData.success) {
          throw new Error(resData.error || "Failed to process text analysis.");
        }

        renderAnalysisResult(resData.data);
        const isUncertainToast = resData.data.is_uncertain || resData.data.prediction === "Uncertain";
        const predBadge = isUncertainToast
          ? "UNCERTAIN — NEEDS VERIFICATION"
          : (resData.data.prediction === "Real" ? "MODEL PREDICTION: REAL" : "MODEL PREDICTION: FAKE");
        const toastType = isUncertainToast ? "info" : (resData.data.prediction === "Real" ? "success" : "error");
        showToast(`Analysis Complete: ${predBadge} (${resData.data.confidence_display})`, toastType);

        // Smooth scroll to result
        if (resultContainer) {
          resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      } catch (err) {
        console.error("Analysis Error:", err);
        showToast(err.message || "An error occurred while evaluating the input.", "error");
      } finally {
        setButtonLoading(analyzeBtn, false);
      }
    });
  }
}

/* ==========================================================================
   Render Analysis Result (Stage 4 Real Machine Learning Telemetry)
   ========================================================================== */
function renderAnalysisResult(data) {
  const container = document.getElementById("analysis-result-container");
  if (!container) return;

  const isUncertain = data.is_uncertain || data.prediction === "Uncertain";
  const isReal = data.prediction === "Real" || (!isUncertain && data.raw_prediction === "Real");

  let badgeClass, cardBorderClass, badgeIcon, badgeText;
  if (isUncertain) {
    badgeClass = "badge-uncertain";
    cardBorderClass = "result-uncertain";
    badgeIcon = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
    badgeText = "UNCERTAIN — NEEDS VERIFICATION";
  } else if (isReal) {
    badgeClass = "badge-real";
    cardBorderClass = "result-real";
    badgeIcon = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    badgeText = "MODEL PREDICTION: REAL";
  } else {
    badgeClass = "badge-fake";
    cardBorderClass = "result-fake";
    badgeIcon = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
    badgeText = "MODEL PREDICTION: FAKE";
  }

  const probabilities = data.probabilities || { Fake: 0.5, Real: 0.5 };
  const fakePct = Math.round((probabilities.Fake || 0) * 100);
  const realPct = Math.round((probabilities.Real || 0) * 100);
  const metrics = data.metrics || {};
  const explanation = data.explanation || {};
  const realSignals = explanation.top_real_signals || [];
  const fakeSignals = explanation.top_fake_signals || [];
  const matchedCount = explanation.total_features_matched || 0;
  const netBias = explanation.net_lexical_bias !== undefined ? explanation.net_lexical_bias : 0;
  const intercept = explanation.baseline_intercept !== undefined ? explanation.baseline_intercept : -0.534;

  const uncertainCalloutHtml = isUncertain ? `
    <div class="uncertain-notice-callout">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
      </svg>
      <div class="uncertain-notice-text">
        <h4>High Ambiguity / Needs Independent Editorial Verification</h4>
        <p>
          Neither class probability reached the 65.0% certainty threshold (Real: <strong>${(probabilities.Real * 100).toFixed(1)}%</strong>, Fake: <strong>${(probabilities.Fake * 100).toFixed(1)}%</strong>).
          This article exhibits weak or mixed lexical cues. Predictions represent statistical signals only, not factual certainty.
        </p>
      </div>
    </div>
  ` : "";

  const realChipsHtml = realSignals.length > 0
    ? realSignals.map(s => `
        <div class="signal-chip signal-chip-real" title="Weight: ${s.weight > 0 ? '+' : ''}${s.weight} | TF-IDF: ${s.tfidf}">
          <span class="signal-token">${escapeHtml(s.token)}</span>
          <span class="signal-score-badge signal-score-real">+${s.contribution.toFixed(4)}</span>
        </div>
      `).join("")
    : `<div class="signal-empty">No positive (Real) lexical signals matched in dictionary.</div>`;

  const fakeChipsHtml = fakeSignals.length > 0
    ? fakeSignals.map(s => `
        <div class="signal-chip signal-chip-fake" title="Weight: ${s.weight > 0 ? '+' : ''}${s.weight} | TF-IDF: ${s.tfidf}">
          <span class="signal-token">${escapeHtml(s.token)}</span>
          <span class="signal-score-badge signal-score-fake">${s.contribution.toFixed(4)}</span>
        </div>
      `).join("")
    : `<div class="signal-empty">No negative (Fake) lexical signals matched in dictionary.</div>`;

  container.innerHTML = `
    <div class="result-card ${cardBorderClass}">
      <div class="result-card-header">
        <div class="result-headline">
          <span class="result-status-badge ${badgeClass}">
            ${badgeIcon}
            ${badgeText}
          </span>
          <span class="stat-pill stat-pill-active" style="border-color: rgba(56, 189, 248, 0.4); color: #38BDF8;" title="Model pattern-matching confidence within training corpus, not factual certainty">
            Model Confidence: ${escapeHtml(data.confidence_display || 'N/A')}
          </span>
          <span class="stat-pill stat-pill-neutral">
            Latency: ${data.latency_ms || 0}ms
          </span>
        </div>
        <span class="result-meta-time">Processed: ${escapeHtml(data.timestamp)}</span>
      </div>

      ${uncertainCalloutHtml}

      <!-- Probability Distribution Bar -->
      <div class="prob-meter-card">
        <div class="prob-meter-header">
          <span>Binary Credibility Distribution</span>
          <span>Model: ${escapeHtml(data.model_name || 'Linear SVM')}</span>
        </div>
        <div class="prob-bar-container">
          <div class="prob-bar-fake" style="width: ${fakePct}%;" title="Fake: ${fakePct}%"></div>
          <div class="prob-bar-real" style="width: ${realPct}%;" title="Real: ${realPct}%"></div>
        </div>
        <div class="prob-labels-grid">
          <span class="prob-label-fake">Fake News Probability: <strong>${(probabilities.Fake * 100).toFixed(1)}%</strong></span>
          <span class="prob-label-real">Real News Probability: <strong>${(probabilities.Real * 100).toFixed(1)}%</strong></span>
        </div>
      </div>

      <!-- Feature Telemetry Grid -->
      <div class="result-features-grid">
        <div class="feature-box">
          <div class="feature-title">Character Count</div>
          <div class="feature-num">${metrics.characters_count || 0}</div>
        </div>
        <div class="feature-box">
          <div class="feature-title">Word Count</div>
          <div class="feature-num">${metrics.words_count || 0}</div>
        </div>
        <div class="feature-box">
          <div class="feature-title">Sentence Structure</div>
          <div class="feature-num">${metrics.sentences_count || 0}</div>
        </div>
        <div class="feature-box">
          <div class="feature-title">Uppercase Ratio</div>
          <div class="feature-num">${metrics.uppercase_ratio || 0}%</div>
        </div>
        <div class="feature-box">
          <div class="feature-title">Lexical Diversity</div>
          <div class="feature-num">${metrics.lexical_diversity || 0}%</div>
        </div>
        <div class="feature-box">
          <div class="feature-title">Avg Word Length</div>
          <div class="feature-num">${metrics.avg_word_length || 0}</div>
        </div>
        <div class="feature-box">
          <div class="feature-title">Punctuation Markers</div>
          <div class="feature-num">!(${metrics.exclamation_count || 0}) ?(${metrics.question_count || 0})</div>
        </div>
        <div class="feature-box">
          <div class="feature-title">Estimated Reading Time</div>
          <div class="feature-num">${metrics.reading_time_display || '0s'}</div>
        </div>
      </div>

      <!-- Explainable AI (XAI) Feature Attribution Section -->
      <div class="explanation-card">
        <div class="explanation-header">
          <div>
            <div class="explanation-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
              <span>Influential Lexical Signals (Explainable AI)</span>
            </div>
            <div class="explanation-subtitle">
              Approximate lexical contribution (w&#772;<sub>j</sub> &middot; x<sub>j</sub>) based on underlying LinearSVC decision weights across calibration folds.
            </div>
          </div>
          <div class="explanation-meta-pills">
            <span class="explanation-pill">Features Matched: <strong>${matchedCount}</strong></span>
            <span class="explanation-pill">Net Decision Bias: <strong>${netBias > 0 ? '+' : ''}${netBias.toFixed(3)}</strong></span>
            <span class="explanation-pill">Intercept (b&#772;<sub>0</sub>): <strong>${intercept.toFixed(3)}</strong></span>
          </div>
        </div>

        <div class="signals-dual-grid">
          <!-- Real Signals Column -->
          <div class="signal-column signal-col-real">
            <div class="signal-col-header">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
              <span>Evidence Pushing Toward Real</span>
            </div>
            <div class="signal-chips-list">
              ${realChipsHtml}
            </div>
          </div>

          <!-- Fake Signals Column -->
          <div class="signal-column signal-col-fake">
            <div class="signal-col-header">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
              <span>Evidence Pushing Toward Fake</span>
            </div>
            <div class="signal-chips-list">
              ${fakeChipsHtml}
            </div>
          </div>
        </div>

        <div class="explanation-disclaimer">
          <strong>Methodology Note:</strong> Attributions represent approximate lexical contributions (w&#772;<sub>j</sub> &middot; x<sub>j</sub>) based on the underlying LinearSVC weights averaged across 3 cross-validation folds. Positive values push the uncalibrated decision boundary toward Real News; negative values push toward Fake News. They do <em>not</em> directly quantify percentage-point changes in the final calibrated probability, and reflect corpus co-occurrences in WELFake rather than verified factual truth.
        </div>
      </div>

      <!-- Academic Integrity Standard Notice & Disclaimer -->
      <div class="integrity-notice">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <div class="integrity-notice-text">
          <h4>Advisory & Operational Disclaimer</h4>
          <p>
            <strong>NewsSentinel identifies linguistic patterns learned from its training data. It does not independently verify sources, events, quotations or factual claims.</strong>
            Predictions are ML-based statistical signals, not definitive fact-checking. Model confidence indicates training distribution alignment, not factual certainty.
          </p>
        </div>
      </div>
    </div>
  `;

  container.style.display = "block";
}

/* ==========================================================================
   Quick Sample Text Loaders
   ========================================================================== */
function initSampleLoaders() {
  const sample1Btn = document.getElementById("sample-headline-1");
  const sample2Btn = document.getElementById("sample-headline-2");
  const headlineInput = document.getElementById("news-headline");
  const textarea = document.getElementById("news-input");

  if (!textarea) return;

  if (sample1Btn) {
    sample1Btn.addEventListener("click", () => {
      if (headlineInput) {
        headlineInput.value = "Senate passes bipartisan budget bill to avert government shutdown";
      }
      textarea.value = "WASHINGTON (Reuters) - The U.S. Senate voted overwhelmingly on Thursday to approve a compromise spending bill, sending the measure to the president's desk and avoiding a looming federal agency shutdown ahead of the midnight deadline.";
      textarea.dispatchEvent(new Event("input"));
      showToast("Loaded Sample Real News (Reuters) into analyzer", "info");
    });
  }

  if (sample2Btn) {
    sample2Btn.addEventListener("click", () => {
      if (headlineInput) {
        headlineInput.value = "SHOCKING! SATANIC PEDOPHILE RING EXPOSED IN SECRET UNDERGROUND BUNKER [VIDEO]";
      }
      textarea.value = "UNBELIEVABLE! Top government officials were caught red handed operating a secret satanic pedophile dungeon beneath a local pizza restaurant. Millions of paid Soros crisis actors and chemtrail operators have been mobilized to suppress this shocking truth!";
      textarea.dispatchEvent(new Event("input"));
      showToast("Loaded Sensational Clickbait sample into analyzer", "info");
    });
  }
}

/* ==========================================================================
   Button Loading Helper
   ========================================================================== */
function setButtonLoading(btn, isLoading) {
  if (isLoading) {
    btn.dataset.originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner"></div><span>Evaluating Veracity...</span>`;
  } else {
    btn.disabled = false;
    if (btn.dataset.originalText) {
      btn.innerHTML = btn.dataset.originalText;
    }
  }
}

/* ==========================================================================
   Toast Notification Utility
   ========================================================================== */
function showToast(message, type = "info") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  let iconSvg = "";
  if (type === "success") {
    iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
  } else if (type === "error") {
    iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F43F5E" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`;
  } else {
    iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;
  }

  toast.innerHTML = `${iconSvg}<span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

/* Simple HTML escaping utility */
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/* ==========================================================================
   Global Theme Switcher (Dark Mode / Light Mode)
   ========================================================================== */
function initThemeSwitcher() {
  const toggleBtn = document.getElementById("theme-toggle-btn");
  const labelEl = document.getElementById("theme-toggle-label");
  if (!toggleBtn) return;

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (labelEl) {
      labelEl.textContent = theme === "light" ? "Light" : "Dark";
    }
    toggleBtn.setAttribute("aria-label", `Switch to ${theme === "light" ? "Dark" : "Light"} Mode`);
    toggleBtn.setAttribute("title", `Currently ${theme === "light" ? "Light" : "Dark"} Mode (click to toggle)`);
  }

  // Restore saved preference on load
  let initialTheme = "dark";
  try {
    const saved = localStorage.getItem("newssentinel_theme");
    if (saved === "light" || saved === "dark") {
      initialTheme = saved;
    } else {
      initialTheme = document.documentElement.getAttribute("data-theme") || "dark";
    }
  } catch (e) {
    initialTheme = document.documentElement.getAttribute("data-theme") || "dark";
  }
  applyTheme(initialTheme);

  toggleBtn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = current === "light" ? "dark" : "light";
    try {
      localStorage.setItem("newssentinel_theme", nextTheme);
    } catch (e) {}
    applyTheme(nextTheme);
  });
}

/* ==========================================================================
   History Page Controller: Classification Filters, Pagination & Detail Modal
   ========================================================================== */
function initHistoryController() {
  const historyDataScript = document.getElementById("initial-history-data");
  const tableBody = document.getElementById("history-table-body");
  const dataTable = document.getElementById("history-data-table");
  const noMatchCard = document.getElementById("history-no-filter-match");
  const paginationWrap = document.getElementById("history-pagination-wrap");
  const searchInput = document.getElementById("history-search-input");
  const resetFiltersBtn = document.getElementById("btn-reset-filters");
  const modalBackdrop = document.getElementById("article-detail-modal");

  // If not on history page or no table present, return early
  if (!tableBody && !historyDataScript) return;

  let allRecords = [];
  try {
    if (historyDataScript && historyDataScript.textContent.trim()) {
      allRecords = JSON.parse(historyDataScript.textContent);
    }
  } catch (e) {
    console.error("Failed to parse initial history dataset:", e);
    allRecords = [];
  }

  // Fast database ID index map
  const recordsById = new Map();
  allRecords.forEach(rec => {
    recordsById.set(rec.id, rec);
  });

  // State
  let activeFilter = "all"; // 'all', 'real', 'fake', 'uncertain'
  let searchQuery = "";
  let currentPage = 1;
  const PAGE_SIZE = 10;

  // Filter tab buttons & count badges
  const filterButtons = document.querySelectorAll(".filter-tab-btn");
  const countAll = document.getElementById("count-all");
  const countReal = document.getElementById("count-real");
  const countFake = document.getElementById("count-fake");
  const countUncertain = document.getElementById("count-uncertain");

  function updateCounts() {
    const total = allRecords.length;
    const realCount = allRecords.filter(r => r.prediction === "Real" && !r.is_uncertain).length;
    const fakeCount = allRecords.filter(r => r.prediction === "Fake" && !r.is_uncertain).length;
    const uncertainCount = allRecords.filter(r => r.is_uncertain || r.prediction === "Uncertain").length;

    if (countAll) countAll.textContent = total;
    if (countReal) countReal.textContent = realCount;
    if (countFake) countFake.textContent = fakeCount;
    if (countUncertain) countUncertain.textContent = uncertainCount;
  }
  updateCounts();

  function getFilteredRecords() {
    let list = allRecords;

    // 1. Classification filter
    if (activeFilter === "real") {
      list = list.filter(r => r.prediction === "Real" && !r.is_uncertain);
    } else if (activeFilter === "fake") {
      list = list.filter(r => r.prediction === "Fake" && !r.is_uncertain);
    } else if (activeFilter === "uncertain") {
      list = list.filter(r => r.is_uncertain || r.prediction === "Uncertain");
    }

    // 2. Search query filter
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      list = list.filter(r => {
        const headline = (r.headline || "").toLowerCase();
        const snippet = (r.snippet || "").toLowerCase();
        const fullText = (r.full_text || "").toLowerCase();
        const idStr = String(r.id);
        return headline.includes(q) || snippet.includes(q) || fullText.includes(q) || idStr.includes(q);
      });
    }

    return list;
  }

  function renderTable() {
    if (!tableBody) return;

    const filtered = getFilteredRecords();
    const totalCount = filtered.length;

    if (totalCount === 0) {
      if (dataTable && dataTable.parentElement) dataTable.parentElement.style.display = "none";
      if (paginationWrap) paginationWrap.style.display = "none";
      if (noMatchCard) {
        noMatchCard.style.display = "flex";
        const msgEl = document.getElementById("filter-empty-message");
        if (msgEl) {
          if (searchQuery && activeFilter !== "all") {
            msgEl.textContent = `No records matching "${searchQuery}" in the "${activeFilter.toUpperCase()}" classification.`;
          } else if (searchQuery) {
            msgEl.textContent = `No historical records matching search query "${searchQuery}".`;
          } else {
            msgEl.textContent = `No historical records found for the "${activeFilter.toUpperCase()}" classification.`;
          }
        }
      }
      return;
    }

    if (dataTable && dataTable.parentElement) dataTable.parentElement.style.display = "block";
    if (noMatchCard) noMatchCard.style.display = "none";
    if (paginationWrap) paginationWrap.style.display = "flex";

    const totalPages = Math.max(1, Math.ceil(totalCount / PAGE_SIZE));
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const startIndex = (currentPage - 1) * PAGE_SIZE;
    const endIndex = Math.min(startIndex + PAGE_SIZE, totalCount);
    const pageRecords = filtered.slice(startIndex, endIndex);

    // Build table rows
    tableBody.innerHTML = pageRecords.map(rec => {
      const isUncertain = rec.is_uncertain || rec.prediction === "Uncertain";
      const predClass = isUncertain ? "uncertain" : (rec.prediction === "Real" ? "real" : "fake");
      const badgeClass = isUncertain ? "badge-uncertain" : (rec.prediction === "Real" ? "badge-real" : "badge-fake");
      const badgeText = isUncertain
        ? "UNCERTAIN — NEEDS VERIFICATION"
        : (rec.prediction === "Real" ? "MODEL PREDICTION: REAL" : "MODEL PREDICTION: FAKE");
      const wordsCount = rec.metrics && rec.metrics.words_count ? rec.metrics.words_count : 0;
      const snippetText = escapeHtml(rec.snippet || rec.headline || "Untitled Record");
      const confDisplay = escapeHtml(rec.confidence_display || "N/A");
      const latency = rec.latency_ms !== undefined ? `${rec.latency_ms}ms` : "--";

      return `
        <tr class="history-row-clickable"
            data-id="${rec.id}"
            data-prediction="${predClass}"
            tabindex="0"
            role="button"
            aria-label="View complete analysis details for record #${rec.id}">
          <td style="font-family: var(--font-mono); font-size: 0.78rem;">#${rec.id} &bull; ${escapeHtml(rec.timestamp)}</td>
          <td style="max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            <span style="font-weight: 500; color: var(--text-main);">${snippetText}</span>
          </td>
          <td style="font-family: var(--font-mono);">${wordsCount}</td>
          <td>
            <span class="result-status-badge ${badgeClass}" style="font-size: 0.72rem; padding: 3px 10px;">
              ${badgeText}
            </span>
          </td>
          <td style="font-family: var(--font-mono); font-weight: 600;">${confDisplay}</td>
          <td style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted);">${latency}</td>
          <td style="text-align: right;">
            <button class="row-inspect-btn" type="button" aria-label="Inspect details for record #${rec.id}">
              <span>View</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </button>
          </td>
        </tr>
      `;
    }).join("");

    renderPagination(totalCount, totalPages, startIndex + 1, endIndex);
  }

  function renderPagination(totalCount, totalPages, start, end) {
    const pagStart = document.getElementById("pag-start");
    const pagEnd = document.getElementById("pag-end");
    const pagTotal = document.getElementById("pag-total");
    const prevBtn = document.getElementById("pag-prev");
    const nextBtn = document.getElementById("pag-next");
    const pagesList = document.getElementById("pag-pages-list");

    if (pagStart) pagStart.textContent = start;
    if (pagEnd) pagEnd.textContent = end;
    if (pagTotal) pagTotal.textContent = totalCount;

    if (prevBtn) prevBtn.disabled = currentPage <= 1;
    if (nextBtn) nextBtn.disabled = currentPage >= totalPages;

    if (pagesList) {
      let pagesHtml = "";
      for (let p = 1; p <= totalPages; p++) {
        if (totalPages <= 7 || p === 1 || p === totalPages || (p >= currentPage - 1 && p <= currentPage + 1)) {
          pagesHtml += `
            <button class="pag-page-number ${p === currentPage ? 'active' : ''}" data-page="${p}" type="button">
              ${p}
            </button>
          `;
        } else if (p === currentPage - 2 || p === currentPage + 2) {
          pagesHtml += `<span style="padding: 0 4px; color: var(--text-muted);">&bull;&bull;&bull;</span>`;
        }
      }
      pagesList.innerHTML = pagesHtml;
    }
  }

  // Tab Filtering event listeners
  filterButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      filterButtons.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      activeFilter = btn.dataset.filter || "all";
      currentPage = 1;
      renderTable();
    });
  });

  // Live Search Input
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      searchQuery = e.target.value.trim();
      currentPage = 1;
      renderTable();
    });
  }

  // Reset Filters & Search
  if (resetFiltersBtn) {
    resetFiltersBtn.addEventListener("click", () => {
      activeFilter = "all";
      searchQuery = "";
      if (searchInput) searchInput.value = "";
      filterButtons.forEach(b => {
        const isAll = (b.dataset.filter || "all") === "all";
        b.classList.toggle("active", isAll);
        b.setAttribute("aria-selected", isAll ? "true" : "false");
      });
      currentPage = 1;
      renderTable();
    });
  }

  // Pagination Controls Click Handling
  if (paginationWrap) {
    paginationWrap.addEventListener("click", (e) => {
      const pageBtn = e.target.closest(".pag-page-number");
      if (pageBtn) {
        const p = parseInt(pageBtn.dataset.page, 10);
        if (p && p !== currentPage) {
          currentPage = p;
          renderTable();
        }
        return;
      }

      if (e.target.closest("#pag-prev") && currentPage > 1) {
        currentPage--;
        renderTable();
        return;
      }

      const next = e.target.closest("#pag-next");
      if (next && !next.disabled) {
        const filtered = getFilteredRecords();
        const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
        if (currentPage < totalPages) {
          currentPage++;
          renderTable();
        }
        return;
      }
    });
  }

  // Clickable Table Rows Event Delegation (Lookup by Database ID)
  if (tableBody) {
    tableBody.addEventListener("click", (e) => {
      const row = e.target.closest(".history-row-clickable");
      if (!row) return;

      // Prevent accidental modal opening when clicking interactive elements other than .row-inspect-btn
      const interactiveEl = e.target.closest("a, input, select");
      if (interactiveEl) return;

      const recordId = parseInt(row.dataset.id, 10);
      const record = recordsById.get(recordId);
      if (record) {
        openDetailModal(record);
      }
    });

    tableBody.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        const row = e.target.closest(".history-row-clickable");
        if (row && document.activeElement === row) {
          e.preventDefault();
          const recordId = parseInt(row.dataset.id, 10);
          const record = recordsById.get(recordId);
          if (record) openDetailModal(record);
        }
      }
    });
  }

  // Modal Open Function
  function openDetailModal(record) {
    if (!modalBackdrop) return;

    const isUncertain = record.is_uncertain || record.prediction === "Uncertain";
    const badgeClass = isUncertain ? "badge-uncertain" : (record.prediction === "Real" ? "badge-real" : "badge-fake");
    const badgeText = isUncertain
      ? "UNCERTAIN — NEEDS VERIFICATION"
      : (record.prediction === "Real" ? "MODEL PREDICTION: REAL" : "MODEL PREDICTION: FAKE");

    // Header pills
    const recIdEl = document.getElementById("modal-record-id");
    const statusBadgeEl = document.getElementById("modal-status-badge");
    const confPill = document.getElementById("modal-confidence-pill");
    const latencyPill = document.getElementById("modal-latency-pill");

    if (recIdEl) recIdEl.textContent = `#${record.id}`;
    if (statusBadgeEl) {
      statusBadgeEl.className = `result-status-badge ${badgeClass}`;
      statusBadgeEl.textContent = badgeText;
    }
    if (confPill) confPill.textContent = `Model Confidence: ${record.confidence_display || 'N/A'}`;
    if (latencyPill) latencyPill.textContent = `Latency: ${record.latency_ms || 0}ms`;

    // Headline
    const headlineEl = document.getElementById("modal-headline");
    if (headlineEl) {
      if (record.headline && record.headline.trim()) {
        headlineEl.textContent = record.headline.trim();
      } else {
        headlineEl.innerHTML = '<span style="color: var(--text-muted); font-weight: normal; font-size: 0.95rem;">(No separate headline submitted)</span>';
      }
    }

    // Probabilities Bar & Percentages
    const probs = record.probabilities || {};
    const fakeP = probs.Fake !== undefined ? probs.Fake : (record.prediction === "Fake" ? 0.95 : 0.05);
    const realP = probs.Real !== undefined ? probs.Real : (record.prediction === "Real" ? 0.95 : 0.05);
    const fakePct = (fakeP * 100).toFixed(1);
    const realPct = (realP * 100).toFixed(1);

    const barFake = document.getElementById("modal-prob-bar-fake");
    const barReal = document.getElementById("modal-prob-bar-real");
    const txtFake = document.getElementById("modal-prob-fake-pct");
    const txtReal = document.getElementById("modal-prob-real-pct");

    if (barFake) barFake.style.width = `${fakePct}%`;
    if (barReal) barReal.style.width = `${realPct}%`;
    if (txtFake) txtFake.textContent = `${fakePct}%`;
    if (txtReal) txtReal.textContent = `${realPct}%`;

    // Full Original Article Text (Safe text escaping & fallback for older records)
    const fullTextEl = document.getElementById("modal-full-text-content");
    if (fullTextEl) {
      if (record.full_text && record.full_text.trim().length > 0) {
        fullTextEl.textContent = record.full_text;
      } else {
        fullTextEl.innerHTML = '<div class="old-record-notice">Full article text unavailable for this older record</div>';
      }
    }

    // Uncertainty Notice Callout
    const uncertainNoticeEl = document.getElementById("modal-uncertainty-notice");
    if (uncertainNoticeEl) {
      if (isUncertain) {
        uncertainNoticeEl.style.display = "block";
        uncertainNoticeEl.innerHTML = `
          <div class="uncertain-notice-callout">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            <div class="uncertain-notice-text">
              <h4>High Ambiguity / Needs Independent Editorial Verification</h4>
              <p>
                Neither class probability reached the 65.0% certainty threshold (Real: <strong>${realPct}%</strong>, Fake: <strong>${fakePct}%</strong>).
                This article exhibits weak or mixed lexical cues. Predictions represent statistical signals only, not factual certainty.
              </p>
            </div>
          </div>
        `;
      } else {
        uncertainNoticeEl.style.display = "none";
        uncertainNoticeEl.innerHTML = "";
      }
    }

    // Explainable AI (XAI) Attribution Signals
    const xaiSection = document.getElementById("modal-xai-section");
    const xaiGrid = document.getElementById("modal-xai-grid");
    if (xaiSection && xaiGrid) {
      const exp = record.explanation || {};
      const realSignals = exp.top_real_signals || exp.top_positive || [];
      const fakeSignals = exp.top_fake_signals || exp.top_negative || [];

      if (realSignals.length > 0 || fakeSignals.length > 0) {
        xaiSection.style.display = "block";
        const realChipsHtml = realSignals.length > 0
          ? realSignals.map(s => `
              <div class="signal-chip signal-chip-real" title="Weight: ${s.weight > 0 ? '+' : ''}${s.weight !== undefined ? s.weight : 0}">
                <span class="signal-token">${escapeHtml(s.token)}</span>
                <span class="signal-score-badge signal-score-real">+${Number(s.contribution || 0).toFixed(4)}</span>
              </div>
            `).join("")
          : '<div class="signal-empty">No positive (Real) signals identified in dictionary.</div>';

        const fakeChipsHtml = fakeSignals.length > 0
          ? fakeSignals.map(s => `
              <div class="signal-chip signal-chip-fake" title="Weight: ${s.weight > 0 ? '+' : ''}${s.weight !== undefined ? s.weight : 0}">
                <span class="signal-token">${escapeHtml(s.token)}</span>
                <span class="signal-score-badge signal-score-fake">${Number(s.contribution || 0).toFixed(4)}</span>
              </div>
            `).join("")
          : '<div class="signal-empty">No negative (Fake) signals identified in dictionary.</div>';

        xaiGrid.innerHTML = `
          <div class="signal-column signal-col-real">
            <div class="signal-col-header">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
              <span>Evidence Pushing Toward Real</span>
            </div>
            <div class="signal-chips-list">${realChipsHtml}</div>
          </div>
          <div class="signal-column signal-col-fake">
            <div class="signal-col-header">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
              <span>Evidence Pushing Toward Fake</span>
            </div>
            <div class="signal-chips-list">${fakeChipsHtml}</div>
          </div>
        `;
      } else {
        xaiSection.style.display = "none";
        xaiGrid.innerHTML = "";
      }
    }

    // Metadata Grid
    const tsEl = document.getElementById("modal-timestamp");
    const modelEl = document.getElementById("modal-model-name");
    const wordCountEl = document.getElementById("modal-word-count");
    const readingTimeEl = document.getElementById("modal-reading-time");

    if (tsEl) tsEl.textContent = record.timestamp || "--";
    if (modelEl) modelEl.textContent = record.model_name || "Linear SVM (Calibrated)";
    if (wordCountEl) {
      wordCountEl.textContent = record.metrics && record.metrics.words_count !== undefined
        ? record.metrics.words_count
        : (record.full_text ? (record.full_text.match(/\b[a-zA-Z0-9_\'-]+\b/g) || []).length : 0);
    }
    if (readingTimeEl) {
      readingTimeEl.textContent = record.metrics && record.metrics.reading_time_display
        ? record.metrics.reading_time_display
        : "--";
    }

    modalBackdrop.style.display = "flex";
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    if (!modalBackdrop) return;
    modalBackdrop.style.display = "none";
    document.body.style.overflow = "";
  }

  // Modal Close event listeners
  const closeBtn = document.getElementById("modal-close-btn");
  const footerCloseBtn = document.getElementById("modal-footer-close-btn");

  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (footerCloseBtn) footerCloseBtn.addEventListener("click", closeModal);

  if (modalBackdrop) {
    modalBackdrop.addEventListener("click", (e) => {
      if (e.target === modalBackdrop) closeModal();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modalBackdrop && modalBackdrop.style.display === "flex") {
      closeModal();
    }
  });

  // Initial table render
  renderTable();
}
