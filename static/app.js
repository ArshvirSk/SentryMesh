/* SentryMesh SPA logic -- five pages, interactive selection rule, live charts.
   Monochrome: black/white/gray + single red accent for attacks only. */

const $ = (sel) => document.querySelector(sel);
const main = $("#main");
const fmt = (n, d = 0) => (n == null ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d }));
const pct = (x) => (x == null ? "—" : (100 * x).toFixed(1) + "%");
const ts = (t) => new Date(t * 1000).toLocaleTimeString();

/* ------- theme (monochrome: no new colors, just inverted gray ramp + adjusted red) ------- */
const THEME = {
  dark: {
    text: "#ececec", textRGB: "236, 236, 236", muted: "#8a8a8a", faint: "#5c5c5c",
    border: "#262626", strong: "#3f3f3f", gray: ["#5c5c5c", "#8a8a8a", "#3f3f3f"],
    red: "#e5484d", redRGB: "229, 72, 77", white: "#ececec", black: "#0a0a0a",
  },
  light: {
    text: "#1a1a1a", textRGB: "26, 26, 26", muted: "#6b6b6b", faint: "#9a9a9a",
    border: "#e2e2e2", strong: "#bdbdbd", gray: ["#9a9a9a", "#6b6b6b", "#bdbdbd"],
    red: "#ce2c31", redRGB: "206, 44, 49", white: "#1a1a1a", black: "#ffffff",
  },
};
const theme = () => THEME[document.documentElement.dataset.theme === "light" ? "light" : "dark"];

function setTheme(mode) {
  document.documentElement.dataset.theme = mode;
  try { localStorage.setItem("sm-theme", mode); } catch (e) {}
  const btn = $("#theme-toggle");
  if (btn) btn.textContent = mode === "light" ? "◐ DARK MODE" : "◐ LIGHT MODE";
  if (labModels && page === "modellab") { updateRuleUI(ruleSelect()); drawCumVar(labModels.pca); }
  if (page === "live") refreshLiveCharts();
  if (page === "overview") renderOvCharts();
}

let lastWindow = null;   // cached /api/window payload for chart redraws
function refreshLiveCharts() {
  if (lastWindow) drawDbscan(lastWindow);
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("sm-theme"); } catch (e) {}
  document.documentElement.dataset.theme = saved === "light" ? "light" : "dark";
  const btn = $("#theme-toggle");
  if (btn) btn.textContent = saved === "light" ? "◐ DARK MODE" : "◐ LIGHT MODE";
  btn?.addEventListener("click", () =>
    setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
}
initTheme();

let page = "overview";
let timers = [];
let liveFilterClass = null;   // set by Overview distribution click-through
let liveFilterIp = null;      // set by "watch this device" from Investigate

function clearTimers() { timers.forEach(clearInterval); timers = []; }
function poll(fn, ms) { fn(); timers.push(setInterval(fn, ms)); }

document.querySelectorAll(".nav-item").forEach((el) =>
  el.addEventListener("click", () => go(el.dataset.page)));

function go(p) {
  page = p;
  clearTimers();
  document.querySelectorAll(".nav-item").forEach((el) =>
    el.classList.toggle("active", el.dataset.page === p));
  ({ overview: pageOverview, modellab: pageModelLab, live: pageLive,
     investigate: pageInvestigate, how: pageHow })[p]();
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}
const post = (path, body) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});

/* ------- tiny SVG helpers ------- */
function svgEl(tag, attrs) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function svgChart(container, w, h, ml, mr, mt, mb) {
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "xMidYMid meet" });
  const g = svgEl("g", { transform: `translate(${ml},${mt})` });
  svg.appendChild(g);
  container.innerHTML = "";
  container.appendChild(svg);
  return { svg, g, iw: w - ml - mr, ih: h - mt - mb };
}
function axis(g, iw, ih, xticks, yticks, xfmt, yfmt) {
  for (const [frac, label] of yticks) {
    const y = ih * (1 - frac);
    g.appendChild(svgEl("line", { x1: 0, y1: y, x2: iw, y2: y, class: "gridline" }));
    const t = svgEl("text", { x: -6, y: y + 3, "text-anchor": "end" });
    t.textContent = label; g.appendChild(t);
  }
  for (const [frac, label] of xticks) {
    const x = iw * frac;
    const t = svgEl("text", { x, y: ih + 14, "text-anchor": "middle" });
    t.textContent = label; g.appendChild(t);
  }
  g.appendChild(svgEl("line", { x1: 0, y1: ih, x2: iw, y2: ih, class: "axis" }));
  g.appendChild(svgEl("line", { x1: 0, y1: 0, x2: 0, y2: ih, class: "axis" }));
}

/* ============================== MODEL LAB ============================== */
const TIEBREAKERS = {
  latency: "lowest inference latency",
  size: "smallest model size",
  accuracy: "highest raw accuracy",
  kappa: "highest Cohen's kappa",
};
let lab = { tolerance: 0.02, tiebreaker: "latency", variant: "full_features" };
let labModels = null;
let selectFetchTimer = null;
let labSort = { key: "macro_f1", dir: -1 };   // sortable comparison table

/* Sort specs for the candidate table; `text` cols sort case-insensitively. */
const LAB_COLS = {
  "Model": { key: "name", text: true },
  "Accuracy": { key: "accuracy" },
  "Macro-F1": { key: "macro_f1" },
  "Precision": { key: "precision_macro" },
  "Recall (sens.)": { key: "recall_macro" },
  "Kappa": { key: "kappa" },
  "MCC": { key: "mcc" },
  "ROC-AUC ovR": { key: "roc_auc_ovr", nulls: -Infinity },
  "CV macro-F1": { key: "cv_macro_f1_mean", nulls: -Infinity },
  "Size": { key: "model_size_kb" },
  "Latency": { key: "latency_ms_per_record" },
};

function sortLabRows(rows) {
  const spec = LAB_COLS[labSort.key];
  if (!spec) return rows;
  const val = (r) => (r[spec.key] == null ? (spec.nulls ?? -Infinity)
    : spec.text ? String(r[spec.key]).toLowerCase() : r[spec.key]);
  return [...rows].sort((a, b) => {
    const va = val(a), vb = val(b);
    const c = spec.text ? va.localeCompare(vb) : va - vb;
    return c * labSort.dir;
  });
}

async function pageModelLab() {
  main.innerHTML = `
    <div class="page-title">Model Lab <span class="muted" style="font-size:12px">· cloud role</span></div>
    <div class="page-sub">Trains all six syllabus candidates on the same split, evaluates them uniformly, then selects the deployable winner by the tolerance-band rule.</div>
    <div class="row">
      <button class="btn" id="btn-train">Train Models</button>
      <span class="mono-note" id="train-status"></span>
    </div>
    <div class="spacer"></div>
    <div id="lab-body" class="muted">No training run yet. Click <b>Train Models</b> to launch the cloud-role sweep (6 candidates × 5-fold CV). This takes a couple of minutes; the page updates when it finishes.</div>`;

  $("#btn-train").addEventListener("click", async () => {
    $("#btn-train").disabled = true;
    const r = await post("/api/train");
    $("#train-status").textContent = r.status === "started" ? "training sweep running…" :
      r.status === "already_running" ? "already running…" : "failed to start";
    watchTraining();
  });

  const m = await api("/api/models");
  if (m.trained) { labModels = m; renderLab(); } else watchTraining();
}

function watchTraining() {
  const iv = setInterval(async () => {
    const st = await api("/api/train/status");
    const el = $("#train-status");
    if (!el) { clearInterval(iv); return; }
    if (st.running) { el.textContent = `training sweep running… (${((Date.now() / 1000 - st.started_at) | 0)}s)`; $("#btn-train").disabled = true; }
    else {
      clearInterval(iv);
      $("#btn-train").disabled = false;
      if (st.error) { el.textContent = "error: " + st.error; return; }
      const m = await api("/api/models");
      if (m.trained) { labModels = m; renderLab(); }
      if (st.elapsed_s) el.textContent = `finished in ${st.elapsed_s}s — winner deployed to edge`;
    }
  }, 2000);
}

/* Client-side mirror of the server's tolerance-band rule: instant feedback
   while dragging the slider; the server /api/select call (debounced) refreshes
   the authoritative sentence-level reasoning. */
function ruleSelect() {
  const rs = labModels.results;
  const best = Math.max(...rs.map((r) => r.macro_f1));
  const qualified = rs.filter((r) => r.macro_f1 >= best - lab.tolerance);
  const key = { latency: (r) => r.latency_ms_per_record, size: (r) => r.model_size_bytes, accuracy: (r) => -r.accuracy, kappa: (r) => -r.kappa }[lab.tiebreaker];
  const winner = [...qualified].sort((a, b) => key(a) - key(b))[0];
  return { best, qualified, winner };
}

async function refetchReasoning() {
  const out = await post("/api/select", { tolerance: lab.tolerance, tiebreaker: lab.tiebreaker });
  if (out.selection && $("#sel-reasoning")) {
    $("#sel-reasoning").textContent = out.selection.reasoning;
    $("#sel-qualified").textContent = `${out.selection.qualified.length} qualified: ${out.selection.qualified.join(", ")}`;
  }
}

function onRuleChange() {
  if (!labModels) return;
  $("#tol-val").textContent = (lab.tolerance * 100).toFixed(0) + " pp";
  updateRuleUI(ruleSelect());
  clearTimeout(selectFetchTimer);
  selectFetchTimer = setTimeout(refetchReasoning, 250);
}

function updateRuleUI(sel) {
  const m = labModels;
  // table: winner badge + qualified outline
  document.querySelectorAll("#cmp-table tbody tr[data-name]").forEach((tr) => {
    const name = tr.dataset.name;
    const isWin = name === sel.winner.name;
    const inBand = sel.qualified.some((q) => q.name === name);
    tr.classList.toggle("winner", isWin);
    tr.style.outline = inBand && !isWin ? "1px solid var(--border-strong)" : "";
    tr.style.outlineOffset = "-1px";
    const wt = tr.querySelector(".win-tag"), qt = tr.querySelector(".q-tag");
    if (wt) wt.style.display = isWin ? "" : "none";
    if (qt) qt.style.display = inBand && !isWin ? "" : "none";
  });
  // reasoning summary block
  $("#sel-best").textContent = sel.best.toFixed(4);
  $("#sel-band").textContent = `±${(lab.tolerance * 100).toFixed(0)} pp → ${sel.qualified.length} qualified`;
  $("#sel-tb").textContent = TIEBREAKERS[lab.tiebreaker];
  $("#sel-qualified").textContent = `${sel.qualified.length} qualified: ${sel.qualified.map((q) => q.name).join(", ")}`;
  drawScatter(sel);
}

function renderLab() {
  const m = labModels;
  const dep = m.deployment || { model: m.winner.name, variant: m.deployed_variant, source: "automatic" };
  const sel = ruleSelect();

  const dep0 = m.deployment || { model: m.winner.name };
  const rows = sortLabRows(m.results).map((r) => `
    <tr data-name="${r.name}" class="${r.name === sel.winner.name ? "winner" : ""}">
      <td class="name">${r.name}
        <span class="tag win win-tag" style="display:${r.name === sel.winner.name ? "" : "none"}">RULE PICK</span>
        <span class="tag q-tag" style="display:none">IN BAND</span>
        <span class="tag">${r.module}</span></td>
      <td>${r.accuracy.toFixed(4)}</td>
      <td>${r.macro_f1.toFixed(4)}</td>
      <td>${r.precision_macro.toFixed(4)}</td>
      <td>${r.recall_macro.toFixed(4)}</td>
      <td>${r.kappa.toFixed(4)}</td>
      <td>${r.mcc.toFixed(4)}</td>
      <td>${r.roc_auc_ovr == null ? "—" : r.roc_auc_ovr.toFixed(4)}</td>
      <td>${r.cv_macro_f1_mean == null ? "—" : r.cv_macro_f1_mean.toFixed(3) + " ± " + r.cv_macro_f1_std.toFixed(3)}</td>
      <td>${fmt(r.model_size_kb, 1)} KB</td>
      <td>${r.latency_ms_per_record.toFixed(3)} ms</td>
      <td><button class="btn mini deploy-btn ${r.name === dep0.model ? "current" : ""}" data-model="${r.name}">
        ${r.name === dep0.model ? "DEPLOYED" : "Deploy"}</button></td>
    </tr>`).join("");

  $("#lab-body").innerHTML = `
    <div class="banner ${dep.source === "manual" ? "override" : ""}" id="dep-banner">
      <span>EDGE DEPLOYMENT:</span>
      <span class="who">${dep.model}</span>
      <span class="tag">${dep.variant}</span>
      ${dep.source === "manual"
        ? '<span class="badge manual">MANUAL OVERRIDE</span><span class="muted">— differs from the rule\'s pick</span>'
        : '<span class="badge on">RULE AUTOMATIC</span><span class="muted">— selected by the tolerance-band rule</span>'}
      <span style="flex:1"></span>
      ${dep.source === "manual" ? '<button class="btn small ghost" id="btn-reset-auto">Reset to Automatic</button>' : ""}
    </div>
    <div class="spacer"></div>
    <div class="controls">
      <div class="ctl-group">
        <span class="ctl-label">Tolerance band · macro-F1 within <span class="ctl-value" id="tol-val">${(lab.tolerance * 100).toFixed(0)} pp</span> of best</span>
        <input type="range" id="tol-slider" min="0" max="10" step="1" value="${(lab.tolerance * 100).toFixed(0)}">
      </div>
      <div class="ctl-group">
        <span class="ctl-label">Tiebreaker among qualifying models</span>
        <div class="seg" id="tb-seg">
          ${Object.entries(TIEBREAKERS).map(([k, v]) =>
            `<button data-tb="${k}" class="${lab.tiebreaker === k ? "active" : ""}">${v}</button>`).join("")}
        </div>
      </div>
      <div class="ctl-group" style="flex:1">
        <span class="ctl-label">Deploy variant (applies to whichever model is deployed)</span>
        <div class="seg" id="var-seg">
          <button data-var="full_features" class="${lab.variant === "full_features" ? "active" : ""}">full_features</button>
          <button data-var="pca_reduced" class="${lab.variant === "pca_reduced" ? "active" : ""}">pca_reduced</button>
        </div>
      </div>
    </div>
    <div class="spacer"></div>
    <div class="panel"><h3>Candidate comparison — ${m.dataset.n_rows.toLocaleString()} rows, ${m.dataset.n_features} features, ${m.results[0].cv_folds}-fold CV</h3>
      <div class="mono-note" style="margin:2px 0 8px">click a column header to sort · deploy/rule highlighting follows the models, not the row order</div>
      <table id="cmp-table">
        <tr>${Object.keys(LAB_COLS).map((h) =>
          `<th class="sortable${labSort.key === h ? (labSort.dir === 1 ? " sorted-asc" : " sorted-desc") : ""}" data-col="${h}">${h}<span class="sort-arrow">${labSort.key === h ? (labSort.dir === 1 ? " ↑" : " ↓") : ""}</span></th>`).join("")}
        <th></th></tr><tbody>${rows}</tbody></table>
      <div class="mono-note" style="margin-top:6px">Precision / Recall are macro-averaged across the four classes; per-class recall is sensitivity in detection literature.</div></div>
    <div class="spacer"></div>
    <div class="grid c2">
      <div class="panel"><h3>Latency vs macro-F1 — band moves with the slider</h3><div id="scatter" class="chart"></div>
        <div class="legend"><span><span class="dot" style="background:var(--text)"></span>rule pick</span>
        <span><span class="dot" style="background:var(--border-strong)"></span>candidate</span>
        <span><span class="dot" style="background:var(--red)"></span>manually deployed</span>
        <span><span class="sw" style="background:var(--band)"></span>qualifying band</span></div></div>
      <div class="panel"><h3>PCA cumulative explained variance</h3><div id="cumvar" class="chart"></div>
        <div class="legend"><span><span class="sw" style="background:var(--text)"></span>cumulative variance</span>
        <span><span class="sw" style="background:var(--muted);stroke-dasharray:4 3"></span>${(m.pca.variance_target * 100).toFixed(0)}% target</span>
        <span><span class="sw" style="background:var(--red)"></span>chosen k = ${m.pca.n_features_reduced}</span></div></div>
    </div>
    <div class="spacer"></div>
    <div class="panel"><h3>Selection reasoning <span class="mono-note" id="sel-qualified"></span></h3>
      <div class="reason" id="sel-reasoning">${m.selection.reasoning}</div>
      <div class="spacer"></div>
      <div class="kv">
        <div class="k">Best macro-F1</div><div class="v" id="sel-best">${m.selection.best_macro_f1.toFixed(4)}</div>
        <div class="k">Tolerance band</div><div class="v" id="sel-band">±${(m.selection.tolerance * 100).toFixed(0)} pp → ${m.selection.qualified.length} qualified</div>
        <div class="k">Tiebreaker</div><div class="v" id="sel-tb">${TIEBREAKERS[lab.tiebreaker]}</div>
        <div class="k">Deployed</div><div class="v">${dep.model} · ${dep.variant} · ${dep.source}</div>
      </div></div>
    <div class="spacer"></div>
    <div class="panel"><h3>Weka-style error suite — deployed artifact (lower is better)</h3>
      <div class="grid c4" id="err-grid">${errGridHtml(m, dep)}</div>
      <div class="mono-note" style="margin-top:8px">MAE / RMSE over all (record, class) probability cells · RAE / RRSE normalised by the zero-R class-prior predictor — the same error analysis the source paper reports (Figs. 7/10)</div></div>
    <div class="spacer"></div>
    <div class="grid c2">
      <div class="panel"><h3>Confusion matrix heatmap — ${dep.model}</h3>
        <div class="mono-note" style="margin-bottom:10px">${dep.variant} · accuracy ${dep.accuracy} · macro-F1 ${dep.macro_f1} (deployed artifact, refit)</div>
        ${heatmapHtml(m, dep)}</div>
      <div class="panel"><h3>PCA: full vs reduced features</h3>
        ${pcaTableHtml(m)}
        <div class="spacer"></div><div class="reason">${m.pca.explanation}</div></div>
    </div>`;

  // ---- wire interactivity ----
  $("#tol-slider").addEventListener("input", (e) => {
    lab.tolerance = parseInt(e.target.value, 10) / 100;
    onRuleChange();
  });
  document.querySelectorAll("#tb-seg button").forEach((b) => b.addEventListener("click", () => {
    lab.tiebreaker = b.dataset.tb;
    document.querySelectorAll("#tb-seg button").forEach((x) => x.classList.toggle("active", x === b));
    onRuleChange();
  }));
  document.querySelectorAll("#var-seg button").forEach((b) => b.addEventListener("click", () => {
    lab.variant = b.dataset.var;
    document.querySelectorAll("#var-seg button").forEach((x) => x.classList.toggle("active", x === b));
    deployModel(dep.source === "manual" ? dep.model : sel.winner.name, lab.variant, dep.source);
  }));
  document.querySelectorAll(".deploy-btn").forEach((b) => b.addEventListener("click", () =>
    deployModel(b.dataset.model, lab.variant, "manual")));
  const resetBtn = $("#btn-reset-auto");
  if (resetBtn) resetBtn.addEventListener("click", () =>
    deployModel(null, lab.variant, "automatic"));

  document.querySelectorAll("#cmp-table th.sortable").forEach((th) => th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (labSort.key === col) labSort.dir = -labSort.dir;
    else { labSort = { key: col, dir: LAB_COLS[col].text ? 1 : -1 };
      // sensible defaults: metrics descend, cost columns ascend
      if (col === "Model") labSort.dir = 1;
      if (col === "Size" || col === "Latency") labSort.dir = 1;
    }
    renderLab();
  }));

  updateRuleUI(sel);
  drawCumVar(m.pca);
}

async function deployModel(name, variant, source) {
  const body = source === "automatic"
    ? { source, variant, tolerance: lab.tolerance, tiebreaker: lab.tiebreaker }
    : { model: name, variant, source };
  const r = await post("/api/deploy", body);
  if (r.error) { alert(r.error); return; }
  const m = await api("/api/models");
  if (m.trained) { labModels = m; if (page === "modellab") renderLab(); }
}

function heatmapHtml(m, dep) {
  const byName = Object.fromEntries(m.results.map((r) => [r.name, r]));
  const src = byName[dep.model] || m.winner;
  const cm = src.confusion_matrix;
  const cls = ["Benign", "DoS", "Probing", "Auth"];
  const T = theme();
  const max = Math.max(...cm.flat());
  let cells = `<div class="hm-head"></div>${cls.map((c) => `<div class="hm-head">${c}</div>`).join("")}`;
  cm.forEach((row, i) => {
    cells += `<div class="hm-head" style="justify-content:flex-end;padding-right:8px">${cls[i]}</div>`;
    row.forEach((v, j) => {
      const intensity = v / max;
      let bg, fg;
      if (i === j) { const a = 0.12 + 0.75 * intensity; bg = `rgba(${T.textRGB},${a})`; fg = intensity > 0.45 ? T.black : "var(--text)"; }
      else if (v > 0) { const a = 0.12 + 0.78 * intensity; bg = `rgba(${T.redRGB},${a})`; fg = intensity > 0.5 ? T.black : "var(--red)"; }
      else { bg = "var(--panel-2)"; fg = "var(--faint)"; }
      cells += `<div class="hm-cell" style="background:${bg};color:${fg}" title="actual ${cls[i]} → predicted ${cls[j]}: ${v}">${v}</div>`;
    });
  });
  return `<div class="hm" style="grid-template-columns:64px repeat(${cls.length},1fr)">${cells}</div>
    <div class="legend"><span><span class="dot" style="background:rgba(${T.textRGB},0.7)"></span>correct (diagonal)</span>
    <span><span class="dot" style="background:rgba(${T.redRGB},0.7)"></span>misclassifications (intensity ∝ count)</span></div>`;
}

function errGridHtml(m, dep) {
  const byName = Object.fromEntries(m.results.map((r) => [r.name, r]));
  const src = byName[dep.model] || m.winner;
  const e = src.errors || { mae: "—", rmse: "—", rae: "—", rrse: "—" };
  return [["MAE", e.mae], ["RMSE", e.rmse], ["RAE", e.rae], ["RRSE", e.rrse]].map(([k, v]) =>
    `<div class="panel stat" style="background:var(--panel-2)"><div class="label">${k}</div><div class="value mono" style="font-size:20px">${v}</div></div>`).join("");
}

function pcaTableHtml(m) {
  const pca = m.pca;
  const rows = [
    ["Features", pca.n_features_full, pca.n_features_reduced],
    ["Variance retained", "100%", pct(pca.variance_retained)],
    ["Accuracy", pca.full.accuracy.toFixed(4), pca.reduced.accuracy.toFixed(4)],
    ["Macro-F1", pca.full.macro_f1.toFixed(4), pca.reduced.macro_f1.toFixed(4)],
    ["Latency (ms/rec)", pca.full.latency_ms_per_record.toFixed(3), pca.reduced.latency_ms_per_record.toFixed(3)],
    ["Model size (KB)", fmt(pca.full.model_size_kb, 1), fmt(pca.reduced.model_size_kb, 1)],
  ].map(([k, a, b]) => `<tr><td class="name">${k}</td><td>${a}</td><td>${b}</td></tr>`).join("");
  return `<table><tr><th></th><th>Full</th><th>PCA-reduced</th></tr>${rows}</table>`;
}

function drawScatter(sel) {
  const elc = $("#scatter");
  if (!elc) return;
  const m = labModels;
  const dep = m.deployment || { model: m.winner.name, source: "automatic" };
  const W = 660, H = 270, ml = 52, mr = 20, mt = 14, mb = 34;
  const { g, iw, ih } = svgChart(elc, W, H, ml, mr, mt, mb);
  const rs = m.results;
  const xs = rs.map((r) => Math.log10(Math.max(r.latency_ms_per_record, 1e-4)));
  const xmin = Math.min(...xs) - 0.15, xmax = Math.max(...xs) + 0.15;
  const ymin = Math.min(...rs.map((r) => r.macro_f1)) - 0.03;
  const ymax = 1.0;
  const X = (v) => ((Math.log10(Math.max(v, 1e-4)) - xmin) / (xmax - xmin)) * iw;
  const Y = (v) => ih - ((v - ymin) / (ymax - ymin)) * ih;

  // qualifying band: y >= best - tolerance
  const bandY = Y(sel.best - lab.tolerance);
  g.appendChild(svgEl("rect", { x: 0, y: 0, width: iw, height: bandY, class: "band" }));
  g.appendChild(svgEl("line", { x1: 0, y1: bandY, x2: iw, y2: bandY, class: "marker" }));
  const bl = svgEl("text", { x: iw - 4, y: bandY - 5, "text-anchor": "end", class: "marker-label" });
  bl.textContent = `band ≥ ${(sel.best - lab.tolerance).toFixed(3)}`; g.appendChild(bl);

  axis(g, iw, ih,
    [0, 0.25, 0.5, 0.75, 1].map((f) => [f, Math.pow(10, xmin + f * (xmax - xmin)).toExponential(0)]),
    [0, 0.5, 1].map((f) => [f, (ymin + f * (ymax - ymin)).toFixed(2)]));

  rs.forEach((r) => {
    const isWin = r.name === sel.winner.name;
    const isDep = r.name === dep.model;
    const t = theme();
    g.appendChild(svgEl("circle", {
      cx: X(r.latency_ms_per_record), cy: Y(r.macro_f1), r: isWin ? 7 : 5,
      fill: isWin ? t.white : isDep ? t.red : t.strong,
      stroke: isWin ? t.black : "none", "stroke-width": 1,
    }));
    const title = svgEl("title");
    title.textContent = `${r.name}\nmacro-F1 ${r.macro_f1.toFixed(4)} · kappa ${r.kappa.toFixed(4)} · MCC ${r.mcc.toFixed(4)}\nlatency ${r.latency_ms_per_record.toFixed(3)} ms · size ${fmt(r.model_size_kb, 1)} KB\nMAE ${r.errors.mae} · RMSE ${r.errors.rmse}`;
    g.lastChild.appendChild(title);
    const lbl = svgEl("text", {
      x: X(r.latency_ms_per_record) + 10,
      y: Y(r.macro_f1) + 4,
      fill: isWin ? t.white : t.muted,
      style: isWin ? "font-weight:700" : "",
    });
    lbl.textContent = r.name.replace(" (bagging)", "").replace(" (CART/Gini)", "").replace("Gradient Boosting ", "").replace("Voting Ensemble ", "");
    g.appendChild(lbl);
  });
  const xl = svgEl("text", { x: iw / 2, y: ih + 28, "text-anchor": "middle" });
  xl.textContent = "inference latency (ms/record, log scale)"; g.appendChild(xl);
}

function drawCumVar(pca) {
  const elc = $("#cumvar");
  if (!elc || !pca.cumvar_curve || !pca.cumvar_curve.length) return;
  const W = 660, H = 230, ml = 46, mr = 16, mt = 12, mb = 30;
  const { g, iw, ih } = svgChart(elc, W, H, ml, mr, mt, mb);
  const curve = pca.cumvar_curve;
  const kmax = curve[curve.length - 1].k;
  const X = (k) => ((k - 1) / Math.max(kmax - 1, 1)) * iw;
  const Y = (v) => ih - v * ih;

  // 95% target line
  const y95 = Y(pca.variance_target);
  g.appendChild(svgEl("line", { x1: 0, y1: y95, x2: iw, y2: y95, class: "marker" }));
  const t95 = svgEl("text", { x: 4, y: y95 - 5, class: "marker-label" });
  t95.textContent = `${(pca.variance_target * 100).toFixed(0)}% target`; g.appendChild(t95);

  // chosen k vertical marker
  const xk = X(pca.n_features_reduced);
  g.appendChild(svgEl("line", { x1: xk, y1: 0, x2: xk, y2: ih, stroke: theme().red, "stroke-width": 1, "stroke-dasharray": "4 3" }));
  const tk = svgEl("text", { x: xk + 4, y: 12, fill: theme().red });
  tk.textContent = `k = ${pca.n_features_reduced} → ${pct(pca.variance_retained)}`; g.appendChild(tk);

  axis(g, iw, ih,
    [0, 0.5, 1].map((f) => [f, String(Math.round(1 + f * (kmax - 1)))]),
    [0, 0.25, 0.5, 0.75, 1].map((f) => [f, (f * 100).toFixed(0) + "%"]));

  const d = curve.map((p, i) => `${i ? "L" : "M"}${X(p.k).toFixed(1)},${Y(p.variance).toFixed(1)}`).join(" ");
  g.appendChild(svgEl("path", { d, fill: "none", stroke: theme().white, "stroke-width": 1.5 }));
  curve.forEach((p) => g.appendChild(svgEl("circle", { cx: X(p.k), cy: Y(p.variance), r: 2.5, fill: theme().white })));

  const xl = svgEl("text", { x: iw / 2, y: ih + 24, "text-anchor": "middle" });
  xl.textContent = "number of principal components"; g.appendChild(xl);
}

/* ============================== OVERVIEW ============================== */
let ovRange = "all";   // '1m' | '5m' | 'all'
let ovRps = 20;

async function pageOverview() {
  main.innerHTML = `
    <div class="page-title">Overview</div>
    <div class="page-sub">Two-tier pipeline — cloud trains and selects once; the fog/edge tier runs only the winner.</div>
    <div class="controls">
      <div class="ctl-group"><span class="ctl-label">Simulator</span>
        <div class="row" style="gap:8px">
          <button class="btn small" id="ov-start">Start</button>
          <button class="btn small ghost" id="ov-pause">Pause</button>
          <button class="btn small ghost" id="ov-reset">Reset</button>
          <span class="badge" id="ov-sim">off</span>
        </div></div>
      <div class="ctl-group"><span class="ctl-label">Speed</span>
        <div class="seg" id="speed-seg">
          <button data-rps="20">1x · 20/s</button>
          <button data-rps="100">5x · 100/s</button>
          <button data-rps="400">20x · 400/s</button>
        </div></div>
      <div class="ctl-group"><span class="ctl-label">Chart window</span>
        <div class="seg" id="range-seg">
          <button data-r="1m">last 1 min</button>
          <button data-r="5m">last 5 min</button>
          <button data-r="all" class="active">since start</button>
        </div></div>
    </div>
    <div class="spacer"></div>
    <div class="grid c5">
      <div class="panel stat"><div class="label">Records processed</div><div class="value mono" id="ov-total">0</div><div class="sub" id="ov-rate"></div></div>
      <div class="panel stat"><div class="label">Attacks detected</div><div class="value attack mono" id="ov-attacks">0</div><div class="sub" id="ov-att-pct"></div></div>
      <div class="panel stat"><div class="label">Live accuracy</div><div class="value mono" id="ov-acc">—</div><div class="sub" id="ov-acc-sub">on unseen stream</div></div>
      <div class="panel stat"><div class="label">Deployed model</div><div class="value mono" id="ov-model" style="font-size:14px; line-height:1.3">—</div><div class="sub" id="ov-variant"></div></div>
      <div class="panel stat"><div class="label">Avg edge latency</div><div class="value mono" id="ov-lat">—</div><div class="sub" id="ov-mem">ms / record · CPU —</div></div>
    </div>
    <div class="spacer"></div>
    <div class="grid c2">
      <div class="panel"><h3>Attack-type distribution <span class="mono-note" id="dist-window"></span></h3><div id="ov-dist" class="muted">no data yet</div>
        <div class="spacer"></div>
        <div class="mono-note" style="margin-bottom:6px">per-class live accuracy on the unseen stream · click an attack class to open it filtered in Live Detection</div>
        <div id="ov-acc-by" class="muted">no scored records yet</div></div>
      <div class="panel"><h3>Traffic over time</h3><div class="bars-over-time" id="ov-time"></div>
        <div class="mono-note" style="margin-top:8px">grey = total · red = attacks</div></div>
    </div>
    <div class="spacer"></div>
    <div class="panel"><h3>Traffic over time by class</h3><div id="ov-lines" class="chart"></div>
      <div class="legend" id="ov-lines-legend"></div></div>`;

  $("#ov-start").addEventListener("click", () => post("/api/sim/start"));
  $("#ov-pause").addEventListener("click", () => post("/api/sim/stop"));
  $("#ov-reset").addEventListener("click", () => post("/api/sim/reset"));
  document.querySelectorAll("#speed-seg button").forEach((b) => b.addEventListener("click", async () => {
    ovRps = parseInt(b.dataset.rps, 10);
    document.querySelectorAll("#speed-seg button").forEach((x) => x.classList.toggle("active", x === b));
    await post("/api/sim/speed", { rps: ovRps });
  }));
  document.querySelectorAll("#range-seg button").forEach((b) => b.addEventListener("click", () => {
    ovRange = b.dataset.r;
    document.querySelectorAll("#range-seg button").forEach((x) => x.classList.toggle("active", x === b));
    renderOvCharts();
  }));

  poll(async () => {
    const s = await api("/api/state");
    const f = await api("/api/feed");
    $("#ov-total").textContent = fmt(f.total);
    const attacks = (f.counts.DoS || 0) + (f.counts.Probing || 0) + (f.counts.Authentication || 0) + (f.counts.ANOMALOUS || 0);
    $("#ov-attacks").textContent = fmt(attacks);
    $("#ov-att-pct").textContent = f.total ? pct(attacks / f.total) + " of traffic" : "—";
    const dep = s.model || {};
    $("#ov-model").textContent = s.deployed ? dep.name : "no model deployed";
    $("#ov-variant").textContent = s.deployed
      ? `${dep.variant} · macro-F1 ${dep.macro_f1} · ${dep.source === "manual" ? "manual override" : "rule automatic"}`
      : "run Train Models in Model Lab";
    $("#ov-lat").textContent = f.avg_latency_ms != null ? f.avg_latency_ms.toFixed(3) : "—";
    const accEl = $("#ov-acc");
    if (accEl) {
      accEl.textContent = f.live_accuracy != null ? (100 * f.live_accuracy).toFixed(2) + "%" : "—";
      $("#ov-acc-sub").textContent = f.live_scored ? `${fmt(f.live_scored)} scored on the unseen stream regime` : "on unseen stream";
    }
    const res = f.resources || {};
    const memEl = $("#ov-mem");
    if (memEl && res.cpu_pct != null) {
      memEl.textContent = `ms / record · CPU ${res.cpu_pct.toFixed(1)}% · RSS ${res.mem_mb.toFixed(0)} MB`;
    }
    $("#ov-sim").textContent = f.sim_running ? "running" : "off";
    $("#ov-sim").className = "badge" + (f.sim_running ? " on" : "");
    $("#sidebar-model").textContent = s.deployed ? dep.name : "no model";
    renderClassAccuracy(f.live_accuracy_by_class);
    renderDist(f.counts, attacks);
    await renderOvCharts();
  }, 1500);
}

/* Per-class live accuracy bars (Overview): click an attack class → Live Detection
   pre-filtered to it. Benign is intentionally not clickable — the whole point is
   jumping to the attack traffic. */
function renderClassAccuracy(byCls) {
  const el = $("#ov-acc-by");
  if (!el) return;
  if (!byCls || !Object.values(byCls).some((c) => c.n > 0)) {
    el.textContent = "no scored records yet";
    return;
  }
  el.innerHTML = Object.entries(byCls).map(([cls, c]) => {
    const acc = c.n ? c.correct / c.n : null;
    const clickable = cls !== "Benign" && c.n > 0;
    const misses = c.n - c.correct;
    return `<div class="bar-row ${clickable ? "clickable" : ""}" ${clickable ? `onclick="goLiveFiltered('${cls}')" title="${misses} missed of ${fmt(c.n)} — show ${cls} in live feed"` : ""}>
      <div class="bl">${cls}</div>
      <div class="bar-track"><div class="bar-fill acc${cls !== "Benign" && misses ? " red" : ""}" style="width:${acc == null ? 0 : 100 * acc}%"></div></div>
      <div class="bv">${acc == null ? "—" : (100 * acc).toFixed(1) + "%"}<span class="acc-n">· ${fmt(c.n)}</span></div></div>`;
  }).join("");
}

async function renderOvCharts() {
  const since = ovRange === "1m" ? Date.now() / 1000 - 60 : ovRange === "5m" ? Date.now() / 1000 - 300 : null;
  const t = await api("/api/timeseries" + (since ? `?since=${Math.floor(since / 2) * 2}` : ""));
  const buckets = t.buckets;
  $("#dist-window").textContent = ovRange === "all" ? "· since start" : `· ${ovRange === "1m" ? "last 1 min" : "last 5 min"}`;

  // distribution over window
  const totals = { Benign: 0, DoS: 0, Probing: 0, Authentication: 0, ANOMALOUS: 0 };
  buckets.forEach((b) => { for (const k in totals) totals[k] += b.classes[k] || 0; });
  const max = Math.max(1, ...Object.values(totals));
  const rows = Object.entries(totals);
  $("#ov-dist").innerHTML = rows.map(([k, v]) => {
    const clickable = k !== "Benign" && v > 0;
    return `<div class="bar-row ${clickable ? "clickable" : ""}" ${clickable ? `onclick="goLiveFiltered('${k}')" title="show ${k} in live feed"` : ""}>
      <div class="bl">${k}</div>
      <div class="bar-track"><div class="bar-fill${k !== "Benign" && v ? " red" : ""}" style="width:${(100 * v) / max}%"></div></div>
      <div class="bv">${fmt(v)}</div></div>`;
  }).join("");

  // total/attack bars
  const timeEl = $("#ov-time");
  if (buckets.length) {
    const maxB = Math.max(1, ...buckets.map((b) => b.total));
    timeEl.innerHTML = buckets.slice(-60).map((b) => {
      const att = (b.classes.DoS || 0) + (b.classes.Probing || 0) + (b.classes.Authentication || 0) + (b.classes.ANOMALOUS || 0);
      return `<div class="tb" style="height:${(100 * b.total) / maxB}%">${att ? `<div class="att" style="height:${(100 * att) / b.total}%"></div>` : ""}</div>`;
    }).join("");
  } else timeEl.innerHTML = "";

  drawClassLines(buckets);
}

const LINE_STYLES = {
  Benign: { stroke: "muted", width: 1.2, swatch: "solid" },
  DoS: { stroke: "red", width: 2, swatch: "solid" },
  Probing: { stroke: "red", width: 1.4, swatch: "dashed" },
  Authentication: { stroke: "red", width: 1.4, swatch: "dotted" },
  ANOMALOUS: { stroke: "red", width: 2, swatch: "longdash", opacity: 0.8 },
};

function drawClassLines(buckets) {
  const elc = $("#ov-lines");
  if (!elc) return;
  const classes = ["Benign", "DoS", "Probing", "Authentication", "ANOMALOUS"];
  $("#ov-lines-legend").innerHTML = classes.map((c) => {
    const s = LINE_STYLES[c];
    const col = theme()[s.stroke];
    const st = s.swatch === "dashed" ? `border-top:2px dashed ${col}`
      : s.swatch === "dotted" ? `border-top:2px dotted ${col}`
      : s.swatch === "longdash" ? `border-top:2px dashed ${col}`
      : `background:${col};height:${Math.max(s.width, 3)}px`;
    return `<span><span class="sw" style="${st};opacity:${s.opacity || 1}"></span>${c}</span>`;
  }).join("");
  if (!buckets.length) { elc.innerHTML = '<div class="muted" style="padding:8px">no data yet — start the simulator</div>'; return; }
  const W = 920, H = 200, ml = 42, mr = 12, mt = 10, mb = 26;
  const { g, iw, ih } = svgChart(elc, W, H, ml, mr, mt, mb);
  const t0 = buckets[0].t, t1 = buckets[buckets.length - 1].t;
  const maxC = Math.max(1, ...buckets.flatMap((b) => classes.map((c) => b.classes[c] || 0)));
  const X = (tt) => (t1 === t0 ? 0 : ((tt - t0) / (t1 - t0))) * iw;
  const Y = (v) => ih - (v / maxC) * ih;

  axis(g, iw, ih,
    [0, 0.5, 1].map((f) => [f, ts(t0 + f * (t1 - t0))]),
    [0, 0.5, 1].map((f) => [f, String(Math.round(f * maxC))]));

  classes.forEach((c) => {
    const st = LINE_STYLES[c];
    let d = "", pen = false;
    buckets.forEach((b) => {
      const v = b.classes[c] || 0;
      d += `${pen ? "L" : "M"}${X(b.t).toFixed(1)},${Y(v).toFixed(1)} `;
      pen = true;
    });
    const attrs = { d, fill: "none", stroke: theme()[st.stroke], "stroke-width": st.width };
    if (st.dash) attrs["stroke-dasharray"] = st.dash;
    if (st.faint) attrs.opacity = 0.75;
    g.appendChild(svgEl("path", attrs));
  });
}

function renderDist(counts) { /* replaced by windowed version in renderOvCharts */ }

window.goLiveFiltered = function (cls) {
  liveFilterClass = cls;
  liveFilterIp = null;
  go("live");
};

window.goLiveWatchIp = function (ip) {
  liveFilterIp = ip;
  liveFilterClass = null;
  go("live");
};

/* ============================== LIVE DETECTION ============================== */
let liveCls = { Benign: true, DoS: true, Probing: true, Authentication: true, ANOMALOUS: true };
let liveQ = "";
let livePaused = false;       // freezes the feed list; DBSCAN viz keeps updating
let lastFeedHtml = "";
let liveWatchIp = null;       // device-history merge while watching one device

async function pageLive() {
  if (liveFilterClass) {
    Object.keys(liveCls).forEach((k) => (liveCls[k] = k === liveFilterClass));
    liveQ = "";
  }
  const pre = liveFilterClass;
  const preIp = liveFilterIp;
  liveFilterClass = null;
  liveFilterIp = null;
  if (preIp) {
    liveQ = preIp.toLowerCase();
    liveWatchIp = preIp;
    // watching a device means ALL of its traffic — clear any leftover class filter
    Object.keys(liveCls).forEach((k) => (liveCls[k] = true));
  }

  main.innerHTML = `
    <div class="page-title">Live Detection <span class="muted" style="font-size:12px">· fog/edge role</span></div>
    <div class="page-sub">Simulated gateway stream, classified record-by-record by the deployed winner, with the DBSCAN anomaly backstop.</div>
    <div class="row">
      <button class="btn" id="btn-start">Start simulator</button>
      <button class="btn ghost" id="btn-stop">Stop</button>
      <span class="badge" id="lv-state">stopped</span>
      <span class="mono-note" id="lv-tally"></span>
    </div>
    <div class="spacer"></div>
    <div class="grid c2">
      <div class="panel"><h3>DBSCAN window — what the anomaly backstop sees</h3>
        <div id="dbscan" class="chart" style="max-width:520px"></div>
        <div class="mono-note" id="dbscan-note" style="margin-top:8px">grey = dense cluster · red = unclustered (anomalous) · viz-only 2D PCA of the rolling window</div></div>
      <div class="panel"><h3>Filter</h3>
        <div class="filterbar" id="lv-filters">
          ${Object.keys(liveCls).map((c) =>
            `<label class="chk ${liveCls[c] ? "checked" : ""}"><input type="checkbox" data-c="${c}" ${liveCls[c] ? "checked" : ""}>${c}</label>`).join("")}
          <input class="search" id="lv-q" placeholder="filter by device/IP…" style="flex:1;min-width:160px" value="${preIp ? preIp : ""}">
        </div>
        <div class="mono-note" id="lv-hint" style="margin-bottom:8px">${pre ? `pre-filtered to ${pre} — re-check boxes to widen` : preIp ? `watching ${preIp} — clear the search box to widen` : "all classes shown"}</div></div>
    </div>
    <div class="spacer"></div>
    <div class="panel"><h3>Live feed</h3>
      <div class="row" style="margin-bottom:10px">
        <button class="btn small ghost" id="btn-pause-feed"></button>
        <span class="mono-note" id="feed-note">freezes the list so you can read a record — the stream, counters and DBSCAN viz keep running</span>
      </div>
      <div class="feed"><div class="feed-row head"><div>time</div><div>source</div><div>predicted</div><div>conf</div><div>latency</div></div>
      <div id="lv-feed"></div></div></div>`;

  $("#btn-start").addEventListener("click", () => post("/api/sim/start"));
  $("#btn-stop").addEventListener("click", () => post("/api/sim/stop"));
  const pauseBtn = $("#btn-pause-feed");
  pauseBtn.textContent = livePaused ? "▶ Resume" : "⏸ Pause feed";
  pauseBtn.addEventListener("click", () => {
    livePaused = !livePaused;
    pauseBtn.textContent = livePaused ? "▶ Resume" : "⏸ Pause feed";
  });
  document.querySelectorAll("#lv-filters input[type=checkbox]").forEach((c) =>
    c.addEventListener("change", () => {
      liveCls[c.dataset.c] = c.checked;
      c.parentElement.classList.toggle("checked", c.checked);
    }));
  $("#lv-q").addEventListener("input", (e) => {
    liveQ = e.target.value.toLowerCase();
    // editing the search away from the watched IP disables the history merge
    if (liveWatchIp && liveQ !== liveWatchIp.toLowerCase()) liveWatchIp = null;
  });

  poll(async () => {
    const [f, w] = await Promise.all([api("/api/feed"), api("/api/window")]);
    if (!$("#lv-feed")) return;    $("#lv-state").textContent = f.sim_running ? "running" : "stopped";
    $("#lv-state").className = "badge" + (f.sim_running ? " on" : "");
    $("#lv-tally").textContent =
      `total ${fmt(f.total)} · benign ${fmt(f.counts.Benign || 0)} · DoS ${fmt(f.counts.DoS || 0)} · probing ${fmt(f.counts.Probing || 0)} · auth ${fmt(f.counts.Authentication || 0)} · anomalous ${fmt(f.anomalies)}` +
      (f.live_accuracy != null ? ` · live accuracy ${(100 * f.live_accuracy).toFixed(2)}% of ${fmt(f.live_scored)}` : "");
    lastWindow = w;
    drawDbscan(w);

  // While watching one device, render its stored history (up to 200 events)
  // instead of the global feed — a single device is far too sparse to stay
  // inside the 120-row recent window. The history grows as the simulator
  // classifies new records, so the watched feed still updates live.
  let feedRows = f.feed;
  if (liveWatchIp) {
    const d = await api("/api/device/" + encodeURIComponent(liveWatchIp));
    if (d && d.found) feedRows = [...d.events].sort((a, b) => b.ts - a.ts).slice(0, 120);
  }
  const shown = feedRows.filter((r) =>
    (liveCls[r.pred] || (r.anomalous && liveCls.ANOMALOUS)) &&
    (!liveQ || r.src_ip.toLowerCase().includes(liveQ)));
  const T = theme();
  const html = shown.map((r) => {
    const attack = r.pred !== "Benign";
    const anomRow = r.anomalous && liveCls.ANOMALOUS && !liveCls[r.pred];
    return `<div class="feed-row${attack && !anomRow ? " attack" : ""}${r.anomalous ? " anomaly" : ""}">
      <div>${ts(r.ts)}</div><div>${r.src_ip}</div>
      <div class="cls">${anomRow ? "ANOMALOUS" : r.pred}${r.anomalous && !anomRow ? ` <span class="tag" style="border-color:${T.red};color:${T.red}">DBSCAN</span>` : ""}</div>
      <div>${(r.confidence * 100).toFixed(1)}%</div><div>${r.latency_ms.toFixed(2)} ms</div></div>`;
  }).join("") || `<div class="feed-row warn-note"><div colspan="5">no records match the current filter</div><div></div><div></div><div></div><div></div></div>`;
  if (!livePaused || html !== lastFeedHtml) {
    lastFeedHtml = html;
    $("#lv-feed").innerHTML = html;
  }
  }, 1200);
}

function drawDbscan(w) {
  const elc = $("#dbscan");
  if (!elc) return;
  if (!w.ready || !w.points.length) {
    elc.innerHTML = '<div class="muted" style="padding:10px">collecting window… (needs ≥10 records)</div>';
    return;
  }
  const W = 500, H = 340;
  const T = theme();
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}` });
  const border = svgEl("rect", { x: 0.5, y: 0.5, width: W - 1, height: H - 1, fill: "var(--panel-2)", stroke: "var(--border)" });
  svg.appendChild(border);
  const xs = w.points.map((p) => p.x), ys = w.points.map((p) => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = 8;
  const X = (v) => pad + ((v - xmin) / Math.max(xmax - xmin, 1e-9)) * (W - 2 * pad);
  const Y = (v) => H - pad - ((v - ymin) / Math.max(ymax - ymin, 1e-9)) * (H - 2 * pad);
  const nAnom = w.points.filter((p) => p.c === -1).length;
  w.points.forEach((p) => {
    const anom = p.c === -1;
    const known = p.c >= 0;
    svg.appendChild(svgEl("rect", {
      x: X(p.x) - (anom ? 4 : 2.5), y: Y(p.y) - (anom ? 4 : 2.5),
      width: anom ? 8 : 5, height: anom ? 8 : 5,
      fill: anom ? T.red : known ? T.gray[p.c % T.gray.length] : T.border,
      stroke: known ? "none" : anom ? "none" : T.strong, "stroke-width": 0.5,
    }));
  });
  elc.innerHTML = "";
  elc.appendChild(svg);
  $("#dbscan-note").innerHTML =
    `${w.points.length} pts · window ${w.n_window} · <span style="color:var(--red)">${nAnom} anomalous</span>` +
    (w.labels_stale ? ' · <span class="faint">labels from last backstop pass</span>' : "") +
    ` · eps ${w.eps}, min_samples ${w.min_samples}`;
}

/* ============================== INVESTIGATE ============================== */
async function pageInvestigate() {
  main.innerHTML = `
    <div class="page-title">Investigate Device</div>
    <div class="page-sub">Full classification history and class breakdown per source device/IP seen by the edge tier.</div>
    <div class="row"><input class="search" id="iv-q" placeholder="search or enter IP, e.g. 10.0.3.42"></div>
    <div class="spacer"></div>
    <div class="grid c2"><div class="panel"><h3>Devices seen</h3><div id="iv-list" class="feed"></div></div>
    <div class="panel"><h3>Device detail</h3><div id="iv-detail" class="muted">select a device</div></div></div>`;

  $("#iv-q").addEventListener("keydown", async (e) => {
    if (e.key === "Enter") showDevice($("#iv-q").value.trim());
  });

  poll(async () => {
    const d = await api("/api/devices");
    const el = $("#iv-list");
    if (!el) return;
    const q = ($("#iv-q").value || "").toLowerCase();
    const rows = d.devices.filter((x) => !q || x.src_ip.toLowerCase().includes(q)).slice(0, 40);
    el.innerHTML = rows.map((x) => {
      const att = (x.counts.DoS || 0) + (x.counts.Probing || 0) + (x.counts.Authentication || 0) + (x.counts.ANOMALOUS || 0);
      return `<div class="feed-row" style="cursor:pointer" onclick="showDevice('${x.src_ip}')">
        <div>${x.src_ip}</div><div>${fmt(x.total)} records</div>
        <div style="color:${att ? "var(--red)" : "inherit"}">${att ? att + " flagged" : "clean"}</div>
        <div><button class="btn mini" onclick="event.stopPropagation();goLiveWatchIp('${x.src_ip}')" title="watch this device in the live feed">watch ▸</button></div><div></div></div>`;
    }).join("") || '<div class="muted" style="padding:8px">no devices yet — start the simulator</div>';
  }, 1500);

  window.showDevice = showDevice;
}

async function showDevice(ip) {
  const r = await api("/api/device/" + encodeURIComponent(ip));
  const el = $("#iv-detail");
  if (!el) return;
  if (!r.found) { el.innerHTML = `<span class="muted">no history for ${ip}</span>`; return; }
  const cls = ["Benign", "DoS", "Probing", "Authentication", "ANOMALOUS"];
  const dist = cls.map((c) => [c, r.counts[c] || 0]);
  const max = Math.max(1, ...dist.map((d) => d[1]));
  const events = [...r.events].sort((a, b) => a.ts - b.ts);
  const t0 = events[0]?.ts ?? 0, t1 = events[events.length - 1]?.ts ?? 1;
  const strip = events.map((e) => {
    const x = t1 === t0 ? 50 : ((e.ts - t0) / (t1 - t0)) * 100;
    const attack = e.pred !== "Benign" || e.anomalous;
    return `<div class="sdot" style="left:${x}%;background:${attack ? "var(--red)" : "var(--muted)"}" title="${ts(e.ts)} · ${e.anomalous ? "ANOMALOUS/" : ""}${e.pred} · ${(e.confidence * 100).toFixed(1)}%"></div>`;
  }).join("");
  el.innerHTML = `
    <div class="kv" style="margin-bottom:12px">
      <div class="k">Source IP</div><div class="v">${r.src_ip}</div>
      <div class="k">Total records</div><div class="v">${fmt(r.total)}</div>
      <div class="k">Actions</div><div class="v"><button class="btn mini" onclick="goLiveWatchIp('${r.src_ip}')">watch in live feed ▸</button></div>
    </div>
    <div class="mono-note" style="margin-bottom:6px">classification timeline (grey benign · red attack)</div>
    <div class="strip">${strip || '<span class="muted" style="font-size:10px;padding:8px;display:inline-block">no events</span>'}</div>
    <div class="strip-axis"><span>${events.length ? ts(t0) : "—"}</span><span>${events.length ? ts(t1) : "—"}</span></div>
    <div class="spacer"></div>
    ${dist.map(([k, v]) => `<div class="bar-row"><div class="bl">${k}</div>
      <div class="bar-track"><div class="bar-fill${k !== "Benign" && v ? " red" : ""}" style="width:${(100 * v) / max}%"></div></div>
      <div class="bv">${fmt(v)}</div></div>`).join("")}
    <div class="spacer"></div><h3 style="font-size:11px;color:var(--muted);text-transform:uppercase;font-family:var(--mono)">Recent events</h3>
    <div class="feed">${r.events.slice(0, 30).map((e) =>
      `<div class="feed-row${e.pred !== "Benign" ? " attack" : ""}${e.anomalous ? " anomaly" : ""}">
       <div>${ts(e.ts)}</div><div>${e.pred}${e.anomalous ? " ⚠" : ""}</div><div>${(e.confidence * 100).toFixed(1)}%</div><div>${e.latency_ms.toFixed(2)} ms</div><div></div></div>`).join("")}</div>`;
}

/* ============================== HOW IT WORKS ============================== */
const HOW = [
  ["Overall architecture", "Module 1 — Introduction to ML",
   "The two-tier cloud/edge split is the module's 'steps in developing an ML application' applied end to end. Training several model families and comparing them — instead of committing to one — is a practical answer to the bias-variance tradeoff and overfitting risk; held-out test error measures generalization, and the tolerance-band selection deliberately accepts a slightly underfit winner when it is far cheaper to run."],
  ["Logistic Regression & Decision Tree candidates", "Module 2 — Regression and Trees",
   "Two of the five candidates come straight from this module: a regularized Logistic Regression and a single CART decision tree grown with the Gini index. The whole evaluation harness is also Module 2: every candidate is scored with the confusion matrix, per-class precision/recall, F-measure and ROC-AUC (one-vs-rest)."],
  ["Random Forest & XGBoost candidates + k-fold CV", "Module 3 — Ensemble Learning",
   "The project's core training strategy is this module: Random Forest (bagging of trees) and Gradient Boosting via XGBoost (boosting of trees) are trained alongside the others with stratified k-fold cross-validation, then compared — 'comparing ensemble methods' made literal, with the ensemble-vs-single-model accuracy/latency tradeoff as the deciding evidence."],
  ["SVM (RBF kernel) & multiclass framing", "Module 4 — Classification",
   "The fifth candidate is an RBF-kernel SVM, chosen because IoT traffic classes are not linearly separable — the kernel trick handles that. The task is genuinely four-class (Benign / DoS / Probing / Authentication) multiclass classification handled natively, not a binary one-vs-rest framing extended after the fact."],
  ["DBSCAN anomaly safety net", "Module 5 — Clustering",
   "At the edge tier, a density-based DBSCAN pass runs over a rolling window of recent traffic. Records that fall in no dense cluster are flagged 'ANOMALOUS — unclassified': a zero-day backstop for attacks the supervised model never saw in training, using the module's density-clustering concepts. The Live Detection page projects this window to 2D so the clusters and outliers are directly visible."],
  ["PCA for the edge deployment", "Module 6 — Dimensionality Reduction",
   "Before deploying, PCA is fitted on the training features (retaining ≥95% variance) and the winner is re-evaluated on the reduced feature set. The deployed model uses full or reduced features depending on which wins the measured accuracy-vs-latency tradeoff — the module's dimensionality-reduction idea made data-driven, with the cumulative explained-variance curve shown in Model Lab."],
  ["Synthetic dataset with a real-data path", "Design decision — why not a downloaded capture",
   "A seeded synthetic generator (seed 42, 72/16/8/4% class skew) is the experimental instrument, not a fallback: it makes every reported number bit-for-bit reproducible, guarantees train/test separation (no capture duplicates leaking across the split), carries provably exact labels, and lets the fog stream replay a known 25% drift — so live generalization on Overview is measured against a known shift, not hoped for. The tradeoff is stated plainly: results speak for relative model behaviour under controlled conditions. The full TON_IoT and CICIoT2023 loaders ship in datasets.py — drop a CSV into data/ and the identical pipeline trains, selects, and live-scores real traffic."],
];

function pageHow() {
  main.innerHTML = `
    <div class="page-title">How It Works</div>
    <div class="page-sub">Every implementation detail in SentryMesh maps to a CSC701 (Rev-2019 'C' Scheme) syllabus module — plus the dataset decision, documented like the rest.</div>
    <div class="grid c2">${HOW.map(([t, m, d], i) => `
      <div class="how-card"><div class="how-num">${String(i + 1).padStart(2, "0")}</div>
        <div><span class="how-mod">${m}</span><h4>${t}</h4><p>${d}</p></div></div>`).join("")}</div>`;
}

go("overview");
