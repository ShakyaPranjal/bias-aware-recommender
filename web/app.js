/* Bias-Aware Recommender, dashboard logic. Author: Pranjal Man Shakya */
(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const pct = (x) => Math.round(100 * x);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const EMOTION_COLORS = { joy: "#D9902A", trust: "#4F8A5B", surprise: "#2C8C88", fear: "#5A4A86", sadness: "#3F5F93", anger: "#A8413F" };
  const STORE_KEY = "biasrec.myRatings.v1";

  const state = { mode: "user", userId: null, exploration: 0.55, data: null, catalog: [], featured: [], myRatings: loadRatings(), charts: {} };

  // ---------------------------------------------------------------- storage
  function loadRatings() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; } catch { return []; }
  }
  function saveRatings() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state.myRatings)); } catch { /* storage unavailable */ }
  }

  // ---------------------------------------------------------------- api
  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let msg = `Request failed (${res.status})`;
      try { msg = (await res.json()).detail || msg; } catch { /* keep default */ }
      throw new Error(msg);
    }
    return res.json();
  }

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => t.classList.remove("show"), 3200);
  }

  // ---------------------------------------------------------------- loading
  async function refresh() {
    document.body.classList.add("loading");
    try {
      if (state.mode === "custom") {
        state.data = await api("/api/analyze", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ratings: state.myRatings.map(({ movieId, rating }) => ({ movieId, rating })), exploration: state.exploration, k: 12 }),
        });
      } else {
        state.data = await api(`/api/users/${state.userId}/analysis?exploration=${state.exploration}&k=12`);
      }
      render();
    } catch (err) {
      toast(err.message);
    } finally {
      document.body.classList.remove("loading");
    }
  }

  let debounce;
  function onExplore(e) {
    state.exploration = e.target.value / 100;
    $("#explore-out").textContent = exploreLabel(state.exploration);
    clearTimeout(debounce);
    debounce = setTimeout(refresh, 220);
  }
  function exploreLabel(x) {
    if (x < 0.2) return "Stay comfortable";
    if (x < 0.45) return "Gentle nudge";
    if (x < 0.7) return "Balanced";
    if (x < 0.9) return "Push me";
    return "Full stretch";
  }

  // ---------------------------------------------------------------- render
  function render() {
    const d = state.data;
    const f = state.featured.find((u) => u.userId === state.userId);
    $("#viewer-name").textContent = state.mode === "custom" ? "Your profile"
      : `MovieLens viewer #${state.userId}${f ? `, ${f.label.toLowerCase()}` : ""}`;
    const strongest = [...d.biases].sort((a, b) => b.score - a.score)[0];
    $("#hero-lede").textContent = strongest.score >= 0.6
      ? `Strongest tendency: ${strongest.name.toLowerCase()}. ${strongest.evidence}`
      : "No strong bias stands out. The picks below keep your range wide.";

    $("#vitals").innerHTML = `
      <div><dt>films rated</dt><dd>${d.profile.n_ratings}</dd></div>
      <div><dt>average rating</dt><dd>${d.profile.mean_rating.toFixed(1)}★</dd></div>
      <div><dt>usual genres</dt><dd>${esc(d.profile.top_genres.slice(0, 2).join(", "))}</dd></div>`;

    const w = d.weights;
    $("#weights").textContent =
      `Score = ${w.alpha.toFixed(2)}·content + ${w.beta.toFixed(2)}·collaborative + ${w.gamma.toFixed(2)}·bias adjustment + ${w.delta.toFixed(2)}·serendipity`;

    renderMap(d);
    renderBiases(d);
    renderRecs(d);
    renderCompare(d);
    renderFingerprint(d);
  }

  function chartDefaults() {
    Chart.defaults.font.family = css("--sans");
    Chart.defaults.color = css("--muted");
    Chart.defaults.borderColor = css("--line");
  }

  function renderMap(d) {
    const pts = (arr) => arr.map(([x, y]) => ({ x, y }));
    const recPts = d.map.recommended.map(([x, y], i) => ({ x, y, title: d.recommendations[i]?.title }));
    const tradPts = d.map.traditional.map(([x, y], i) => ({ x, y, title: d.baseline[i]?.title }));
    const datasets = [
      { label: "Catalogue", order: 5, data: pts(state.catalog), pointRadius: 1.8, borderWidth: 0, pointBorderWidth: 0, pointBackgroundColor: css("--line"), backgroundColor: css("--line"), pointHoverRadius: 0 },
      { label: "Watched", order: 4, data: pts(d.map.watched), pointRadius: 4, borderWidth: 0, pointBorderWidth: 0, pointBackgroundColor: hexAlpha(css("--you"), 0.85), backgroundColor: hexAlpha(css("--you"), 0.85) },
      { label: "Traditional picks", order: 3, data: tradPts, pointRadius: 6, pointBackgroundColor: "transparent", pointBorderColor: css("--trad"), pointBorderWidth: 2, backgroundColor: "transparent", borderColor: css("--trad"), borderWidth: 2 },
      { label: "Bias-aware picks", order: 2, data: recPts, pointRadius: 7.5, pointHoverRadius: 10, pointBackgroundColor: css("--grow"), pointBorderColor: css("--surface"), pointBorderWidth: 2, backgroundColor: css("--grow"), borderColor: css("--surface"), borderWidth: 2 },
      { label: "Your centre", order: 1, data: [{ x: d.map.centroid[0], y: d.map.centroid[1], title: `Centre of your comfort zone: ${d.map.centre_title}` }], pointStyle: "crossRot", pointRadius: 9, pointHoverRadius: 11, borderColor: css("--ink"), borderWidth: 2.5 },
    ];
    if (state.charts.map) {
      state.charts.map.data.datasets.forEach((ds, i) => { ds.data = datasets[i].data; });
      state.charts.map.update();
      return;
    }
    state.charts.map = new Chart($("#map"), {
      type: "scatter",
      data: { datasets },
      options: {
        maintainAspectRatio: false, animation: { duration: 650, easing: "easeOutCubic" },
        layout: { padding: 14 },
        scales: { x: { display: false }, y: { display: false } },
        plugins: {
          legend: { display: false },
          tooltip: {
            filter: (item) => item.datasetIndex >= 2 && Boolean(item.raw.title),
            callbacks: { label: (item) => item.raw.title },
          },
        },
      },
    });
  }

  function hexAlpha(hex, a) {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }

  function levelColor(score) { return score >= 0.7 ? css("--high") : score >= 0.4 ? css("--mid") : css("--low"); }

  function renderBiases(d) {
    $("#bias-list").innerHTML = d.biases.map((b) => `
      <li class="bias-item">
        <div class="bias-name">${esc(b.name)} <span class="level level-${b.level}">${b.level}</span></div>
        <div class="bias-score" aria-label="percentile ${pct(b.score)}">${pct(b.score)}</div>
        <div class="bar" aria-hidden="true"><i style="width:${pct(b.score)}%;background:${levelColor(b.score)}"></i></div>
        <p class="bias-evidence">${esc(b.evidence)}</p>
      </li>`).join("");

    const labels = d.biases.map((b) => b.name.replace(" bias", ""));
    const after = d.metrics.bias_aware.list_bias_scores || {};
    const data = {
      labels,
      datasets: [
        { label: "Your history", data: d.biases.map((b) => pct(b.score)), borderColor: css("--you"), backgroundColor: css("--you-soft"), borderWidth: 2, pointRadius: 3 },
        { label: "Your recommendations", data: d.biases.map((b) => (b.key in after ? pct(after[b.key]) : null)), borderColor: css("--grow"), backgroundColor: "transparent", borderDash: [5, 4], borderWidth: 2, pointRadius: 2, spanGaps: true },
      ],
    };
    if (state.charts.radar) { state.charts.radar.data = data; state.charts.radar.update(); return; }
    state.charts.radar = new Chart($("#radar"), {
      type: "radar", data,
      options: {
        maintainAspectRatio: true,
        scales: { r: { min: 0, max: 100, ticks: { stepSize: 25, backdropColor: "transparent", display: false }, pointLabels: { font: { size: 12, weight: "500" }, color: css("--ink") }, grid: { color: css("--line") }, angleLines: { color: css("--line") } } },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, usePointStyle: true } } },
      },
    });
  }

  function renderRecs(d) {
    const w = d.weights;
    $("#recs").innerHTML = d.recommendations.map((r) => {
      const c = r.components;
      const taste = w.alpha * c.content + w.beta * c.cf;
      const growth = w.gamma * c.bias_adj + w.delta * c.serendipity;
      const tasteShare = Math.round((100 * taste) / (taste + growth || 1));
      const role = r.explanation.role.toLowerCase();
      const emo = EMOTION_COLORS[r.emotion] || "#777";
      return `
      <li class="rec">
        <div class="tone" style="background:linear-gradient(160deg, ${emo}, ${hexAlpha(emo, 0.55)})" title="Dominant emotion: ${esc(r.emotion)}">${esc(r.emotion)}</div>
        <div>
          <div class="rec-title"><h3>${esc(r.title)}</h3><span class="year">${r.year ?? ""}</span><span class="role-chip role-${role}">${esc(r.explanation.role)}</span></div>
          <p class="rec-genres">${esc(r.genres.join(", "))}</p>
          <p class="rec-reason">${esc(r.explanation.reason)}</p>
          <p class="rec-fit">${esc(r.explanation.fit)}</p>
          <div class="mix" aria-hidden="true"><i style="width:${tasteShare}%;background:var(--you)"></i><i style="width:${100 - tasteShare}%;background:var(--grow)"></i></div>
          <p class="mix-legend">${tasteShare}% taste match, ${100 - tasteShare}% growth</p>
          ${r.overview ? `<details class="overview"><summary>Plot</summary><p>${esc(r.overview)}</p></details>` : ""}
          ${state.mode === "custom" ? `<button class="btn btn-quiet add-btn" data-add="${r.movieId}" data-title="${esc(r.title)}" data-year="${r.year ?? ""}">I've seen this, add it</button>` : ""}
        </div>
      </li>`;
    }).join("");
  }

  function renderCompare(d) {
    const a = d.metrics.bias_aware, t = d.metrics.traditional;
    const rows = [
      ["Genre diversity after watching", "Normalised genre entropy of history + picks. Higher is broader.", a.genre_entropy_after, t.genre_entropy_after, true, (x) => x.toFixed(3)],
      ["Intra-list similarity", "How alike the picks are to each other. Lower is more varied.", a.intra_list_similarity, t.intra_list_similarity, false, (x) => x.toFixed(2)],
      ["Novelty", "Average self-information of the picks, in bits. Higher means less obvious.", a.novelty, t.novelty, true, (x) => x.toFixed(2)],
      ["Bias Reduction Index", "How strongly the list leans against your biases. Above 0 counters them.", a.bias_reduction_index, t.bias_reduction_index, true, (x) => (x >= 0 ? "+" : "") + x.toFixed(2)],
    ];
    $("#metrics").innerHTML = `
      <thead><tr><th scope="col">Metric</th><th scope="col" class="num">Bias-aware</th><th scope="col" class="num">Traditional</th></tr></thead>
      <tbody>${rows.map(([name, hint, va, vt, higher, fmt]) => {
        const win = higher ? va > vt : va < vt;
        return `<tr><th scope="row">${name}<span class="hint">${hint}</span></th><td class="num ${win ? "win" : ""}">${fmt(va)}</td><td class="num">${fmt(vt)}</td></tr>`;
      }).join("")}</tbody>`;
    $("#trad-list").innerHTML = d.baseline.map((m) => `<li><span>${esc(m.title)}</span> ${m.year ?? ""}</li>`).join("");
  }

  function renderFingerprint(d) {
    const rows = d.profile.genre_fingerprint.filter((g) => g.you > 0.005 || g.typical > 0.02 || g.recommended > 0.005)
      .sort((a, b) => b.you - a.you);
    const data = {
      labels: rows.map((g) => g.genre),
      datasets: [
        { label: "You", data: rows.map((g) => pct(g.you)), backgroundColor: css("--you"), borderRadius: 3 },
        { label: "Typical viewer", data: rows.map((g) => pct(g.typical)), backgroundColor: css("--trad"), borderRadius: 3 },
        { label: "Your picks", data: rows.map((g) => pct(g.recommended)), backgroundColor: css("--grow"), borderRadius: 3 },
      ],
    };
    if (state.charts.fp) { state.charts.fp.data = data; state.charts.fp.update(); }
    else {
      state.charts.fp = new Chart($("#fingerprint"), {
        type: "bar", data,
        options: {
          indexAxis: "y", maintainAspectRatio: false,
          scales: { x: { ticks: { callback: (v) => `${v}%` }, grid: { color: css("--line") } }, y: { grid: { display: false }, ticks: { color: css("--ink") } } },
          plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } }, tooltip: { callbacks: { label: (i) => `${i.dataset.label}: ${i.raw}%` } } },
          datasets: { bar: { barPercentage: 0.85, categoryPercentage: 0.8 } },
        },
      });
    }
    const emo = Object.entries(d.profile.emotions).sort((a, b) => b[1] - a[1]);
    const max = emo[0][1];
    $("#emotions").innerHTML = emo.map(([k, v]) => `
      <li><span>${k}</span><div class="bar"><i style="width:${(100 * v) / max}%;background:${EMOTION_COLORS[k]}"></i></div><span>${pct(v)}%</span></li>`).join("");
    $("#favs").innerHTML = d.profile.favourites.slice(0, 6).map((m) => `<li><span>${esc(m.title)}</span><span>${m.year ?? ""}</span></li>`).join("");
  }

  // ---------------------------------------------------------------- builder
  function renderRated() {
    $("#rated-count").textContent = state.myRatings.length;
    $("#analyze-me").disabled = state.myRatings.length < 5;
    $("#rated").innerHTML = state.myRatings.length ? state.myRatings.map((m, i) => `
      <li>
        <span>${esc(m.title)} <small>${m.year ?? ""}</small></span>
        <span class="stars" role="radiogroup" aria-label="Rating for ${esc(m.title)}">
          ${[1, 2, 3, 4, 5].map((s) => `<button type="button" role="radio" aria-checked="${s === m.rating}" aria-label="${s} star${s > 1 ? "s" : ""}" class="${s <= m.rating ? "on" : ""}" data-i="${i}" data-s="${s}">★</button>`).join("")}
          <button type="button" class="btn-icon" data-remove="${i}" aria-label="Remove ${esc(m.title)}">✕</button>
        </span>
      </li>`).join("") : `<li><small>No films yet. Search above and pick a few you've seen.</small></li>`;
  }

  let searchTimer;
  async function onSearch(e) {
    const q = e.target.value.trim();
    clearTimeout(searchTimer);
    if (!q) { $("#search-results").innerHTML = ""; return; }
    searchTimer = setTimeout(async () => {
      try {
        const { results } = await api(`/api/movies/search?q=${encodeURIComponent(q)}&limit=8`);
        const have = new Set(state.myRatings.map((m) => m.movieId));
        $("#search-results").innerHTML = results.length ? results.map((m) => `
          <li><button type="button" data-pick="${m.movieId}" data-title="${esc(m.title)}" data-year="${m.year ?? ""}" ${have.has(m.movieId) ? "disabled" : ""}>
            ${esc(m.title)} <small>${m.year ?? ""} · ${esc(m.genres.slice(0, 3).join(", "))}</small></button></li>`).join("")
          : `<li><small>No films match “${esc(q)}”.</small></li>`;
      } catch (err) { toast(err.message); }
    }, 180);
  }

  function addRating(movieId, title, year, rating = 4) {
    if (state.myRatings.some((m) => m.movieId === movieId)) return false;
    state.myRatings.push({ movieId, title, year: year ? Number(year) : null, rating });
    saveRatings();
    return true;
  }

  // ---------------------------------------------------------------- init
  async function init() {
    chartDefaults();
    $("#exploration").value = Math.round(state.exploration * 100);
    $("#explore-out").textContent = exploreLabel(state.exploration);
    $("#exploration").addEventListener("input", onExplore);

    try {
      const [meta, map] = await Promise.all([api("/api/meta"), api("/api/map?n=1500")]);
      state.featured = meta.featured_users;
      state.catalog = map.points;
      const opts = meta.featured_users.map((u) => `<option value="${u.userId}">Viewer ${u.userId}, ${esc(u.label.toLowerCase())}</option>`);
      opts.push(`<option value="custom">My own profile</option>`);
      $("#viewer-select").innerHTML = opts.join("");
      state.userId = meta.featured_users[0].userId;
      if (state.myRatings.length >= 5 && new URLSearchParams(location.search).get("me") === "1") {
        state.mode = "custom";
        $("#viewer-select").value = "custom";
      }
    } catch (err) {
      $("#viewer-name").textContent = "The recommender isn't running.";
      $("#hero-lede").textContent = `${err.message}. Start the server with "uvicorn biasrec.api:app" and reload.`;
      return;
    }

    $("#viewer-select").addEventListener("change", (e) => {
      if (e.target.value === "custom") {
        if (state.myRatings.length < 5) { openBuilder(); return; }
        state.mode = "custom";
      } else {
        state.mode = "user";
        state.userId = Number(e.target.value);
      }
      refresh();
    });

    const dlg = $("#builder");
    function openBuilder() { renderRated(); dlg.showModal(); $("#movie-search").focus(); }
    $("#open-builder").addEventListener("click", openBuilder);
    dlg.addEventListener("close", () => {
      if (state.mode !== "custom") $("#viewer-select").value = String(state.userId);
    });
    $("#movie-search").addEventListener("input", onSearch);
    $("#search-results").addEventListener("click", (e) => {
      const b = e.target.closest("[data-pick]");
      if (!b) return;
      addRating(Number(b.dataset.pick), b.dataset.title, b.dataset.year);
      b.disabled = true;
      renderRated();
    });
    $("#rated").addEventListener("click", (e) => {
      const star = e.target.closest("[data-s]");
      const rm = e.target.closest("[data-remove]");
      if (star) { state.myRatings[Number(star.dataset.i)].rating = Number(star.dataset.s); }
      if (rm) { state.myRatings.splice(Number(rm.dataset.remove), 1); }
      if (star || rm) { saveRatings(); renderRated(); }
    });
    $("#clear-ratings").addEventListener("click", () => { state.myRatings = []; saveRatings(); renderRated(); });
    $("#analyze-me").addEventListener("click", () => {
      state.mode = "custom";
      $("#viewer-select").value = "custom";
      dlg.close();
      refresh();
    });
    $("#recs").addEventListener("click", (e) => {
      const b = e.target.closest("[data-add]");
      if (!b) return;
      if (addRating(Number(b.dataset.add), b.dataset.title, b.dataset.year, 4)) {
        toast(`Added ${b.dataset.title} at 4★. Rate it in “Build my own profile”.`);
        refresh();
      }
    });

    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      Object.values(state.charts).forEach((c) => c.destroy());
      state.charts = {};
      chartDefaults();
      if (state.data) render();
    });

    refresh();
  }

  init();
})();
