/* Bias-Aware Recommender, in-browser inference engine. Author: Pranjal Man Shakya
 *
 * A line-by-line port of the inference path in biasrec/ (bias.py, models.py,
 * engine.py, explain.py). The model is trained in Python and exported by
 * scripts/build_static.py; this file loads it and answers the same API the
 * FastAPI server does, so the dashboard runs with no server at all.
 * tests/test_static_parity.py checks that both give identical results.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === "object" && module.exports) module.exports = mod;
  else root.BiasRecStatic = mod.browser();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const BIAS_INFO = {
    negativity: ["Negativity bias", "Prefers dark, tragic or violent stories over hopeful ones."],
    novelty_avoidance: ["Novelty avoidance", "Stays in the same genres, eras and mainstream titles."],
    confirmation: ["Confirmation bias", "Keeps returning to the same themes and rates them highest."],
    recency: ["Recency bias", "Favours new releases and rates recent watches more generously."],
    hero_projection: ["Hero projection", "Gravitates to hero / saviour arcs that mirror an ideal self."],
    comfort_zone: ["Comfort zone", "History sits in a tight cluster of very similar films."],
  };
  const BIAS_KEYS = Object.keys(BIAS_INFO);
  const NEGATIVE_VALENCE = -0.15;

  // ------------------------------------------------------------ numerics
  const sum = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
  const mean = (a) => sum(a) / a.length;
  const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  function std(a) { const m = mean(a); let s = 0; for (const x of a) s += (x - m) * (x - m); return Math.sqrt(s / a.length); }
  function median(a) {
    const s = Float64Array.from(a).sort(); const n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  function argmax(a) { let j = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[j]) j = i; return j; }
  /** indices sorting x high to low, ties in original order (numpy desc_order) */
  function descOrder(x) { return Array.from(x, (_, i) => i).sort((a, b) => (x[b] - x[a]) || (a - b)); }
  function ascOrder(x) { return Array.from(x, (_, i) => i).sort((a, b) => (x[a] - x[b]) || (a - b)); }
  function minmax(x) {
    let lo = Infinity, hi = -Infinity;
    for (const v of x) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const out = new Float64Array(x.length);
    if (hi - lo < 1e-12) return out;
    for (let i = 0; i < x.length; i++) out[i] = (x[i] - lo) / (hi - lo);
    return out;
  }
  function entropy(p) {
    let tot = 0; for (const v of p) if (v > 0) tot += v;
    if (tot === 0) return 0;
    let h = 0; for (const v of p) if (v > 0) { const q = v / tot; h -= q * Math.log(q); }
    return h;
  }
  const normalizedEntropy = (counts, n) => (n <= 1 ? 0 : entropy(counts) / Math.log(n));
  function rankdata(a) {
    const order = ascOrder(a); const ranks = new Float64Array(a.length);
    let i = 0;
    while (i < a.length) {
      let j = i;
      while (j + 1 < a.length && a[order[j + 1]] === a[order[i]]) j++;
      for (let t = i; t <= j; t++) ranks[order[t]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return ranks;
  }
  function corrcoef(a, b) {
    const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return sab / Math.sqrt(saa * sbb);
  }
  function spearman(a, b) {
    if (a.length < 3 || std(a) === 0 || std(b) === 0) return 0;
    return corrcoef(rankdata(a), rankdata(b));
  }
  /** numpy.quantile, method="linear", reproduced operation for operation */
  function quantile(values, q) {
    const a = Float64Array.from(values).sort(); const n = a.length;
    const vi = n * q + (1 + q * (1 - 1 - 1)) - 1;
    let prev = Math.floor(vi), next = prev + 1;
    if (vi >= n - 1) { prev = n - 1; next = n - 1; }
    if (vi < 0) { prev = 0; next = 0; }
    const g = vi - Math.floor(vi);
    const lo = a[prev], hi = a[next], diff = hi - lo;
    return g >= 0.5 ? hi - diff * (1 - g) : lo + diff * g;
  }
  /** Python's round() to an integer: ties go to the even neighbour */
  function pyRound(x) { const f = Math.floor(x); const d = x - f; return d === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(x); }
  /** Python's f"{x:.{nd}f}": exact decimal value, exact ties go to even */
  function fixed(x, nd) {
    const exact = Math.abs(x).toFixed(100).split(".")[1];   // exact binary value, up to 100 digits
    const isTie = exact[nd] === "5" && /^0*$/.test(exact.slice(nd + 1));
    if (!isTie) return x.toFixed(nd);
    const down = (Math.sign(x) * Math.floor(Math.abs(x) * 10 ** nd)) / 10 ** nd;
    const lastDigit = Number(down.toFixed(nd).slice(-1));
    return lastDigit % 2 === 0 ? down.toFixed(nd) : x.toFixed(nd);
  }
  const round2 = (x) => Number(fixed(x, 2));
  const round4 = (x) => Number(fixed(x, 4));
  const pct = (x) => `${pyRound(100 * x)}%`;
  function join(items) {
    items = items.filter(Boolean);
    if (items.length <= 1) return items.join("");
    return items.slice(0, -1).join(", ") + " and " + items[items.length - 1];
  }

  // --------------------------------------------------------------- engine
  class Engine {
    constructor(meta, movies, buffer) {
      this.meta = meta;
      this.mv = movies;
      const B = (name) => {
        const { offset, dtype, shape } = meta.binary[name];
        const T = { float32: Float32Array, uint8: Uint8Array, uint16: Uint16Array, int32: Int32Array, uint32: Uint32Array }[dtype];
        const len = shape.reduce((p, c) => p * c, 1);
        return { data: new T(buffer, offset, len), shape };
      };
      const mat = (name) => { const { data, shape } = B(name); return { data: Float64Array.from(data), rows: shape[0], cols: shape[1] || 1 }; };
      this.n = movies.movieId.length;
      this.E = mat("embedding"); this.V = mat("cf_V"); this.T = mat("themes"); this.EM = mat("emotions");
      this.G = mat("genre_matrix"); this.map = mat("map");
      this.valence = Float64Array.from(B("valence").data);
      this.hero = Float64Array.from(B("hero").data);
      this.genres = meta.genres; this.themeNames = meta.themes; this.emotionNames = meta.emotions;

      const u = B("user_ids").data, off = B("user_offsets").data;
      this.ridx = B("r_idx").data; this.rhalf = B("r_half_stars").data; this.rts = B("r_ts").data;
      this.userRange = new Map();
      for (let i = 0; i < u.length; i++) this.userRange.set(u[i], [off[i], off[i + 1]]);

      this.years = Float64Array.from(movies.yearFilled);
      this.decades = this.years.map((y) => Math.floor(y / 10) * 10);
      this.decadeValues = meta.decade_values;
      const nr = movies.nRatings; let maxN = 1; for (const x of nr) if (x > maxN) maxN = x;
      this.pop = Float64Array.from(nr, (x) => Math.log1p(x) / Math.log1p(maxN));
      this.support = Uint8Array.from(nr, (x) => (x >= meta.min_support ? 1 : 0));
      this.qualityNorm = minmax(movies.quality);
      this.population = meta.population;
      this.idByMovie = new Map(movies.movieId.map((id, i) => [id, i]));
      this.titleLower = movies.title.map((t) => t.toLowerCase());

      // CF item factors for "viewers who liked X" anchors
      const k = this.V.cols, sq = meta.cf_s.map(Math.sqrt);
      this.itemF = new Float64Array(this.n * k);
      for (let i = 0; i < this.n; i++) {
        let nrm = 0;
        for (let c = 0; c < k; c++) { const v = this.V.data[i * k + c] * sq[c]; this.itemF[i * k + c] = v; nrm += v * v; }
        nrm = Math.sqrt(nrm) || 1;
        for (let c = 0; c < k; c++) this.itemF[i * k + c] /= nrm;
      }
    }

    row(M, i) { return M.data.subarray(i * M.cols, (i + 1) * M.cols); }
    dotRows(M, i, j) { let s = 0; const c = M.cols, a = i * c, b = j * c; for (let t = 0; t < c; t++) s += M.data[a + t] * M.data[b + t]; return s; }
    colSums(M, idx) { const out = new Float64Array(M.cols); for (const i of idx) for (let c = 0; c < M.cols; c++) out[c] += M.data[i * M.cols + c]; return out; }
    colMeans(M, idx) { const s = this.colSums(M, idx); return s.map((v) => v / idx.length); }

    // ------------------------------------------------------- bias engine
    rawMetrics(idx, ratings, ts) {
      const d = {};
      const gc = this.colSums(this.G, idx);
      const val = idx.map((i) => this.valence[i]);
      const nb = mean(val.map((v) => (v < NEGATIVE_VALENCE ? 1 : 0)));
      d.negative_share = nb; d.mean_valence = mean(val); d.emotions = this.colMeans(this.EM, idx);

      const hGenre = normalizedEntropy(gc, this.G.cols);
      const decCounts = this.decadeValues.map((dv) => idx.reduce((s, i) => s + (this.decades[i] === dv ? 1 : 0), 0));
      const hEra = normalizedEntropy(decCounts, Math.min(this.decadeValues.length, 8));
      const mainstream = mean(idx.map((i) => this.pop[i]));
      const noveltyAvoid = (1 - hGenre) * 0.4 + (1 - Math.min(hEra, 1)) * 0.25 + mainstream * 0.35;
      Object.assign(d, { genre_entropy: hGenre, era_entropy: Math.min(hEra, 1), mainstream, genre_counts: gc, decade_counts: decCounts });

      const nT = this.T.cols;
      const dominant = idx.map((i) => argmax(this.row(this.T, i)));
      const themeCounts = new Float64Array(nT); for (const t of dominant) themeCounts[t]++;
      const themeConc = 1 - normalizedEntropy(themeCounts, nT);
      const topThemes = descOrder(themeCounts).slice(0, 3);
      const inside = dominant.map((t) => topThemes.includes(t));
      let gap = 0;
      if (inside.some(Boolean) && inside.some((x) => !x)) {
        gap = mean(ratings.filter((_, i) => inside[i])) - mean(ratings.filter((_, i) => !inside[i]));
      }
      const confirmation = 0.6 * themeConc + 0.4 * clip(gap / 1.5, 0, 1);
      Object.assign(d, { theme_concentration: themeConc, theme_rating_gap: gap, top_themes: topThemes, theme_counts: themeCounts });

      let drift = 0, fresh = 0;
      if (ts && ts.length === idx.length) {
        drift = Math.max(0, spearman(ts, ratings));
        fresh = mean(idx.map((i, t) => (1970 + ts[t] / (365.25 * 86400) - this.years[i] <= 2.5 ? 1 : 0)));
      }
      const recency = 0.45 * drift + 0.55 * fresh;
      Object.assign(d, { rating_drift: drift, fresh_release_share: fresh, median_year: median(idx.map((i) => this.years[i])) });

      const w = ratings.map((r) => Math.max(r, 0.5));
      const hero = sum(idx.map((i, t) => this.hero[i] * w[t])) / sum(w);
      d.hero_share = mean(idx.map((i) => (this.hero[i] >= 0.5 ? 1 : 0)));

      const c = this.colMeans(this.E, idx);
      const cz = mean(idx.map((i) => { const r = this.row(this.E, i); let s = 0; for (let t = 0; t < r.length; t++) s += (r[t] - c[t]) ** 2; return Math.sqrt(s); }));
      d.comfort_index = cz;

      return { raw: { negativity: nb, novelty_avoidance: noveltyAvoid, confirmation, recency, hero_projection: hero, comfort_zone: cz }, details: d };
    }

    percentile(key, value) {
      const pop = this.population[key];
      if (!pop || !pop.length) return 0.5;
      let lo = 0, hi = pop.length;                       // searchsorted(side="right")
      while (lo < hi) { const mid = (lo + hi) >> 1; if (pop[mid] <= value + 1e-9) lo = mid + 1; else hi = mid; }
      const p = lo / pop.length;
      return key === "comfort_zone" ? 1 - p : p;
    }
    typical(key) { return median(this.population[key]); }

    analyzeProfile(idx, ratings, ts) {
      const { raw, details } = this.rawMetrics(idx, ratings, ts);
      details.scores = Object.fromEntries(BIAS_KEYS.map((k) => [k, this.percentile(k, raw[k])]));
      return { idx, ratings, ts, raw, details };
    }
    strengths(p) { return Object.fromEntries(BIAS_KEYS.map((k) => [k, clip((p.details.scores[k] - 0.35) / 0.65, 0, 1) ** 1.3])); }

    counterVectors(p) {
      const d = p.details, n = this.n, G = this.G;
      const positivity = this.valence.map((v) => clip((v - d.mean_valence) / 0.9, 0, 1));

      const gs = d.genre_counts.map((v) => v / Math.max(sum(d.genre_counts), 1));
      const gmax = Math.max(Math.max(...gs), 1e-9);
      const gfam = gs.map((v) => v / gmax);
      const ds = d.decade_counts.map((v) => v / Math.max(sum(d.decade_counts), 1));
      const dmax = Math.max(Math.max(...ds), 1e-9);
      const decLookup = new Map(this.decadeValues.map((dv, i) => [dv, ds[i] / dmax]));
      const novelty = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let fam = 0, cnt = 0;
        for (let c = 0; c < G.cols; c++) { const g = G.data[i * G.cols + c]; fam += g * gfam[c]; cnt += g; }
        const gn = 1 - fam / Math.max(cnt, 1);
        const en = 1 - (decLookup.get(this.decades[i]) ?? 0);
        novelty[i] = clip(0.5 * gn + 0.2 * en + 0.3 * (1 - this.pop[i]), 0, 1);
      }

      const tp = this.colMeans(this.T, p.idx);
      const tn = Math.sqrt(sum(tp.map((v) => v * v))) + 1e-9;
      const tpn = tp.map((v) => v / tn);
      const td = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const r = this.row(this.T, i); let nrm = 0, dot = 0;
        for (let c = 0; c < r.length; c++) nrm += r[c] * r[c];
        nrm = Math.sqrt(nrm) + 1e-9;
        for (let c = 0; c < r.length; c++) dot += (r[c] / nrm) * tpn[c];
        td[i] = 1 - dot;
      }
      let tdMax = -Infinity; for (const v of td) if (v > tdMax) tdMax = v;
      const confirmation = td.map((v) => clip(v / (tdMax + 1e-9), 0, 1));

      const older = this.years.map((y) => clip((d.median_year - y) / 30, 0, 1));

      const cen = this.colMeans(this.E, p.idx);
      const dist = new Float64Array(n);
      for (let i = 0; i < n; i++) { const r = this.row(this.E, i); let s = 0; for (let t = 0; t < r.length; t++) s += (r[t] - cen[t]) ** 2; dist[i] = Math.sqrt(s); }
      let dmin = Infinity, dmx = -Infinity; for (const v of dist) { if (v < dmin) dmin = v; if (v > dmx) dmx = v; }
      const comfort = dist.map((v) => (v - dmin) / (dmx - dmin + 1e-9));

      return { negativity: positivity, novelty_avoidance: novelty, confirmation, recency: older,
        hero_projection: this.hero.map((h) => 1 - h), comfort_zone: comfort };
    }

    // ------------------------------------------------------------ models
    contentScore(idx, ratings) {
      const m = mean(ratings);
      let w = ratings.map((r) => r - m);
      if (w.every((x) => Math.abs(x) <= 1e-8)) w = ratings.map(() => 1);
      w = w.map((x) => (x > 0 ? x : 0.35 * x));
      const k = this.E.cols, v = new Float64Array(k);
      idx.forEach((i, t) => { const r = this.row(this.E, i); for (let c = 0; c < k; c++) v[c] += w[t] * r[c]; });
      const nrm = Math.sqrt(sum(v.map((x) => x * x)));
      if (nrm > 0) for (let c = 0; c < k; c++) v[c] /= nrm;
      const sims = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) { const r = this.row(this.E, i); let s = 0; for (let c = 0; c < k; c++) s += r[c] * v[c]; sims[i] = s; }
      return minmax(sims);
    }
    cfRaw(idx, ratings) {
      const k = this.V.cols, z = new Float64Array(k);
      idx.forEach((i, t) => { const r = this.row(this.V, i); for (let c = 0; c < k; c++) z[c] += ratings[t] * r[c]; });
      const out = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) { const r = this.row(this.V, i); let s = 0; for (let c = 0; c < k; c++) s += r[c] * z[c]; out[i] = s; }
      return out;
    }
    mostSimilarRated(target, idx, ratings) {
      const liked = idx.filter((_, t) => ratings[t] >= 3.5);
      if (!liked.length) return null;
      const k = this.V.cols; let best = -Infinity, bj = 0;
      liked.forEach((i, j) => { let s = 0; for (let c = 0; c < k; c++) s += this.itemF[i * k + c] * this.itemF[target * k + c]; if (s > best) { best = s; bj = j; } });
      return [liked[bj], best];
    }

    // ------------------------------------------------------------ scoring
    static weights(x) {
      x = clip(x, 0, 1);
      return { alpha: 0.12 * (1 - x) + 0.05, beta: 0.68 * (1 - x) + 0.15, gamma: 0.45 * x + 0.05, delta: 0.25 * x + 0.05 };
    }
    eligible(p) { const m = Uint8Array.from(this.support); for (const i of p.idx) m[i] = 0; return m; }
    rankNorm(x, mask) {
      const ids = []; for (let i = 0; i < x.length; i++) if (mask[i]) ids.push(i);
      const out = new Float64Array(x.length);
      ids.sort((a, b) => (x[a] - x[b]) || (a - b)).forEach((i, r) => { out[i] = (r + 1) / ids.length; });
      return out;
    }
    scores(p, w, x) {
      const elig = this.eligible(p);
      const content = this.rankNorm(this.contentScore(p.idx, p.ratings), elig);
      const cf = this.rankNorm(this.cfRaw(p.idx, p.ratings), elig);
      const strengths = this.strengths(p), counters = this.counterVectors(p);
      const total = sum(Object.values(strengths));
      const biasAdj = new Float64Array(this.n);
      if (total > 1e-9) for (let i = 0; i < this.n; i++) { let s = 0; for (const k of BIAS_KEYS) s += strengths[k] * counters[k][i]; biasAdj[i] = s / total; }
      const stretch = counters.comfort_zone, target = 0.30 + 0.30 * x;
      const ser = new Float64Array(this.n), score = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) {
        ser[i] = this.qualityNorm[i] * Math.exp(-((stretch[i] - target) ** 2) / (2 * 0.14 ** 2));
        score[i] = w.alpha * content[i] + w.beta * cf[i] + w.gamma * biasAdj[i] + w.delta * ser[i];
      }
      return { score, content, cf, bias_adj: biasAdj, serendipity: ser, stretch, strengths, counters, elig };
    }
    mmr(score, candidates, k, lam) {
      const pool = descOrder(candidates.map((i) => score[i])).slice(0, 400).map((j) => candidates[j]);
      if (!pool.length) return [];
      const s = minmax(pool.map((i) => score[i]));
      const maxSim = new Float64Array(pool.length), avail = new Uint8Array(pool.length).fill(1), chosen = [];
      for (let it = 0; it < Math.min(k, pool.length); it++) {
        let j = -1, best = -Infinity;
        for (let t = 0; t < pool.length; t++) { if (!avail[t]) continue; const v = (1 - lam) * s[t] - lam * maxSim[t]; if (j < 0 || v > best) { best = v; j = t; } }
        chosen.push(pool[j]); avail[j] = 0;
        for (let t = 0; t < pool.length; t++) maxSim[t] = Math.max(maxSim[t], this.dotRows(this.E, pool[t], pool[j]));
      }
      return chosen;
    }
    recommend(p, k, x) {
      const w = Engine.weights(x), S = this.scores(p, w, x), elig = S.elig;
      const rel = S.cf, eligVals = [];
      for (let i = 0; i < this.n; i++) if (elig[i]) eligVals.push(rel[i]);
      const thresh = eligVals.length ? quantile(eligVals, 1 - (0.05 + 0.40 * x)) : 0;
      const cand = []; for (let i = 0; i < this.n; i++) if (elig[i] && rel[i] >= thresh) cand.push(i);
      return { picks: this.mmr(S.score, cand, k, 0.30 + 0.25 * x), S, w };
    }
    baseline(p, k) {
      const cf = this.cfRaw(p.idx, p.ratings), elig = this.eligible(p);
      for (let i = 0; i < this.n; i++) if (!elig[i]) cf[i] = -Infinity;
      return descOrder(cf).slice(0, k);
    }

    // ------------------------------------------------------------ metrics
    biasReduction(p, recs) {
      if (recs.length < 3) return [0, {}];
      const lst = this.analyzeProfile(recs, recs.map(() => 4.0), null).details.scores;
      let w = this.strengths(p); const keys = BIAS_KEYS.filter((k) => k !== "recency");
      let total = sum(keys.map((k) => w[k]));
      if (total < 1e-9) { w = Object.fromEntries(keys.map((k) => [k, 1])); total = keys.length; }
      return [sum(keys.map((k) => w[k] * (p.details.scores[k] - lst[k]))) / total, lst];
    }
    listMetrics(p, recs) {
      const nG = this.G.cols;
      const before = normalizedEntropy(this.colSums(this.G, p.idx), nG);
      const after = normalizedEntropy(this.colSums(this.G, [...p.idx, ...recs]), nG);
      let ils = 0;
      if (recs.length > 1) { let s = 0, c = 0; for (let i = 0; i < recs.length; i++) for (let j = i + 1; j < recs.length; j++) { s += this.dotRows(this.E, recs[i], recs[j]); c++; } ils = s / c; }
      const novelty = recs.length ? mean(recs.map((i) => -Math.log2(clip(this.mv.nRatings[i] / this.meta.n_users, 1e-6, 1)))) : 0;
      const [bri, lst] = this.biasReduction(p, recs);
      return { genre_entropy_before: before, genre_entropy_after: after, intra_list_similarity: ils, novelty, bias_reduction_index: bri, list_bias_scores: lst };
    }

    // -------------------------------------------------------- explanation
    topGenres(p, n = 3) { const c = p.details.genre_counts; return descOrder(c).slice(0, n).filter((i) => c[i] > 0).map((i) => this.genres[i]); }
    biasEvidence(key, p) {
      const d = p.details;
      if (key === "negativity") return `${pct(d.negative_share)} of your history has a dark or negative tone (typical viewer: ${pct(this.typical("negativity"))}).`;
      if (key === "novelty_avoidance") return `Your viewing is concentrated in ${join(this.topGenres(p, 2))}, genre diversity is ${pct(d.genre_entropy)} of the maximum, and ${pct(d.mainstream)} of your picks are mainstream hits.`;
      if (key === "confirmation") {
        const theme = this.themeNames[d.top_themes[0]], gap = d.theme_rating_gap;
        return `You keep returning to themes like “${theme}”.` + (gap > 0.15 ? ` You rate those themes ${fixed(gap, 1)}★ higher than everything else.` : "");
      }
      if (key === "recency") {
        const drift = d.rating_drift;
        const trend = drift >= 0.1 ? `, and your ratings drift upward over time (ρ=${fixed(drift, 2)})` : ", and your ratings stay steady over time";
        if (!p.ts) return "Watch dates are needed to measure recency bias; profiles built here don't have them.";
        return `${pct(d.fresh_release_share)} of your watches were new releases at the time${trend}.`;
      }
      if (key === "hero_projection") return `${pct(d.hero_share)} of what you watch centres on a hero or saviour arc` + (d.scores.hero_projection >= 0.5 ? ", and you rate those films highly." : ".");
      if (key === "comfort_zone") return `Your comfort-zone index is ${fixed(d.comfort_index, 2)} (typical viewer: ${fixed(this.typical("comfort_zone"), 2)}); lower means your films sit close together in story space.`;
      return "";
    }
    explainRec(m, p, comp, contrib, used) {
      const mv = this.mv, title = mv.title[m];
      const anchor = this.mostSimilarRated(m, p.idx, p.ratings);
      const anchorTitle = anchor ? mv.title[anchor[0]] : null;
      let main = BIAS_KEYS.reduce((a, b) => (contrib[b] > contrib[a] ? b : a), BIAS_KEYS[0]);
      if (contrib[main] < 0.06) main = null;
      const variant = main ? used[main] || 0 : 0;
      const valence = this.valence[m];
      const lead = this.emotionNames[argmax(this.row(this.EM, m))];
      const genres = mv.genres[m].map((g) => this.genres[g]);
      const userTop = this.topGenres(p, 3);
      const newG = genres.filter((g) => !userTop.includes(g)).map((g) => g.toLowerCase());
      const decade = mv.year[m] == null ? "an earlier era" : `${Math.floor(mv.year[m] / 10) * 10}s`;
      const theme = this.themeNames[argmax(this.row(this.T, m))];
      const tone = valence > 0.25 ? "hopeful" : valence > 0 ? "lighter" : "less bleak";
      const like = anchorTitle ? ` while keeping the depth of ${anchorTitle}` : "";
      const options = {
        negativity: [
          `You often watch dark, heavy films. ${title} offers a ${tone} story led by ${lead}${like}.`,
          `Most of your history is intense. This one runs on ${lead} instead, a real change of emotional register.`,
          `A ${tone} counterweight to your darker picks, without losing substance.`,
        ],
        novelty_avoidance: [
          newG.length ? `You rarely step outside ${join(userTop.slice(0, 2))}. This ${join(newG.slice(0, 2))} pick from the ${decade} widens your range without feeling disconnected.`
            : `A less-travelled title from the ${decade} that people with your taste rate well.`,
          newG.length ? `${title} brings ${join(newG.slice(0, 2))} into a history that is mostly ${join(userTop.slice(0, 2))}.`
            : "Fewer people have seen this one, but it rewards viewers with your taste.",
          `A ${decade} film in a direction your history hasn't taken yet.`,
        ],
        confirmation: [
          `It explores “${theme}”, a perspective that rarely appears in your history, so it challenges rather than confirms your usual worldview.`,
          `Built around “${theme}”, which your ratings barely touch. Worth seeing the world from that angle.`,
        ],
        recency: [
          `You lean toward new releases. This ${decade} film has stood the test of time and shows you a different era of filmmaking.`,
          `Older than almost everything you rate, and a reminder that the ${decade} still have something to say.`,
        ],
        hero_projection: [
          "Instead of a lone hero saving the day, this story is about ordinary, flawed people: a mirror rather than an ideal.",
          "No chosen one here. The drama comes from people making hard choices at human scale.",
        ],
        comfort_zone: [
          "It sits just outside your usual cluster of films: familiar enough to enjoy, different enough to stretch you.",
          anchorTitle ? `Different territory in story space, connected to your taste through ${anchorTitle}.`
            : "Different territory in story space, but close enough to your taste to land.",
          "A step sideways from what you usually watch: new ground, similar craft.",
        ],
      };
      const reason = main ? options[main][variant % options[main].length]
        : "A strong match for your taste" + (anchorTitle ? `, close to ${anchorTitle}.` : ".");
      const q = mv.quality[m];
      const fit = anchorTitle && main ? `Why it still fits: viewers who liked ${anchorTitle} rate it highly (${fixed(q, 1)}★ average).`
        : `Rated ${fixed(q, 1)}★ on average by ${mv.nRatings[m]} viewers.`;
      const st = comp.stretch;
      return { reason, fit, counters: main, role: st < 0.42 ? "Bridge" : st > 0.62 ? "Stretch" : "Expand" };
    }

    // ------------------------------------------------------------- output
    card(m) {
      const mv = this.mv;
      return { movieId: mv.movieId[m], title: mv.title[m], year: mv.year[m], genres: mv.genres[m].map((g) => this.genres[g]),
        overview: mv.overview[m] || "", rating: round2(mv.meanRating[m]), n_ratings: mv.nRatings[m],
        valence: round2(this.valence[m]), emotion: this.emotionNames[argmax(this.row(this.EM, m))] };
    }
    pt(i) { return [round4(this.map.data[i * 2]), round4(this.map.data[i * 2 + 1])]; }

    analyze(p, k = 12, x = 0.5) {
      const { picks, S, w } = this.recommend(p, k, x);
      const base = this.baseline(p, k);
      const used = Object.fromEntries(BIAS_KEYS.map((b) => [b, 0]));
      const recs = picks.map((m) => {
        const contrib = Object.fromEntries(BIAS_KEYS.map((b) => [b, S.strengths[b] * S.counters[b][m] * 0.6 ** used[b]]));
        const comp = { content: S.content[m], cf: S.cf[m], bias_adj: S.bias_adj[m], serendipity: S.serendipity[m], stretch: S.stretch[m] };
        const expl = this.explainRec(m, p, comp, contrib, used);
        if (expl.counters) used[expl.counters]++;
        return { ...this.card(m), score: S.score[m], components: comp, explanation: expl };
      });
      const sc = p.details.scores;
      const biases = BIAS_KEYS.map((b) => ({ key: b, name: BIAS_INFO[b][0], description: BIAS_INFO[b][1], score: sc[b], raw: p.raw[b],
        level: sc[b] >= 0.7 ? "high" : sc[b] >= 0.4 ? "moderate" : "low", evidence: this.biasEvidence(b, p) }));
      const d = p.details;
      const userShare = d.genre_counts.map((v) => v / Math.max(sum(d.genre_counts), 1));
      const recCounts = this.colSums(this.G, picks);
      const recShare = recCounts.map((v) => v / Math.max(sum(recCounts), 1));
      const liked = descOrder(p.ratings).slice(0, 8).map((t) => p.idx[t]);
      const E = this.E, c = this.colMeans(E, p.idx);
      const centre = p.idx[argmax(p.idx.map((i) => { const r = this.row(E, i); let s = 0; for (let t = 0; t < r.length; t++) s += r[t] * c[t]; return s; }))];
      return {
        profile: {
          n_ratings: p.idx.length, mean_rating: round2(mean(p.ratings)), top_genres: this.topGenres(p, 3),
          favourites: liked.map((i) => this.card(i)),
          emotions: Object.fromEntries(this.emotionNames.map((e, i) => [e, d.emotions[i]])),
          genre_fingerprint: this.genres.map((g, i) => ({ genre: g, you: userShare[i], typical: this.meta.catalog_share[i], recommended: recShare[i] })),
        },
        biases, weights: w, exploration: x, recommendations: recs,
        baseline: base.map((m) => this.card(m)),
        metrics: { bias_aware: this.listMetrics(p, picks), traditional: this.listMetrics(p, base) },
        map: { watched: p.idx.slice(0, 400).map((i) => this.pt(i)), recommended: picks.map((i) => this.pt(i)),
          traditional: base.map((i) => this.pt(i)), centroid: this.pt(centre), centre_title: this.mv.title[centre] },
      };
    }

    userProfile(uid) {
      const r = this.userRange.get(uid);
      if (!r) return null;
      const idx = [], ratings = [], ts = [];
      for (let i = r[0]; i < r[1]; i++) { idx.push(this.ridx[i]); ratings.push(this.rhalf[i] / 2); ts.push(this.rts[i]); }
      return this.analyzeProfile(idx, ratings, ts);
    }
    customProfile(items) {
      const pairs = items.filter((it) => this.idByMovie.has(Number(it.movieId))).map((it) => [this.idByMovie.get(Number(it.movieId)), Number(it.rating)]);
      if (!pairs.length) throw new Error("none of the rated movies are in the catalogue");
      return this.analyzeProfile(pairs.map((x) => x[0]), pairs.map((x) => x[1]), null);
    }
    search(q, limit = 12) {
      q = q.toLowerCase().trim();
      if (!q) return [];
      const hits = []; this.titleLower.forEach((t, i) => { if (t.includes(q)) hits.push(i); });
      return descOrder(hits.map((i) => this.mv.nRatings[i])).slice(0, limit).map((j) => this.card(hits[j]));
    }

    /** Same routes and payloads as biasrec/api.py */
    handle(path, opts = {}) {
      const url = new URL(path, "http://local");
      const qp = (k, d) => (url.searchParams.has(k) ? Number(url.searchParams.get(k)) : d);
      const fail = (status, detail) => { const e = new Error(detail); e.status = status; throw e; };
      if (url.pathname === "/api/health") return { status: "ok", version: "1.0.0", loaded: true, mode: "static" };
      if (url.pathname === "/api/meta") {
        const m = this.meta;
        return { n_movies: m.n_movies, n_ratings: m.n_ratings, n_users: m.n_users, genres: m.genres, themes: m.themes, featured_users: m.featured_users, with_overview: m.with_overview };
      }
      if (url.pathname === "/api/map") return { points: this.meta.catalog_map.slice(0, qp("n", 1500)).map((i) => this.pt(i)) };
      if (url.pathname === "/api/movies/search") return { results: this.search(url.searchParams.get("q") || "", qp("limit", 12)) };
      const um = url.pathname.match(/^\/api\/users\/(\d+)\/analysis$/);
      if (um) {
        const p = this.userProfile(Number(um[1]));
        if (!p) fail(404, `user ${um[1]} not found`);
        return { ...this.analyze(p, qp("k", 12), qp("exploration", 0.5)), userId: Number(um[1]) };
      }
      if (url.pathname === "/api/analyze") {
        const body = JSON.parse(opts.body || "{}");
        if (!body.ratings || body.ratings.length < 3) fail(422, "Rate at least 3 films.");
        let p;
        try { p = this.customProfile(body.ratings); } catch (e) { fail(400, e.message); }
        return this.analyze(p, body.k ?? 12, body.exploration ?? 0.5);
      }
      fail(404, "Not found");
    }
  }

  async function load(base) {
    const get = (f) => fetch(base + f).then((r) => { if (!r.ok) throw new Error(`Could not load ${f} (${r.status})`); return r; });
    // model.bin, or model.b64.txt on hosts that only serve text files
    const model = fetch(base + "model.bin").then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error("no bin"))))
      .catch(() => get("model.b64.txt").then((r) => r.text()).then((t) => {
        const s = atob(t.trim()), bytes = new Uint8Array(s.length);
        for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
        return bytes.buffer;
      }));
    const [meta, movies, buf] = await Promise.all([get("meta.json").then((r) => r.json()), get("movies.json").then((r) => r.json()), model]);
    return new Engine(meta, movies, buf);
  }

  return {
    Engine, load, quantile, rankdata, pyRound, fixed,
    browser() {
      const script = document.currentScript;
      const base = (script && script.dataset.model) || "data/";
      let ready = null;
      return {
        ready: () => (ready = ready || load(base)),
        async request(path, opts) {
          const engine = await (ready = ready || load(base));
          await new Promise((r) => setTimeout(r, 0));   // let the UI paint before heavy work
          return engine.handle(path, opts);
        },
      };
    },
  };
});
