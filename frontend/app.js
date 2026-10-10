/* CropCurve frontend. Plain JS + Chart.js, served by the FastAPI backend. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const COLORS = { none: "#7a4a2b", today: "#3c6b4a", delay: "#a8792a", limit: "#98a38b", grid: "#e4ddcf", text: "#4d574b" };

  const state = {
    health: null, diseases: [], byId: {}, samples: [],
    file: null, fileUrl: null, coords: null, wmode: "auto",
    result: null, chart: null, specimen: null,
  };

  // ---------- helpers ----------
  const pct = (x) => Math.round(x * 100);
  const healthPct = (s) => Math.round((1 - s) * 1000) / 10;
  const parseDate = (iso) => { const [y, m, d] = iso.split("-").map(Number); return new Date(y, m - 1, d); };
  const shortDay = (iso) => { const d = parseDate(iso); return `${d.toLocaleDateString(undefined, { weekday: "short" })} ${d.getDate()}`; };
  const longDay = (iso) => parseDate(iso).toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });
  const addDays = (iso, n) => { const d = parseDate(iso); d.setDate(d.getDate() + n);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  const fullName = (d) => (d.is_healthy ? `Healthy ${d.crop.toLowerCase()}` : `${d.crop}, ${d.name.charAt(0).toLowerCase()}${d.name.slice(1)}`);

  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    let body = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) {
      let msg = body?.detail;
      if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
      throw new Error(msg || `The server returned an error (${res.status}).`);
    }
    return body;
  }

  // ---------- startup ----------
  async function init() {
    try {
      const [health, diseases, samples] = await Promise.all([
        api("/api/health"), api("/api/diseases"), api("/api/samples").catch(() => []),
      ]);
      state.health = health;
      state.diseases = diseases;
      state.byId = Object.fromEntries(diseases.map((d) => [d.id, d]));
      state.samples = samples;
      $("demo-note").hidden = health.model_loaded;
      $("delay").max = health.max_spray_delay;
      if (!health.weather_key_configured) {
        setWeatherMode("manual");
        $("locate-status").textContent = "Live forecast isn’t set up on this server yet, so set the weather yourself.";
      }
      renderCatalogue();
      renderSamples();
    } catch {
      showFormError("Can’t reach the CropCurve server. Is the backend running?");
    }
  }

  // ---------- catalogue ("What it recognises") ----------
  function renderCatalogue() {
    const crops = {};
    state.diseases.forEach((d) => (crops[d.crop] ||= { latin: d.crop_scientific, items: [] }).items.push(d));
    const names = Object.keys(crops).sort();
    $("count-crops").textContent = names.length;
    $("count-conditions").textContent = state.diseases.filter((d) => !d.is_healthy).length;
    $("crop-list").innerHTML = names.map((c) => {
      const items = crops[c].items.filter((d) => !d.is_healthy);
      const text = items.length
        ? items.map((d) => (d.has_forecast ? `<span class="has-forecast">${esc(d.name)}</span>` : esc(d.name))).join(", ")
        : "Healthy leaves only";
      return `<div class="crop"><h3>${esc(c)} <em>${esc(crops[c].latin)}</em></h3><p>${text}</p></div>`;
    }).join("");
  }

  // ---------- samples + specimen sheet ----------
  function renderSamples() {
    if (!state.samples.length) return;
    // one sample per condition for the strip, diseased leaves first
    const seen = new Set();
    const picks = state.samples
      .filter((s) => (seen.has(s.id) ? false : seen.add(s.id)))
      .sort((a, b) => Number(a.is_healthy) - Number(b.is_healthy) || Number(b.has_forecast) - Number(a.has_forecast));
    const strip = picks.slice(0, 5);
    $("sample-row").innerHTML = strip.map((s, i) =>
      `<button type="button" data-i="${i}" title="${esc(fullName(s))}" aria-label="Try a ${esc(fullName(s).toLowerCase())} leaf"><img src="${s.url}" alt=""></button>`).join("");
    $("sample-row").querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => useSample(strip[Number(b.dataset.i)])));
    $("samples").hidden = false;

    const diseased = picks.filter((s) => !s.is_healthy && s.has_forecast);
    const pool = diseased.length ? diseased : picks;
    showSpecimen(pool[Math.floor(Math.random() * pool.length)]);
  }

  function showSpecimen(s) {
    state.specimen = s;
    const img = $("specimen-img");
    img.src = s.url;
    img.alt = `${fullName(s)} leaf`;
    img.hidden = false;
    $("specimen-drawing").setAttribute("hidden", "");  // SVG elements ignore the .hidden property
    $("specimen-latin").textContent = s.crop_scientific;
    $("specimen-name").textContent = fullName(s);
    $("specimen-note").textContent = "Reference leaf from the training set";
    $("specimen-use").hidden = false;
  }
  $("specimen-use").addEventListener("click", () => state.specimen && useSample(state.specimen));

  async function useSample(s) {
    try {
      const blob = await (await fetch(s.url)).blob();
      setFile(new File([blob], s.file, { type: blob.type || "image/jpeg" }));
      $("upload").scrollIntoView({ behavior: "smooth", block: "center" });
    } catch {
      showFormError("Couldn’t load that sample photo.");
    }
  }

  // ---------- photo ----------
  const upload = $("upload");
  $("photo").addEventListener("change", (e) => e.target.files[0] && setFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => upload.addEventListener(ev, (e) => { e.preventDefault(); upload.classList.add("dragging"); }));
  ["dragleave", "dragend"].forEach((ev) => upload.addEventListener(ev, () => upload.classList.remove("dragging")));
  upload.addEventListener("drop", (e) => {
    e.preventDefault();
    upload.classList.remove("dragging");
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  function setFile(file) {
    if (!file.type.startsWith("image/")) { showFormError("That file isn’t a photo. Choose a JPG or PNG."); return; }
    state.file = file;
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
    state.fileUrl = URL.createObjectURL(file);
    $("preview").src = state.fileUrl;
    $("file-name").textContent = file.name;
    $("upload-empty").hidden = true;
    $("upload-filled").hidden = false;
    hideFormError();
  }

  // ---------- weather controls ----------
  function setWeatherMode(mode) {
    state.wmode = mode;
    document.querySelector(`input[name="wmode"][value="${mode}"]`).checked = true;
    $("panel-auto").hidden = mode !== "auto";
    $("panel-manual").hidden = mode !== "manual";
  }
  document.querySelectorAll('input[name="wmode"]').forEach((r) => r.addEventListener("change", () => setWeatherMode(r.value)));

  $("locate").addEventListener("click", () => {
    const status = $("locate-status");
    if (!navigator.geolocation) { status.textContent = "This browser can’t share your location. Type a place name instead."; return; }
    status.className = "hint";
    status.textContent = "Finding your location…";
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        state.coords = { lat: pos.coords.latitude, lon: pos.coords.longitude };
        $("city").value = "";
        status.className = "hint ok";
        status.textContent = `Using your location (${state.coords.lat.toFixed(2)}, ${state.coords.lon.toFixed(2)}).`;
      },
      () => { status.textContent = "Location access was blocked. Type a village or town instead."; },
      { timeout: 10000, maximumAge: 600000 },
    );
  });
  $("city").addEventListener("input", () => { if ($("city").value.trim()) { state.coords = null; $("locate-status").textContent = ""; } });

  const sliders = [["temp", "temp-out", " °C"], ["humidity", "hum-out", "%"], ["rain", "rain-out", " mm"]];
  const syncSliders = () => sliders.forEach(([i, o, u]) => ($(o).textContent = $(i).value + u));
  sliders.forEach(([i]) => $(i).addEventListener("input", () => {
    syncSliders();
    document.querySelectorAll("#presets button").forEach((b) => b.setAttribute("aria-pressed", "false"));
  }));
  document.querySelectorAll("#presets button").forEach((btn) => {
    btn.setAttribute("aria-pressed", "false");
    btn.addEventListener("click", () => {
      $("temp").value = btn.dataset.t; $("humidity").value = btn.dataset.h; $("rain").value = btn.dataset.r;
      syncSliders();
      document.querySelectorAll("#presets button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    });
  });

  // ---------- submit ----------
  function showFormError(msg) { const el = $("form-error"); el.textContent = msg; el.hidden = false; }
  function hideFormError() { $("form-error").hidden = true; }

  $("check-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    hideFormError();
    if (!state.file) { showFormError("Add a photo of the leaf first."); return; }

    const fd = new FormData();
    fd.append("image", state.file, state.file.name);
    fd.append("extent", document.querySelector('input[name="extent"]:checked').value);
    fd.append("weather_mode", state.wmode);
    if (state.wmode === "auto") {
      if (state.coords) { fd.append("lat", state.coords.lat); fd.append("lon", state.coords.lon); }
      else if ($("city").value.trim()) fd.append("city", $("city").value.trim());
    }
    fd.append("temp", $("temp").value);       // manual values double as the fallback
    fd.append("humidity", $("humidity").value);
    fd.append("rain_mm", $("rain").value);

    const btn = $("submit");
    btn.disabled = true;
    btn.textContent = "Checking the leaf…";
    try {
      state.result = await api("/api/analyze", { method: "POST", body: fd });
      render(state.result, true);
    } catch (err) {
      $("results").hidden = true;   // never leave a half-rendered result on screen
      $("check").hidden = false;
      showFormError(err.message || "Something went wrong showing the result.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Check this leaf";
    }
  });

  $("again").addEventListener("click", () => {
    $("results").hidden = true;
    $("check").hidden = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  async function switchDisease(id) {
    const r = state.result;
    try {
      const data = await api("/api/simulate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ disease_id: id, extent: r.extent, weather_days: r.weather.days }),
      });
      state.result = { ...r, ...data, diagnosis: { ...r.diagnosis, chosen: id } };
      render(state.result, false);
    } catch (err) {
      alert(err.message);
    }
  }

  // ---------- results ----------
  function render(data, scroll) {
    $("check").hidden = true;
    $("results").hidden = false;
    $("result-photo").src = state.fileUrl;
    renderDiagnosis(data);
    renderVerdict(data);
    renderCurve(data);
    renderRecommendation(data);
    renderWeather(data.weather);
    if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderDiagnosis(data) {
    const dx = data.diagnosis, d = data.disease;
    const chosen = dx.chosen || dx.disease_id;
    const conf = (dx.top.find((t) => t.disease_id === chosen) || dx.top[0]).confidence;
    $("dx-latin").textContent = d.crop_scientific;
    $("dx-crop").textContent = d.crop;
    $("dx-name").textContent = d.is_healthy ? "Healthy leaf" : d.name;
    $("dx-pathogen").textContent = d.is_healthy ? "No signs of disease." : `Caused by ${d.pathogen}`;
    $("dx-bar").style.width = `${pct(conf)}%`;
    $("dx-conf").textContent = `${pct(conf)}% match with ${d.is_healthy ? "healthy leaves" : "this condition"} in our reference set`;
    $("dx-symptoms").textContent = d.symptoms;
    $("dx-favoured").textContent = d.favoured_by;
    $("dx-favoured").previousElementSibling.hidden = $("dx-favoured").hidden = d.is_healthy;
    $("dx-warnings").innerHTML = dx.quality_warnings.map((w) => `<li>${esc(w)}</li>`).join("");

    const alt = $("dx-alt");
    alt.innerHTML = "";
    if (dx.low_confidence) {
      const others = dx.top.filter((t) => t.disease_id !== chosen && t.confidence >= 0.1).slice(0, 1);
      const box = document.createElement("div");
      box.className = "alt";
      box.innerHTML = "<p>We’re not certain about this one. Compare the photo with the description below, or retake it closer and in daylight.</p>";
      others.forEach((o) => {
        const info = state.byId[o.disease_id];
        const b = document.createElement("button");
        b.type = "button"; b.className = "link";
        b.textContent = `See the result for ${info ? fullName(info).toLowerCase() : o.disease_id} instead (${pct(o.confidence)}%)`;
        b.addEventListener("click", () => switchDisease(o.disease_id));
        box.appendChild(b);
      });
      alt.appendChild(box);
    }
  }

  function renderVerdict(data) {
    const { simulation: sim, weather, disease: d } = data;
    const v = $("verdict");
    const days = weather.days;
    let status, title, sub;

    if (d.is_healthy) {
      status = "healthy";
      title = `This ${d.crop.toLowerCase()} leaf looks healthy.`;
      sub = "Nothing to spray. Keep checking your plants twice a week, especially after rain.";
    } else if (!sim) {
      status = "unknown";
      if (d.pathogen_type === "viral") {
        title = "Remove infected plants early.";
        sub = "No spray cures a virus. The steps below slow its spread to healthy plants.";
      } else {
        title = `Treat ${d.name.toLowerCase()} early.`;
        sub = "We haven’t modelled how this condition spreads with the weather yet, so there’s no forecast. The care steps below still apply, and most leaf problems are far easier to control in the first few days.";
      }
    } else {
      const a = sim.action;
      const limit = pct(sim.threshold);
      const maxDelay = Object.keys(sim.treated).length - 1;
      status = a.status;
      switch (a.status) {
        case "low_risk":
          title = "No spray needed yet.";
          sub = `With this weather, damage stays under ${limit}% of the leaf area for the next ${sim.horizon_days} days, even untreated. Check again in three days, or after rain.`;
          break;
        case "act_by":
          if (a.latest_spray_day === 0) {
            title = "Spray today.";
            sub = `Waiting even a day lets damage pass ${limit}% of the leaf area within two weeks.`;
          } else {
            title = `Spray by ${longDay(days[a.latest_spray_day].date)}.`;
            sub = a.latest_spray_day >= maxDelay
              ? `You have at least ${plural(a.latest_spray_day, "day")}, though spraying sooner keeps more of the crop healthy.`
              : `You have ${plural(a.latest_spray_day, "day")}. Spray any later and damage passes ${limit}% of the leaf area.`;
          }
          break;
        case "spread_warning": {
          const k = a.untreated_crosses_threshold_on_day;
          title = `Act before ${longDay(addDays(days[0].date, k))}.`;
          sub = `Untreated, damage passes ${limit}% of the leaf area around then. Treating before that day matters most.`;
          break;
        }
        default:
          title = "Spray today, and remove infected leaves.";
          sub = `Damage will pass ${limit}% of the leaf area even with a spray today. Spraying now still slows it a lot.`;
      }
    }
    v.className = `verdict ${status}`;
    void v.offsetWidth;
    v.classList.add("reveal");
    $("verdict-title").textContent = title;
    $("verdict-sub").textContent = sub;
  }

  function renderCurve(data) {
    const sim = data.simulation;
    $("curve-block").hidden = !sim;
    if (!sim) return;

    const hasTreat = Object.keys(sim.treated).length > 0;
    document.querySelectorAll("[data-treat]").forEach((el) => (el.hidden = !hasTreat));
    const days = data.weather.days;
    const labels = sim.no_action.map((_, i) => (i === 0 ? "Today" : shortDay(addDays(days[0].date, i))));

    let delay = 3;
    if (hasTreat) {
      const maxDelay = Object.keys(sim.treated).length - 1;
      $("delay").max = maxDelay;
      if (sim.action.status === "act_by" && sim.action.latest_spray_day !== null) delay = sim.action.latest_spray_day + 2;
      delay = Math.max(1, Math.min(delay, maxDelay));
      $("delay").value = delay;
    }

    if (!window.Chart) { $("chart-fallback").hidden = false; updateReadout(); return; }
    if (Chart.defaults?.font) Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;

    const datasets = [line("No treatment", sim.no_action.map(healthPct), COLORS.none, { borderWidth: 2.5 })];
    if (hasTreat) {
      datasets.push(line("Spray today", sim.treated["0"].map(healthPct), COLORS.today, { borderWidth: 2.5 }));
      datasets.push(line("Spray later", sim.treated[String(delay)].map(healthPct), COLORS.delay, { borderDash: [6, 5], borderWidth: 2 }));
    }
    datasets.push(line("Damage limit", sim.no_action.map(() => healthPct(sim.threshold)), COLORS.limit,
      { borderDash: [2, 4], borderWidth: 1.5, pointHoverRadius: 0 }));
    const minY = Math.max(0, Math.floor(Math.min(...sim.no_action.map(healthPct)) / 10) * 10 - 10);

    if (state.chart) state.chart.destroy();
    state.chart = new Chart($("chart"), {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 600 },
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y}% healthy` } } },
        scales: {
          y: { min: minY, max: 100, border: { display: false }, grid: { color: COLORS.grid },
               ticks: { color: COLORS.text, callback: (v) => `${v}%` },
               title: { display: true, text: "Healthy leaf area", color: COLORS.text } },
          x: { grid: { display: false }, border: { color: COLORS.grid },
               ticks: { color: COLORS.text, maxRotation: 0, autoSkipPadding: 14 } },
        },
      },
    });
    updateReadout();
  }

  function line(label, data, color, extra = {}) {
    return { label, data, borderColor: color, backgroundColor: color, tension: .3, pointRadius: 0, pointHoverRadius: 4, fill: false, ...extra };
  }

  $("delay").addEventListener("input", () => {
    const sim = state.result?.simulation;
    if (!sim || !sim.treated["0"]) return;
    if (state.chart) { state.chart.data.datasets[2].data = sim.treated[$("delay").value].map(healthPct); state.chart.update("none"); }
    updateReadout();
  });

  function updateReadout() {
    const { simulation: sim, weather } = state.result;
    const end = (arr) => healthPct(arr[arr.length - 1]);
    if (!sim.treated["0"]) {
      $("readout").innerHTML = `In ${sim.horizon_days} days, about <strong>${end(sim.no_action)}%</strong> of the leaf area is still healthy if nothing is done.`;
      return;
    }
    const d = Number($("delay").value);
    $("delay-out").textContent = plural(d, "day");
    $("key-delay").textContent = `Spray in ${plural(d, "day")}`;
    const when = d === 0 ? "today" : `on ${shortDay(weather.days[d].date)}`;
    $("readout").innerHTML =
      `In ${sim.horizon_days} days the healthy leaf area is <strong>${end(sim.treated["0"])}%</strong> if you spray today, ` +
      `<strong>${end(sim.treated[String(d)])}%</strong> if you spray ${esc(when)}, and <strong>${end(sim.no_action)}%</strong> if you don’t spray.`;
  }

  function renderRecommendation(data) {
    const rec = data.recommendation;
    $("care").textContent = rec.cultural_practices;
    $("treat-block").hidden = data.disease.is_healthy;
    const body = $("rec-body");
    if (!rec.primary) { body.innerHTML = rec.notes.map((n) => `<p class="rec-note">${esc(n)}</p>`).join(""); return; }
    const p = rec.primary;
    const traits = [
      p.mode === "systemic" ? "Absorbed into the leaf" : "Protects the leaf surface",
      `lasts about ${p.protection_days} days`,
      p.rainfast ? "stays on in rain" : "washes off in heavy rain",
      `${p.cost_level} cost`,
      `resistance group ${p.frac_group}`,
    ].join(", ");
    body.innerHTML = `
      <p class="product">${esc(p.product)}</p>
      <p class="rate">Mix <strong>${esc(p.typical_rate)}</strong>.</p>
      <p class="traits">${esc(traits)}.</p>
      ${p.why.length ? `<ul class="why">${p.why.map((w) => `<li>Chosen because ${esc(w)}.</li>`).join("")}</ul>` : ""}
      <p class="rec-note">${esc(p.notes)}</p>
      <ul class="rec-list">${rec.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
      ${rec.alternatives.length ? `<div class="alts"><p>If you can’t get it:</p>${rec.alternatives
        .map((a) => `<p><strong>${esc(a.product)}</strong>, ${esc(a.typical_rate)}</p>`).join("")}</div>` : ""}`;
  }

  function renderWeather(w) {
    const parts = [];
    if (w.warning) parts.push(w.warning);
    if (w.source === "live") {
      parts.push(`Forecast for ${w.location}.`);
      if (w.live_days < w.days.length) parts.push(`Days in grey italics after day ${w.live_days} repeat the forecast average.`);
    } else {
      parts.push("Conditions you set, applied to every day.");
    }
    $("weather-meta").textContent = parts.join(" ");
    const cls = (d) => (d.extrapolated ? ' class="est"' : "");
    const row = (label, f) => `<tr><td>${label}</td>${w.days.map((d) => `<td${cls(d)}>${f(d)}</td>`).join("")}</tr>`;
    $("days").innerHTML =
      `<thead><tr><th></th>${w.days.map((d) => `<th${cls(d)}>${esc(shortDay(d.date))}</th>`).join("")}</tr></thead><tbody>` +
      row("Temp.", (d) => `${Math.round(d.temp)}°`) +
      row("Humidity", (d) => `${Math.round(d.humidity)}%`) +
      row("Rain", (d) => (d.rain_mm > 0 ? `${d.rain_mm} mm` : "–")) + "</tbody>";
  }

  // Phone menu: open/close the sprout menu.
  const navToggle = document.querySelector(".nav-toggle");
  const siteNav = document.getElementById("site-nav");
  if (navToggle && siteNav) {
    const setMenu = (open) => {
      siteNav.classList.toggle("open", open);
      navToggle.setAttribute("aria-expanded", String(open));
      navToggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
      navToggle.querySelector(".nav-toggle-label").textContent = open ? "Close" : "Menu";
    };
    navToggle.addEventListener("click", () => setMenu(!siteNav.classList.contains("open")));
    siteNav.addEventListener("click", (e) => { if (e.target.closest("a")) setMenu(false); });
    document.addEventListener("click", (e) => { if (!e.target.closest(".masthead")) setMenu(false); });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && siteNav.classList.contains("open")) { setMenu(false); navToggle.focus(); }
    });
  }

  syncSliders();
  init();
})();
