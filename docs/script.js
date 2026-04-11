'use strict';

const DATA = window.SITE_DATA;

const CAT_COLORS = {
  PT_FT_RI: '#34d399', PT_FT: '#fbbf24', FT_RI: '#818cf8',
  PT_RI: '#f97316', PT_only: '#fb923c', FT_only: '#4ade80',
  RI_only: '#c084fc', None: '#334155',
};
const CAT_ORDER = ['PT_FT_RI','PT_FT','FT_RI','PT_RI','PT_only','FT_only','RI_only','None'];

/* ═══ INIT ═══ */
document.addEventListener('DOMContentLoaded', () => {
  renderCategoryChart();
  renderFeatureGrid();
  renderTrainingChart();
  setupModal();
  setupTabs();
});

/* ═══ CANVAS HELPERS ═══ */
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  let w = rect.width;
  let h = rect.height;
  // Fallbacks if layout hasn't happened yet
  if (!w || w < 10) w = canvas.parentElement?.clientWidth || canvas.parentElement?.offsetWidth || 800;
  if (!h || h < 10) h = parseInt(canvas.style.height) || 120;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

/* ═══ DRAW TIME SERIES WITH ACTIVATION GRADIENT ═══ */
function drawTimeSeries(canvas, values, activations, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { top: 10, bottom: 10, left: 5, right: 5 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  // Filter out NaN/Infinity for range calculation
  const cleanVals = values.map(v => (isFinite(v) ? v : null));
  const finiteVals = cleanVals.filter(v => v !== null);
  const vMin = finiteVals.length ? Math.min(...finiteVals) : 0;
  const vMax = finiteVals.length ? Math.max(...finiteVals) : 1;
  const vRange = vMax - vMin || 1;

  const aMax = Math.max(...activations, 0.01);

  const xScale = plotW / (values.length - 1);
  const yScale = plotH / vRange;

  // Draw base line first (always visible), skipping NaN
  ctx.beginPath();
  let drawing = false;
  for (let i = 0; i < values.length; i++) {
    if (!isFinite(values[i])) { drawing = false; continue; }
    const x = pad.left + i * xScale;
    const y = pad.top + plotH - (values[i] - vMin) * yScale;
    if (!drawing) { ctx.moveTo(x, y); drawing = true; } else { ctx.lineTo(x, y); }
  }
  ctx.strokeStyle = 'rgba(148,163,184,0.35)';
  ctx.lineWidth = (opts.lineWidth || 1.5) * 0.8;
  ctx.stroke();

  // Draw activation overlay with gradient color
  for (let i = 0; i < values.length - 1; i++) {
    if (!isFinite(values[i]) || !isFinite(values[i + 1])) continue;
    const act = (activations[i] + activations[i + 1]) / 2;
    if (act <= 0) continue;

    const x1 = pad.left + i * xScale;
    const y1 = pad.top + plotH - (values[i] - vMin) * yScale;
    const x2 = pad.left + (i + 1) * xScale;
    const y2 = pad.top + plotH - (values[i + 1] - vMin) * yScale;

    const t = Math.min(act / aMax, 1);
    // Yellow to red gradient
    const r = 255;
    const g = Math.round(200 * (1 - t * t));
    const b = Math.round(40 * (1 - t));

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = `rgb(${r},${g},${b})`;
    ctx.lineWidth = (opts.lineWidth || 1.5) + 1.5 * t;
    ctx.stroke();
  }

  // Mark peak timestep
  if (opts.peak !== undefined && opts.peak >= 0 && opts.peak < values.length) {
    const px = pad.left + opts.peak * xScale;
    const py = pad.top + plotH - (values[opts.peak] - vMin) * yScale;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,60,60,0.9)';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

/* ═══ MINI PREVIEW (for cards) ═══ */
function drawMiniPreview(canvas, feature) {
  if (!feature.windows || !feature.windows.length) return;
  const w = feature.windows[0];
  const acts = w.activations_pt.map((a, i) =>
    a + (w.activations_ft[i] || 0) + (w.activations_ri[i] || 0)
  );
  drawTimeSeries(canvas, w.raw_values, acts, { lineWidth: 1 });
}

/* ═══ CATEGORY CHART ═══ */
function renderCategoryChart() {
  const canvas = document.getElementById('category-chart');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);

  const layers = Object.keys(DATA.layer_stats).map(Number).sort((a, b) => a - b);
  const barW = (w - 80) / layers.length;
  const pad = { top: 30, bottom: 40, left: 50, right: 30 };
  const plotH = h - pad.top - pad.bottom;

  // Y axis
  ctx.fillStyle = '#64748b';
  ctx.font = '11px Inter';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = i * 1000;
    const y = pad.top + plotH - (val / 4096) * plotH;
    ctx.fillText(val.toString(), pad.left - 8, y + 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.strokeStyle = '#1e3a5f';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  // Bars
  layers.forEach((layer, i) => {
    const stats = DATA.layer_stats[String(layer)];
    const x = pad.left + i * barW + 2;
    const bw = barW - 4;
    let yOffset = 0;

    CAT_ORDER.forEach(cat => {
      const count = stats[cat] || 0;
      const barH = (count / 4096) * plotH;
      ctx.fillStyle = CAT_COLORS[cat];
      ctx.globalAlpha = cat === 'None' ? 0.15 : 0.8;
      ctx.fillRect(x, pad.top + plotH - yOffset - barH, bw, barH);
      yOffset += barH;
    });
    ctx.globalAlpha = 1;

    // X label
    ctx.fillStyle = '#64748b';
    ctx.font = '9px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(layer.toString(), x + bw / 2, h - pad.bottom + 14);
  });

  // X axis label
  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Layer', w / 2, h - 5);

  // Legend
  const legend = document.getElementById('cat-legend');
  if (legend) {
    legend.innerHTML = CAT_ORDER.filter(c => c !== 'None').map(cat =>
      `<div class="legend-item"><div class="legend-dot" style="background:${CAT_COLORS[cat]}"></div>${cat}</div>`
    ).join('');
  }
}

/* ═══ FEATURE GRID ═══ */
function renderFeatureGrid() {
  const grid = document.getElementById('feature-grid');
  if (!grid || !DATA.top_features) return;

  DATA.top_features.forEach((feat, idx) => {
    const isCross = feat.category !== 'FT_RI';
    const card = document.createElement('div');
    card.className = 'feature-card';
    card.dataset.type = isCross ? 'cross' : 'ts';
    card.dataset.idx = idx;

    const scoreClass = `score-${feat.score}`;
    const catBadge = feat.category === 'PT_FT' ? 'badge-ptft'
      : feat.category === 'FT_RI' ? 'badge-ftri' : 'badge-ptftri';

    card.innerHTML = `
      <div class="fc-header">
        <span class="fc-name">${feat.name}</span>
        <span class="fc-score ${scoreClass}">${feat.score}/10</span>
      </div>
      <div class="fc-meta">Layer ${feat.layer} &middot; Feature ${feat.feature_id} &middot; <span class="badge ${catBadge}">${feat.category}</span></div>
      <div class="fc-desc">${feat.interpretation.split('.').slice(0, 2).join('.') + '.'}</div>
      <div class="fc-preview"><canvas data-preview="${idx}" style="width:100%;height:80px;display:block"></canvas></div>
    `;
    grid.appendChild(card);

    // Draw preview after append (double rAF for layout)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const c = card.querySelector(`canvas[data-preview="${idx}"]`);
        if (c) drawMiniPreview(c, feat);
      });
    });

    card.addEventListener('click', () => openModal(idx));
  });
}

/* ═══ TABS ═══ */
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.dataset.filter;
      document.querySelectorAll('.feature-card').forEach(card => {
        if (filter === 'all') {
          card.classList.remove('hidden');
        } else if (filter === 'cross') {
          card.classList.toggle('hidden', card.dataset.type !== 'cross');
        } else {
          card.classList.toggle('hidden', card.dataset.type !== 'ts');
        }
      });
    });
  });
}

/* ═══ MODAL ═══ */
function setupModal() {
  const modal = document.getElementById('feature-modal');
  modal.querySelector('.modal-backdrop').addEventListener('click', closeModal);
  modal.querySelector('.modal-close').addEventListener('click', closeModal);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
}

function closeModal() {
  document.getElementById('feature-modal').classList.add('hidden');
  document.body.style.overflow = '';
}

function openModal(idx) {
  const feat = DATA.top_features[idx];
  const modal = document.getElementById('feature-modal');
  const body = document.getElementById('modal-body');
  document.body.style.overflow = 'hidden';

  const catBadge = feat.category === 'PT_FT' ? 'badge-ptft'
    : feat.category === 'FT_RI' ? 'badge-ftri' : 'badge-ptftri';

  let html = `
    <div class="modal-title">${feat.name}</div>
    <div class="modal-subtitle">Layer ${feat.layer} &middot; Feature ${feat.feature_id} &middot;
      <span class="badge ${catBadge}">${feat.category}</span> &middot;
      Score ${feat.score}/10</div>
    <p style="color:#94a3b8;margin-bottom:1.5rem;font-size:0.9rem">${feat.interpretation}</p>
  `;

  // Time series windows
  if (feat.windows && feat.windows.length) {
    html += `<div class="modal-section"><h3>Top Activating Time Series Windows</h3>`;
    feat.windows.forEach((w, i) => {
      html += `
        <div class="ts-plot-container">
          <div class="ts-plot-label">#${i + 1} &middot; activation = ${w.activation_value.toFixed(2)} &middot; peak at t=${w.peak_timestep}</div>
          <canvas data-modal-ts="${idx}-${i}" style="width:100%;height:120px;display:block"></canvas>
        </div>`;
    });
    html += `</div>`;
  }

  // Wiki spans
  if (feat.wiki_spans && feat.wiki_spans.length) {
    html += `<div class="modal-section"><h3>Top Activating WikiText Spans</h3>`;
    feat.wiki_spans.forEach((ws, i) => {
      // Highlight the peak region in the text
      const hs = ws.highlight_start || 0;
      const he = ws.highlight_end || 0;
      let displayText;
      if (hs < he && he <= ws.text.length) {
        const before = escapeHtml(ws.text.slice(0, hs));
        const highlighted = escapeHtml(ws.text.slice(hs, he));
        const after = escapeHtml(ws.text.slice(he));
        displayText = `${before}<span class="wiki-highlight">${highlighted}</span>${after}`;
      } else {
        displayText = escapeHtml(ws.text);
      }
      html += `
        <div class="wiki-span">
          <div class="wiki-span-act">#${i + 1} &middot; activation = ${ws.activation_value.toFixed(2)} &middot; peak token ${ws.peak_token_idx}</div>
          <div>${displayText}</div>
        </div>`;
    });
    html += `</div>`;
  }

  // Stats
  if (feat.stats) {
    const s = feat.stats;
    html += `
      <div class="modal-section"><h3>Activation Statistics</h3>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;font-size:0.8rem;font-family:'JetBrains Mono',monospace">
          <div style="color:#f59e0b">PT rate: ${(s.rate_pt || 0).toFixed(3)}<br>mean: ${(s.mean_act_pt || 0).toFixed(3)}</div>
          <div style="color:#10b981">FT rate: ${(s.rate_ft || 0).toFixed(3)}<br>mean: ${(s.mean_act_ft || 0).toFixed(3)}</div>
          <div style="color:#a78bfa">RI rate: ${(s.rate_ri || 0).toFixed(3)}<br>mean: ${(s.mean_act_ri || 0).toFixed(3)}</div>
        </div>
      </div>`;
  }

  body.innerHTML = html;
  modal.classList.remove('hidden');

  // Render TS plots in modal (use timeout to ensure layout is fully complete)
  setTimeout(() => {
    feat.windows.forEach((w, i) => {
      const c = body.querySelector(`canvas[data-modal-ts="${idx}-${i}"]`);
      if (!c) return;
      const acts = w.activations_pt.map((a, j) =>
        a + (w.activations_ft[j] || 0) + (w.activations_ri[j] || 0)
      );
      drawTimeSeries(c, w.raw_values, acts, { peak: w.peak_timestep, lineWidth: 1.5 });
    });
  }, 50);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/* ═══ TRAINING CHART ═══ */
function renderTrainingChart() {
  const canvas = document.getElementById('training-chart');
  if (!canvas || !DATA.train_log) return;
  const { ctx, w, h } = setupCanvas(canvas);

  const layers = Object.keys(DATA.train_log).map(Number).filter(l => !isNaN(DATA.train_log[l].final_loss)).sort((a, b) => a - b);
  const pad = { top: 30, bottom: 40, left: 60, right: 30 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const barW = plotW / layers.length;

  const maxLoss = Math.max(...layers.map(l => DATA.train_log[String(l)].final_loss || 0));

  // Grid
  ctx.fillStyle = '#64748b';
  ctx.font = '11px Inter';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = (maxLoss * i / 4).toFixed(2);
    const y = pad.top + plotH - (i / 4) * plotH;
    ctx.fillText(val, pad.left - 8, y + 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.strokeStyle = '#1e3a5f';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  // Bars
  layers.forEach((layer, i) => {
    const info = DATA.train_log[String(layer)];
    const loss = info.final_loss || 0;
    const x = pad.left + i * barW + 2;
    const bw = barW - 4;
    const barH = (loss / maxLoss) * plotH;

    // Color by loss magnitude
    const t = loss / maxLoss;
    const r = Math.round(56 + 199 * t);
    const g = Math.round(189 - 100 * t);
    const b = Math.round(248 - 200 * t);
    ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
    ctx.fillRect(x, pad.top + plotH - barH, bw, barH);

    // X label
    ctx.fillStyle = '#64748b';
    ctx.font = '9px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(layer.toString(), x + bw / 2, h - pad.bottom + 14);
  });

  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Layer', w / 2, h - 5);
  ctx.save();
  ctx.translate(15, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Final Loss', 0, 0);
  ctx.restore();
}
