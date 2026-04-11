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
  renderArchDiagram();
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

/* ═══ ARCHITECTURE DIAGRAM (animated canvas) ═══ */
function renderArchDiagram() {
  const canvas = document.getElementById('arch-canvas');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);

  const colors = { pt: '#f59e0b', ft: '#10b981', ri: '#a78bfa', enc: '#38bdf8', z: '#34d399', loss: '#fb7185' };
  const rows = [
    { label: 'PT', color: colors.pt, y: h * 0.18 },
    { label: 'FT', color: colors.ft, y: h * 0.50 },
    { label: 'RI', color: colors.ri, y: h * 0.82 },
  ];
  const colX = { input: w * 0.08, enc: w * 0.30, latent: w * 0.52, dec: w * 0.72, loss: w * 0.90 };
  const boxW = w * 0.12;
  const boxH = 36;
  const encH = h * 0.72;

  function roundRect(x, y, bw, bh, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + bw - r, y); ctx.quadraticCurveTo(x + bw, y, x + bw, y + r);
    ctx.lineTo(x + bw, y + bh - r); ctx.quadraticCurveTo(x + bw, y + bh, x + bw - r, y + bh);
    ctx.lineTo(x + r, y + bh); ctx.quadraticCurveTo(x, y + bh, x, y + bh - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function hex2rgba(hex, a) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${a})`;
  }

  // Animated dash offset
  let dashOffset = 0;

  function draw() {
    ctx.clearRect(0, 0, w, h);
    dashOffset -= 0.5;

    // ── Connecting lines (animated dashed) ──
    rows.forEach(row => {
      // Input → Encoder
      drawDashedArrow(ctx, colX.input + boxW + 4, row.y, colX.enc - boxW/2 - 4, row.y, row.color, dashOffset);
      // Encoder → Latent
      drawDashedArrow(ctx, colX.enc + boxW/2 + 4, row.y, colX.latent - boxW/2 - 4, row.y, colors.enc, dashOffset);
      // Latent → Decoder
      drawDashedArrow(ctx, colX.latent + boxW/2 + 4, row.y, colX.dec - boxW/2 - 4, row.y, colors.z, dashOffset);
      // Decoder → Loss
      drawDashedArrow(ctx, colX.dec + boxW/2 + 4, row.y, colX.loss - boxW/2 - 4, row.y, row.color, dashOffset);
    });

    // ── Input boxes ──
    rows.forEach(row => {
      const x = colX.input, y = row.y - boxH/2;
      roundRect(x, y, boxW, boxH, 6);
      ctx.fillStyle = hex2rgba(row.color, 0.1);
      ctx.fill();
      ctx.strokeStyle = hex2rgba(row.color, 0.5);
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = row.color;
      ctx.font = '600 13px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('x' + row.label, x + boxW/2, y + 16);
      ctx.fillStyle = '#64748b';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText('1024-dim', x + boxW/2, y + 30);
    });

    // ── Shared Encoder (tall box) ──
    const encX = colX.enc - boxW/2, encY = h * 0.14 - 10;
    roundRect(encX, encY, boxW, encH, 8);
    ctx.fillStyle = hex2rgba(colors.enc, 0.06);
    ctx.fill();
    ctx.strokeStyle = hex2rgba(colors.enc, 0.5);
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.stroke();

    // "SHARED" label
    ctx.fillStyle = '#0a1220';
    ctx.fillRect(encX + boxW/2 - 30, encY - 8, 60, 16);
    ctx.fillStyle = colors.enc;
    ctx.font = '700 9px Inter';
    ctx.textAlign = 'center';
    ctx.letterSpacing = '0.1em';
    ctx.fillText('SHARED', encX + boxW/2, encY + 4);

    // Encoder text
    ctx.fillStyle = colors.enc;
    ctx.font = '600 14px Inter';
    ctx.fillText('Encoder', encX + boxW/2, encY + encH/2 - 20);
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('1024→2048', encX + boxW/2, encY + encH/2);
    ctx.fillText('ReLU', encX + boxW/2, encY + encH/2 + 14);
    ctx.fillText('2048→4096', encX + boxW/2, encY + encH/2 + 28);
    ctx.fillText('TopK(k=64)', encX + boxW/2, encY + encH/2 + 42);

    // ── Sparse latent (dot grids) ──
    rows.forEach(row => {
      const cx = colX.latent, cy = row.y;
      const gridW = boxW * 0.9, gridH = boxH * 0.7;
      const x0 = cx - gridW/2, y0 = cy - gridH/2;

      roundRect(cx - boxW/2, cy - boxH/2, boxW, boxH, 6);
      ctx.fillStyle = hex2rgba(colors.z, 0.08);
      ctx.fill();
      ctx.strokeStyle = hex2rgba(colors.z, 0.4);
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.stroke();

      // Draw dot grid (some lit, most dim = sparse)
      const cols = 16, rws = 3;
      const dotR = 2.5;
      const spacingX = gridW / (cols + 1);
      const spacingY = gridH / (rws + 1);
      // Pseudo-random active dots (deterministic per row)
      const seed = row.label.charCodeAt(0);
      for (let r = 0; r < rws; r++) {
        for (let c = 0; c < cols; c++) {
          const dx = x0 + (c + 1) * spacingX;
          const dy = y0 + (r + 1) * spacingY;
          const isActive = ((seed * 7 + c * 13 + r * 31) % 10) < 2; // ~20% lit
          ctx.beginPath();
          ctx.arc(dx, dy, dotR, 0, Math.PI * 2);
          if (isActive) {
            ctx.fillStyle = colors.z;
            ctx.globalAlpha = 0.9;
          } else {
            ctx.fillStyle = '#334155';
            ctx.globalAlpha = 0.4;
          }
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      // Label
      ctx.fillStyle = colors.z;
      ctx.font = '500 9px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText('z' + row.label + ' (sparse)', cx, cy + boxH/2 + 12);
    });

    // ── Decoder boxes ──
    rows.forEach(row => {
      const x = colX.dec - boxW/2, y = row.y - boxH/2;
      roundRect(x, y, boxW, boxH, 6);
      ctx.fillStyle = hex2rgba(row.color, 0.07);
      ctx.fill();
      ctx.strokeStyle = hex2rgba(row.color, 0.35);
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.stroke();
      ctx.fillStyle = row.color;
      ctx.font = '600 13px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('D' + row.label, x + boxW/2, y + 16);
      ctx.fillStyle = '#64748b';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText('4096→1024', x + boxW/2, y + 30);
    });

    // ── Loss boxes ──
    rows.forEach(row => {
      const x = colX.loss - boxW/2, y = row.y - boxH/2;
      roundRect(x, y, boxW, boxH, 6);
      ctx.fillStyle = hex2rgba(colors.loss, 0.08);
      ctx.fill();
      ctx.strokeStyle = hex2rgba(colors.loss, 0.35);
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.stroke();
      ctx.fillStyle = colors.loss;
      ctx.font = '600 12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('MSE' + row.label, x + boxW/2, y + 22);
    });

    requestAnimationFrame(draw);
  }

  draw();
}

function drawDashedArrow(ctx, x1, y1, x2, y2, color, offset) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.lineDashOffset = offset;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  // Arrowhead
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const aLen = 7;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - aLen * Math.cos(angle - 0.4), y2 - aLen * Math.sin(angle - 0.4));
  ctx.lineTo(x2 - aLen * Math.cos(angle + 0.4), y2 - aLen * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.7;
  ctx.fill();
  ctx.globalAlpha = 1;
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

  // Apply initial filter based on active tab
  const activeTab = document.querySelector('.tab-btn.active');
  if (activeTab) {
    const filter = activeTab.dataset.filter;
    document.querySelectorAll('.feature-card').forEach(card => {
      if (filter === 'all') {
        card.classList.remove('hidden');
      } else if (filter === 'cross') {
        card.classList.toggle('hidden', card.dataset.type !== 'cross');
      } else {
        card.classList.toggle('hidden', card.dataset.type !== 'ts');
      }
    });
  }
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
