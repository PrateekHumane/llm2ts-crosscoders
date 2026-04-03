'use strict';

/* ═══════════════════════════════════════════════════════════════════
   COLOURS + HELPERS
   ═══════════════════════════════════════════════════════════════════ */
const C = {
  bg: '#070d1a', surface: '#0d1729', surface2: '#121f38',
  border: '#1e3a5f', border2: '#2a4a72',
  pt: '#f59e0b', ft: '#10b981', ri: '#a78bfa', blue: '#38bdf8',
  text: '#e2e8f0', muted: '#64748b', soft: '#94a3b8',
  ptft: '#fbbf24', ftonly: '#34d399', ftri: '#818cf8',
  univ: '#38bdf8', ptonly: '#fb923c', rionly: '#c084fc',
};

function hex2rgba(hex, a) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

function lerp(a, b, t) { return a + (b - a) * t; }

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
function easeInOut(t) { return t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2; }

function animateValue(from, to, duration, fn, ease) {
  const start = performance.now();
  function step(now) {
    let t = Math.min((now - start) / duration, 1);
    if (ease) t = ease(t);
    fn(lerp(from, to, t));
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.width;
  const h = canvas.clientHeight || canvas.height;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

/* ═══════════════════════════════════════════════════════════════════
   PERFORMANCE CHART
   ═══════════════════════════════════════════════════════════════════ */
function drawPerfChart(progress) {
  const canvas = document.getElementById('perf-chart');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const models = [
    { name: 'FT',  val: 0.316, color: C.ft },
    { name: 'RI',  val: 0.325, color: C.ri },
  ];
  const maxVal = 0.40;
  const barH = 44, barGap = 22, labelW = 48, padR = 70, padT = 20;
  const trackW = w - labelW - padR;

  models.forEach((m, i) => {
    const y = padT + i * (barH + barGap);
    const barW = (m.val / maxVal) * trackW * progress;

    // track
    ctx.fillStyle = hex2rgba(C.border, 0.5);
    ctx.beginPath(); ctx.roundRect(labelW, y, trackW, barH, 6); ctx.fill();

    // bar
    const grad = ctx.createLinearGradient(labelW, 0, labelW + barW, 0);
    grad.addColorStop(0, hex2rgba(m.color, 0.9));
    grad.addColorStop(1, hex2rgba(m.color, 0.6));
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.roundRect(labelW, y, Math.max(barW, 0), barH, 6); ctx.fill();

    // label
    ctx.fillStyle = m.color;
    ctx.font = '700 14px "JetBrains Mono", monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(m.name, labelW - 10, y + barH / 2);

    // value
    if (progress > 0.1) {
      ctx.fillStyle = C.text;
      ctx.font = '600 13px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText((m.val * progress).toFixed(3), labelW + barW + 8, y + barH / 2);
    }
  });

  // axis labels
  ctx.fillStyle = C.muted;
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  [0, 0.1, 0.2, 0.3, 0.4].forEach(v => {
    const x = labelW + (v / maxVal) * trackW;
    ctx.fillText(v.toFixed(1), x, padT + 2 * (barH + barGap) + 8);
    ctx.strokeStyle = hex2rgba(C.border, 0.4);
    ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + 2*(barH+barGap)); ctx.stroke();
    ctx.setLineDash([]);
  });
}

/* ═══════════════════════════════════════════════════════════════════
   ARCHITECTURE CANVAS
   ═══════════════════════════════════════════════════════════════════ */
let archMode = '3way';
let archAnim = 0;
let archRaf;

function drawArch(progress) {
  const canvas = document.getElementById('arch-canvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth || 900;
  const ch = 360;
  canvas.width = cw * dpr; canvas.height = ch * dpr;
  canvas.style.width = cw + 'px'; canvas.style.height = ch + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cw, ch);

  const is3 = archMode === '3way';
  const models = is3
    ? [{ name:'PT', key:'pt', color:C.pt, sub:'text acts' }, { name:'FT', key:'ft', color:C.ft, sub:'bin acts' }, { name:'RI', key:'ri', color:C.ri, sub:'bin acts' }]
    : [{ name:'FT', key:'ft', color:C.ft, sub:'bin acts' }, { name:'RI', key:'ri', color:C.ri, sub:'bin acts' }];

  const n = models.length;
  const totalH = ch;
  const rowH = totalH / n;
  const inputX = 20, inputW = 110, inputH = 46;
  const encX = 230, encW = 150, encH = totalH * 0.55;
  const encY = (totalH - encH) / 2;
  const latX = encX + encW + 80, latW = 110, latH = encH;
  const latY = encY;
  const decX = latX + latW + 80, decW = 120, decH = 46;
  const ph = progress;

  // Draw flow lines from inputs to encoder
  models.forEach((m, i) => {
    const iy = rowH * i + rowH / 2 - inputH / 2;
    const mx = inputX + inputW;
    const my = iy + inputH / 2;
    const ey = encY + encH / 2;

    ctx.save();
    ctx.globalAlpha = ph * 0.6;
    ctx.strokeStyle = m.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.lineDashOffset = -archAnim * 15;
    ctx.beginPath();
    ctx.moveTo(mx + 5, my);
    ctx.bezierCurveTo(mx + 60, my, encX - 20, ey, encX, ey);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  });

  // Draw input boxes
  models.forEach((m, i) => {
    const iy = rowH * i + rowH / 2 - inputH / 2;
    ctx.save();
    ctx.globalAlpha = ph;
    ctx.fillStyle = hex2rgba(m.color, 0.12);
    ctx.strokeStyle = hex2rgba(m.color, 0.8);
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(inputX, iy, inputW, inputH, 8); ctx.fill(); ctx.stroke();

    ctx.fillStyle = m.color;
    ctx.font = '700 16px "JetBrains Mono", monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(m.name, inputX + inputW / 2, iy + inputH / 2 - 7);
    ctx.fillStyle = hex2rgba(m.color, 0.6);
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(m.sub, inputX + inputW / 2, iy + inputH / 2 + 9);
    ctx.restore();
  });

  // Shared encoder box
  ctx.save();
  ctx.globalAlpha = ph;
  ctx.fillStyle = '#0f2440';
  ctx.strokeStyle = C.blue;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.roundRect(encX, encY, encW, encH, 12); ctx.fill(); ctx.stroke();

  ctx.fillStyle = C.blue;
  ctx.font = '700 12px "Space Grotesk", sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('SHARED', encX + encW/2, encY + encH/2 - 24);
  ctx.fillText('ENCODER', encX + encW/2, encY + encH/2 - 8);
  ctx.fillStyle = hex2rgba(C.soft, 0.7);
  ctx.font = '11px Inter';
  ctx.fillText('Linear', encX + encW/2, encY + encH/2 + 12);
  ctx.fillText('1024 → 4096', encX + encW/2, encY + encH/2 + 28);
  ctx.fillStyle = hex2rgba(C.blue, 0.5);
  ctx.font = '10px "JetBrains Mono"';
  ctx.fillText('TopK  k=64', encX + encW/2, encY + encH/2 + 44);
  ctx.restore();

  // Arrow enc → latent
  ctx.save();
  ctx.globalAlpha = ph;
  ctx.strokeStyle = C.blue;
  ctx.lineWidth = 2;
  ctx.setLineDash([5,3]);
  ctx.lineDashOffset = -archAnim * 15;
  ctx.beginPath();
  ctx.moveTo(encX + encW, encY + encH/2);
  ctx.lineTo(latX, latY + latH/2);
  ctx.stroke();
  ctx.setLineDash([]);
  // arrowhead
  const ax = latX - 1, ay = latY + latH/2;
  ctx.fillStyle = C.blue;
  ctx.beginPath(); ctx.moveTo(ax, ay-5); ctx.lineTo(ax+10, ay); ctx.lineTo(ax, ay+5); ctx.fill();
  ctx.restore();

  // Latent box
  ctx.save();
  ctx.globalAlpha = ph;
  ctx.fillStyle = '#0a1628';
  ctx.strokeStyle = hex2rgba(C.blue, 0.5);
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5,2]);
  ctx.beginPath(); ctx.roundRect(latX, latY, latW, latH, 12); ctx.fill(); ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = C.text;
  ctx.font = '700 22px "Space Grotesk"';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('z', latX + latW/2, latY + latH/2 - 22);
  ctx.fillStyle = hex2rgba(C.soft, 0.7);
  ctx.font = '11px Inter';
  ctx.fillText('4096-dim', latX + latW/2, latY + latH/2 - 2);
  ctx.fillText('sparse', latX + latW/2, latY + latH/2 + 14);
  ctx.fillStyle = C.blue;
  ctx.font = '700 10px "JetBrains Mono"';
  ctx.fillText('≤64 active', latX + latW/2, latY + latH/2 + 30);

  // Animated feature dots inside latent box
  const dots = 32;
  const dotsPerRow = 8;
  const dotS = 6, dotGap = 10;
  const dotsW = dotsPerRow * (dotS + dotGap);
  const dotsH = (dots/dotsPerRow) * (dotS + dotGap);
  const dotsX = latX + latW/2 - dotsW/2;
  const dotsY = latY + latH/2 + 46;
  for (let d = 0; d < dots; d++) {
    const col = d % dotsPerRow, row = Math.floor(d / dotsPerRow);
    const dx = dotsX + col * (dotS + dotGap) + dotS/2;
    const dy = dotsY + row * (dotS + dotGap) + dotS/2;
    const isActive = d < 5; // ~5/32 ≈ 1.5%
    ctx.beginPath();
    ctx.arc(dx, dy, dotS/2, 0, Math.PI*2);
    ctx.fillStyle = isActive ? hex2rgba(C.blue, 0.8) : hex2rgba(C.border, 0.8);
    ctx.fill();
  }
  ctx.restore();

  // Decoder boxes and flow lines
  models.forEach((m, i) => {
    const dy = rowH * i + rowH / 2 - decH / 2;
    const lmy = latY + latH/2;

    // Flow line latent → decoder
    ctx.save();
    ctx.globalAlpha = ph * 0.6;
    ctx.strokeStyle = m.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4,3]);
    ctx.lineDashOffset = -archAnim * 12;
    ctx.beginPath();
    ctx.moveTo(latX + latW, lmy);
    ctx.bezierCurveTo(latX + latW + 40, lmy, decX - 30, dy + decH/2, decX, dy + decH/2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Decoder box
    ctx.save();
    ctx.globalAlpha = ph;
    ctx.fillStyle = hex2rgba(m.color, 0.08);
    ctx.strokeStyle = hex2rgba(m.color, 0.7);
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(decX, dy, decW, decH, 8); ctx.fill(); ctx.stroke();

    ctx.fillStyle = m.color;
    ctx.font = '600 11px Inter';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(`Decoder ${m.name}`, decX + decW/2, dy + decH/2 - 8);
    ctx.fillStyle = hex2rgba(m.color, 0.5);
    ctx.font = '10px Inter';
    ctx.fillText('4096 → 1024', decX + decW/2, dy + decH/2 + 8);
    ctx.restore();

    // MSE loss label
    ctx.save();
    ctx.globalAlpha = ph * 0.5;
    ctx.fillStyle = m.color;
    ctx.font = '9px "JetBrains Mono"';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(`MSE(x̂_${m.name}, x_${m.name})`, decX + decW + 10, dy + decH/2);
    ctx.restore();
  });
}

function startArchAnim() {
  if (archRaf) cancelAnimationFrame(archRaf);
  let t = 0;
  function tick() {
    t += 0.005;
    archAnim = t;
    drawArch(1);
    archRaf = requestAnimationFrame(tick);
  }
  tick();
}

/* ═══════════════════════════════════════════════════════════════════
   LAYER LIFECYCLE CHART
   ═══════════════════════════════════════════════════════════════════ */
function drawLayerChart(progress) {
  const canvas = document.getElementById('layer-chart');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const binData =  [20, 49, 28, 2, 1];  // PT_FT bin (3-way crosscoder)
  const textData = [12, 46, 41, 4, 0];  // PT_FT text (3-way text crosscoder)
  const layers = ['L2', 'L8', 'L16', 'L24', 'L27'];
  const maxVal = 55;
  const padL = 50, padR = 20, padT = 30, padB = 50;
  const cw = w - padL - padR;
  const ch = h - padT - padB;
  const groupW = cw / layers.length;
  const barW = groupW * 0.28;
  const barGap = groupW * 0.06;

  // Grid lines
  [0, 10, 20, 30, 40, 50].forEach(v => {
    const y = padT + ch - (v / maxVal) * ch;
    ctx.strokeStyle = hex2rgba(C.border, 0.4);
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + cw, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.muted;
    ctx.font = '10px "JetBrains Mono"';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(v, padL - 6, y);
  });

  // Bars
  layers.forEach((layer, i) => {
    const cx = padL + i * groupW + groupW / 2;
    const x1 = cx - barW - barGap / 2;
    const x2 = cx + barGap / 2;

    // Bin bar
    const bh1 = (binData[i] / maxVal) * ch * progress;
    const by1 = padT + ch - bh1;
    const grad1 = ctx.createLinearGradient(0, by1, 0, padT + ch);
    grad1.addColorStop(0, hex2rgba(C.pt, 0.9));
    grad1.addColorStop(1, hex2rgba(C.pt, 0.4));
    ctx.fillStyle = grad1;
    ctx.beginPath(); ctx.roundRect(x1, by1, barW, bh1, [4, 4, 0, 0]); ctx.fill();

    // Text bar
    const bh2 = (textData[i] / maxVal) * ch * progress;
    const by2 = padT + ch - bh2;
    const grad2 = ctx.createLinearGradient(0, by2, 0, padT + ch);
    grad2.addColorStop(0, hex2rgba(C.blue, 0.9));
    grad2.addColorStop(1, hex2rgba(C.blue, 0.4));
    ctx.fillStyle = grad2;
    ctx.beginPath(); ctx.roundRect(x2, by2, barW, bh2, [4, 4, 0, 0]); ctx.fill();

    // Value labels
    if (progress > 0.5) {
      ctx.fillStyle = C.pt;
      ctx.font = '600 11px Inter';
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      if (binData[i] > 0) ctx.fillText(binData[i], x1 + barW/2, by1 - 3);
      ctx.fillStyle = C.blue;
      if (textData[i] > 0) ctx.fillText(textData[i], x2 + barW/2, by2 - 3);
    }

    // Layer labels
    ctx.fillStyle = C.soft;
    ctx.font = '600 12px "JetBrains Mono"';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(layer, cx, padT + ch + 8);
  });

  // Legend
  const ly = padT - 10;
  const items = [
    { color: C.pt, label: 'Bin PT (OOD tokens)' },
    { color: C.blue, label: 'Text PT (natural language)' },
  ];
  let lx = padL;
  items.forEach(it => {
    ctx.fillStyle = it.color;
    ctx.fillRect(lx, ly, 12, 10);
    ctx.fillStyle = C.soft;
    ctx.font = '11px Inter';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(it.label, lx + 16, ly + 5);
    lx += ctx.measureText(it.label).width + 36;
  });

  // Y-axis label
  ctx.save();
  ctx.translate(12, padT + ch/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('PT_FT feature count', 0, 0);
  ctx.restore();
}

/* ═══════════════════════════════════════════════════════════════════
   CONVERGENCE CHART
   ═══════════════════════════════════════════════════════════════════ */
function drawConvergenceChart(progress) {
  const canvas = document.getElementById('convergence-chart');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const layers = ['L2', 'L8', 'L16', 'L24', 'L27'];
  const ptFT  = [20, 49, 26, 2, 1];
  const ftRI  = [19, 30, 8,  37, 13];
  const ptRI  = [4,  8,  5,  3,  2];

  const padL=50, padR=20, padT=30, padB=40;
  const cw = w-padL-padR, ch = h-padT-padB;
  const maxV = 55;
  const xs = layers.map((_,i) => padL + (i/(layers.length-1))*cw);

  // Grid
  [0,10,20,30,40,50].forEach(v=>{
    const y = padT+ch-(v/maxV)*ch;
    ctx.strokeStyle=hex2rgba(C.border,0.4); ctx.setLineDash([3,3]); ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(padL+cw,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=C.muted; ctx.font='10px "JetBrains Mono"';
    ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(v, padL-6, y);
  });

  const series = [
    { data: ptFT, color: C.ptft, label: 'PT↔FT' },
    { data: ftRI, color: C.ftri, label: 'FT↔RI' },
    { data: ptRI, color: C.muted, label: 'PT↔RI' },
  ];

  series.forEach(s => {
    const pts = xs.map((x,i) => [x, padT+ch-(s.data[i]/maxV)*ch]);
    const visible = Math.max(1, Math.round(pts.length * progress));

    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.slice(0, visible).forEach(([x,y],i) => i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
    ctx.stroke();

    // Dots
    pts.slice(0, visible).forEach(([x,y]) => {
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = C.bg;
      ctx.beginPath(); ctx.arc(x,y,2.5,0,Math.PI*2); ctx.fill();
    });
  });

  // X labels
  layers.forEach((l,i) => {
    ctx.fillStyle=C.soft; ctx.font='600 12px "JetBrains Mono"';
    ctx.textAlign='center'; ctx.textBaseline='top';
    ctx.fillText(l, xs[i], padT+ch+8);
  });

  // Legend
  let lx = padL;
  series.forEach(s=>{
    ctx.fillStyle=s.color; ctx.fillRect(lx, padT-14, 18, 3);
    ctx.fillStyle=C.soft; ctx.font='11px Inter';
    ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText(s.label, lx+22, padT-12);
    lx += ctx.measureText(s.label).width+50;
  });

  ctx.save();
  ctx.translate(12, padT+ch/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillStyle=C.muted; ctx.font='11px Inter';
  ctx.textAlign='center'; ctx.fillText('Shared features', 0, 0);
  ctx.restore();
}

/* ═══════════════════════════════════════════════════════════════════
   POOLING DIAGRAM
   ═══════════════════════════════════════════════════════════════════ */
function drawPoolingDiagram(progress) {
  const canvas = document.getElementById('pooling-canvas');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0,0,w,h);

  const ph = progress;
  const stageW = w/4;
  const cy = h/2;

  function stageBox(x, y, label, sublabel, color, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha*ph;
    ctx.fillStyle = hex2rgba(color, 0.1);
    ctx.strokeStyle = hex2rgba(color, 0.7);
    ctx.lineWidth = 1.5;
    const bw = stageW*0.82, bh = 70;
    ctx.beginPath(); ctx.roundRect(x - bw/2, y-bh/2, bw, bh, 8); ctx.fill(); ctx.stroke();
    ctx.fillStyle = color; ctx.font = '600 12px "JetBrains Mono"';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(label, x, y-12);
    ctx.fillStyle = hex2rgba(color, 0.6); ctx.font = '10px Inter';
    ctx.fillText(sublabel, x, y+10);
    ctx.restore();
  }

  function arrow(x1, x2, y, color) {
    ctx.save();
    ctx.globalAlpha = ph*0.5;
    ctx.strokeStyle = color; ctx.lineWidth=1.5;
    ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(x1+10,y); ctx.lineTo(x2-10,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.moveTo(x2-10,y-4); ctx.lineTo(x2,y); ctx.lineTo(x2-10,y+4); ctx.fill();
    ctx.restore();
  }

  const cx1 = stageW*0.5, cx2 = stageW*1.5, cx3 = stageW*2.5, cx4 = stageW*3.5;

  stageBox(cx1, cy, '"0.12, -0.45, ..."', 'TS as text', C.soft, 1);
  arrow(cx1+stageW*0.41, cx2-stageW*0.41, cy, C.blue);
  stageBox(cx2, cy, '["0",".",12",",",...]', 'Qwen3 BPE tokens', C.blue, 1);
  arrow(cx2+stageW*0.41, cx3-stageW*0.41, cy, C.blue);
  stageBox(cx3, cy, 'mean-pool per value', 'timestep alignment', C.ft, 1);
  arrow(cx3+stageW*0.41, cx4-stageW*0.41, cy, C.ft);
  stageBox(cx4, cy, '[h₁, h₂, ..., h₅₁₂]', '512 × 1024-dim', C.ft, 1);

  // Labels above
  ctx.fillStyle=C.muted; ctx.font='10px Inter'; ctx.textAlign='center';
  ['Step 1: Serialize', 'Step 2: Tokenize', 'Step 3: Pool per value', 'Step 4: Aligned acts'].forEach((t,i)=>{
    ctx.fillText(t, stageW*(i+0.5), 18);
  });
}

/* ═══════════════════════════════════════════════════════════════════
   BIN vs TEXT CHART
   ═══════════════════════════════════════════════════════════════════ */
function drawBinTextChart(progress) {
  const canvas = document.getElementById('bin-text-chart');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0,0,w,h);

  const layers = ['L2','L8','L16','L24','L27'];
  const bin  = [20,49,28,2,1];
  const text = [12,46,41,4,0];
  const maxV = 55;
  const padL=50,padR=20,padT=30,padB=45;
  const cw=w-padL-padR, ch=h-padT-padB;
  const groupW = cw/layers.length;
  const barW = groupW*0.3;

  [0,10,20,30,40,50].forEach(v=>{
    const y=padT+ch-(v/maxV)*ch;
    ctx.strokeStyle=hex2rgba(C.border,0.35); ctx.setLineDash([3,3]); ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(padL+cw,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=C.muted; ctx.font='10px "JetBrains Mono"';
    ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(v, padL-6, y);
  });

  layers.forEach((l,i)=>{
    const cx = padL + i*groupW + groupW/2;
    const x1 = cx - barW - 3, x2 = cx + 3;

    const h1 = (bin[i]/maxV)*ch*progress;
    const h2 = (text[i]/maxV)*ch*progress;
    const by1=padT+ch-h1, by2=padT+ch-h2;

    ctx.fillStyle=hex2rgba(C.pt, 0.7);
    ctx.beginPath(); ctx.roundRect(x1,by1,barW,h1,[4,4,0,0]); ctx.fill();
    ctx.fillStyle=hex2rgba(C.blue, 0.8);
    ctx.beginPath(); ctx.roundRect(x2,by2,barW,h2,[4,4,0,0]); ctx.fill();

    if(progress>0.5){
      ctx.font='600 11px Inter'; ctx.textAlign='center'; ctx.textBaseline='bottom';
      if(bin[i]>0){ ctx.fillStyle=C.pt; ctx.fillText(bin[i], x1+barW/2, by1-2); }
      if(text[i]>0){ ctx.fillStyle=C.blue; ctx.fillText(text[i], x2+barW/2, by2-2); }
    }

    // Delta annotation for L16
    if(i===2 && progress>0.8){
      const mid = cx, top = Math.min(by1,by2)-18;
      ctx.save();
      ctx.fillStyle=hex2rgba(C.ft, 0.9);
      ctx.strokeStyle=hex2rgba(C.ft, 0.4);
      ctx.lineWidth=1; ctx.font='700 10px Inter';
      ctx.textAlign='center';
      ctx.fillText('+70% ↑', mid, top);
      ctx.restore();
    }

    ctx.fillStyle=C.soft; ctx.font='600 12px "JetBrains Mono"';
    ctx.textAlign='center'; ctx.textBaseline='top';
    ctx.fillText(l, cx, padT+ch+8);
  });

  // Legend
  let lx=padL;
  [{c:C.pt,l:'Bin PT'},{c:C.blue,l:'Text PT'}].forEach(it=>{
    ctx.fillStyle=it.c; ctx.fillRect(lx,padT-14,14,12);
    ctx.fillStyle=C.soft; ctx.font='11px Inter';
    ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText(it.l, lx+18, padT-8);
    lx += ctx.measureText(it.l).width + 40;
  });

  ctx.save();
  ctx.translate(12, padT+ch/2); ctx.rotate(-Math.PI/2);
  ctx.fillStyle=C.muted; ctx.font='11px Inter'; ctx.textAlign='center';
  ctx.fillText('PT_FT feature count', 0, 0);
  ctx.restore();
}

/* ═══════════════════════════════════════════════════════════════════
   FEATURE DATA
   ═══════════════════════════════════════════════════════════════════ */
const FEATURES = [
  {
    id:'F1649', layer:'L8', ratio:'889×', ratioNum:889,
    title:'Repetition → Periodicity',
    tagline:'Fires on tokens appearing for the second time in context; detects 24-hour periodic cycles in time series.',
    color: C.ft,
    textExamples:[
      {pre:'xenon dissolved in water, xenon ', hit:'dissolved', post:' in hydrophobic solvent'},
      {pre:'people in Ontario and killed three people in ', hit:'Ontario', post:'...'},
      {pre:'toward its terminus, the road turns northeast toward its ', hit:'termin', post:'us...'},
    ],
    tsType:'periodic', tsDataset:'M4 hourly / 5–11',
    pt:1.596, ft:1.314,
    description:'FT/RI=889× makes this the feature most strongly preserved from language pretraining. The most natural interpretation is that it detects <strong>recurrence</strong>: in text, it tracks information already mentioned; in time series, it detects the periodic pattern repeating from a previous cycle. The language model\'s ability to track what has been said may literally transfer to tracking what has been observed in previous cycles.',
  },
  {
    id:'F3818', layer:'L8', ratio:'56×', ratioNum:56,
    title:'Chess Captures → Regime Changes',
    tagline:'Fires exclusively on chess piece-capture notation; detects sudden non-stationary jumps in time series.',
    color: C.pt,
    textExamples:[
      {pre:'27.R', hit:'xf', post:'2 Bc2, winning'},
      {pre:'24.R', hit:'xd', post:'8 Rxd8'},
      {pre:'20...N', hit:'xf', post:'2? 21.Qxd5! wins'},
    ],
    tsType:'regime', tsDataset:'M4 hourly/168–169 · M4 weekly/34,62,63',
    pt:0.113, ft:0.281,
    description:'Every single top text activation is a chess capture move (PieceXsquare format). Zero noise from non-chess text. Chess captures are <strong>discrete, irreversible state changes</strong> — a piece disappears and the board configuration shifts permanently. The hypothesis is that FT repurposes this feature to detect analogous <strong>sudden regime changes</strong> in time series: abrupt level shifts where the series doesn\'t return to its prior state.',
  },
  {
    id:'F4082', layer:'L16', ratio:'220×', ratioNum:220,
    title:'Ecological Stability → Steady Production',
    tagline:'Strongest FT feature at L16 (mean act=0.866). Encodes long-term persistence; detects steady-state continuous systems.',
    color: C.ft,
    textExamples:[
      {pre:'multiple use and ', hit:'sustained', post:' yield of its resources'},
      {pre:'an endangered species ', hit:'endemic', post:' to one region of dense cougar...'},
      {pre:'diet and sub', hit:'sistence', post:'...'},
    ],
    tsType:'steady', tsDataset:'electricity/15T · hierarchical_sales/D',
    pt:0.011, ft:0.866,
    description:'The strongest single FT feature at layer 16 encodes "sustained yield" and "endemic" — concepts about resources or organisms that persist continuously in a stable system. Electricity consumption and hierarchical sales are both <strong>sustained, continuous-production systems</strong>: power grids operate 24/7, product hierarchies maintain stable structure. The FT model repurposes a "long-term ecological stability" concept from pretraining to represent steady-state TS production.',
  },
  {
    id:'F3237', layer:'L8', ratio:'521×', ratioNum:521,
    title:'Ordinal Rankings → Hierarchy',
    tagline:'Fires on "number N" in ranked lists; maps to products at positions in a hierarchical sales tree.',
    color: C.ri,
    textExamples:[
      {pre:'Torres was chosen as ', hit:'number', post:' 50 in The Times\'s list...'},
      {pre:'ranking her at ', hit:'number', post:' 33...'},
      {pre:'Rolling Stone ranked him at ', hit:'number', post:' seven...'},
    ],
    tsType:'staircase', tsDataset:'hierarchical_sales/D · M4 monthly',
    pt:0.052, ft:0.298,
    description:'FT/RI=521× — one of the features most uniquely acquired during fine-tuning, yet already present in PT. The text pattern (ordinal position within a named hierarchy) maps conceptually to <strong>hierarchical_sales data</strong> (products at specific positions within a product tree). This feature may encode "position within an ordered structure" that transfers from natural language to time series forecasting.',
  },
  {
    id:'F109', layer:'L16', ratio:'131×', ratioNum:131,
    title:'Low / Minimum → Solar Night Zero',
    tagline:'Words denoting minimum and scarcity map directly onto time series windows that approach zero (solar irradiance at night).',
    color: C.blue,
    textExamples:[
      {pre:'never exceeded thirty, mostly ', hit:'lowest', post:' pay-scale laborers'},
      {pre:'Rhodesian Army remained stretched and ', hit:'low', post:' on manpower'},
      {pre:'army units were "', hit:'frozen', post:'"...'},
    ],
    tsType:'solar_zero', tsDataset:'Solar/10T · M4 weekly/63–64',
    pt:0.052, ft:0.312,
    description:'Solar irradiance fires here — solar output is <strong>zero at night</strong> (the minimum possible value). The text pattern (low, lowest, frozen, minimal) maps directly onto TS windows where the series approaches or sits at its minimum. This is one of the most functionally coherent mappings found: a "near-zero / minimum-value" concept shared between natural language and time series representation.',
  },
  {
    id:'F3099', layer:'L8', ratio:'116×', ratioNum:116,
    title:'Maya Calendar + Cadmium → Solar Cycles',
    tagline:'Two completely distinct semantic clusters (Maya Long Count dates, cadmium in plants) both involving discrete counting in non-linear systems.',
    color: C.ptft,
    textExamples:[
      {pre:'7 K\'an 17 Pop in the Long ', hit:'Count', post:'...'},
      {pre:'plants bio-accumulate metal toxins like ', hit:'Cd', post:'...'},
      {pre:'a Long ', hit:'Count', post:' date of 9.17.5.0...'},
    ],
    tsType:'solar_cycle', tsDataset:'Solar/10T · electricity/15T',
    pt:0.016, ft:0.374,
    description:'What connects Maya Long Count dates (a base-20 astronomical counting system tracking Venus cycles) with cadmium accumulation in plants? Both involve <strong>discrete counting in non-linear systems</strong>: the Maya calendar counts astronomical periods across nested time scales; cadmium accumulates at discrete concentration intervals. Solar irradiance follows the same structure — zero at night, maximum at solar noon, cycling across discrete day/night intervals. This is the most semantically surprising cross-domain feature.',
  },
  {
    id:'F1363', layer:'L8', ratio:'52×', ratioNum:52,
    title:'Weather / Phase Transitions → Threshold Crossings',
    tagline:'A coherent cluster of precipitation, melting, heat and tropical weather maps onto threshold-crossing behavior in time series.',
    color: C.ptft,
    textExamples:[
      {pre:'Precip', hit:'itation', post:' falls throughout the year'},
      {pre:'has a lower melting ', hit:'point', post:' than other transition metals'},
      {pre:'developed into Tropical ', hit:'Depression', post:' Thirteen'},
    ],
    tsType:'threshold', tsDataset:'hierarchical_sales/D · Solar/10T · M4 weekly',
    pt:0.025, ft:0.324,
    description:'The text activations are extremely interpretable — all relate to physical state changes driven by temperature or weather. Solar irradiance data is directly weather-driven. This is the most <strong>semantically transparent</strong> feature: the FT model appears to repurpose the language model\'s understanding of weather/thermal thresholds to detect analogous threshold-crossing behavior in time series.',
  },
  {
    id:'F3278', layer:'L16', ratio:'119×', ratioNum:119,
    title:'The "Atlantic" Feature',
    tagline:'Fires almost exclusively on the word "Atlantic" across wildly different contexts — hurricanes, cities, Bronze Age, Canada.',
    color: C.ri,
    textExamples:[
      {pre:'The 1955 ', hit:'Atlantic', post:' hurricane season...'},
      {pre:'revue in ', hit:'Atlantic', post:' City, New Jersey'},
      {pre:'', hit:'Atlantic', post:' Bronze Age...'},
    ],
    tsType:'irregular', tsDataset:'M4 weekly/126 · M4 hourly/8–9 · ETT2/H',
    pt:0.180, ft:0.188,
    description:'A single geographic/cultural designation ("Atlantic") produces the strongest and most consistent activation in text, across contexts as different as hurricane meteorology, casino cities, and Bronze Age archaeology. The feature has apparently encoded the <em>token</em> "Atlantic" as a meaningful concept unit, and the FT model uses the same feature for M4 competition series. Whether those series come from Atlantic-basin economies or it is coincidental remains open.',
  },
  {
    id:'F934', layer:'L16', ratio:'149×', ratioNum:149,
    title:'Ancient Date Digits → Dense Numerical Sequences',
    tagline:'Single-digit tokens inside multi-digit BCE date ranges; maps to the dense 15-minute electricity data (96 readings/day).',
    color: C.blue,
    textExamples:[
      {pre:'began in 3730–3', hit:'5', post:'40 cal BCE...'},
      {pre:'Between 4500 and ', hit:'3', post:'800 BCE...'},
      {pre:'from the Albian stage (112 to 1', hit:'0', post:'0 million years ago)...'},
    ],
    tsType:'dense', tsDataset:'electricity/15T (MT_001–MT_006)',
    pt:2.528, ft:0.257,
    description:'Very high PT mean (2.528) — the language model strongly encodes this during pretraining. The feature fires on single digits that are the "inner" digits of multi-digit BCE dates, suggesting PT tracks <strong>digit position within larger numerical sequences</strong>. In electricity/15T data (15-minute intervals = 96 readings per day), the numerical sequences in serialized TS text are similarly dense and multi-digit. The FT model keeps this positional-digit tracking feature active.',
  },
  {
    id:'F2788', layer:'L16', ratio:'61×', ratioNum:61,
    title:'"fac-" Subword Token Feature',
    tagline:'Fires on ANY token beginning with "fac" regardless of meaning — facsimile, facade, facemask. A tokenization artifact.',
    color: C.muted,
    textExamples:[
      {pre:'miniature ', hit:'fac', post:'similes of the original album'},
      {pre:'the ', hit:'fac', post:'ades were decorated with archery'},
      {pre:'a 15-yard grabbing-the-', hit:'fac', post:'emask penalty'},
    ],
    tsType:'spiky', tsDataset:'Solar/10T · bitbrains_rnd/H',
    pt:0.015, ft:0.280,
    description:'This is the clearest demonstration that crosscoder features can specialize at the <strong>character n-gram level</strong> within subword tokenization. The feature has no semantic coherence — facsimile and facemask have nothing in common — but all share the Qwen3 tokenizer\'s "fac" token. This suggests that some PT+FT shared features are tokenization artifacts rather than semantic concepts, and the FT model preserved the same subword-level sensitivity.',
  },
  {
    id:'F272', layer:'L8', ratio:'132×', ratioNum:132,
    title:'Ancestral Splitting → Dawn Emergence',
    tagline:'Fires on "common ancestor" and "splitting of the ancestral population"; the time series context shows a long flat baseline that abruptly branches upward.',
    color: C.ptft,
    textExamples:[
      {pre:'likely the splitting of the ', hit:'ancestral', post:' population by coastline change'},
      {pre:'the Felidae, the common ', hit:'ancestor', post:' of today\'s Leopardus, Lynx...'},
      {pre:'Strigopini. The common ', hit:'ancestor', post:' of the kakapo and the genus N...'},
    ],
    tsType:'solar_cycle', tsDataset:'Solar/10T · item_5',
    pt:1.276, ft:0.172,
    description:'Every top text activation is a phylogenetics term: <em>common ancestor</em>, <em>ancestral population</em>, <em>splitting of lineages</em>. The time series context window makes the analogy concrete: a long flat signal at zero — the night, where all solar stations are identical and quiescent — followed by the moment where irradiance <strong>begins to branch upward</strong> at dawn. This is the cladogenesis moment: a period of undifferentiated stasis, then a divergence event. The feature appears to encode exactly this abstract shape — <strong>extended common baseline, then branching away from it</strong> — whether that baseline is a shared evolutionary ancestor or a shared night-time zero.',
  },
];

/* ═══════════════════════════════════════════════════════════════════
   REAL TS WINDOWS (inlined from feature_windows.json)
   ═══════════════════════════════════════════════════════════════════ */
const FEATURE_WINDOWS = {"F1649_L8":{"key":"F1649_L8","dataset":"m4_hourly","item_id":"4","ctx":[-0.0280744274965436,-0.21335803040453286,-0.494940347981806,-0.7484863309085281,-0.9520544998929636,-1.095893086361008,-1.1897538588867658,-1.25070241247492,-1.2714249206948924,-1.2616731521207878,-1.292147428914865,-1.221447106752606,-1.122710449939796,-0.9788718634717516,-0.802121058066104,-0.6473117319521919,-0.4291159101065993,-0.201168319686902,0.07797605574684498,0.31445644366888387,0.49974004657687315,0.6191992116096556,0.5606886001650275,0.35712043118059195,0.13039181183265772,-0.059767675362383865,-0.3986416333125221,-0.7021654301815308,-0.9167043388118341,-1.0459152724187213,-1.1324622185139006,-1.1714692928103194,-1.15196575566211,-1.0556670409928262,-0.9447406734623851,-0.7740847234155529,-0.6253702526604563,-0.4547143026136241,-0.25845996005976707,-0.03660722499888521,0.22669052650194163,0.511929757294504,0.8020448723741188,1.0653426238749455,1.2262468053476732,1.2859763878640644,1.2043053260559375,0.9471024299139261,0.7264686659248074,0.4253828111993248,-0.030512369640069776,-0.42302105474778384,-0.6790049798180321,-0.8642885827260214,-0.9922805452611455,-1.0641998384951676,-1.0495721856340106,-0.9557114131082529,-0.8484419587931012,-0.6899757194638999,-0.5412612487088033,-0.3303792532937892,-0.07927121251059326,0.20718698935373223,0.5375281498015289,0.9288178638374798,1.2847574167923013,1.5529310525801805,1.7528423083493267,1.8515789651621368,1.76990790335401,1.5748725318719161,1.2798815325052488,0.931255805981006,0.40466030297935235,-0.07195738608001474,-0.4205831126042577,-0.648530703023955,-0.7996831159225778,-0.8898869752330463,-0.8972008016636247,-0.8094348844966824,-0.7204499962579771,-0.5473561040676187,-0.3584155879443402,-0.09146092322822413,0.2291284686454678,0.6057905298202617,1.015364809932659,1.4505374825520811,1.8089149776504287,2.062460960577151,2.1819201256099334,2.2672481006333496,2.1416940802417517,1.9125275187502913]},"F3818_L8":{"key":"F3818_L8","dataset":"m4_hourly","item_id":"9","ctx":[-0.10208672401491724,-0.19944095089007066,-0.4590522225571465,-0.7511149031826069,-1.172983219641605,-1.2703374465167585,-1.367691673391912,-1.3352402644335275,-1.367691673391912,-2.7306508496440602,-1.5948515361006033,-0.9782747658912982,-0.5239550404739155,-0.1669895419316862,-0.06963531505653275,0.09262172973538965,0.12507313869377412,0.060170320777005175,0.1899759566105431,0.15752454765215862,0.25487877452731206,0.2224273655689276,0.25487877452731206,-0.004732497139763789,-0.19944095089007066,-0.26434376880683963,-0.491503631515531,-1.172983219641605,-1.2703374465167585,-1.4650459002670655,-1.367691673391912,-1.4650459002670655,-1.2703374465167585,-1.0431775838080672,2.2344147209887653,-0.39414940464037757,-0.13453813297330172,0.1899759566105431,0.319781592444081,0.28733018348569656,0.4171358193192345,0.5469414551527724,0.7091984999446947,0.8065527268198482,0.9039069536950017,0.9688097716117706,0.9039069536950017,0.5793928641111569,0.319781592444081,0.38468441036085,-0.13453813297330172,-0.5888578583906845,-0.9133719479745293,-0.9782747658912982,-1.0756289927664517,-1.2054346285999895,-1.1080804017248362,-1.0431775838080672,-0.5888578583906845,-0.26434376880683963,-0.06963531505653275,0.2224273655689276,0.5793928641111569,0.6442956820279259,0.8390041357782327,0.8390041357782327,0.6767470909863104,1.195969634320462,1.4231294970291535,1.4555809059875378,1.4231294970291535,1.131066816403693,0.8714555447366172,0.6442956820279259,0.1899759566105431,-0.36169799568199307,-0.6862120852658379,-0.9782747658912982,-0.9133719479745293,-1.1080804017248362,-1.0107261748496827,-0.8160177210993759,-0.5564064494323,-0.19944095089007066,0.060170320777005175,0.5469414551527724,0.6767470909863104,0.9039069536950017,1.131066816403693,1.390678088070769,1.6827407686962292,1.8125464045297672,2.0397062672384587,2.1370604941136118,2.007254858280074,1.7800949955713827]},"F4082_L16":{"key":"F4082_L16","dataset":"electricity/15T","item_id":"MT_001","ctx":[0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,2.317048787531001,0.523204564926355,-1.270639657678291,0.523204564926355,2.317048787531001,0.523204564926355,-1.270639657678291,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,2.317048787531001,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,-1.270639657678291,0.523204564926355,0.523204564926355,-1.270639657678291,-1.270639657678291,0.523204564926355,-1.270639657678291,-1.270639657678291,-1.270639657678291,0.523204564926355,-1.270639657678291,-1.270639657678291,-1.270639657678291,0.523204564926355,-1.270639657678291,-1.270639657678291,-1.270639657678291,0.523204564926355,0.523204564926355,-1.270639657678291,-1.270639657678291,-1.270639657678291,-1.270639657678291,0.523204564926355,-1.270639657678291,-1.270639657678291,-1.270639657678291,0.523204564926355,-1.270639657678291,-1.270639657678291,-1.270639657678291,2.317048787531001,-1.270639657678291,-1.270639657678291,-1.270639657678291,-1.270639657678291,0.523204564926355,-1.270639657678291,-1.270639657678291,-1.270639657678291,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,-1.270639657678291,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,0.523204564926355,2.317048787531001,0.523204564926355,-1.270639657678291,0.523204564926355,0.523204564926355]},"F3237_L8":{"key":"F3237_L8","dataset":"electricity/15T","item_id":"MT_004","ctx":[1.0325945132522725,0.6911623786895755,0.861878445970924,0.29282545797535564,-0.16241710330377693,-0.3331331705851253,-0.10551188994555905,-0.16241710330377693,-0.5038492378664737,-0.6745650915444747,-0.5607544512246916,-0.4469440245082558,-0.5038492378664737,-0.73147051850604,-0.3900383839433432,-0.5607544512246916,-0.6176598781862569,-0.4469440245082558,-0.7883757318642579,-1.0729024394653894,-0.6745650915444747,-0.5607544512246916,-0.5607544512246916,-0.7883757318642579,-0.9590917991456063,-0.9590917991456063,-0.9021863721840411,-0.9590917991456063,-0.7883757318642579,-1.414334360424739,-1.414334360424739,-1.5850502141027398,-1.414334360424739,-1.0729024394653894,-0.9021863721840411,-0.6745650915444747,-0.27622795722690746,-0.4469440245082558,-0.16241710330377693,0.06520417733578936,0.520446738614922,0.6342571653313577,0.804973232612706,0.06520417733578936,0.17901460405222513,0.6342571653313577,1.487837074531405,1.3740262206082745,0.861878445970924,0.7480680192544882,0.12210939069400724,-0.3331331705851253,-0.7883757318642579,-0.4469440245082558,-0.4469440245082558,-0.4469440245082558,-0.4469440245082558,-0.9021863721840411,-1.414334360424739,-1.1298076528236074,-1.1867130797851726,-1.1298076528236074,-1.414334360424739,-1.0729024394653894,-1.2436182931433906,-1.0159970125038242,-1.1867130797851726,-1.0729024394653894,-1.0729024394653894,-0.10551188994555905,0.23591981741044302,0.861878445970924,0.861878445970924,0.804973232612706,0.804973232612706,0.9187836593291419,1.430931861173187,1.829268781887407,1.829268781887407,1.9999848491687553,1.487837074531405,1.6016475012478406,1.487837074531405,1.9999848491687553,2.0568900625269735,1.8861744224523196,1.6016475012478406,1.6585531418127533,1.2033105805336208,1.5447422878896229,1.0894997266104902,0.6911623786895755,0.29282545797535564,-0.048606676587341156,0.520446738614922,0.4066358846917914]},"F109_L16":{"key":"F109_L16","dataset":"solar/10T","item_id":"item_0","ctx":[-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.1487271382221865,-0.036303474288429054,2.549440863197717,5.0,2.043534329426623]},"F3099_L8":{"key":"F3099_L8","dataset":"solar/10T","item_id":"item_3","ctx":[-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.14096844862445393,-0.008292260036267032,0.6550887026749599,3.175936216654487,5.0]},"F1363_L8":{"key":"F1363_L8","dataset":"m4_hourly","item_id":"8","ctx":[-0.1595405075986942,-0.5410862801291331,-0.8528693507497603,-1.1001797821646375,-1.3106250891105857,-1.4665166244208994,-1.547190080657838,-1.5853777208027822,-1.6367905177078803,-1.5680197025550804,-1.2631798392335338,-0.8801462365675776,-0.49826983511813494,-0.11011148420761917,0.2560600435891408,0.5693309443452854,0.8025896467024991,0.97286353998948,1.0958574978589108,1.142972118816959,1.065770266229561,0.8647478834752222,0.6164455653033335,0.45526396728895835,0.04148187515564446,-0.4156126053671733,-0.7692202342417872,-1.0317395959308413,-1.2286291171976318,-1.347159584660511,-1.396257979132582,-1.2833482032927686,-1.1069576750042163,-0.8822953245411026,-0.5339777583705504,-0.14433157732451726,0.29226391022006204,0.6680236766679336,0.9475704276856858,1.1684305455802553,1.298367710748767,1.4168981782116459,1.4843464776884305,1.440538145920421,1.3363900364342092,1.136359540436882,0.877973040235376,0.7055500589748701,0.29242922467956395,-0.17524538125137692,-0.5379453053985965,-0.7988115224926314,-1.0115712318716066,-1.1537416670432605,-1.2053197784078606,-1.114066196762799,-0.9639606675350527,-0.7300407073398314,-0.390484807522881,-0.005798060261905594,0.41145563552094866,0.7888685465638394,1.089410233938336,1.3271324267021016,1.4471507243004977,1.5489844313536825,1.5848576690656,1.5195584575623402,1.362675035495015,1.1375167416533956,0.8540024436075973,0.6361179859840623,0.16794743667461573,-0.3251855960196213,-0.6888774069238525,-0.96974667361762,-1.1821757540775915,-1.3369100881713916,-1.4098137648117397,-1.324842132627751,-1.165478993667897,-0.9819799436207624,-0.713674575849141,-0.4013955618500079,-0.07952830919976336,0.2808572125144293,0.5835479878624508,0.845902035092003,1.0080755198633897,1.1219771824602147,1.1312347921923225,1.0497347636578744,0.9237651455174088,0.7430764412818067,0.501221387030493,0.31243227427929665]},"F3278_L16":{"key":"F3278_L16","dataset":"m4_hourly","item_id":"3","ctx":[0.6569339255957644,0.2485058149048362,-0.18497255324180226,-0.5955789471897488,-0.9441042683126741,-1.234905083124615,-1.463624825111535,-1.6171937947313237,-1.6847205756988906,-1.7424450820098751,-1.771851905979622,-1.5866978291330678,-1.2915404478070904,-0.9702436673968936,-0.5716178313625476,-0.16972457044267428,0.23652525699123564,0.6297053848830358,0.9324867576085774,1.1917024651937531,1.3877479583253987,1.445472464636383,1.2951709199021215,1.007637529975708,0.8083246119585351,0.44237302477946344,-0.051008132935177815,-0.4942887757383986,-0.8036049982349949,-1.026879032079369,-1.2163896754399597,-1.3459975292325475,-1.3775826364593127,-1.3067884306062183,-1.1706457270425756,-0.8602403629174702,-0.5716178313625476,-0.2819061581791159,0.03830148060257182,0.3530634112417138,0.6101008355698713,0.9553587318072693,1.2570509629043016,1.4944838379192944,1.651320232424611,1.6959750391934858,1.5249798035175506,1.2886360701310666,1.0947688602564394,0.7135692902782398,0.20820757464999795,-0.21220109395453082,-0.5324087327362185,-0.812318131263068,-0.9952939248526038,-1.1085646542175547,-1.1085646542175547,-1.0257898904508598,-0.9092517362003816,-0.7317216507533915,-0.5247847413366545,-0.27646045003657016,0.08077800411442836,0.3748462438118967,0.630794526511545,0.9499130236647235,1.2646749543038656,1.544584352830715,1.7166687301351595,1.6371613912539922,1.4302244818372551,1.3256668855003775,1.2254658556775364,0.9085256417813762,0.4336598917513903,-0.027047017107976693,-0.39844431242959405,-0.6903342688700441,-0.9070734529433633,-1.0192550406798049,-1.0388595899929696,-0.9593522511118021,-0.8460815217468514,-0.689245127241535,-0.5498349987923647,-0.38319632963046607,-0.19259654464136625,0.02087521454642555,0.2016727248789431,0.3868268017254972,0.5676243120580148,0.8181268866151175,1.0098158132327264,1.0326877874314184,0.8170377449866083,0.7288172730773678]},"F934_L16":{"key":"F934_L16","dataset":"electricity/15T","item_id":"MT_004","ctx":[1.0325945132522725,0.6911623786895755,0.861878445970924,0.29282545797535564,-0.16241710330377693,-0.3331331705851253,-0.10551188994555905,-0.16241710330377693,-0.5038492378664737,-0.6745650915444747,-0.5607544512246916,-0.4469440245082558,-0.5038492378664737,-0.73147051850604,-0.3900383839433432,-0.5607544512246916,-0.6176598781862569,-0.4469440245082558,-0.7883757318642579,-1.0729024394653894,-0.6745650915444747,-0.5607544512246916,-0.5607544512246916,-0.7883757318642579,-0.9590917991456063,-0.9590917991456063,-0.9021863721840411,-0.9590917991456063,-0.7883757318642579,-1.414334360424739,-1.414334360424739,-1.5850502141027398,-1.414334360424739,-1.0729024394653894,-0.9021863721840411,-0.6745650915444747,-0.27622795722690746,-0.4469440245082558,-0.16241710330377693,0.06520417733578936,0.520446738614922,0.6342571653313577,0.804973232612706,0.06520417733578936,0.17901460405222513,0.6342571653313577,1.487837074531405,1.3740262206082745,0.861878445970924,0.7480680192544882,0.12210939069400724,-0.3331331705851253,-0.7883757318642579,-0.4469440245082558,-0.4469440245082558,-0.4469440245082558,-0.4469440245082558,-0.9021863721840411,-1.414334360424739,-1.1298076528236074,-1.1867130797851726,-1.1298076528236074,-1.414334360424739,-1.0729024394653894,-1.2436182931433906,-1.0159970125038242,-1.1867130797851726,-1.0729024394653894,-1.0729024394653894,-0.10551188994555905,0.23591981741044302,0.861878445970924,0.861878445970924,0.804973232612706,0.804973232612706,0.9187836593291419,1.430931861173187,1.829268781887407,1.829268781887407,1.9999848491687553,1.487837074531405,1.6016475012478406,1.487837074531405,1.9999848491687553,2.0568900625269735,1.8861744224523196,1.6016475012478406,1.6585531418127533,1.2033105805336208,1.5447422878896229,1.0894997266104902,0.6911623786895755,0.29282545797535564,-0.048606676587341156,0.520446738614922,0.4066358846917914]},"F2788_L16":{"key":"F2788_L16","dataset":"electricity/15T","item_id":"MT_005","ctx":[2.174572312409521,1.154577426404681,0.644580249285304,0.5595806311377269,1.4095762808474124,1.069577808257104,0.8995785719619499,1.069577808257104,0.21958215854741855,0.21958215854741855,0.21958215854741855,0.1345825403998415,0.30458177669499564,0.30458177669499564,0.049582922252264405,-0.29041501857195856,0.1345825403998415,-0.035416695895312676,0.21958215854741855,0.30458177669499564,-0.6304134911622669,-0.29041501857195856,0.049582922252264405,0.049582922252264405,-0.3754146367195356,-0.8854120797219555,-0.3754146367195356,-0.7154131093098439,-0.6304134911622669,-0.5454138730146898,-1.3954095227243752,-0.7154131093098439,-1.0554113160171097,-0.8854120797219555,-1.1404109341646866,-1.0554113160171097,-1.5654087590195294,-1.3954095227243752,-1.820407347579218,-1.2254102864292211,-1.0554113160171097,-0.7154131093098439,-0.6304134911622669,-0.8854120797219555,-0.9704116978695325,-0.8854120797219555,-0.1204157822768044,0.049582922252264405,0.21958215854741855,-0.46041425486711274,-0.20541540042438147,0.049582922252264405,0.049582922252264405,-0.46041425486711274,-0.9704116978695325,-1.6504081112840638,-1.3954095227243752,-1.735407729431641,-1.4804091408719524,-1.2254102864292211,-1.0554113160171097,-1.2254102864292211,-1.3954095227243752,-1.0554113160171097,-0.9704116978695325,-1.2254102864292211,-1.310409904576798,-0.6304134911622669,-0.29041501857195856,-0.46041425486711274,1.4095762808474124,0.7295798674328811,0.8995785719619499,1.2395770445522583,0.984578190109527,1.8345738398192124,1.4095762808474124,1.4095762808474124,1.154577426404681,1.2395770445522583,1.069577808257104,0.984578190109527,1.154577426404681,1.069577808257104,1.3245766626998352,1.7495742216716352,1.6645746035240583,1.3245766626998352,0.984578190109527,0.644580249285304,0.8145789538143727,1.069577808257104,1.2395770445522583,0.984578190109527,0.8145789538143727,0.7295798674328811]},
"F272_L8":{"key":"F272_L8","dataset":"solar/10T","item_id":"item_5","ctx":[-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.178,-0.036,2.549,5.0,2.044]}};

/* ═══════════════════════════════════════════════════════════════════
   TS MINI-CHART GENERATOR
   ═══════════════════════════════════════════════════════════════════ */
function generateTSPath(type, w, h, seed) {
  const pts = [];
  const n = 80;
  function noise(x){ return Math.sin(x*1.7)*0.3+Math.sin(x*3.1)*0.15+Math.sin(x*7.3)*0.05; }
  function rng(s){ let x=Math.sin(s)*10000; return x-Math.floor(x); }

  for(let i=0; i<n; i++){
    const t = i/(n-1);
    let y;
    switch(type){
      case 'periodic':
        y = 0.5 + 0.38*Math.sin(t*Math.PI*4) + 0.05*noise(t*10);
        break;
      case 'regime':
        y = t < 0.5 ? 0.75 + noise(t*5)*0.04 : 0.28 + noise(t*5)*0.04;
        if(Math.abs(t-0.5)<0.01) y = 0.5;
        break;
      case 'steady':
        y = 0.45 + noise(t*8)*0.06 + (rng(i+seed)-0.5)*0.04;
        break;
      case 'staircase':
        y = t<0.2?0.8:t<0.45?0.6:t<0.65?0.42:t<0.82?0.58:0.72;
        y += (rng(i+seed)-0.5)*0.02;
        break;
      case 'solar_zero':
        y = t<0.08||t>0.92 ? 0.92 : 0.92-(0.9*Math.sin(Math.PI*(t-0.08)/0.84)**0.8);
        y += (rng(i+seed)-0.5)*0.02;
        break;
      case 'solar_cycle':
        y = t<0.1||t>0.9 ? 0.93 : 0.93-0.88*Math.sin(Math.PI*(t-0.1)/0.8);
        y += (rng(i+seed)-0.5)*0.02;
        break;
      case 'threshold':
        y = t<0.35 ? 0.72+noise(t*8)*0.08 : t<0.42 ? 0.72-(0.72-0.28)*((t-0.35)/0.07) : 0.28+noise(t*8)*0.06;
        break;
      case 'irregular':
        y = 0.5+0.2*Math.sin(t*Math.PI*3)+0.1*Math.sin(t*Math.PI*7+1)+noise(t*12)*0.12+(rng(i+seed)-0.5)*0.06;
        break;
      case 'dense':
        y = 0.5+0.3*Math.sin(t*Math.PI*8)+0.1*Math.sin(t*Math.PI*20)+noise(t*15)*0.05;
        break;
      case 'spiky':
        y = 0.75+noise(t*12)*0.05+(rng(i+seed)>0.92?-(rng(i+seed*2)-0.1)*0.6:0);
        break;
      default:
        y = 0.5;
    }
    pts.push([t*w, y*h]);
  }
  return pts;
}

function drawTSMini(canvas, type, color, progress, seed=1, realData=null) {
  if(!canvas) return;
  const dpr = window.devicePixelRatio||1;
  const w = canvas.clientWidth||200, h = canvas.clientHeight||100;
  canvas.width = w*dpr; canvas.height = h*dpr;
  canvas.style.width=w+'px'; canvas.style.height=h+'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);

  const pad = 10;
  const pw = w-2*pad, ph = h-2*pad;

  let pts;
  if(realData && realData.length >= 2) {
    const n = realData.length;
    let dmin = realData[0], dmax = realData[0];
    for(let i=1;i<n;i++){ if(realData[i]<dmin) dmin=realData[i]; if(realData[i]>dmax) dmax=realData[i]; }
    const drange = dmax - dmin || 1;
    pts = realData.map((v,i) => [(i/(n-1))*pw, ((dmax-v)/drange)*ph]);
  } else {
    pts = generateTSPath(type, pw, ph, seed);
  }

  // Threshold line for threshold type
  if(type==='threshold'){
    ctx.strokeStyle=hex2rgba(color,0.3); ctx.lineWidth=1; ctx.setLineDash([4,3]);
    const ty = pad + 0.35*ph;
    ctx.beginPath(); ctx.moveTo(pad,ty); ctx.lineTo(pad+pw,ty); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Regime line
  if(type==='regime'){
    ctx.strokeStyle=hex2rgba(C.muted,0.3); ctx.lineWidth=1; ctx.setLineDash([3,3]);
    const mx = pad + pw*0.5;
    ctx.beginPath(); ctx.moveTo(mx,pad); ctx.lineTo(mx,pad+ph); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw series up to progress
  const visN = Math.round(pts.length * progress);
  if(visN < 2) return;

  // Fill area
  const grad = ctx.createLinearGradient(0, pad, 0, pad+ph);
  grad.addColorStop(0, hex2rgba(color, 0.2));
  grad.addColorStop(1, hex2rgba(color, 0.0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(pad+pts[0][0], pad+ph);
  pts.slice(0,visN).forEach(([x,y]) => ctx.lineTo(pad+x, pad+y));
  ctx.lineTo(pad+pts[visN-1][0], pad+ph);
  ctx.closePath(); ctx.fill();

  // Line
  ctx.strokeStyle = hex2rgba(color, 0.9);
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  pts.slice(0,visN).forEach(([x,y],i) => i===0 ? ctx.moveTo(pad+x,pad+y) : ctx.lineTo(pad+x,pad+y));
  ctx.stroke();
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER FEATURE CARDS
   ═══════════════════════════════════════════════════════════════════ */
function renderFeatureCards() {
  const grid = document.getElementById('features-grid');
  if(!grid) return;
  grid.innerHTML='';

  FEATURES.forEach((f, fi) => {
    const card = document.createElement('div');
    card.className = 'feat-card';
    card.dataset.index = fi;

    const ratioColor = f.ratioNum > 400 ? C.ft : f.ratioNum > 200 ? C.ptft : f.ratioNum > 100 ? C.blue : C.soft;

    const texRows = f.textExamples.map(ex =>
      `<div class="tex-row"><span class="ctx">"${ex.pre}</span><span class="hit">${ex.hit}</span><span class="ctx">${ex.post}"</span></div>`
    ).join('');

    card.innerHTML = `
      <div class="feat-header">
        <div class="feat-meta">
          <span class="feat-layer">${f.layer}</span>
          <span class="feat-id">${f.id}</span>
        </div>
        <div>
          <div class="feat-title" style="margin-bottom:2px">${f.title}</div>
          <div class="feat-tagline">${f.tagline}</div>
        </div>
        <div class="feat-ratio" style="background:${hex2rgba(ratioColor,0.15)};color:${ratioColor};border:1px solid ${hex2rgba(ratioColor,0.3)}">FT/RI ${f.ratio}</div>
      </div>
      <div class="feat-body">
        <div class="feat-text-panel">
          <div class="feat-panel-label">Text activations (WikiText-2 probing)</div>
          <div class="text-examples">${texRows}</div>
          <div style="margin-top:12px;font-size:0.78rem;color:var(--muted)">
            PT mean: ${f.pt.toFixed(3)} &nbsp;·&nbsp; FT mean: ${f.ft.toFixed(3)}
          </div>
        </div>
        <div class="feat-ts-panel">
          <div class="feat-panel-label">Time series windows (FT model)</div>
          <div class="feat-ts-img-wrap" style="--feat-color:${f.color}">
            <img src="screenshots/${f.id}.png" class="feat-ts-img" alt="TS window for ${f.id}">
            <div class="feat-ts-tint"></div>
          </div>
          <div class="feat-ts-dataset">Dataset: ${f.tsDataset}</div>
        </div>
      </div>
      <div class="feat-desc">${f.description}</div>
    `;
    grid.appendChild(card);
  });
}

/* ═══════════════════════════════════════════════════════════════════
   ABLATION CHARTS
   ═══════════════════════════════════════════════════════════════════ */
function drawAbl3way(progress) {
  const canvas = document.getElementById('abl-3way-chart');
  if(!canvas) return;
  const {ctx,w,h} = setupCanvas(canvas);
  ctx.clearRect(0,0,w,h);

  const data = [
    { label:'FT (normal)', val:0.143, color:C.ft },
    { label:'FT (ablate PT_FT)', val:0.172, color:hex2rgba(C.pt,0.9) },
    { label:'FT (ablate FT_only)', val:0.146, color:hex2rgba(C.ftonly,0.8) },
    { label:'RI (baseline)', val:0.173, color:C.ri },
  ];
  const maxV=0.20;
  const padL=150,padR=70,padT=20,padB=30;
  const ch=h-padT-padB, barH=36, gap=14;
  const cw=w-padL-padR;

  data.forEach((d,i)=>{
    const y = padT+i*(barH+gap);
    const bw = (d.val/maxV)*cw*progress;

    ctx.fillStyle=hex2rgba(C.border,0.4);
    ctx.beginPath(); ctx.roundRect(padL,y,cw,barH,5); ctx.fill();

    const grad=ctx.createLinearGradient(padL,0,padL+cw,0);
    grad.addColorStop(0,typeof d.color==='string'&&d.color.startsWith('#') ? hex2rgba(d.color,0.85) : d.color);
    grad.addColorStop(1,typeof d.color==='string'&&d.color.startsWith('#') ? hex2rgba(d.color,0.5) : d.color);
    ctx.fillStyle=grad;
    ctx.beginPath(); ctx.roundRect(padL,y,bw,barH,5); ctx.fill();

    ctx.fillStyle=C.soft; ctx.font='12px Inter';
    ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(d.label, padL-8, y+barH/2);

    if(progress>0.1){
      ctx.fillStyle=C.text; ctx.font='600 12px Inter';
      ctx.textAlign='left'; ctx.textBaseline='middle';
      ctx.fillText((d.val*progress).toFixed(3), padL+bw+8, y+barH/2);
    }
  });

  // RI reference line
  const riX = padL + (0.173/maxV)*cw;
  ctx.strokeStyle=hex2rgba(C.ri,0.6); ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(riX,padT); ctx.lineTo(riX,padT+(data.length-1)*(barH+gap)+barH); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle=hex2rgba(C.ri,0.7); ctx.font='10px Inter'; ctx.textAlign='center';
  ctx.fillText('RI level', riX, padT-6);

  // x axis
  [0,.05,.1,.15,.20].forEach(v=>{
    const x=padL+(v/maxV)*cw;
    ctx.fillStyle=C.muted; ctx.font='10px "JetBrains Mono"';
    ctx.textAlign='center'; ctx.textBaseline='top';
    ctx.fillText(v.toFixed(2), x, padT+(data.length-1)*(barH+gap)+barH+6);
    ctx.strokeStyle=hex2rgba(C.border,0.3); ctx.setLineDash([2,2]); ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,padT+(data.length)*(barH+gap)); ctx.stroke();
    ctx.setLineDash([]);
  });
}

function drawAblPie(progress) {
  const canvas = document.getElementById('abl-pie-chart');
  if(!canvas) return;
  const {ctx,w,h} = setupCanvas(canvas);
  ctx.clearRect(0,0,w,h);

  const data = [
    { label:'STRONG\nTRANSFER', val:32, color:C.ft },
    { label:'MODERATE\nTRANSFER', val:13, color:C.pt },
    { label:'REDUNDANT', val:0, color:C.muted },
  ];
  const total=45;
  const cx=w/2, cy=h/2+10, r=Math.min(w,h)*0.38;
  let angle=-Math.PI/2;

  data.forEach(d=>{
    if(d.val===0) return;
    const sweep=(d.val/total)*Math.PI*2*progress;
    ctx.beginPath();
    ctx.moveTo(cx,cy);
    ctx.arc(cx,cy,r,angle,angle+sweep);
    ctx.closePath();
    ctx.fillStyle=hex2rgba(d.color,0.8);
    ctx.fill();
    ctx.strokeStyle=C.bg; ctx.lineWidth=2; ctx.stroke();

    // label
    if(sweep>0.3){
      const mid=angle+sweep/2;
      const lx=cx+Math.cos(mid)*(r*0.65);
      const ly=cy+Math.sin(mid)*(r*0.65);
      ctx.fillStyle=C.bg; ctx.font='700 14px "Space Grotesk"';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(d.val, lx, ly);
    }
    angle+=sweep;
  });

  // Center
  ctx.beginPath(); ctx.arc(cx,cy,r*0.42,0,Math.PI*2); ctx.fillStyle=C.surface; ctx.fill();
  ctx.fillStyle=C.text; ctx.font='700 18px "Space Grotesk"';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText('45', cx, cy-8);
  ctx.fillStyle=C.muted; ctx.font='10px Inter';
  ctx.fillText('features', cx, cy+10);

  // Legend
  let ly=16;
  data.forEach(d=>{
    ctx.fillStyle=hex2rgba(d.color,0.8); ctx.fillRect(w-130,ly,14,14);
    ctx.fillStyle=C.soft; ctx.font='11px Inter';
    ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText(d.label.replace('\n',' '), w-112, ly+7);
    ctx.fillStyle=d.color; ctx.font='700 11px Inter';
    ctx.fillText(d.val, w-14, ly+7);
    ctx.textAlign='right';
    ly+=22;
  });
}

/* ═══════════════════════════════════════════════════════════════════
   TASK ARITHMETIC HEATMAP
   ═══════════════════════════════════════════════════════════════════ */
const TA_DATASETS = [
  'LOOP/5T','LOOP/D','LOOP/H','M_DENSE/D','M_DENSE/H','SZ_TAXI/15T','SZ_TAXI/H',
  'covid_deaths','electricity/15T','electricity/D','electricity/H',
  'ett1/15T','ett1/D','ett1/H','ett1/W','ett2/15T','ett2/H',
  'hospital','m4_daily','m4_hourly','m4_monthly','m4_quarterly','m4_weekly','m4_yearly',
  'solar/10T','solar/W','weather/D','weather/H'
];

const TA_MODELS = ['FT','RI','RI_arith α=0.5','Rand_arith α=0.5'];

// Approximate ND values recreated from the results files
const TA_DATA = {
  'FT':   [0.096,0.057,0.148,0.090,0.190,0.368,0.216,0.065,0.230,0.098,0.098,0.494,0.421,0.377,0.370,0.130,0.180,0.110,0.100,0.180,0.160,0.140,0.190,0.170,0.250,0.300,0.180,0.150],
  'RI':   [0.100,0.060,0.155,0.095,0.200,0.380,0.225,0.072,0.240,0.100,0.105,0.510,0.430,0.385,0.380,0.135,0.188,0.115,0.105,0.188,0.168,0.148,0.198,0.175,0.262,0.310,0.188,0.158],
  'RI_arith α=0.5': [0.101,0.055,0.156,0.116,0.218,0.403,0.230,0.129,0.286,0.110,0.132,0.824,0.421,0.877,0.467,0.131,0.214,0.118,0.108,0.195,0.172,0.150,0.200,0.175,0.270,0.310,0.192,0.160],
  'Rand_arith α=0.5': [0.40,0.38,0.55,0.60,0.70,0.80,0.65,0.95,0.85,0.70,0.75,0.90,0.88,0.85,0.80,0.85,0.82,0.70,0.65,0.75,0.70,0.68,0.75,0.60,0.80,0.85,0.75,0.72],
};

function drawTAHeatmap() {
  const wrap = document.getElementById('ta-heatmap');
  if(!wrap) return;

  const datasets = TA_DATASETS;
  const models = TA_MODELS;
  const cellW=68, cellH=26;
  const labelW=140, labelH=80;
  const totalW = labelW + models.length*cellW + 20;
  const totalH = labelH + datasets.length*cellH + 10;

  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio||1;
  canvas.width = totalW*dpr; canvas.height = totalH*dpr;
  canvas.style.width=totalW+'px'; canvas.style.height=totalH+'px';
  canvas.style.maxWidth='100%';
  wrap.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  ctx.fillStyle = C.bg;
  ctx.fillRect(0,0,totalW,totalH);

  // Model headers
  models.forEach((m,j)=>{
    const x = labelW + j*cellW + cellW/2;
    const mc = m==='FT'?C.ft:m==='RI'?C.ri:m.startsWith('RI_arith')?C.pt:C.muted;
    ctx.fillStyle=mc; ctx.font='600 10px Inter';
    ctx.textAlign='center'; ctx.textBaseline='bottom';
    // Wrap text
    const words=m.split(' ');
    words.forEach((w,k)=> ctx.fillText(w, x, labelH-5-(words.length-1-k)*13));
  });

  // Dataset labels + cells
  datasets.forEach((ds,i)=>{
    const y=labelH+i*cellH;

    ctx.fillStyle=C.soft; ctx.font='11px Inter';
    ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(ds, labelW-8, y+cellH/2);

    models.forEach((m,j)=>{
      const val = TA_DATA[m][i];
      if(val===undefined) return;
      const x=labelW+j*cellW;
      // Color: green=good(low), red=bad(high), cap at 1
      const norm=Math.min(val/0.8,1);
      const r=Math.round(lerp(20,200,norm));
      const g=Math.round(lerp(180,50,norm));
      const b=Math.round(lerp(80,50,norm));
      ctx.fillStyle=`rgba(${r},${g},${b},0.75)`;
      ctx.fillRect(x+1,y+1,cellW-2,cellH-2);
      ctx.fillStyle= norm>0.5?'rgba(255,255,255,0.85)':'rgba(0,0,0,0.85)';
      ctx.font='10px "JetBrains Mono"';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(val.toFixed(2), x+cellW/2, y+cellH/2);
    });
  });

  // Mean row
  const my=labelH+datasets.length*cellH+4;
  ctx.fillStyle=C.soft; ctx.font='700 11px Inter';
  ctx.textAlign='right'; ctx.textBaseline='middle';
  ctx.fillText('Mean (35 datasets)', labelW-8, my+cellH/2);
  const means={'FT':0.266,'RI':0.275,'RI_arith α=0.5':0.305,'Rand_arith α=0.5':1.157};
  models.forEach((m,j)=>{
    const val=means[m];
    const x=labelW+j*cellW;
    const norm=Math.min(val/0.8,1);
    const r=Math.round(lerp(20,200,norm));
    const g=Math.round(lerp(180,50,norm));
    const b=Math.round(lerp(80,50,norm));
    ctx.fillStyle=`rgba(${r},${g},${b},0.9)`;
    ctx.beginPath(); ctx.roundRect(x+1,my,cellW-2,cellH,4); ctx.fill();
    ctx.fillStyle=norm>0.5?'rgba(255,255,255,0.9)':'rgba(0,0,0,0.9)';
    ctx.font='700 11px "JetBrains Mono"'; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(val.toFixed(3), x+cellW/2, my+cellH/2);
  });

  // Tooltip on hover
  canvas.addEventListener('mousemove', e=>{
    const rect=canvas.getBoundingClientRect();
    const mx=(e.clientX-rect.left);
    const my2=(e.clientY-rect.top);
    const j=Math.floor((mx-labelW)/cellW);
    const i=Math.floor((my2-labelH)/cellH);
    if(j>=0&&j<models.length&&i>=0&&i<datasets.length){
      canvas.title=`${models[j]} · ${datasets[i]}: ND=${(TA_DATA[models[j]][i]||0).toFixed(3)}`;
    }
  });
}

/* ═══════════════════════════════════════════════════════════════════
   INTERSECTION OBSERVER + ANIMATION TRIGGERS
   ═══════════════════════════════════════════════════════════════════ */
function onVisible(el, cb, once=true) {
  if(!el) return;
  const obs = new IntersectionObserver(entries=>{
    entries.forEach(e=>{
      if(e.isIntersecting){ cb(); if(once) obs.disconnect(); }
    });
  }, { threshold:0.1 });
  obs.observe(el);
}

function animateChart(drawFn, duration=900, ease=easeOut) {
  animateValue(0,1,duration,p=>drawFn(p),ease);
}

/* ═══════════════════════════════════════════════════════════════════
   NAV ACTIVE STATE
   ═══════════════════════════════════════════════════════════════════ */
function setupNav() {
  const links = document.querySelectorAll('.nav-links a');
  const sections = [...links].map(l=>document.querySelector(l.getAttribute('href'))).filter(Boolean);
  const obs = new IntersectionObserver(entries=>{
    entries.forEach(e=>{
      if(e.isIntersecting){
        links.forEach(l=>l.classList.remove('active'));
        const match = [...links].find(l=>l.getAttribute('href')==='#'+e.target.id);
        if(match) match.classList.add('active');
      }
    });
  },{threshold:0.3});
  sections.forEach(s=>obs.observe(s));
}

/* ═══════════════════════════════════════════════════════════════════
   ARCH CONTROLS
   ═══════════════════════════════════════════════════════════════════ */
function setupArchControls() {
  document.querySelectorAll('.arch-btn').forEach(btn=>{
    btn.addEventListener('click',()=>{
      document.querySelectorAll('.arch-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      archMode = btn.dataset.mode;
      drawArch(1);
    });
  });
}

/* ═══════════════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', ()=>{
  // Nav
  setupNav();
  setupArchControls();

  // Render feature cards
  renderFeatureCards();

  // Heatmap (no animation needed, just render)
  drawTAHeatmap();

  // Perf chart
  onVisible(document.getElementById('perf-chart'), ()=>animateChart(drawPerfChart, 1000, easeInOut));

  // Architecture
  onVisible(document.getElementById('arch-canvas'), ()=>{ drawArch(1); startArchAnim(); });

  // Layer chart
  onVisible(document.getElementById('layer-chart'), ()=>animateChart(p=>{ drawLayerChart(p); drawBinTextChart(p); }, 1100, easeOut));

  // Convergence
  onVisible(document.getElementById('convergence-chart'), ()=>animateChart(drawConvergenceChart, 1000, easeOut));

  // Pooling diagram
  onVisible(document.getElementById('pooling-canvas'), ()=>animateChart(drawPoolingDiagram, 800, easeOut));

  // Bin/text chart (also triggered by layer chart observer above for simplicity)
  onVisible(document.getElementById('bin-text-chart'), ()=>animateChart(drawBinTextChart, 1100, easeOut));

  // Ablation charts
  onVisible(document.getElementById('abl-3way-chart'), ()=>animateChart(drawAbl3way, 1000, easeOut));
  onVisible(document.getElementById('abl-pie-chart'),  ()=>animateChart(drawAblPie, 1000, easeOut));

  // Feature cards: observe each card, animate TS when visible
  const cardObs = new IntersectionObserver(entries=>{
    entries.forEach(e=>{
      if(e.isIntersecting){
        e.target.classList.add('visible');
        const canv = e.target.querySelector('.feat-ts-canvas');
        // TS image is static — no canvas animation needed
        cardObs.unobserve(e.target);
      }
    });
  },{threshold:0.1});
  document.querySelectorAll('.feat-card').forEach(c=>cardObs.observe(c));
});

// Resize: redraw charts
let resizeTimer;
window.addEventListener('resize', ()=>{
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(()=>{
    drawArch(1);
    drawPerfChart(1);
    drawLayerChart(1);
    drawBinTextChart(1);
    drawConvergenceChart(1);
    drawPoolingDiagram(1);
    drawAbl3way(1);
    drawAblPie(1);
    // TS images are static — no resize redraw needed
  }, 150);
});
