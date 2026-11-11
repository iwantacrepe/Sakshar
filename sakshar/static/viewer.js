// Page image + SVG overlay of OCR blocks; click a statement / field / answer to highlight its source.
(function () {
  const dataEl = document.getElementById('blocks-data');
  if (!dataEl) return;
  const blocks = JSON.parse(dataEl.textContent);
  const SVGNS = 'http://www.w3.org/2000/svg';
  const rects = {};

  document.querySelectorAll('.page').forEach(pageEl => {
    const page = +pageEl.dataset.page;
    const svg = pageEl.querySelector('svg');
    blocks.filter(b => b.page === page).forEach(b => {
      const r = document.createElementNS(SVGNS, 'rect');
      const pad = 0.003;
      r.setAttribute('x', b.bbox.x1 - pad);
      r.setAttribute('y', b.bbox.y1 - pad);
      r.setAttribute('width', b.bbox.x2 - b.bbox.x1 + 2 * pad);
      r.setAttribute('height', b.bbox.y2 - b.bbox.y1 + 2 * pad);
      r.setAttribute('rx', 0.002);
      if (b.conf < 0.75) r.classList.add('lowconf');
      const t = document.createElementNS(SVGNS, 'title');
      t.textContent = `${b.id} · ${Math.round(b.conf * 100)}% · ${b.text}`;
      r.appendChild(t);
      svg.appendChild(r);
      rects[b.id] = r;
    });
  });

  const viewer = document.getElementById('viewer');
  document.getElementById('show-boxes')?.addEventListener('change', e =>
    viewer.classList.toggle('show-boxes', e.target.checked));

  function highlight(ids, scroll = true) {
    Object.values(rects).forEach(r => r.classList.remove('hl'));
    const hits = ids.map(id => rects[id]).filter(Boolean);
    hits.forEach(r => r.classList.add('hl'));
    if (scroll && hits.length) {
      const pageEl = hits[0].closest('.page');
      const img = pageEl.querySelector('img');
      const y = parseFloat(hits[0].getAttribute('y')) * img.clientHeight;
      const top = pageEl.getBoundingClientRect().top + window.scrollY + y - 140;
      window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
    }
  }
  window.saksharHighlight = highlight;

  document.addEventListener('click', e => {
    if (e.target.closest('form, button, input, audio, a')) return;
    const el = e.target.closest('[data-cites]');
    if (!el) return;
    document.querySelectorAll('[data-cites].active').forEach(x => x.classList.remove('active'));
    el.classList.add('active');
    highlight((el.dataset.cites || '').split(/\s+/).filter(Boolean));
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.target.matches('.stmt')) e.target.click();
  });

  // tabs
  document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
    document.querySelectorAll('.tab, .tabpane').forEach(x => x.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
    try { localStorage.setItem('sakshar-tab', tab.dataset.tab); } catch (_) {}
  }));
  try {
    const saved = localStorage.getItem('sakshar-tab');
    if (saved) document.querySelector(`.tab[data-tab="${saved}"]`)?.click();
  } catch (_) {}

  // "Listen" on the explanation
  document.addEventListener('click', async e => {
    const btn = e.target.closest('.listen');
    if (!btn) return;
    btn.disabled = true;
    const label = btn.textContent;
    btn.textContent = '…';
    try {
      const body = new FormData();
      body.append('text', btn.dataset.text);
      body.append('lang', btn.dataset.lang);
      const res = await fetch('/api/speak', { method: 'POST', body });
      if (!res.ok) throw new Error(await res.text());
      const { audio } = await res.json();
      new Audio(audio).play();
    } catch (err) {
      alert('Text-to-speech is not available right now.');
    } finally {
      btn.disabled = false;
      btn.textContent = label;
    }
  });
})();
