// Draw the source region of each field that needs review into its <canvas>, straight from the
// deskewed page image (no server-side cropping; bbox is normalized 0..1).
(function () {
  function drawCrops(root) {
    root.querySelectorAll('canvas.crop').forEach(cv => {
      const img = document.querySelector(`.page[data-page="${cv.dataset.page}"] img`);
      if (!img) return;
      const draw = () => {
        const [x1, y1, x2, y2] = cv.dataset.bbox.split(',').map(Number);
        const W = img.naturalWidth, H = img.naturalHeight;
        const padX = 0.01 * W, padY = 0.006 * H;
        const sx = Math.max(0, x1 * W - padX), sy = Math.max(0, y1 * H - padY);
        const sw = Math.min(W - sx, (x2 - x1) * W + 2 * padX), sh = Math.min(H - sy, (y2 - y1) * H + 2 * padY);
        const maxW = cv.parentElement.clientWidth || 320;
        const scale = Math.min(3, maxW / sw, 110 / sh);
        cv.width = Math.round(sw * scale);
        cv.height = Math.round(sh * scale);
        cv.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, cv.width, cv.height);
      };
      img.complete && img.naturalWidth ? draw() : img.addEventListener('load', draw, { once: true });
    });
  }
  drawCrops(document);
  document.body.addEventListener('htmx:afterSwap', e => drawCrops(e.target));
})();
