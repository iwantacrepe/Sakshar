// Tap-to-talk: MediaRecorder (webm/opus) -> POST /api/documents/{id}/voice -> transcript + spoken answer.
(function () {
  const btn = document.getElementById('mic');
  if (!btn) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    btn.disabled = true;
    btn.title = 'Voice recording is not supported in this browser';
    return;
  }
  const labelEl = btn.querySelector('span');
  let rec = null, chunks = [], stream = null, timer = null;

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch (e) {
      alert('Microphone permission is needed to ask by voice.');
      return;
    }
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : '';
    rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    chunks = [];
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    rec.onstop = send;
    rec.start(250);
    btn.classList.add('recording');
    labelEl.textContent = btn.dataset.stop;
    timer = setTimeout(stop, 30000); // hard cap: 30 s
  }

  function stop() {
    clearTimeout(timer);
    if (rec && rec.state !== 'inactive') rec.stop();
    stream?.getTracks().forEach(t => t.stop());
    btn.classList.remove('recording');
    labelEl.textContent = btn.dataset.label;
  }

  async function send() {
    const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
    if (blob.size < 2000) return;
    const body = new FormData();
    body.append('audio', blob, 'question.webm');
    body.append('lang', btn.dataset.lang);
    btn.disabled = true;
    labelEl.textContent = '…';
    try {
      const res = await fetch(`/api/documents/${btn.dataset.doc}/voice`, { method: 'POST', body });
      const html = await res.text();
      const log = document.getElementById('chat-log');
      log.insertAdjacentHTML('beforeend', html);
      htmx.process(log);
      log.lastElementChild?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      const cites = log.lastElementChild?.dataset.cites;
      if (cites && window.saksharHighlight) window.saksharHighlight(cites.split(/\s+/), false);
    } catch (e) {
      alert('Could not send the recording. Please try again.');
    } finally {
      btn.disabled = false;
      labelEl.textContent = btn.dataset.label;
    }
  }

  btn.addEventListener('click', () => (rec && rec.state === 'recording') ? stop() : start());
})();
