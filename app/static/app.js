const $ = id => document.getElementById(id);
const labels = {classical: 'Classical', learned: 'Learned', hybrid: 'Hybrid', optimized: 'Hybrid · FP16', reference: 'Reference'};
const descriptions = {classical: 'Traditional processing', learned: 'Neural reconstruction', hybrid: 'Classical + learned', optimized: 'Reduced precision', reference: 'Clean target render'};
const sceneInfo = {'001': ['Samsung S6', 'Low light'], '003': ['Nexus 6', 'Normal light'], '004': ['Google Pixel', 'Indoor scene'], '005': ['iPhone 7', 'Low light']};
let currentData = null, activeImage = 'hybrid', viewMode = 'compare', isBusy = false, timer = null;

function error(message) {
  $('error-message').textContent = message;
  $('error').hidden = !message;
}
$('dismiss-error').onclick = () => error('');

function busy(value) {
  isBusy = value;
  $('busy').hidden = !value;
  $('studio').setAttribute('aria-busy', String(value));
  document.querySelectorAll('.primary, .sample-card, #quick-start, #adaptive, #fp16').forEach(control => control.disabled = value);
  clearInterval(timer);
  if (value) {
    const started = Date.now();
    $('elapsed').textContent = '0 s';
    timer = setInterval(() => $('elapsed').textContent = `${Math.floor((Date.now() - started) / 1000)} s`, 1000);
  }
}

function zoom(url, caption) {
  $('large-image').src = url;
  $('large-caption').textContent = caption;
  $('lightbox').showModal();
}
$('close-lightbox').onclick = () => $('lightbox').close();
$('lightbox').onclick = event => { if (event.target === $('lightbox')) $('lightbox').close(); };
$('zoom-current').onclick = () => { if (currentData) zoom(currentData.images[activeImage], labels[activeImage]); };

function setSplit(value) {
  const split = Math.max(0, Math.min(100, Number(value)));
  $('compare-range').value = split;
  $('compare-stage').style.setProperty('--split', `${split}%`);
  $('compare-range').setAttribute('aria-valuetext', `${Math.round(split)} percent baseline visible`);
}
$('compare-range').oninput = event => setSplit(event.target.value);
function moveSplit(event) {
  const rect = $('compare-stage').getBoundingClientRect();
  setSplit((event.clientX - rect.left) / rect.width * 100);
}
$('compare-stage').onpointerdown = event => {
  if (viewMode !== 'compare') return;
  $('compare-stage').setPointerCapture(event.pointerId);
  moveSplit(event);
};
$('compare-stage').onpointermove = event => {
  if ($('compare-stage').hasPointerCapture(event.pointerId)) moveSplit(event);
};
function setViewMode(mode) {
  viewMode = mode;
  $('compare-stage').classList.toggle('single', mode === 'single');
  $('mode-compare').setAttribute('aria-pressed', String(mode === 'compare'));
  $('mode-single').setAttribute('aria-pressed', String(mode === 'single'));
  $('compare-range').disabled = mode === 'single';
  $('compare-range').style.visibility = mode === 'single' ? 'hidden' : 'visible';
  $('viewer-help').textContent = mode === 'single' ? 'Selected output' : 'Drag to compare';
}
$('mode-compare').onclick = () => setViewMode('compare');
$('mode-single').onclick = () => setViewMode('single');

function selectImage(key) {
  if (!currentData?.images[key]) return;
  activeImage = key;
  const baseline = key === 'classical' ? 'sensor_preview' : 'classical';
  $('viewer-after').src = currentData.images[key];
  $('viewer-after').alt = `${labels[key]} reconstruction`;
  $('viewer-before').src = currentData.images[baseline];
  $('viewer-before').alt = baseline === 'sensor_preview' ? 'Sensor preview' : 'Classical reconstruction';
  $('left-label').textContent = baseline === 'sensor_preview' ? 'Sensor preview' : 'Classical';
  $('right-label').textContent = labels[key];
  document.querySelectorAll('[data-image]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.image === key)));
  document.querySelectorAll('#metrics tr').forEach(row => row.classList.toggle('selected-row', row.dataset.method === key));
  const metric = currentData.metrics[key];
  $('quick-psnr').textContent = metric ? (metric.psnr === null ? '∞ dB' : `${metric.psnr.toFixed(2)} dB`) : '—';
  $('quick-color').textContent = metric ? metric.delta_e_00.toFixed(2) : '—';
  const milliseconds = currentData.processing_ms[key];
  $('quick-time').textContent = milliseconds === undefined ? '—' : `${milliseconds.toFixed(0)} ms`;
}

function imageCard(key, url) {
  const card = document.createElement('button');
  card.type = 'button'; card.className = 'image-card'; card.dataset.image = key;
  card.setAttribute('aria-pressed', 'false');
  card.setAttribute('aria-label', `View ${labels[key]}`);
  const img = document.createElement('img'); img.src = url; img.alt = ''; img.draggable = false;
  const title = document.createElement('span'); title.className = 'card-caption'; title.textContent = labels[key];
  const detail = document.createElement('small'); detail.textContent = descriptions[key]; title.append(detail);
  card.append(img, title); card.onclick = () => selectImage(key);
  return card;
}

function render(data) {
  currentData = data;
  $('empty').hidden = true; $('results').hidden = false;
  $('output-grid').replaceChildren(); $('viewer-tabs').replaceChildren();
  for (const [key, name] of Object.entries(labels)) {
    if (!data.images[key]) continue;
    $('output-grid').append(imageCard(key, data.images[key]));
    if (key !== 'reference') {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = name;
      button.dataset.image = key; button.setAttribute('aria-pressed', 'false'); button.onclick = () => selectImage(key);
      $('viewer-tabs').append(button);
    }
  }
  $('workspace-caption').textContent = data.metadata.scene_id && sceneInfo[data.metadata.scene_id] ? sceneInfo[data.metadata.scene_id].join(' · ') : 'Your uploaded capture';
  $('dimensions').textContent = `${data.crop_yxhw[3]} × ${data.crop_yxhw[2]} · ${data.metadata.bayer_pattern} Bayer · ${data.adaptive ? 'Adaptive' : 'Fixed'}`;
  $('duration').textContent = `${(data.total_ms / 1000).toFixed(2)} s total`;
  $('download').hidden = false; $('download').href = data.download; $('download').download = 'PrismRAW-comparison.zip';
  $('reference-kind').textContent = data.reference_kind || 'No matching reference was supplied. Quality scores are unavailable; inspect the outputs visually.';
  $('metrics').replaceChildren();
  for (const [key, milliseconds] of Object.entries(data.processing_ms)) {
    const row = document.createElement('tr'); row.dataset.method = key;
    const m = data.metrics[key];
    const values = [labels[key], m ? (m.psnr === null ? '∞' : m.psnr.toFixed(2)) : '—', m ? m.ssim.toFixed(4) : '—', m ? m.delta_e_00.toFixed(2) : '—', m?.lpips !== undefined ? m.lpips.toFixed(4) : '—', `${milliseconds.toFixed(0)} ms`];
    values.forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); });
    $('metrics').append(row);
  }
  $('stage-grid').replaceChildren();
  for (const [name, url] of Object.entries({'Sensor preview': data.images.sensor_preview, ...data.intermediates})) {
    const block = document.createElement('div'), image = document.createElement('img'), caption = document.createElement('p');
    image.src = url; image.alt = name; image.tabIndex = 0; image.setAttribute('role', 'button');
    image.onclick = () => zoom(url, name);
    image.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); zoom(url, name); } };
    caption.textContent = name; block.append(image, caption); $('stage-grid').append(block);
  }
  $('settings').replaceChildren();
  for (const [name, value] of Object.entries(data.settings)) {
    const chip = document.createElement('span'); chip.textContent = `${name.replaceAll('_', ' ')}: ${value.toFixed(3)}`; $('settings').append(chip);
  }
  selectImage(data.images.hybrid ? 'hybrid' : 'classical');
  setSplit(50); setViewMode(viewMode);
}

async function request(url, options) {
  busy(true); error('');
  try {
    const response = await fetch(url, options);
    const text = await response.text();
    let data;
    try { data = JSON.parse(text); } catch { throw Error('The engine could not complete this request. Please try again or use a smaller RAW capture.'); }
    if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Check your file format, camera metadata and reference dimensions.');
    render(data);
    if (window.innerWidth <= 800) document.querySelector('.workspace').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (exception) { error(exception.message || 'Unable to reach the local engine.'); }
  finally { busy(false); }
}

function tab(upload) {
  $('sample-pane').hidden = upload; $('upload-pane').hidden = !upload;
  $('sample-tab').setAttribute('aria-selected', String(!upload)); $('upload-tab').setAttribute('aria-selected', String(upload));
  $('sample-tab').tabIndex = upload ? -1 : 0; $('upload-tab').tabIndex = upload ? 0 : -1;
}
$('sample-tab').onclick = () => tab(false); $('upload-tab').onclick = () => tab(true);
for (const button of [$('sample-tab'), $('upload-tab')]) button.onkeydown = event => {
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); const upload = button.id === 'sample-tab'; tab(upload); $(upload ? 'upload-tab' : 'sample-tab').focus(); }
};
function runSample() {
  if (isBusy || !$('scene').value) return;
  request(`/api/sample/${encodeURIComponent($('scene').value)}?adaptive=${$('adaptive').checked}&include_fp16=${$('fp16').checked}`, {method: 'POST'});
}
$('run-sample').onclick = runSample; $('quick-start').onclick = runSample;

function selectedFile() {
  const file = $('raw-file').files[0];
  $('file-status').hidden = true;
  if (!file) { $('file-status').hidden = true; return; }
  const extension = file.name.split('.').pop().toLowerCase();
  if (!['dng', 'npy', 'npz', 'mat'].includes(extension)) { error('Choose a DNG, NPY, NPZ or MAT sensor file. Ordinary JPEG/PNG photographs are not RAW inputs.'); $('raw-file').value = ''; return; }
  if (file.size > 64 * 1024 * 1024) { error('This sensor file exceeds the 64 MB upload limit.'); $('raw-file').value = ''; return; }
  error(''); const dng = extension === 'dng';
  $('metadata-field').hidden = dng; $('metadata-file').required = !dng; $('dng-note').hidden = !dng;
  $('file-status').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`; $('file-status').hidden = false;
}
$('raw-file').onchange = selectedFile;
for (const event of ['dragenter', 'dragover']) $('drop-zone').addEventListener(event, e => { e.preventDefault(); if (!isBusy) $('drop-zone').classList.add('dragging'); });
for (const event of ['dragleave', 'drop']) $('drop-zone').addEventListener(event, e => { e.preventDefault(); $('drop-zone').classList.remove('dragging'); });
$('drop-zone').addEventListener('drop', event => {
  if (isBusy || !event.dataTransfer.files.length) return;
  const transfer = new DataTransfer(); transfer.items.add(event.dataTransfer.files[0]); $('raw-file').files = transfer.files; selectedFile();
});
$('upload-pane').onsubmit = event => {
  event.preventDefault(); if (isBusy) return;
  const form = new FormData(event.target); form.set('adaptive', $('adaptive').checked); form.set('include_fp16', $('fp16').checked);
  if ($('raw-file').files[0]?.name.toLowerCase().endsWith('.dng')) form.delete('metadata');
  for (const key of ['metadata', 'reference']) if (!form.get(key)?.size) form.delete(key);
  request('/api/upload', {method: 'POST', body: form});
};

$('show-evaluation').onclick = async () => {
  const box = $('evaluation');
  if (!box.hidden) { box.hidden = true; $('show-evaluation').setAttribute('aria-expanded', 'false'); return; }
  try {
    const response = await fetch('/api/evaluation'); if (!response.ok) throw Error(); const data = await response.json(); box.replaceChildren();
    const note = document.createElement('p');
    if (!data.metrics) { note.textContent = data.status || 'Evaluation unavailable'; box.append(note); }
    else {
      note.textContent = `Held-out scenes ${data.scene_ids.join(', ')} · ${data.patch_count} patches · fixed clean-RAW rendering. CPU resource measurements use a separate 256 × 256 capture.`; box.append(note);
      const table = document.createElement('table'), head = document.createElement('tr');
      ['Pipeline', 'PSNR ↑', 'SSIM ↑', 'ΔE00 ↓', 'LPIPS ↓', 'Model size', 'CPU median', 'Peak process RAM'].forEach(value => { const cell = document.createElement('th'); cell.textContent = value; head.append(cell); }); table.append(head);
      for (const [key, name] of Object.entries(labels)) {
        const m = data.metrics[key]; if (!m) continue; const b = data.benchmark?.methods[key], row = document.createElement('tr');
        [name, m.psnr === null ? '∞' : m.psnr.toFixed(2), m.ssim.toFixed(4), m.delta_e_00.toFixed(2), m.lpips.toFixed(4), b?.model_bytes ? `${(b.model_bytes / 1e6).toFixed(3)} MB` : '—', b ? `${b.median_ms.toFixed(0)} ms` : '—', b ? `${b.peak_process_rss_mib.toFixed(0)} MiB` : '—'].forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }); table.append(row);
      }
      box.append(table); const footnote = document.createElement('p'); footnote.textContent = 'Local research results, not official rankings or smartphone measurements. FP16 reduces stored weights but is slower on this CPU. Camera colour calibration is approximate.'; box.append(footnote);
    }
    box.hidden = false; $('show-evaluation').setAttribute('aria-expanded', 'true');
  } catch { error('The measured evaluation report could not be loaded. Please try again.'); }
};

function chooseScene(id) {
  $('scene').value = id;
  document.querySelectorAll('[data-scene]').forEach(card => card.setAttribute('aria-pressed', String(card.dataset.scene === id)));
}
$('scene').onchange = () => chooseScene($('scene').value);
fetch('/api/health').then(response => { if (!response.ok) throw Error(); return response.json(); }).then(data => {
  $('connection').textContent = `Local engine · ${data.device}`;
  $('scene').replaceChildren(); $('sample-cards').replaceChildren();
  data.samples.forEach(sample => {
    const option = document.createElement('option'); option.value = sample.id; option.textContent = sample.name; $('scene').append(option);
    const info = sceneInfo[sample.id] || [sample.name, 'Sample capture'];
    const card = document.createElement('button'); card.type = 'button'; card.className = 'sample-card'; card.dataset.scene = sample.id; card.setAttribute('aria-pressed', 'false');
    const img = document.createElement('img'); img.src = `/static/samples/${sample.id}.jpg`; img.alt = ''; img.draggable = false;
    const text = document.createElement('span'); text.className = 'sample-text'; const title = document.createElement('strong'), subtitle = document.createElement('small'); title.textContent = info[0]; subtitle.textContent = info[1]; text.append(title, subtitle); card.append(img, text); card.onclick = () => chooseScene(sample.id); $('sample-cards').append(card);
  });
  if (data.samples.length) { chooseScene(data.samples[0].id); $('run-sample').disabled = false; $('quick-start').disabled = false; }
  else { $('run-sample').disabled = true; $('quick-start').disabled = true; $('connection').textContent = 'Local engine · sample assets missing'; }
  if (!data.models.learned || !data.models.hybrid) $('connection').textContent = 'Local engine · classical ready';
}).catch(() => error('The local engine is unavailable. Start run_local.ps1 and reload this page.'));
