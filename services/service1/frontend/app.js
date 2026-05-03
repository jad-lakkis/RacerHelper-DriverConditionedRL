'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const trackSelect     = document.getElementById('track-select');
const lapInput        = document.getElementById('lap-input');
const lapError        = document.getElementById('lap-error');
const riskSlider      = document.getElementById('risk-slider');
const riskDisplay     = document.getElementById('risk-display');
const replayFile      = document.getElementById('replay-file');
const uploadText      = document.getElementById('upload-text');
const profilePanel    = document.getElementById('profile-panel');
const profileLoading  = document.getElementById('profile-loading');
const profileResults  = document.getElementById('profile-results');
const maxHoursSlider  = document.getElementById('max-hours');
const hoursDisplay    = document.getElementById('hours-display');
const generateBtn     = document.getElementById('generate-btn');
const generateResult  = document.getElementById('generate-result');
const jobPanel        = document.getElementById('job-panel');
const jobIdDisplay    = document.getElementById('job-id-display');
const jobStatusBadge  = document.getElementById('job-status-badge');
const jobResultsCount = document.getElementById('job-results-count');
const jobMetrics      = document.getElementById('job-metrics');
const jobBestLap      = document.getElementById('job-best-lap');
const jobDistance     = document.getElementById('job-distance');
const epsilonDisplay  = document.getElementById('epsilon-display');
const jobMessage      = document.getElementById('job-message');
const jobError        = document.getElementById('job-error');
const jobResultsList  = document.getElementById('job-results-list');
const stopBtn         = document.getElementById('stop-btn');

// ── State ─────────────────────────────────────────────────────────────────────
let profile      = null;
let currentJobId = null;
let pollTimer    = null;

const POLL_INTERVAL_MS = 10 * 60 * 1000;  // 10 minutes

// ── Lap time ──────────────────────────────────────────────────────────────────
const LAP_RE = /^\d{1,2}:\d{2}\.\d{3}$/;

function parseLapMs(str) {
  if (!LAP_RE.test(str.trim())) return null;
  const [minPart, rest] = str.trim().split(':');
  const [secPart, msPart] = rest.split('.');
  const ms = parseInt(minPart, 10) * 60000
           + parseInt(secPart, 10) * 1000
           + parseInt(msPart,  10);
  return isNaN(ms) || ms <= 0 ? null : ms;
}

function msToLapStr(ms) {
  const m   = Math.floor(ms / 60000);
  const s   = Math.floor((ms % 60000) / 1000);
  const rem = ms % 1000;
  return `${m}:${String(s).padStart(2,'0')}.${String(rem).padStart(3,'0')}`;
}

function validateLap() {
  const ms = parseLapMs(lapInput.value);
  if (!ms) {
    lapInput.classList.add('invalid');
    lapError.textContent = 'Use format M:SS.mmm, e.g. 1:26.290';
    return false;
  }
  lapInput.classList.remove('invalid');
  lapError.textContent = '';
  return true;
}

lapInput.addEventListener('input', () => {
  if (lapInput.value) validateLap();
  else { lapInput.classList.remove('invalid'); lapError.textContent = ''; }
  refreshGenerateBtn();
});

// ── Sliders ───────────────────────────────────────────────────────────────────
riskSlider.addEventListener('input', () => {
  riskDisplay.textContent = riskSlider.value + '%';
});

maxHoursSlider.addEventListener('input', () => {
  const h = parseFloat(maxHoursSlider.value);
  hoursDisplay.textContent = Number.isInteger(h) ? `${h} h` : `${h} h`;
});

// ── Load tracks ───────────────────────────────────────────────────────────────
async function loadTracks() {
  try {
    const res  = await fetch('/api/tracks');
    const data = await res.json();
    trackSelect.innerHTML = data
      .map(t => `<option value="${t.id}">${t.label}</option>`)
      .join('');
  } catch {
    trackSelect.innerHTML = '<option value="">Failed to load tracks</option>';
  }
  refreshGenerateBtn();
}

// ── Profile bars ──────────────────────────────────────────────────────────────
function setBar(barId, valId, value, min, max) {
  const pct = ((value - min) / (max - min)) * 100;
  document.getElementById(barId).style.width = Math.max(0, Math.min(100, pct)) + '%';
  document.getElementById(valId).textContent =
    value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
}

function setOusBar(value) {
  const bar = document.getElementById('bar-ous');
  const val = document.getElementById('val-ous');
  const halfPct = Math.abs(value) / 5 * 50;
  if (value >= 0) {
    bar.style.left  = '50%';
    bar.style.width = halfPct + '%';
  } else {
    bar.style.width = halfPct + '%';
    bar.style.left  = (50 - halfPct) + '%';
  }
  val.textContent = value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
}

function renderProfile(p) {
  setBar('bar-braking', 'val-braking', p.braking_aggression, 0, 1);
  setOusBar(p.oversteer_understeer_score);
  setBar('bar-corner', 'val-corner', p.corner_entry_speed_ratio, 0, 1);
  document.getElementById('val-corner').textContent =
    `${p.corner_entry_speed_ratio.toFixed(2)} (${p.corner_entry_speed_level})`;
}

// ── Replay upload ─────────────────────────────────────────────────────────────
replayFile.addEventListener('change', async () => {
  const file = replayFile.files[0];
  if (!file) return;

  uploadText.textContent = file.name;
  profile = null;
  profilePanel.classList.add('visible');
  profileLoading.classList.add('visible');
  profileResults.classList.remove('visible');
  refreshGenerateBtn();

  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch('/api/driver-profile', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    profile = data;
    renderProfile(profile);
    profileLoading.classList.remove('visible');
    profileResults.classList.add('visible');
  } catch (err) {
    profileLoading.classList.remove('visible');
    profilePanel.innerHTML =
      `<p class="field-error" style="margin-top:.75rem">&#9888; ${err.message}</p>`;
  }

  refreshGenerateBtn();
});

// ── Generate button state ─────────────────────────────────────────────────────
function refreshGenerateBtn() {
  const lapOk = parseLapMs(lapInput.value) !== null;
  generateBtn.disabled = !(lapOk && profile !== null && trackSelect.value);
}

trackSelect.addEventListener('change', refreshGenerateBtn);

// ── Generate / start training ─────────────────────────────────────────────────
generateBtn.addEventListener('click', async () => {
  if (!validateLap()) return;

  const payload = {
    track:                       trackSelect.value,
    best_lap_ms:                 parseLapMs(lapInput.value),
    risk_tolerance:              parseInt(riskSlider.value, 10) / 100,
    braking_aggression:          profile.braking_aggression,
    oversteer_understeer_score:  profile.oversteer_understeer_score,
    corner_entry_speed_ratio:    profile.corner_entry_speed_ratio,
    max_training_hours:          parseFloat(maxHoursSlider.value),
  };

  generateBtn.disabled    = true;
  generateBtn.textContent = 'Submitting…';
  generateResult.classList.remove('visible');
  jobPanel.classList.remove('visible');
  stopPolling();

  try {
    const res  = await fetch('/api/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      generateResult.innerHTML =
        `<div class="result-badge result-badge--err">&#9888; Error</div>
         <pre class="result-json">${JSON.stringify(data, null, 2)}</pre>`;
      generateResult.classList.add('visible');
    } else {
      currentJobId = data.job_id;
      showJobPanel(currentJobId, payload.max_training_hours);
      startPolling(currentJobId);
    }
  } catch (err) {
    generateResult.innerHTML =
      `<div class="result-badge result-badge--err">&#9888; Network Error</div>
       <p style="font-size:.875rem;color:#f87171">${err.message}</p>`;
    generateResult.classList.add('visible');
  }

  generateBtn.textContent = 'Start Training';
  refreshGenerateBtn();
});

// ── Job panel ─────────────────────────────────────────────────────────────────
function showJobPanel(jobId, maxHours) {
  jobIdDisplay.textContent    = jobId;
  jobStatusBadge.textContent  = 'pending';
  jobStatusBadge.className    = 'job-status-badge status--pending';
  jobResultsCount.textContent = '';
  jobBestLap.textContent      = '—';
  jobDistance.textContent     = '—';
  epsilonDisplay.textContent  = '—';
  jobMessage.textContent      = '';
  jobError.textContent        = '';
  jobResultsList.innerHTML    = '';
  jobMetrics.style.display    = 'none';
  stopBtn.style.display       = 'none';
  jobPanel.classList.add('visible');
}

function updateJobPanel(data) {
  const s = data.status;
  jobStatusBadge.textContent = s;
  jobStatusBadge.className   = `job-status-badge status--${s}`;

  // Live metrics
  if (data.best_race_ms !== null && data.best_race_ms !== undefined) {
    jobMetrics.style.display = 'grid';
    jobBestLap.textContent   = msToLapStr(data.best_race_ms);
  }
  if (data.last_distance !== null && data.last_distance !== undefined) {
    jobDistance.textContent = data.last_distance.toFixed(4);
  }

  if (data.results_count > 0) {
    jobResultsCount.textContent = `${data.results_count} file${data.results_count > 1 ? 's' : ''} saved`;
  }

  if (data.message) jobMessage.textContent = data.message;
  if (data.error)   jobError.textContent   = data.error;

  stopBtn.style.display = (s === 'running') ? 'inline-flex' : 'none';

  const terminal = ['completed', 'stopped', 'timeout', 'failed'];
  if (terminal.includes(s)) {
    stopPolling();
    fetchResultsList(data.job_id || currentJobId);
  }
}

async function fetchResultsList(jobId) {
  try {
    const res  = await fetch(`/api/job/${jobId}/result`);
    const data = await res.json();
    if (!data.files || data.files.length === 0) {
      jobResultsList.innerHTML = '<p class="no-results">No result files found.</p>';
      return;
    }
    jobResultsList.innerHTML = data.files
      .map(f =>
        `<a class="result-file-link"
            href="/api/job/${jobId}/result/${encodeURIComponent(f)}"
            download="${f}">${f}</a>`
      ).join('');
  } catch {
    jobResultsList.innerHTML = '<p class="no-results">Could not load results.</p>';
  }
}

// ── Polling (every 10 minutes) ────────────────────────────────────────────────
function startPolling(jobId) {
  pollOnce(jobId);
  pollTimer = setInterval(() => pollOnce(jobId), POLL_INTERVAL_MS);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollOnce(jobId) {
  try {
    const res  = await fetch(`/api/job/${jobId}/status`);
    if (!res.ok) return;
    const data = await res.json();
    updateJobPanel(data);
  } catch { /* network blip — keep polling */ }
}

// ── Stop button ───────────────────────────────────────────────────────────────
stopBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  stopBtn.disabled    = true;
  stopBtn.textContent = 'Stopping…';
  try {
    const res  = await fetch(`/api/job/${currentJobId}/stop`, { method: 'POST' });
    const data = await res.json();
    updateJobPanel({ ...data, job_id: currentJobId, results_count: 0 });
  } catch (err) {
    jobError.textContent = `Stop failed: ${err.message}`;
  }
  stopBtn.disabled    = false;
  stopBtn.textContent = 'Stop training';
});

// ── Boot ──────────────────────────────────────────────────────────────────────
loadTracks();
