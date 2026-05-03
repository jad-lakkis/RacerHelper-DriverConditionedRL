'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const trackSelect   = document.getElementById('track-select');
const lapInput      = document.getElementById('lap-input');
const lapError      = document.getElementById('lap-error');
const riskSlider    = document.getElementById('risk-slider');
const riskDisplay   = document.getElementById('risk-display');
const replayFile    = document.getElementById('replay-file');
const uploadText    = document.getElementById('upload-text');
const profilePanel  = document.getElementById('profile-panel');
const profileLoading= document.getElementById('profile-loading');
const profileResults= document.getElementById('profile-results');
const generateBtn   = document.getElementById('generate-btn');
const generateResult= document.getElementById('generate-result');

// ── State ─────────────────────────────────────────────────────────────────────
let profile = null;   // extracted driver profile object

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

// ── Risk slider ───────────────────────────────────────────────────────────────
riskSlider.addEventListener('input', () => {
  riskDisplay.textContent = riskSlider.value + '%';
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
}

// ── Profile bars ──────────────────────────────────────────────────────────────
function setBar(barId, valId, value, min, max) {
  const pct = ((value - min) / (max - min)) * 100;
  document.getElementById(barId).style.width  = Math.max(0, Math.min(100, pct)) + '%';
  document.getElementById(valId).textContent  =
    value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
}

function setOusBar(value) {
  // Centered bar: value in [-5, 5]
  const bar = document.getElementById('bar-ous');
  const val = document.getElementById('val-ous');
  const halfPct = Math.abs(value) / 5 * 50;   // 0–50 %
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
  generateResult.classList.remove('visible');
  refreshGenerateBtn();

  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch('/api/driver-profile', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

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

// ── Generate ──────────────────────────────────────────────────────────────────
function refreshGenerateBtn() {
  const lapOk = parseLapMs(lapInput.value) !== null;
  const hasProfile = profile !== null;
  generateBtn.disabled = !(lapOk && hasProfile && trackSelect.value);
}

trackSelect.addEventListener('change', refreshGenerateBtn);

generateBtn.addEventListener('click', async () => {
  if (!validateLap()) return;

  const payload = {
    track:                       trackSelect.value,
    best_lap_ms:                 parseLapMs(lapInput.value),
    risk_tolerance:              parseInt(riskSlider.value, 10) / 100,
    braking_aggression:          profile.braking_aggression,
    oversteer_understeer_score:  profile.oversteer_understeer_score,
    corner_entry_speed_ratio:    profile.corner_entry_speed_ratio,
  };

  generateBtn.disabled = true;
  generateBtn.textContent = 'Sending…';
  generateResult.classList.remove('visible');

  try {
    const res  = await fetch('/api/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    generateResult.innerHTML = res.ok
      ? `<div class="result-badge result-badge--ok">&#10003; ${data.status}</div>
         <p style="font-size:.875rem;color:var(--muted);margin-bottom:.75rem">${data.message}</p>
         <pre class="result-json">${JSON.stringify(data.hyperparameters, null, 2)}</pre>`
      : `<div class="result-badge result-badge--err">&#9888; Error</div>
         <pre class="result-json">${JSON.stringify(data, null, 2)}</pre>`;
  } catch (err) {
    generateResult.innerHTML =
      `<div class="result-badge result-badge--err">&#9888; Network Error</div>
       <p style="font-size:.875rem;color:#f87171">${err.message}</p>`;
  }

  generateResult.classList.add('visible');
  generateBtn.textContent = 'Generate Replay';
  refreshGenerateBtn();
});

// ── Boot ──────────────────────────────────────────────────────────────────────
loadTracks();
