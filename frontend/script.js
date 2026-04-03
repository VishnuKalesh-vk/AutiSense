/**
 * script.js — Frontend logic for the Autism Risk Analysis Web App.
 *
 * Flow:
 *   1. User picks an image file → show preview + enable Analyse button.
 *   2. User clicks Analyse → POST multipart/form-data to /predict.
 *   3. API responds → animate score ring, display results.
 *   4. Errors (bad file type, network, server) are shown gracefully.
 *
 * DISCLAIMER: Educational use only. Not a medical diagnosis tool.
 */

'use strict';

// ── Configuration ────────────────────────────────────────────
const API_URL = 'http://127.0.0.1:8000/predict';

const ACCEPTED_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/gif',
  'image/webp',
  'image/bmp',
]);

const CIRCUMFERENCE = 314.16; // 2 * π * 50  (SVG ring radius = 50)

// ── DOM references ───────────────────────────────────────────
const imageInput       = document.getElementById('imageInput');
const imagePreview     = document.getElementById('imagePreview');
const previewWrapper   = document.getElementById('previewWrapper');
const dropZone         = document.getElementById('dropZone');
const dropZoneInner    = document.getElementById('dropZoneInner');
const fileName         = document.getElementById('fileName');
const analyzeBtn       = document.getElementById('analyzeBtn');

const resultIdle       = document.getElementById('resultIdle');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultCard       = document.getElementById('resultCard');

const scoreRingFill    = document.getElementById('scoreRingFill');
const riskBadge        = document.getElementById('riskBadge');
const riskScore        = document.getElementById('riskScore');
const riskLevel        = document.getElementById('riskLevel');
const confidence       = document.getElementById('confidence');
const modelSource      = document.getElementById('modelSource');
const apiDisclaimer    = document.getElementById('apiDisclaimer');
const resetBtn         = document.getElementById('resetBtn');

const errorMessage     = document.getElementById('errorMessage');

// ── Helpers ──────────────────────────────────────────────────
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

function showError(msg) {
  errorMessage.textContent = `⚠ ${msg}`;
  show(errorMessage);
}

function clearError() {
  errorMessage.textContent = '';
  hide(errorMessage);
}

/** Animate the SVG score ring to a given 0–1 value and colour. */
function animateRing(score, level) {
  const offset = CIRCUMFERENCE * (1 - score);
  scoreRingFill.style.strokeDashoffset = offset;
  const colour = level === 'Low' ? '#00e676' : level === 'Medium' ? '#ffca28' : '#ff5252';
  scoreRingFill.style.stroke = colour;
}

/** Reset ring to empty state. */
function resetRing() {
  scoreRingFill.style.transition = 'none';
  scoreRingFill.style.strokeDashoffset = CIRCUMFERENCE;
  scoreRingFill.style.stroke = '#4f8ef7';
  // Re-enable transition after reset tick
  requestAnimationFrame(() => {
    scoreRingFill.style.transition =
      'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.5s ease';
  });
}

/** Switch the result panel to idle state. */
function showIdle() {
  hide(loadingIndicator);
  hide(resultCard);
  show(resultIdle);
}

/** Switch the result panel to loading state. */
function showLoading() {
  hide(resultIdle);
  hide(resultCard);
  show(loadingIndicator);
}

/** Switch the result panel to result state. */
function showResults() {
  hide(loadingIndicator);
  hide(resultIdle);
  show(resultCard);
}

/** Reset the entire UI back to its initial state. */
function resetUI() {
  imageInput.value = '';
  imagePreview.src = '';
  fileName.textContent = '';
  hide(previewWrapper);
  show(dropZoneInner);

  analyzeBtn.disabled = true;
  analyzeBtn.setAttribute('aria-disabled', 'true');

  clearError();
  resetRing();
  showIdle();
}

// ── File handling ─────────────────────────────────────────────
function handleFile(file) {
  clearError();
  showIdle();

  if (!file) {
    hide(previewWrapper);
    show(dropZoneInner);
    analyzeBtn.disabled = true;
    analyzeBtn.setAttribute('aria-disabled', 'true');
    return;
  }

  if (!ACCEPTED_TYPES.has(file.type)) {
    showError(
      `"${file.name}" is not a supported image type. ` +
      'Please upload a JPEG, PNG, GIF, WEBP, or BMP file.'
    );
    hide(previewWrapper);
    show(dropZoneInner);
    analyzeBtn.disabled = true;
    analyzeBtn.setAttribute('aria-disabled', 'true');
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  imagePreview.src = objectUrl;
  imagePreview.onload = () => URL.revokeObjectURL(objectUrl);
  fileName.textContent = file.name;

  hide(dropZoneInner);
  show(previewWrapper);

  analyzeBtn.disabled = false;
  analyzeBtn.setAttribute('aria-disabled', 'false');
}

// ── Event: file input change ─────────────────────────────────
imageInput.addEventListener('change', function () {
  handleFile(this.files[0] || null);
});

// ── Drag-and-drop ─────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0] || null;
  if (file) handleFile(file);
});

// ── Event: analyse button ─────────────────────────────────────
analyzeBtn.addEventListener('click', async function () {
  const file = imageInput.files[0] || null;

  // If file came via drag-and-drop, imageInput.files may be empty.
  // Retrieve the last handled file from the preview src as a fallback isn't
  // possible; instead we ask the user to re-select if input is empty.
  if (!file) {
    showError('Please select an image first.');
    return;
  }

  clearError();
  showLoading();
  analyzeBtn.disabled = true;
  analyzeBtn.setAttribute('aria-disabled', 'true');

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(API_URL, { method: 'POST', body: formData });

    if (!response.ok) {
      let detail = `Server returned ${response.status}`;
      try {
        const errBody = await response.json();
        if (errBody.detail) detail = errBody.detail;
      } catch (_) { /* use default */ }
      throw new Error(detail);
    }

    const data = await response.json();
    const level = data.risk_level;
    const score = data.risk_score;

    // Populate result fields
    riskScore.textContent   = `${(score * 100).toFixed(1)}%`;
    riskBadge.textContent   = level;
    riskBadge.setAttribute('data-level', level);
    riskLevel.textContent   = level;
    confidence.textContent  = data.confidence;
    if (modelSource)    modelSource.textContent    = data.model_source;
    if (apiDisclaimer)  apiDisclaimer.textContent  = data.message;

    showResults();
    // Trigger ring animation on next paint so CSS transition fires
    requestAnimationFrame(() => animateRing(score, level));

  } catch (err) {
    showIdle();
    if (err instanceof TypeError) {
      showError('Could not reach the API. Make sure the backend is running at ' + API_URL);
    } else {
      showError(`Analysis failed: ${err.message}`);
    }
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.setAttribute('aria-disabled', 'false');
  }
});

// ── Event: reset button ───────────────────────────────────────
resetBtn.addEventListener('click', resetUI);
