// ==============================================================
// PharmAgentAI — Frontend App Logic
// ==============================================================

// Dynamic API base: on Railway the frontend and API are on the same origin.
// Locally the server is on port 8080. This handles both cases.
const API = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost'
    ? 'http://127.0.0.1:8080'
    : window.location.origin;   // On Railway: https://pharmagentai.up.railway.app


// ── DOM refs ───────────────────────────────────────────────────
const viewGenerate  = document.getElementById('view-generate');
const viewHistory   = document.getElementById('view-history');
const navGenerate   = document.getElementById('nav-generate');
const navHistory    = document.getElementById('nav-history');
const form          = document.getElementById('generate-form');
const inputDrug     = document.getElementById('drug-name');
const inputEmail    = document.getElementById('notify-email');
const btnGenerate   = document.getElementById('btn-generate');
const statusCard    = document.getElementById('status-card');
const statusBadge   = document.getElementById('status-badge');
const badgeText     = document.getElementById('badge-text');
const resultCard    = document.getElementById('result-card');
const claimText     = document.getElementById('claim-text');
const resultMeta    = document.getElementById('result-meta');
const loopBar       = document.getElementById('loop-bar');
const loopText      = document.getElementById('loop-text');
const historyTbody  = document.getElementById('history-tbody');

// ── Navigation ─────────────────────────────────────────────────
navGenerate.addEventListener('click', e => {
    e.preventDefault();
    showView('generate');
});

navHistory.addEventListener('click', e => {
    e.preventDefault();
    showView('history');
    loadHistory();
});

function showView(name) {
    viewGenerate.classList.toggle('active', name === 'generate');
    viewHistory.classList.toggle('active', name === 'history');
    viewGenerate.classList.toggle('hidden', name !== 'generate');
    viewHistory.classList.toggle('hidden', name === 'generate');
    navGenerate.classList.toggle('active', name === 'generate');
    navHistory.classList.toggle('active', name === 'history');
}

// ── Generate Form Submit ────────────────────────────────────────
form.addEventListener('submit', async e => {
    e.preventDefault();

    const drug  = inputDrug.value.trim();
    const email = inputEmail.value.trim();
    if (!drug) return;

    // Reset UI
    btnGenerate.disabled = true;
    btnGenerate.querySelector('.btn-text').textContent = 'Starting...';
    statusCard.classList.remove('hidden');
    resultCard.classList.add('hidden');
    resetSteps();

    // Set badge to processing
    statusBadge.className = 'status-badge processing';
    badgeText.textContent = 'Processing';

    try {
        const res = await fetch(`${API}/runs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ drug_name: drug, tone: 'scientific', notify_email: email })
        });

        if (!res.ok) throw new Error('API trigger failed');

        const data = await res.json();
        btnGenerate.querySelector('.btn-text').textContent = 'Running...';

        // Begin polling
        pollRun(data.run_id, drug);

    } catch (err) {
        showToast('❌ Could not reach the API server. Is it running?');
        resetBtn();
    }
});

// ── Step Simulation ─────────────────────────────────────────────
// We animate the steps optimistically while waiting for the pipeline.
// The real data comes from the DB when the pipeline completes.
let stepTimer = null;
let currentStep = 1;

function resetSteps() {
    currentStep = 1;
    if (stepTimer) clearInterval(stepTimer);

    [1, 2, 3].forEach(i => {
        const el = document.getElementById(`step-${i}`);
        el.className = 'step' + (i === 1 ? ' active' : ' step-waiting');
    });
    document.querySelectorAll('.step-connector').forEach(c => c.classList.remove('done'));
    document.getElementById('step-1-status').innerHTML = '<div class="spinner"></div>';
    document.getElementById('step-2-status').innerHTML = '<div class="waiting-dot">···</div>';
    document.getElementById('step-3-status').innerHTML = '<div class="waiting-dot">···</div>';
    loopBar.classList.remove('hidden');
    loopText.textContent = 'Loop 1 of 3 max';
}

function advanceStep() {
    if (currentStep < 3) {
        // Mark current step done
        document.getElementById(`step-${currentStep}`).className = 'step done';
        document.getElementById(`step-${currentStep}-status`).innerHTML = '<div class="check">✓</div>';

        // Mark connector done
        document.querySelectorAll('.step-connector')[currentStep - 1]?.classList.add('done');

        currentStep++;

        // Activate next step
        document.getElementById(`step-${currentStep}`).className = 'step active';
        document.getElementById(`step-${currentStep}-status`).innerHTML = '<div class="spinner"></div>';
    }
}

// ── Polling Logic ───────────────────────────────────────────────
// Polls GET /runs/{run_id} every 3 seconds.
// 404 = still running, 200 = done.

function pollRun(runId, drug) {
    let attempts = 0;
    const MAX = 40; // 40 × 3s = 2 minutes

    // Advance steps every ~10 seconds to simulate progress
    let stepInterval = setInterval(() => advanceStep(), 10000);

    const interval = setInterval(async () => {
        attempts++;

        if (attempts > MAX) {
            clearInterval(interval);
            clearInterval(stepInterval);
            showToast('⏱ Pipeline is taking longer than 2 minutes. Please try again.');
            resetBtn();
            return;
        }

        try {
            const res = await fetch(`${API}/runs/${runId}`);

            if (res.status === 404) {
                // Still running — update loop count if changed
                return;
            }

            if (res.ok) {
                clearInterval(interval);
                clearInterval(stepInterval);
                const run = await res.json();
                handleComplete(run, drug);
            }

        } catch (err) {
            // Network error — keep retrying silently
        }
    }, 3000);
}

function handleComplete(run, drug) {
    // Mark all steps done
    [1, 2, 3].forEach(i => {
        document.getElementById(`step-${i}`).className = 'step done';
        document.getElementById(`step-${i}-status`).innerHTML = '<div class="check">✓</div>';
    });
    document.querySelectorAll('.step-connector').forEach(c => c.classList.add('done'));

    // Update badge
    statusBadge.className = 'status-badge completed';
    badgeText.textContent = 'Completed';
    loopBar.classList.add('hidden');

    // Show result card
    claimText.textContent = run.final_claim || 'No claim generated.';
    resultMeta.textContent = `${drug} · ${run.loop_count} review loop${run.loop_count !== 1 ? 's' : ''}`;
    resultCard.classList.remove('hidden');

    // Scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    resetBtn();
    showToast('✓ Claim approved and ready!');
}

// ── History View ────────────────────────────────────────────────
async function loadHistory() {
    historyTbody.innerHTML = '<tr><td colspan="4" class="loading-row">Loading...</td></tr>';

    try {
        const res = await fetch(`${API}/runs`);
        const runs = await res.json();

        if (!runs.length) {
            historyTbody.innerHTML = '<tr><td colspan="4" class="loading-row">No runs yet.</td></tr>';
            return;
        }

        historyTbody.innerHTML = '';
        runs.forEach(r => {
            const snippet = (r.final_claim || '—').substring(0, 90) + '...';
            const tag = r.status === 'completed'
                ? `<span class="tag-success">Approved</span>`
                : `<span class="tag-escalated">${r.status}</span>`;

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${r.drug_name}</strong></td>
                <td>${tag}</td>
                <td>${r.loop_count}</td>
                <td style="max-width:360px; white-space:normal; font-size:0.82rem">${snippet}</td>
            `;
            historyTbody.appendChild(tr);
        });
    } catch (err) {
        historyTbody.innerHTML = '<tr><td colspan="4" class="loading-row" style="color:#f87171">Failed to load history. Is the server running?</td></tr>';
    }
}

// ── Utilities ───────────────────────────────────────────────────
function resetBtn() {
    btnGenerate.disabled = false;
    btnGenerate.querySelector('.btn-text').textContent = 'Generate Claim';
}

function resetForm() {
    inputDrug.value = '';
    inputEmail.value = '';
    statusCard.classList.add('hidden');
    resultCard.classList.add('hidden');
    resetBtn();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function copyClaimText() {
    navigator.clipboard.writeText(claimText.textContent).then(() => {
        showToast('📋 Claim copied to clipboard!');
    });
}

let toastTimer;
function showToast(msg) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add('hidden'), 3500);
}
