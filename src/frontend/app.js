// ==============================================================
// PharmAgentAI - Frontend Logic (Vanilla JS)
// ==============================================================
// PURPOSE: Handles DOM manipulation, routing (views), and API
//          communication (fetch, polling) without React/Vue.
// ==============================================================

const API_BASE = 'http://127.0.0.1:8080';

// ── DOM Elements ───────────────────────────────────────────────
const viewGenerate = document.getElementById('view-generate');
const viewHistory = document.getElementById('view-history');
const navNewRun = document.getElementById('nav-new-run');
const navHistory = document.getElementById('nav-history');

const formGenerate = document.getElementById('generate-form');
const inputDrugName = document.getElementById('drug-name');
const btnGenerate = document.getElementById('btn-generate');
const statusContainer = document.getElementById('status-container');
const runStatusBadge = document.getElementById('run-status-badge');
const resultBox = document.getElementById('result-box');
const finalClaimText = document.getElementById('final-claim-text');

const historyTbody = document.getElementById('history-tbody');

// ── Navigation (Simple SPA Router) ──────────────────────────────
navNewRun.addEventListener('click', (e) => {
    e.preventDefault();
    viewGenerate.classList.remove('hidden');
    viewHistory.classList.add('hidden');
    navNewRun.classList.add('active');
    navHistory.classList.remove('active');
});

navHistory.addEventListener('click', (e) => {
    e.preventDefault();
    viewGenerate.classList.add('hidden');
    viewHistory.classList.remove('hidden');
    navNewRun.classList.remove('active');
    navHistory.classList.add('active');
    loadHistory(); // Fetch data when view is opened
});


// ── Generate Claim Flow ─────────────────────────────────────────

formGenerate.addEventListener('submit', async (e) => {
    e.preventDefault();
    const drugName = inputDrugName.value.trim();
    if (!drugName) return;

    // UI Reset
    btnGenerate.disabled = true;
    btnGenerate.textContent = 'Starting Pipeline...';
    statusContainer.classList.remove('hidden');
    resultBox.classList.add('hidden');
    runStatusBadge.className = 'badge badge-pending';
    runStatusBadge.textContent = 'Running';
    
    // Reset steps
    document.getElementById('step-med').className = 'step active';
    document.getElementById('step-comm').className = 'step';
    document.getElementById('step-reg').className = 'step';

    try {
        // Trigger the backend API
        const response = await fetch(`${API_BASE}/runs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ drug_name: drugName, tone: 'scientific' })
        });
        
        if (!response.ok) throw new Error('API Error');
        
        const data = await response.json();
        
        // Start polling for completion
        pollRunStatus(data.run_id);

    } catch (error) {
        alert("Failed to start run. Is the backend running?");
        btnGenerate.disabled = false;
        btnGenerate.textContent = 'Generate & Verify';
    }
});


// ── HTTP Polling for Background Tasks ──────────────────────────
// HOW THIS WORKS:
// 1. POST /runs returns a run_id UUID immediately (202 Accepted)
// 2. We poll GET /runs/{run_id} every 2.5 seconds
// 3. While pipeline is running: server returns 404 (not in DB yet)
// 4. When pipeline finishes: server returns 200 with the full run details
// 5. We clear the interval and display the result
// 
// WHY 404 = "still running"?
// The DB record is only created AFTER the pipeline completes.
// A missing record = pipeline still in progress. This is the
// "poll until it exists" pattern — simple and reliable.

async function pollRunStatus(runId) {
    let attempts = 0;
    const maxAttempts = 60; // 60 * 2.5s = 2.5 minutes max wait
    
    const interval = setInterval(async () => {
        attempts++;
        if (attempts > maxAttempts) {
            clearInterval(interval);
            alert("Pipeline is taking longer than expected. Please check the terminal.");
            btnGenerate.disabled = false;
            btnGenerate.textContent = 'Generate & Verify';
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/runs/${runId}`);
            
            if (response.status === 404) {
                // Still running — update simulated progress
                updateSimulatedProgress();
                return;
            }

            if (response.ok) {
                // Pipeline completed!
                clearInterval(interval);
                const runDetails = await response.json();
                handleRunCompleted(runDetails);
            }
        } catch (error) {
            console.error("Polling error", error);
        }
    }, 2500);
}


let simStep = 1;
function updateSimulatedProgress() {
    // Just a visual flair for the user while polling
    simStep++;
    if (simStep === 2) {
        document.getElementById('step-med').classList.add('completed');
        document.getElementById('step-comm').classList.add('active');
    } else if (simStep >= 3) {
        document.getElementById('step-comm').classList.add('completed');
        document.getElementById('step-reg').classList.add('active');
    }
}


function handleRunCompleted(runDetails) {
    // UI Updates
    document.getElementById('step-reg').classList.add('completed');
    
    runStatusBadge.className = 'badge badge-success';
    runStatusBadge.textContent = 'COMPLETED';
    
    finalClaimText.textContent = runDetails.final_claim || "No claim generated.";
    resultBox.classList.remove('hidden');
    
    btnGenerate.disabled = false;
    btnGenerate.textContent = 'Generate New Claim';
    inputDrugName.value = '';
    simStep = 1;
}


// ── Load History Flow ──────────────────────────────────────────

async function loadHistory() {
    historyTbody.innerHTML = '<tr><td colspan="4">Loading...</td></tr>';
    
    try {
        const response = await fetch(`${API_BASE}/runs`);
        const runs = await response.json();
        
        historyTbody.innerHTML = '';
        
        if (runs.length === 0) {
            historyTbody.innerHTML = '<tr><td colspan="4">No past runs found.</td></tr>';
            return;
        }

        runs.forEach(run => {
            const tr = document.createElement('tr');
            
            // Truncate long claims
            let claimSnippet = run.final_claim || "";
            if (claimSnippet.length > 80) claimSnippet = claimSnippet.substring(0, 80) + "...";

            tr.innerHTML = `
                <td><strong>${run.drug_name}</strong></td>
                <td><span class="badge badge-success">${run.status}</span></td>
                <td>${run.loop_count}</td>
                <td><small>${claimSnippet}</small></td>
            `;
            historyTbody.appendChild(tr);
        });

    } catch (error) {
        historyTbody.innerHTML = '<tr><td colspan="4" style="color:red">Failed to load history.</td></tr>';
    }
}
