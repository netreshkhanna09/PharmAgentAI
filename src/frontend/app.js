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

async function pollRunStatus(runId) {
    // We poll the GET /runs endpoint to check if our run is completed
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/runs`);
            const runs = await response.json();
            
            // Find our run in the DB
            const ourRun = runs.find(r => r.run_id === runId);
            
            if (ourRun) {
                // If it exists in the DB, it has finished!
                // (Our API only saves to DB after the LangGraph pipeline completes)
                clearInterval(interval);
                handleRunCompleted(ourRun);
            } else {
                // Not in DB yet, still running.
                // Simulate step progression for UX (since graph is synchronous in backend)
                updateSimulatedProgress();
            }
        } catch (error) {
            console.error("Polling error", error);
        }
    }, 2500); // Poll every 2.5 seconds
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
