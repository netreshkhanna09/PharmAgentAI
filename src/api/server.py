# ==============================================================
# FastAPI Service Layer
# ==============================================================
# PURPOSE:
#   Wraps the PharmAgentAI LangGraph pipeline in a REST API.
#   Allows external systems (frontends, schedulers, microservices)
#   to trigger runs and fetch results via HTTP.
#
# KEY CONCEPTS:
#   - Pydantic models for request/response validation
#   - BackgroundTasks for non-blocking execution (LLM calls are slow)
#   - Async endpoints for high concurrency
#   - RESTful resource design (/runs, /drugs)
# ==============================================================

import uuid
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.orchestrator.graph import pharma_graph
from src.models.schemas import AgentState
from src.memory.postgres_memory import _engine
from src.agents.action_agent import send_claim_notification
from sqlalchemy import text


# ── Pydantic API Models ─────────────────────────────────────────

class RunRequest(BaseModel):
    """Payload for POST /runs"""
    drug_name: str
    tone: str = "scientific"
    notify_email: Optional[str] = None   # ← email for Action Agent notification

class RunResponse(BaseModel):
    """Response for POST /runs"""
    run_id: str
    status: str
    message: str

class RunDetails(BaseModel):
    """Response for GET /runs/{run_id}"""
    run_id: str
    drug_name: str
    status: str
    final_claim: Optional[str] = None
    loop_count: int


# ── FastAPI App Setup ───────────────────────────────────────────

app = FastAPI(
    title="PharmAgentAI API",
    description="Automated FDA Compliance & Claim Generation System",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add CORS Middleware so frontend can make API requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root redirect so both / and /ui/ work
from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/ui/")


# ── Background Task Runner ──────────────────────────────────────

def _execute_pipeline(drug_name: str, tone: str, run_id: str, notify_email: str = None):
    """
    Runs the LangGraph pipeline in a background thread.
    CRITICAL: run_id is passed via LangGraph 'config' so the DB
    saves the record with THIS id — the same one the client polls.
    After completion, triggers the Action Agent to send email notification.
    """
    print(f"\n[API Background] Starting pipeline for {drug_name} (run_id={run_id[:8]}...)")
    initial_state = AgentState(
        user_request=f"Generate a {tone} FDA-compliant marketing claim for {drug_name}",
        drug_name=drug_name,
        claim_draft=None,
        trial_data=None,
        compliance_result=None,
        loop_count=0
    )
    try:
        pharma_graph.invoke(
            initial_state,
            config={
                "recursion_limit": 10,
                "configurable": {
                    "thread_id": run_id,
                    "run_id": run_id,
                },
            }
        )
        print(f"[API Background] Pipeline completed for {drug_name}")

        # ── Action Agent: Email Notification ──────────────────
        # Fetch the completed run from DB to get the final claim text
        if notify_email:
            try:
                with _engine.connect() as conn:
                    row = conn.execute(
                        text("SELECT final_claim, loop_count FROM agent_runs WHERE id = :id"),
                        {"id": run_id}
                    ).fetchone()
                if row and row.final_claim:
                    send_claim_notification(
                        recipient_email=notify_email,
                        drug_name=drug_name,
                        final_claim=row.final_claim,
                        loop_count=row.loop_count,
                        run_id=run_id,
                    )
            except Exception as email_err:
                print(f"[Action Agent] Email step failed: {email_err}")

    except Exception as e:
        print(f"[API Background] Pipeline failed for {drug_name}: {e}")


# ── REST API Endpoints ──────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe for load balancers (e.g. AWS ALB, Kubernetes)."""
    return {"status": "ok", "system": "PharmAgentAI"}


@app.post("/runs", response_model=RunResponse, status_code=202, tags=["Pipeline"])
async def trigger_run(request: RunRequest, bg_tasks: BackgroundTasks):
    """
    Triggers a new pipeline run.
    Returns 202 Accepted immediately; pipeline runs in background.

    THE FLOW:
    1. Generate UUID here (run_id)
    2. Return run_id to client immediately
    3. Background task starts → passes run_id into graph config
    4. graph.py saves DB record with THAT run_id
    5. Client polls GET /runs/{run_id} until it appears in DB
    """
    run_id = str(uuid.uuid4())
    bg_tasks.add_task(
        _execute_pipeline,
        request.drug_name,
        request.tone,
        run_id,
        request.notify_email,   # ← passed to Action Agent for email notification
    )
    return RunResponse(
        run_id=run_id,
        status="pending",
        message=f"Pipeline triggered for {request.drug_name}. Poll GET /runs/{run_id} for status."
    )



@app.get("/runs", response_model=List[RunDetails], tags=["Database"])
async def list_runs():
    """
    Lists all past pipeline runs from the episodic memory database.
    """
    try:
        with _engine.connect() as conn:
            result = conn.execute(
                text("SELECT id AS run_id, drug_name, status, final_claim, loop_count FROM agent_runs ORDER BY created_at DESC LIMIT 50")
            )
            rows = result.fetchall()
            
        return [
            RunDetails(
                run_id=row.run_id,
                drug_name=row.drug_name,
                status=row.status,
                final_claim=row.final_claim,
                loop_count=row.loop_count
            )
            for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}", response_model=RunDetails, tags=["Database"])
async def get_run(run_id: str):
    """
    Gets the status of a specific pipeline run by its UUID.
    
    THE POLLING CONTRACT:
    - Returns 404 while the pipeline is still running (not in DB yet)
    - Returns 200 with full details once the pipeline completes
    - Frontend polls this endpoint every 2-3 seconds after POST /runs
    """
    try:
        with _engine.connect() as conn:
            result = conn.execute(
                text("SELECT id AS run_id, drug_name, status, final_claim, loop_count FROM agent_runs WHERE id = :run_id"),
                {"run_id": run_id}
            )
            row = result.fetchone()

        if row is None:
            # Not in DB yet = still running in background
            raise HTTPException(status_code=404, detail="Run not yet complete (still processing)")

        return RunDetails(
            run_id=row.run_id,
            drug_name=row.drug_name,
            status=row.status,
            final_claim=row.final_claim,
            loop_count=row.loop_count
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/drugs/{drug_name}/failures", tags=["Database"])
async def get_drug_failures(drug_name: str):
    """
    Retrieves the compliance failure history for a specific drug.
    This exposes what the Episodic Memory system has learned.
    """
    try:
        with _engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT failure_type, fda_rule, violation_details, created_at AS timestamp
                    FROM compliance_failures 
                    WHERE LOWER(drug_name) = :drug
                    ORDER BY created_at DESC
                """),
                {"drug": drug_name.lower()}
            )
            rows = result.fetchall()
            
        return [
            {
                "type": row.failure_type,
                "rule": row.fda_rule,
                "details": row.violation_details,
                "timestamp": row.timestamp
            }
            for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Static Frontend (MUST be last so API routes take priority) ──
import os as _os
_frontend_path = _os.path.join(_os.path.dirname(__file__), "..", "frontend")
app.mount("/ui", StaticFiles(directory=_frontend_path, html=True), name="frontend")
