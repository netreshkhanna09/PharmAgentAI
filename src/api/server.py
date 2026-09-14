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
from sqlalchemy import text


# ── Pydantic API Models ─────────────────────────────────────────

class RunRequest(BaseModel):
    """Payload for POST /runs"""
    drug_name: str
    tone: str = "scientific"

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


# ── Background Task Runner ──────────────────────────────────────

def _execute_pipeline(drug_name: str, tone: str):
    """
    Synchronous wrapper to run the LangGraph pipeline.
    This runs in a background thread so it doesn't block the API.
    """
    print(f"\n[API Background] Starting pipeline for {drug_name}...")
    initial_state = AgentState(
        drug_name=drug_name,
        claim_draft=None,
        trial_data=None,
        compliance_result=None,
        loop_count=0
    )
    
    # Run the graph
    try:
        # Note: In production, we'd pass the run_id to the graph to save it
        # properly in the DB. For simplicity, the graph generates its own run_id.
        pharma_graph.invoke(initial_state, config={"recursion_limit": 10})
        print(f"[API Background] Pipeline completed for {drug_name}")
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
    Returns 202 Accepted immediately and runs the LLMs in the background.
    """
    run_id = str(uuid.uuid4())
    
    # Add to background tasks (executed AFTER response is sent)
    bg_tasks.add_task(_execute_pipeline, request.drug_name, request.tone)
    
    return RunResponse(
        run_id=run_id,
        status="accepted",
        message=f"Pipeline triggered for {request.drug_name}. Check logs for output."
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
