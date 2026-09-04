# ==============================================================
# PharmAgentAI — Pydantic Data Models (Structured Outputs)
# ==============================================================
# WHY: We force the LLM to return strict JSON matching these 
# schemas. This prevents parsing errors and makes the entire 
# pipeline type-safe and reliable.
# ==============================================================

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# --------------------------------------------------------------
# CLINICAL TRIAL DATA MODEL
# Represents structured data fetched from ClinicalTrials.gov
# --------------------------------------------------------------
class ClinicalTrialData(BaseModel):
    """Structured data returned by the Medical Affairs Agent."""

    nct_id: str = Field(description="ClinicalTrials.gov identifier e.g. NCT04512345")
    drug_name: str = Field(description="Name of the drug being studied")
    primary_endpoint: str = Field(description="Primary clinical endpoint measured")
    p_value: float = Field(description="Statistical p-value of primary endpoint")
    hazard_ratio: Optional[float] = Field(
        default=None, description="Hazard ratio (for survival endpoints)"
    )
    confidence_interval: Optional[str] = Field(
        default=None, description="95% CI e.g. '0.61-0.83'"
    )
    patient_population: str = Field(description="Target patient population")
    study_phase: str = Field(description="Clinical trial phase e.g. Phase III")


# --------------------------------------------------------------
# CLAIM DRAFT MODEL
# Represents the output of the Commercial Agent
# --------------------------------------------------------------
class ClaimDraft(BaseModel):
    """Marketing claim draft produced by the Commercial Agent."""

    claim_text: str = Field(description="The full marketing claim text")
    supporting_data: str = Field(description="Key statistics referenced in claim")
    tone: Literal["promotional", "scientific", "balanced"] = Field(
        description="Tone category of the claim"
    )


# --------------------------------------------------------------
# COMPLIANCE RESULT MODEL
# Represents the output of the Regulatory Compliance Agent
# This is the MOST important model — it drives routing decisions
# --------------------------------------------------------------
class ComplianceResult(BaseModel):
    """
    Structured output from the Regulatory Compliance Agent.

    status: PASS or FAIL
    failure_type: 
        - 'factual'   → Routes back to Medical Affairs Agent
        - 'tone'      → Routes back to Commercial Agent
        - 'citation'  → Missing citation, routes to Commercial Agent
        - None        → Status is PASS
    """

    status: Literal["PASS", "FAIL"] = Field(
        description="Whether the claim passed FDA compliance check"
    )
    failure_type: Optional[Literal["factual", "tone", "citation"]] = Field(
        default=None,
        description="Category of failure — drives orchestrator routing"
    )
    violation_details: Optional[str] = Field(
        default=None, description="Specific violation found in the claim"
    )
    fda_rule_referenced: Optional[str] = Field(
        default=None, description="Which FDA guideline rule was violated"
    )
    confidence_score: float = Field(
        description="How confident the agent is in its assessment (0.0 to 1.0)"
    )


# --------------------------------------------------------------
# AGENT STATE MODEL
# The central state object that flows through the LangGraph graph
# Think of this as the "shared memory" of the entire agent run
# --------------------------------------------------------------
class AgentState(BaseModel):
    """
    Central state object passed between all nodes in the LangGraph graph.

    Every agent reads from and writes to this state.
    The Orchestrator reads this state to make routing decisions.
    """

    # --- Input ---
    user_request: str = Field(description="Original user request")
    drug_name: str = Field(description="Drug name extracted from request")

    # --- Medical Affairs Agent Output ---
    trial_data: Optional[ClinicalTrialData] = Field(default=None)

    # --- Commercial Agent Output ---
    claim_draft: Optional[ClaimDraft] = Field(default=None)

    # --- Regulatory Agent Output ---
    compliance_result: Optional[ComplianceResult] = Field(default=None)

    # --- Orchestration Control ---
    loop_count: int = Field(
        default=0,
        description="Number of revision loops completed. Max=3 before human escalation."
    )
    escalated_to_human: bool = Field(
        default=False,
        description="True if loop_count exceeded max and human review was triggered"
    )

    # --- Final Output ---
    final_claim: Optional[str] = Field(
        default=None,
        description="The final approved claim text, ready for output"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
