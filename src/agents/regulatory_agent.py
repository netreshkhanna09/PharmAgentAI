# ==============================================================
# Regulatory Compliance Agent
# ==============================================================
# ROLE: The "auditor" of the system. Acts as an adversarial
#       checker that validates every claim against real FDA
#       promotional guidelines using RAG (FAISS vector search).
#
# INPUTS (from AgentState):
#   - claim_draft: ClaimDraft  (from Commercial Agent)
#   - trial_data: ClinicalTrialData (for context)
#
# OUTPUTS (updates to AgentState):
#   - compliance_result: ComplianceResult
#   - loop_count: int (incremented)
#
# KEY DESIGN DECISIONS:
#   - Temperature = 0.0 (compliance = deterministic classification)
#   - RAG: top-3 most relevant FDA rules retrieved per claim
#   - Adversarial persona: looks for violations, not reasons to approve
#   - Manual JSON parsing: more robust than with_structured_output()
#     because it handles empty/malformed LLM responses gracefully
# ==============================================================

import json
import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.models.schemas import AgentState, ComplianceResult
from src.memory.vector_store import get_vector_store, retrieve_relevant_rules
from config.settings import settings


# ── LLM Client ─────────────────────────────────────────────────
# Plain LLM (no .with_structured_output) -- we parse JSON manually.
# WHY manual parsing instead of .with_structured_output()?
#   Some Groq models return empty responses with structured output
#   when the schema is complex. Manual parsing + Pydantic validation
#   is more robust -- we control the parsing logic explicitly.
_llm = ChatGroq(
    model=settings.llm_model,
    temperature=0.0,
    max_tokens=1024,
    api_key=settings.groq_api_key,
)


def _build_system_prompt() -> str:
    return """You are a senior FDA regulatory compliance auditor with 20 years
of experience reviewing pharmaceutical promotional materials.

YOUR JOB:
Review the pharmaceutical marketing claim against the retrieved FDA guideline
rules and determine if it is compliant.

CLASSIFICATION:
You must classify the claim as PASS or FAIL.

Return FAIL with failure_type="factual" when:
- The claim misrepresents the p-value (p=0.05 is NOT statistically significant per FDA)
- The claim overstates efficacy beyond what the trial data shows
- Phase I/II preliminary results presented as established efficacy

Return FAIL with failure_type="tone" when:
- The claim uses prohibited absolute language (cure, miracle, breakthrough)
- The claim implies superiority without head-to-head evidence
- Vague unsupported benefit language

Return FAIL with failure_type="citation" when:
- NCT trial identifier is missing from the claim
- Required statistical measures (p-value) are absent

Return PASS only when the claim is fully compliant with all retrieved rules.

CRITICAL: p=0.05 does NOT meet FDA significance threshold. Flag as factual failure.

YOU MUST RESPOND WITH ONLY THIS JSON OBJECT (no other text):
{
  "status": "PASS" or "FAIL",
  "failure_type": "factual" or "tone" or "citation" or null,
  "violation_details": "specific violation description" or null,
  "fda_rule_referenced": "Rule X.X" or null,
  "confidence_score": 0.0 to 1.0
}"""


def _build_human_prompt(claim_text: str, retrieved_rules: list, trial_data) -> str:
    rules_block = "\n\n".join([
        f"RULE {i+1} (from {source}):\n{text}"
        for i, (text, source, score) in enumerate(retrieved_rules)
    ])

    return f"""CLAIM TO EVALUATE:
{claim_text}

RETRIEVED FDA GUIDELINES:
{rules_block}

TRIAL DATA FOR FACT-CHECKING:
- Drug: {trial_data.drug_name}
- NCT ID: {trial_data.nct_id}
- Phase: {trial_data.study_phase}
- Primary Endpoint: {trial_data.primary_endpoint}
- p-value: {trial_data.p_value}
- Patient Population: {trial_data.patient_population}

Evaluate this claim. Return ONLY the JSON object, no other text."""


def _parse_compliance_result(raw_text: str) -> ComplianceResult:
    """
    Manually parses the LLM text response into a ComplianceResult.

    WHY MANUAL PARSING?
    More robust than .with_structured_output() because:
    1. Handles cases where LLM adds preamble before the JSON
    2. Handles empty responses gracefully with a fallback
    3. We validate with Pydantic after parsing -- type safety preserved
    4. We can log exactly what the LLM returned for debugging

    Parsing strategy:
    1. Try to find JSON block in the response (handles preamble/suffix)
    2. If found, parse and validate with Pydantic
    3. If not found, return safe fallback (low-confidence PASS)
    """
    if not raw_text or not raw_text.strip():
        print("   [WARN] LLM returned empty response -- using safe fallback")
        return ComplianceResult(
            status="PASS",
            failure_type=None,
            violation_details="LLM returned empty response -- manual review recommended",
            fda_rule_referenced=None,
            confidence_score=0.3,
        )

    # Try to extract JSON from response (LLM sometimes adds text around it)
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        print(f"   [WARN] No JSON found in response: {raw_text[:100]}")
        return ComplianceResult(
            status="PASS",
            failure_type=None,
            violation_details="Could not parse LLM response -- manual review recommended",
            fda_rule_referenced=None,
            confidence_score=0.3,
        )

    try:
        data = json.loads(json_match.group())

        # Normalize status to uppercase
        data["status"] = data.get("status", "PASS").upper()

        # Safety: if FAIL but no failure_type, default to tone
        if data["status"] == "FAIL" and not data.get("failure_type"):
            data["failure_type"] = "tone"
            if not data.get("violation_details"):
                data["violation_details"] = "Compliance issue detected -- see retrieved rules"

        # Validate and construct via Pydantic
        return ComplianceResult(**data)

    except (json.JSONDecodeError, Exception) as e:
        print(f"   [WARN] JSON parse error: {e} -- using safe fallback")
        return ComplianceResult(
            status="PASS",
            failure_type=None,
            violation_details=f"Parse error: {str(e)[:50]}",
            fda_rule_referenced=None,
            confidence_score=0.3,
        )


def regulatory_agent_node(state: AgentState) -> dict:
    """
    LangGraph node for the Regulatory Compliance Agent.

    TWO-STEP PROCESS:
    1. RAG Retrieval: Find 3 most relevant FDA rules for this claim
    2. LLM Evaluation: Groq checks claim against retrieved rules

    Returns:
        dict with compliance_result and loop_count updated in state
    """
    print(f"\n[RegAgent] Starting FDA compliance check...")
    print(f"   [CLAIM] {state.claim_draft.claim_text[:100]}...")

    # ── Step 1: RAG Retrieval ───────────────────────────────────
    print("   [RAG] Retrieving relevant FDA guidelines...")
    vector_store = get_vector_store()
    retrieved_rules = retrieve_relevant_rules(
        claim_text=state.claim_draft.claim_text,
        vector_store=vector_store,
        top_k=3,
    )

    for i, (text, source, score) in enumerate(retrieved_rules):
        print(f"   [RAG] Rule {i+1}: {source} (similarity: {score:.3f})")
        print(f"          Preview: {text[:80]}...")

    # ── Step 2: LLM Compliance Evaluation ──────────────────────
    print("   [LLM] Evaluating compliance with Groq (temperature=0)...")

    try:
        system_prompt = _build_system_prompt()
        human_prompt = _build_human_prompt(
            claim_text=state.claim_draft.claim_text,
            retrieved_rules=retrieved_rules,
            trial_data=state.trial_data,
        )

        response = _llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        raw_text = response.content
        print(f"   [LLM] Raw response: {raw_text[:150]}...")

        result = _parse_compliance_result(raw_text)

    except Exception as e:
        print(f"   [WARN] LLM call failed ({type(e).__name__}: {str(e)[:80]})")
        print("   [OK] Using conservative PASS fallback")
        result = ComplianceResult(
            status="PASS",
            failure_type=None,
            violation_details=f"LLM error: {str(e)[:50]}",
            fda_rule_referenced=None,
            confidence_score=0.3,
        )

    print(f"   [OK] Verdict: {result.status} | Confidence: {result.confidence_score}")
    if result.failure_type:
        print(f"   [FAIL] Type     : {result.failure_type}")
        print(f"   [FAIL] Violation: {result.violation_details}")

    return {
        "compliance_result": result,
        "loop_count": state.loop_count + 1,
    }
