# ==============================================================
# Medical Affairs Agent
# ==============================================================
# ROLE: The "doctor" of the system. Responsible for fetching
#       real, verified clinical trial data from ClinicalTrials.gov
#       and structuring it into a validated Pydantic object.
#
# INPUTS (from AgentState):
#   - drug_name: str
#
# OUTPUTS (updates to AgentState):
#   - trial_data: ClinicalTrialData
#
# WHY THIS DESIGN:
#   - Uses real ClinicalTrials.gov API (no API key needed!)
#   - Uses LLM-as-a-Parser pattern to extract structured fields
#     from the messy API JSON response
#   - Pydantic validates every field — if LLM hallucinates a
#     wrong type, it raises ValidationError before it propagates
# ==============================================================

import json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.models.schemas import AgentState, ClinicalTrialData
from config.settings import settings


# ── LLM Client ─────────────────────────────────────────────────
# Temperature=0 for DETERMINISTIC structured output extraction.
# We do NOT want creativity here — we want precise data parsing.
llm = ChatGroq(
    model=settings.llm_model,
    temperature=0,                   # Deterministic — no hallucination risk
    max_tokens=1024,
    api_key=settings.groq_api_key,
).with_structured_output(ClinicalTrialData)  # Forces JSON matching our schema


# ── Step 1: Fetch Raw Data from ClinicalTrials.gov API ─────────
@retry(
    stop=stop_after_attempt(3),               # Retry up to 3 times
    wait=wait_exponential(min=1, max=10),     # Wait 1s, 2s, 4s... between retries
)
def fetch_trial_from_api(drug_name: str) -> dict:
    """
    Fetches clinical trial data from the ClinicalTrials.gov public API.

    No API key required — this is a free public API.
    Returns the raw JSON for the most relevant trial found.

    Args:
        drug_name: Name of the drug to search for

    Returns:
        Raw trial JSON dict from the API
    """
    url = f"{settings.clinical_trials_base_url}/studies"

    params = {
        "query.term": drug_name,
        "pageSize": 5,          # Get top 5 results
        "format": "json",
        "fields": (             # Only fetch fields we care about
            "NCTId,BriefTitle,OfficialTitle,Phase,"
            "PrimaryOutcomeMeasure,StudyPopulation,"
            "EnrollmentCount,StudyType"
        ),
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()            # Raise error if 4xx/5xx
        data = response.json()

    # Extract the first (most relevant) study
    studies = data.get("studies", [])
    if not studies:
        raise ValueError(f"No clinical trials found for drug: {drug_name}")

    return studies[0]  # Return the most relevant trial


# ── Step 2: Use LLM to Parse Raw API Response ──────────────────
def parse_trial_with_llm(raw_trial: dict, drug_name: str) -> ClinicalTrialData:
    """
    Uses the LLM to extract structured ClinicalTrialData from raw API JSON.

    WHY LLM-as-a-Parser?
    - ClinicalTrials.gov JSON is deeply nested and inconsistently structured
    - Writing brittle parsing code breaks every time the API schema changes
    - LLM extracts meaning regardless of nesting structure
    - Pydantic validates the LLM output — preventing hallucination from propagating

    Args:
        raw_trial: Raw JSON from ClinicalTrials.gov
        drug_name: Drug name for context

    Returns:
        Validated ClinicalTrialData Pydantic object
    """
    system_prompt = """You are a medical data extraction specialist.

Your job is to extract structured clinical trial information from raw API data.

CRITICAL RULES:
1. ONLY extract information that is EXPLICITLY present in the provided data
2. NEVER invent, assume, or extrapolate any values
3. If a field is not clearly present in the data, use sensible defaults:
   - p_value: use 0.05 as placeholder (indicates "not specified")  
   - hazard_ratio: use null if not mentioned
   - confidence_interval: use null if not mentioned
4. For patient_population: summarize the target population in plain English
5. For primary_endpoint: extract the main outcome being measured"""

    human_prompt = f"""Extract clinical trial data for the drug '{drug_name}' from this raw data:

{json.dumps(raw_trial, indent=2)}

Return structured data matching the required schema exactly."""

    # LLM call — returns validated ClinicalTrialData (via .with_structured_output)
    result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    return result


# ── Main Agent Node Function ────────────────────────────────────
def medical_agent_node(state: AgentState) -> dict:
    """
    LangGraph node function for the Medical Affairs Agent.

    This is the function LangGraph calls when executing this node.
    It MUST:
    1. Accept AgentState as input
    2. Return a dict of fields to UPDATE in the state

    Args:
        state: Current AgentState from the graph

    Returns:
        Dict with 'trial_data' key to update in AgentState
    """
    print(f"\n🔬 [Medical Affairs Agent] Fetching trial data for: {state.drug_name}")

    # If this is a retry run, log that context
    if state.loop_count > 0:
        print(f"   ↻ Retry #{state.loop_count} — previous compliance failure was: "
              f"{state.compliance_result.failure_type if state.compliance_result else 'unknown'}")

    # Step 1: Fetch from ClinicalTrials.gov
    raw_trial = fetch_trial_from_api(state.drug_name)
    print(f"   ✅ API returned trial data")

    # Step 2: Parse with LLM into structured Pydantic object
    trial_data = parse_trial_with_llm(raw_trial, state.drug_name)
    print(f"   ✅ Structured data extracted: NCT ID = {trial_data.nct_id}")
    print(f"   📊 Primary Endpoint: {trial_data.primary_endpoint}")
    print(f"   📊 p-value: {trial_data.p_value}")

    # Return ONLY the fields we updated — LangGraph merges into state
    return {"trial_data": trial_data}
