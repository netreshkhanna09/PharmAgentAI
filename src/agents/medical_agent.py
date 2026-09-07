# ==============================================================
# Medical Affairs Agent
# ==============================================================
import json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.models.schemas import AgentState, ClinicalTrialData
from config.settings import settings


# LLM Client -- Temperature=0 for deterministic structured extraction
_llm_base = ChatGroq(
    model=settings.llm_model,
    temperature=0,
    max_tokens=1024,
    api_key=settings.groq_api_key,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def fetch_trial_from_api(drug_name: str) -> dict:
    """Fetches clinical trial data from ClinicalTrials.gov (free public API)."""
    url = f"{settings.clinical_trials_base_url}/studies"
    params = {
        "query.term": drug_name,
        "pageSize": 3,
        "format": "json",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    studies = data.get("studies", [])
    if not studies:
        raise ValueError(f"No clinical trials found for: {drug_name}")
    return studies[0]


def parse_trial_direct(raw_trial: dict, drug_name: str) -> dict:
    """
    Directly extracts fields from ClinicalTrials.gov JSON structure.
    This is the reliable base layer -- always works, no LLM needed.
    """
    protocol = raw_trial.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    design_module = protocol.get("designModule", {})
    outcomes_module = protocol.get("outcomesModule", {})
    eligibility_module = protocol.get("eligibilityModule", {})

    nct_id = id_module.get("nctId", "NCT-UNKNOWN")

    phases = design_module.get("phases", [])
    phase = phases[0].replace("_", " ").title() if phases else "Unknown Phase"

    primary_outcomes = outcomes_module.get("primaryOutcomes", [])
    endpoint = (
        primary_outcomes[0].get("measure", "Overall Survival")
        if primary_outcomes else "Overall Survival"
    )

    population = eligibility_module.get("studyPopulation", "")
    if not population:
        criteria = eligibility_module.get("eligibilityCriteria", "Adult patients")
        population = criteria[:100]

    return {
        "nct_id": nct_id,
        "drug_name": drug_name,
        "primary_endpoint": endpoint,
        "p_value": 0.05,
        "hazard_ratio": None,
        "confidence_interval": None,
        "patient_population": population,
        "study_phase": phase,
    }


def enrich_with_llm(parsed_data: dict, raw_trial: dict, drug_name: str) -> ClinicalTrialData:
    """
    Uses LLM to enrich parsed data with missing fields.
    Falls back to direct-parsed data gracefully if LLM fails.
    """
    try:
        structured_llm = _llm_base.with_structured_output(ClinicalTrialData)

        system_prompt = (
            "You are a clinical data extraction specialist. "
            "Extract structured trial data from the provided JSON. "
            "RULES: "
            "1. Use ONLY data explicitly present in the JSON. "
            "2. For p_value: use 0.05 if not clearly stated. "
            "3. For hazard_ratio and confidence_interval: use null if not present. "
            "4. Keep patient_population under 150 characters. "
            f"5. drug_name must be exactly: {drug_name}"
        )

        human_prompt = (
            f"Drug name: {drug_name}\n\n"
            f"Pre-parsed baseline data:\n{json.dumps(parsed_data, indent=2)}\n\n"
            f"Raw API data for enrichment:\n{json.dumps(raw_trial, indent=2)[:3000]}\n\n"
            "Return a complete ClinicalTrialData object."
        )

        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        print("   [LLM] Enrichment: SUCCESS")
        return result

    except Exception as e:
        print(f"   [WARN] LLM enrichment skipped ({type(e).__name__}: {str(e)[:80]})")
        print("   [OK] Using direct-parsed data -- system continues normally")
        return ClinicalTrialData(**parsed_data)


def medical_agent_node(state: AgentState) -> dict:
    """
    LangGraph node for the Medical Affairs Agent.
    Input:  full AgentState
    Output: dict of updated fields (only trial_data)
    """
    print(f"\n[MedAgent] Fetching trial data for: {state.drug_name}")

    if state.loop_count > 0:
        prev_failure = (
            state.compliance_result.failure_type
            if state.compliance_result else "unknown"
        )
        print(f"   [RETRY] Loop #{state.loop_count} -- previous failure: {prev_failure}")

    # Step 1: Fetch from ClinicalTrials.gov
    raw_trial = fetch_trial_from_api(state.drug_name)
    print("   [OK] API returned trial data")

    # Step 2: Direct parse (reliable base)
    parsed_data = parse_trial_direct(raw_trial, state.drug_name)
    print(f"   [OK] Direct parse complete -- NCT ID: {parsed_data['nct_id']}")

    # Step 3: LLM enrichment (with fallback)
    trial_data = enrich_with_llm(parsed_data, raw_trial, state.drug_name)
    print(f"   [DATA] Primary Endpoint : {trial_data.primary_endpoint}")
    print(f"   [DATA] Study Phase      : {trial_data.study_phase}")
    print(f"   [DATA] p-value          : {trial_data.p_value}")

    return {"trial_data": trial_data}
