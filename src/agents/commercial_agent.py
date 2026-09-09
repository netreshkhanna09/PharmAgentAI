# ==============================================================
# Commercial Agent
# ==============================================================
# ROLE: The "copywriter" of the system. Takes verified clinical
#       trial data and writes a professional, compelling, and
#       scientifically-accurate pharmaceutical marketing claim.
#
# INPUTS (from AgentState):
#   - trial_data: ClinicalTrialData  (from Medical Agent)
#   - compliance_result: Optional    (on retry -- knows WHY it failed)
#
# OUTPUTS (updates to AgentState):
#   - claim_draft: ClaimDraft
#
# KEY DESIGN DECISIONS:
#   - Temperature = 0.3 (slight creativity for compelling copy,
#     but grounded -- not fully deterministic like data extraction)
#   - Structured Output forces ClaimDraft schema every time
#   - On retry: receives rejection reason to fix specific issue
#   - NEVER touches raw API data -- only uses typed ClinicalTrialData
#     (prevents hallucination of statistics)
# ==============================================================

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.models.schemas import AgentState, ClaimDraft, ClinicalTrialData
from config.settings import settings


# ── LLM Client ─────────────────────────────────────────────────
# Temperature = 0.3 for the Commercial Agent.
#
# WHY 0.3 and not 0.0?
# We want deterministic DATA agents (Medical, Regulatory) at 0.0.
# We want slightly varied, human-sounding COPY from the Commercial
# Agent -- 0.3 gives creativity while staying grounded.
#
# WHY not 0.7 or higher?
# Higher temperature risks the LLM diverging from the trial data
# facts. Since we're working with regulated pharmaceutical claims,
# even slight creative liberties with statistics are dangerous.
_llm = ChatGroq(
    model=settings.llm_model,
    temperature=0.3,
    max_tokens=1024,
    api_key=settings.groq_api_key,
).with_structured_output(ClaimDraft)


def _build_system_prompt() -> str:
    """
    Builds the 5-layer system prompt for the Commercial Agent.

    Layer 1: PERSONA DEFINITION
    Layer 2: TASK DEFINITION
    Layer 3: CONSTRAINTS (hallucination prevention)
    Layer 4: OUTPUT FORMAT
    Layer 5: STYLE GUIDE (few-shot anchor)
    """
    return """You are a senior pharmaceutical marketing copywriter with 15 years
of experience writing FDA-compliant promotional materials for oncology,
cardiology, and rare disease drugs.

TASK:
Write a single, professional pharmaceutical marketing claim based ONLY on
the clinical trial data provided. The claim must be suitable for use in
promotional materials submitted to the FDA.

STRICT RULES (violations result in regulatory rejection):
1. Use ONLY statistics explicitly provided in the trial data. NEVER invent numbers.
2. NEVER use the words: "cure", "miracle", "breakthrough", "revolutionary".
3. NEVER claim superiority over a competitor drug unless comparison data is given.
4. ALWAYS include the specific p-value or statistical measure when citing significance.
5. Claims must be qualified with the trial population context (e.g., "in patients with...").
6. Use "statistically significant" only when p < 0.05 is confirmed in the data.
7. Use precise scientific language -- avoid vague terms like "greatly improved".

TONE GUIDANCE:
- For tone="scientific": Use clinical/academic language. Cite exact figures.
- For tone="promotional": Accessible language for HCP audience. Still cite key stat.
- For tone="balanced": Mix of both. Lead with patient benefit, follow with data.

OUTPUT:
Return ONLY a ClaimDraft JSON object. No preamble. No explanation. Just the object.

STYLE EXAMPLE (for reference):
Input:  drug=Pembrolizumab, endpoint=Overall Survival, p=0.02, HR=0.71, phase=Phase III
Output claim: "Pembrolizumab demonstrated a statistically significant improvement in
Overall Survival versus placebo (HR 0.71; p=0.02) in a Phase III trial of adult
patients with advanced non-small cell lung cancer."
"""


def _build_human_prompt(
    trial_data: ClinicalTrialData,
    rejection_reason: str | None = None
) -> str:
    """
    Builds the human-turn prompt with trial data and optional retry context.

    On retry runs, we inject the SPECIFIC rejection reason.
    This is how the agent learns from its previous failure.
    """
    # Format the trial data cleanly for the LLM
    data_block = f"""VERIFIED CLINICAL TRIAL DATA (use ONLY these facts):
- Drug Name       : {trial_data.drug_name}
- NCT ID          : {trial_data.nct_id}
- Study Phase     : {trial_data.study_phase}
- Primary Endpoint: {trial_data.primary_endpoint}
- p-value         : {trial_data.p_value}
- Hazard Ratio    : {trial_data.hazard_ratio if trial_data.hazard_ratio else 'Not reported'}
- 95% CI          : {trial_data.confidence_interval if trial_data.confidence_interval else 'Not reported'}
- Patient Pop.    : {trial_data.patient_population}"""

    # On a retry, add the specific rejection reason
    # This is the "Reflexion" pattern -- agent learns from its failure
    if rejection_reason:
        retry_block = f"""

IMPORTANT -- PREVIOUS CLAIM WAS REJECTED:
Rejection reason: {rejection_reason}

You MUST fix this specific issue in your new draft.
Do NOT repeat the same mistake. Adjust the tone/wording accordingly."""
    else:
        retry_block = ""

    return f"""{data_block}{retry_block}

Write a compelling, FDA-compliant pharmaceutical marketing claim for {trial_data.drug_name}.
Choose tone="scientific" for this draft."""


def commercial_agent_node(state: AgentState) -> dict:
    """
    LangGraph node for the Commercial Agent.

    Signature contract:
    - Input:  full AgentState
    - Output: dict with only 'claim_draft' key updated

    On retry loops, extracts the rejection reason from compliance_result
    and passes it to the LLM so it fixes the SPECIFIC issue.
    """
    print(f"\n[CommAgent] Drafting claim for: {state.drug_name}")

    # ── Extract rejection context on retry runs ─────────────────
    # This implements the "Reflexion" pattern:
    # Agent receives its past failure and reasons about how to fix it
    rejection_reason = None
    if state.loop_count > 0 and state.compliance_result:
        rejection_reason = state.compliance_result.violation_details
        print(f"   [RETRY] Loop #{state.loop_count} -- fixing: {rejection_reason}")

    # ── Build prompts ───────────────────────────────────────────
    system_prompt = _build_system_prompt()
    human_prompt = _build_human_prompt(state.trial_data, rejection_reason)

    # ── Call LLM with structured output ────────────────────────
    # The LLM MUST return a ClaimDraft object.
    # If it tries to return anything else, Pydantic raises ValidationError.
    print("   [LLM] Calling Groq for claim generation...")

    try:
        claim_draft = _llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        print("   [OK] Claim generated successfully")
        print(f"   [DRAFT] Tone: {claim_draft.tone}")
        print(f"   [DRAFT] {claim_draft.claim_text[:120]}...")

    except Exception as e:
        # Fallback: build a basic claim from structured data
        # No hallucination possible -- we assemble from typed fields only
        print(f"   [WARN] LLM call failed ({type(e).__name__}). Using structured fallback.")

        claim_text = (
            f"{state.trial_data.drug_name} demonstrated statistically significant "
            f"improvement in {state.trial_data.primary_endpoint} "
            f"(p={state.trial_data.p_value}) in a "
            f"{state.trial_data.study_phase} trial in "
            f"{state.trial_data.patient_population}."
        )

        claim_draft = ClaimDraft(
            claim_text=claim_text,
            supporting_data=f"p={state.trial_data.p_value}, NCT: {state.trial_data.nct_id}",
            tone="scientific",
        )

        print(f"   [OK] Fallback claim built from structured data")

    return {"claim_draft": claim_draft}
