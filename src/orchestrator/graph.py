# ==============================================================
# LangGraph Orchestrator -- PharmAgentAI State Machine
# ==============================================================

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.schemas import AgentState
from src.agents.medical_agent import medical_agent_node


def commercial_agent_node(state: AgentState) -> dict:
    """Commercial Agent -- drafts marketing claim. PLACEHOLDER for Day 3."""
    print(f"\n[CommAgent] Drafting claim for: {state.drug_name}")

    draft_text = (
        f"{state.drug_name} demonstrated statistically significant improvement "
        f"in {state.trial_data.primary_endpoint} "
        f"(p={state.trial_data.p_value}) in a {state.trial_data.study_phase} trial "
        f"enrolling {state.trial_data.patient_population}."
    )

    print("   [OK] Draft claim created")
    print(f"   [DRAFT] {draft_text[:100]}...")

    from src.models.schemas import ClaimDraft
    return {
        "claim_draft": ClaimDraft(
            claim_text=draft_text,
            supporting_data=f"p={state.trial_data.p_value}",
            tone="scientific",
        )
    }


def regulatory_agent_node(state: AgentState) -> dict:
    """Regulatory Compliance Agent -- validates claim. PLACEHOLDER for Day 4."""
    print(f"\n[RegAgent] Checking FDA compliance...")
    print(f"   [CLAIM] {state.claim_draft.claim_text[:80]}...")

    from src.models.schemas import ComplianceResult
    result = ComplianceResult(
        status="PASS",
        failure_type=None,
        violation_details=None,
        fda_rule_referenced=None,
        confidence_score=0.92,
    )

    print(f"   [OK] Compliance Status: {result.status} (confidence: {result.confidence_score})")

    return {
        "compliance_result": result,
        "loop_count": state.loop_count + 1,
    }


def generate_output_node(state: AgentState) -> dict:
    """Output Generator -- finalizes the approved claim. PLACEHOLDER for Day 5."""
    print(f"\n[OUTPUT] Claim approved! Finalizing...")
    print(f"\n{'='*60}")
    print("FINAL APPROVED CLAIM:")
    print(f"{'='*60}")
    print(state.claim_draft.claim_text)
    print(f"{'='*60}\n")
    return {"final_claim": state.claim_draft.claim_text}


def human_review_node(state: AgentState) -> dict:
    """Human-in-the-Loop -- escalates after max retries."""
    print(f"\n[ALERT] Max retries ({state.loop_count}) exceeded!")
    print("   [ESCALATE] Routing to human reviewer for manual approval.")
    return {"escalated_to_human": True}


def route_after_compliance(state: AgentState) -> str:
    """
    Conditional routing function -- called by LangGraph after regulatory_agent.
    Returns the NAME of the next node to execute.

    Priority order:
    1. Circuit breaker (max loops)
    2. PASS -> generate output
    3. FAIL: factual -> medical agent
    4. FAIL: tone/citation -> commercial agent
    """
    if state.loop_count >= settings.max_retry_loops:
        print(f"\n[RED] Max retries reached ({state.loop_count}). Escalating to human.")
        return "human_review"

    if state.compliance_result.status == "PASS":
        print("\n[GREEN] Compliance PASSED. Generating output.")
        return "generate_output"

    failure_type = state.compliance_result.failure_type

    if failure_type == "factual":
        print("\n[YELLOW] Factual failure. Re-routing to Medical Agent.")
        return "medical_agent"

    if failure_type in ("tone", "citation"):
        print(f"\n[YELLOW] {failure_type.capitalize()} failure. Re-routing to Commercial Agent.")
        return "commercial_agent"

    return "human_review"


def build_graph():
    """
    Assembles and compiles the PharmAgentAI LangGraph state machine.

    Graph topology:
        START -> medical_agent -> commercial_agent -> regulatory_agent
                                                           |
                                        [PASS] -> generate_output -> END
                                     [FAIL: factual] -> medical_agent (loop)
                                     [FAIL: tone] -> commercial_agent (loop)
                                     [loop >= 3] -> human_review -> END
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("medical_agent",    medical_agent_node)
    graph.add_node("commercial_agent", commercial_agent_node)
    graph.add_node("regulatory_agent", regulatory_agent_node)
    graph.add_node("generate_output",  generate_output_node)
    graph.add_node("human_review",     human_review_node)

    # Add fixed edges
    graph.add_edge(START,              "medical_agent")
    graph.add_edge("medical_agent",    "commercial_agent")
    graph.add_edge("commercial_agent", "regulatory_agent")
    graph.add_edge("generate_output",  END)
    graph.add_edge("human_review",     END)

    # Add the smart conditional edge (the orchestrator's core logic)
    graph.add_conditional_edges(
        "regulatory_agent",
        route_after_compliance,
        {
            "generate_output":  "generate_output",
            "medical_agent":    "medical_agent",
            "commercial_agent": "commercial_agent",
            "human_review":     "human_review",
        }
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


from config.settings import settings

pharma_graph = build_graph()
