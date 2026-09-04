# ==============================================================
# LangGraph Orchestrator — PharmAgentAI State Machine
# ==============================================================
# This file defines the ENTIRE agent graph topology:
# - All nodes (agents)
# - All edges (transitions)
# - All conditional routing logic
#
# Think of this as the "circuit board" connecting all agents.
# The graph is compiled ONCE at startup, then run for each request.
# ==============================================================

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.schemas import AgentState
from src.agents.medical_agent import medical_agent_node


# ── Placeholder nodes (built in future days) ───────────────────
# We define skeleton functions so the graph compiles today.
# These will be replaced with real implementations on Days 3-4.

def commercial_agent_node(state: AgentState) -> dict:
    """
    Commercial Agent — drafts marketing claim from trial data.
    PLACEHOLDER: Full implementation in Day 3.
    """
    print(f"\n✍️  [Commercial Agent] Drafting claim for: {state.drug_name}")

    # Temporary hardcoded draft for testing the graph flow
    draft_text = (
        f"{state.drug_name} demonstrated statistically significant improvement "
        f"in {state.trial_data.primary_endpoint} "
        f"(p={state.trial_data.p_value}) in a {state.trial_data.study_phase} trial "
        f"enrolling {state.trial_data.patient_population}."
    )

    print(f"   ✅ Draft claim created")
    print(f"   📝 Claim: {draft_text[:100]}...")

    # Import here to avoid circular imports
    from src.models.schemas import ClaimDraft
    return {
        "claim_draft": ClaimDraft(
            claim_text=draft_text,
            supporting_data=f"p={state.trial_data.p_value}",
            tone="scientific",
        )
    }


def regulatory_agent_node(state: AgentState) -> dict:
    """
    Regulatory Compliance Agent — validates claim against FDA rules.
    PLACEHOLDER: Full RAG implementation in Day 4.
    """
    print(f"\n⚖️  [Regulatory Agent] Checking FDA compliance...")
    print(f"   📋 Claim: {state.claim_draft.claim_text[:80]}...")

    # Temporary: always PASS for now so we can test graph flow
    # Real implementation uses FAISS + FDA guidelines vector store
    from src.models.schemas import ComplianceResult
    result = ComplianceResult(
        status="PASS",
        failure_type=None,
        violation_details=None,
        fda_rule_referenced=None,
        confidence_score=0.92,
    )

    print(f"   ✅ Compliance Status: {result.status} (confidence: {result.confidence_score})")

    return {
        "compliance_result": result,
        "loop_count": state.loop_count + 1,   # Always increment loop counter
    }


def generate_output_node(state: AgentState) -> dict:
    """
    Output Generation — finalizes the approved claim.
    PLACEHOLDER: Full PDF + email implementation in Day 5.
    """
    print(f"\n📄 [Output Generator] Claim approved! Finalizing...")
    print(f"\n{'='*60}")
    print(f"✅ FINAL APPROVED CLAIM:")
    print(f"{'='*60}")
    print(f"{state.claim_draft.claim_text}")
    print(f"{'='*60}\n")

    return {"final_claim": state.claim_draft.claim_text}


def human_review_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop — escalates to human reviewer after max retries.
    """
    print(f"\n🚨 [Human Review] Max retries ({state.loop_count}) exceeded!")
    print(f"   ↗️  Escalating to human reviewer for manual approval.")
    print(f"   📋 Last claim draft: {state.claim_draft.claim_text[:100]}...")

    return {"escalated_to_human": True}


# ── Conditional Routing Function ───────────────────────────────
def route_after_compliance(state: AgentState) -> str:
    """
    The brain of the orchestrator.
    Called by LangGraph after regulatory_agent_node completes.
    Returns the NAME of the next node to execute.

    Routing Logic (in priority order):
    1. Check max retries FIRST (circuit breaker)
    2. If PASS → generate output
    3. If FAIL: factual → re-fetch trial data (medical agent)
    4. If FAIL: tone → re-draft claim (commercial agent)
    5. If FAIL: citation → re-draft claim (commercial agent)
    """
    # ── Circuit Breaker (highest priority) ─────────────────────
    if state.loop_count >= settings.max_retry_loops:
        print(f"\n🔴 [Router] Max retries reached ({state.loop_count}). Escalating to human.")
        return "human_review"

    # ── Happy Path ──────────────────────────────────────────────
    if state.compliance_result.status == "PASS":
        print(f"\n🟢 [Router] Compliance PASSED. Generating output.")
        return "generate_output"

    # ── Failure Routing ─────────────────────────────────────────
    failure_type = state.compliance_result.failure_type

    if failure_type == "factual":
        print(f"\n🟡 [Router] Factual failure. Re-routing to Medical Agent.")
        return "medical_agent"                  # Re-fetch better trial data

    elif failure_type in ("tone", "citation"):
        print(f"\n🟡 [Router] {failure_type.capitalize()} failure. Re-routing to Commercial Agent.")
        return "commercial_agent"               # Re-draft the claim

    # Fallback (should never hit this in practice)
    return "human_review"


# ── Build the Graph ─────────────────────────────────────────────
def build_graph():
    """
    Assembles and compiles the PharmAgentAI LangGraph state machine.

    Graph topology:
        START → medical_agent → commercial_agent → regulatory_agent
                                      ↑                   │
                                      │    [FAIL: tone]    │
                                      └────────────────────┤
                              ↑                            │
                   [FAIL: factual]                         │
                   medical_agent ←─────────────────────────┤
                                                           │
                                              [PASS] → generate_output → END
                                           [loop≥3] → human_review → END

    Returns:
        Compiled LangGraph application ready to invoke
    """
    # Step 1: Create the graph with our typed state schema
    graph = StateGraph(AgentState)

    # Step 2: Add all nodes (agent functions)
    graph.add_node("medical_agent",    medical_agent_node)
    graph.add_node("commercial_agent", commercial_agent_node)
    graph.add_node("regulatory_agent", regulatory_agent_node)
    graph.add_node("generate_output",  generate_output_node)
    graph.add_node("human_review",     human_review_node)

    # Step 3: Add edges (define the flow)
    graph.add_edge(START,               "medical_agent")     # Entry point
    graph.add_edge("medical_agent",     "commercial_agent")  # Always go to commercial after medical
    graph.add_edge("commercial_agent",  "regulatory_agent")  # Always go to regulatory after commercial
    graph.add_edge("generate_output",   END)                 # Output → Done
    graph.add_edge("human_review",      END)                 # Human review → Done

    # Step 4: Add conditional edge (the smart routing)
    graph.add_conditional_edges(
        "regulatory_agent",          # FROM this node
        route_after_compliance,      # CALL this function
        {                            # MAP results to node names
            "generate_output": "generate_output",
            "medical_agent":   "medical_agent",
            "commercial_agent": "commercial_agent",
            "human_review":    "human_review",
        }
    )

    # Step 5: Compile with in-memory checkpointing (for state persistence)
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app


# Import settings needed for routing
from config.settings import settings

# Singleton compiled graph — import this in your runner/API
pharma_graph = build_graph()
