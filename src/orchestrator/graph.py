# ==============================================================
# LangGraph Orchestrator -- PharmAgentAI State Machine
# ==============================================================
# This file defines the ENTIRE agent graph topology.
# Think of this as the circuit board connecting all agents.
#
# GRAPH TOPOLOGY:
#   START
#     -> medical_agent      (fetches real trial data)
#     -> commercial_agent   (writes LLM-generated claim)
#     -> regulatory_agent   (validates claim vs FDA rules)
#          |
#          |--> [PASS]            -> generate_output -> END
#          |--> [FAIL: factual]   -> medical_agent   (loop)
#          |--> [FAIL: tone]      -> commercial_agent (loop)
#          |--> [loop >= max]     -> human_review     -> END
# ==============================================================

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.schemas import AgentState
from src.agents.medical_agent import medical_agent_node
from src.agents.commercial_agent import commercial_agent_node
from src.agents.regulatory_agent import regulatory_agent_node
from src.memory.postgres_memory import save_run, save_claim, save_failure, get_run_stats
from src.output.report_generator import generate_pdf_report
from config.settings import settings


# regulatory_agent_node is now imported from src.agents.regulatory_agent
# It uses FAISS + sentence-transformers + Groq for real FDA compliance checking


# ── Output Generator (Placeholder -- Full PDF implementation Day 6) ──
def generate_output_node(state: AgentState, config: dict = None) -> dict:
    """
    Generates final output from approved claim.
    Saves run + claim to episodic memory (SQLite/PostgreSQL).
    Generates PDF audit report.

    WHY config parameter?
    LangGraph passes a 'config' dict when invoking the graph.
    The API puts the pre-generated run_id into config['configurable']['run_id']
    so the DB record uses the same UUID the client is polling for.
    """
    print(f"\n[OUTPUT] Claim approved! Finalizing...")
    print(f"\n{'='*60}")
    print("FINAL APPROVED CLAIM:")
    print(f"{'='*60}")
    print(state.claim_draft.claim_text)
    print(f"{'='*60}")
    print(f"Supporting Data: {state.claim_draft.supporting_data}")
    print(f"Tone           : {state.claim_draft.tone}")
    print(f"NCT ID         : {state.trial_data.nct_id}")
    print(f"{'='*60}\n")

    # ── Resolve run_id: use API-provided or generate new ───────────
    # config is {'configurable': {'run_id': '...'}} when called from API
    api_run_id = None
    if config and "configurable" in config:
        api_run_id = config["configurable"].get("run_id")

    # ── Save to Episodic Memory ─────────────────────────────────
    run_id = "unknown"
    try:
        run_id = save_run(
            drug_name=state.drug_name,
            status="completed",
            loop_count=state.loop_count,
            final_claim=state.claim_draft.claim_text,
            run_id=api_run_id,   # ← pass API's UUID so client can poll it
        )
        save_claim(
            run_id=run_id,
            drug_name=state.drug_name,
            claim_text=state.claim_draft.claim_text,
            tone=state.claim_draft.tone or "scientific",
            compliance_status="PASS",
            loop_number=state.loop_count,
        )
        print(f"[Memory] Run saved to DB (run_id: {run_id[:8]}...)")
    except Exception as e:
        print(f"[Memory] Warning: Could not save to DB: {e}")

    # ── Generate PDF Report ─────────────────────────────────────
    try:
        stats = get_run_stats(state.drug_name)
        pdf_path = generate_pdf_report(state, run_id, stats)
        print(f"[OUTPUT] PDF Report Generated: {pdf_path}")
    except Exception as e:
        print(f"[OUTPUT] Warning: Could not generate PDF: {e}")

    return {"final_claim": state.claim_draft.claim_text}



# ── Human Review (Circuit Breaker) ─────────────────────────────
def human_review_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop escalation.
    Saves run + failure to episodic memory before escalating.
    """
    print(f"\n[ALERT] Max retries ({state.loop_count}) exceeded!")
    print("   [ESCALATE] Routing to human reviewer.")
    if state.claim_draft:
        print(f"   [LAST CLAIM] {state.claim_draft.claim_text[:100]}...")
    if state.compliance_result:
        print(f"   [LAST FAILURE] {state.compliance_result.violation_details}")

    # ── Save failure to Episodic Memory ────────────────────────
    try:
        run_id = save_run(
            drug_name=state.drug_name,
            status="escalated",
            loop_count=state.loop_count,
            final_claim=None,
        )
        if state.compliance_result and state.claim_draft:
            save_failure(
                run_id=run_id,
                drug_name=state.drug_name,
                failure_type=state.compliance_result.failure_type,
                violation_details=state.compliance_result.violation_details,
                fda_rule=state.compliance_result.fda_rule_referenced,
                claim_text=state.claim_draft.claim_text,
                confidence_score=state.compliance_result.confidence_score or 0.0,
            )
        print(f"[Memory] Failure saved to DB (run_id: {run_id[:8]}...)")
    except Exception as e:
        print(f"[Memory] Warning: Could not save to DB: {e}")

    return {"escalated_to_human": True}


# ── Conditional Routing Function ───────────────────────────────
def route_after_compliance(state: AgentState) -> str:
    """
    The orchestrator's decision function.
    Called by LangGraph after regulatory_agent_node completes.
    Returns the NAME of the next node.

    PRIORITY ORDER (important -- check circuit breaker FIRST):
    1. loop_count >= max  -> human_review    (circuit breaker)
    2. status == PASS     -> generate_output (happy path)
    3. failure == factual -> medical_agent   (re-fetch data)
    4. failure == tone    -> commercial_agent (re-write copy)
    5. fallback           -> human_review
    """
    # Priority 1: Circuit breaker (ALWAYS check this first)
    if state.loop_count >= settings.max_retry_loops:
        print(f"\n[RED] Circuit breaker: {state.loop_count} loops. Escalating.")
        return "human_review"

    # Priority 2: Happy path
    if state.compliance_result.status == "PASS":
        print("\n[GREEN] Compliance PASSED. Generating output.")
        return "generate_output"

    # Priority 3 & 4: Failure routing
    failure_type = state.compliance_result.failure_type

    if failure_type == "factual":
        print("\n[YELLOW] Factual failure -> re-routing to Medical Agent.")
        return "medical_agent"

    if failure_type in ("tone", "citation"):
        print(f"\n[YELLOW] {failure_type} failure -> re-routing to Commercial Agent.")
        return "commercial_agent"

    # Safety net: failure_type is None but status is FAIL
    # LLM failed to classify -- treat as tone issue and re-draft the claim
    # Better to retry than to immediately escalate to human
    print(f"\n[YELLOW] Unclassified failure (type=None) -> re-routing to Commercial Agent.")
    return "commercial_agent"



# ── Build and Compile the Graph ─────────────────────────────────
def build_graph():
    """
    Assembles and compiles the PharmAgentAI LangGraph state machine.

    WHY StateGraph(AgentState)?
    Passing our Pydantic model as the schema means LangGraph will:
    1. Type-check all node return values against the schema
    2. Merge partial updates (dicts) into the full state automatically
    3. Validate state at each transition -- catching bugs early

    WHY MemorySaver?
    MemorySaver is an in-memory checkpointer that saves state after
    each node execution. This means:
    1. If a node fails, we can resume from the last checkpoint
    2. We can inspect intermediate state for debugging
    3. Human-in-the-loop pauses are possible (Day 6)
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("medical_agent",    medical_agent_node)
    graph.add_node("commercial_agent", commercial_agent_node)   # Real agent!
    graph.add_node("regulatory_agent", regulatory_agent_node)
    graph.add_node("generate_output",  generate_output_node)
    graph.add_node("human_review",     human_review_node)

    # Fixed edges (deterministic transitions)
    graph.add_edge(START,              "medical_agent")
    graph.add_edge("medical_agent",    "commercial_agent")
    graph.add_edge("commercial_agent", "regulatory_agent")
    graph.add_edge("generate_output",  END)
    graph.add_edge("human_review",     END)

    # Conditional edge (dynamic routing based on runtime state)
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

    # Compile with in-memory checkpointing
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Singleton -- import `pharma_graph` anywhere to run the system
pharma_graph = build_graph()
