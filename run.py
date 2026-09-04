# ==============================================================
# PharmAgentAI — Test Runner
# ==============================================================
# Run this script to test the full agent graph end-to-end.
#
# Usage:
#   .\venv\Scripts\python run.py
#
# What it does:
#   1. Initializes the PharmAgentAI LangGraph graph
#   2. Sends a test drug claim request
#   3. Shows each agent executing step by step
#   4. Prints the final approved claim
# ==============================================================

import uuid
from src.orchestrator.graph import pharma_graph
from src.models.schemas import AgentState


def run_pharma_agent(drug_name: str):
    """
    Runs the PharmAgentAI system for a given drug.

    Args:
        drug_name: Name of the drug to generate a compliant claim for
    """
    print(f"\n{'='*60}")
    print(f"🚀 PharmAgentAI — Starting agent run")
    print(f"💊 Drug: {drug_name}")
    print(f"{'='*60}")

    # Build the initial state
    initial_state = AgentState(
        user_request=f"Generate a compliant marketing claim for {drug_name}",
        drug_name=drug_name,
    )

    # Each run needs a unique thread_id for state checkpointing
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Invoke the graph — this runs the full multi-agent pipeline
    final_state = pharma_graph.invoke(initial_state, config=config)

    print(f"\n{'='*60}")
    print(f"✅ Agent run complete!")
    if final_state.get("final_claim"):
        print(f"📋 Total revision loops: {final_state.get('loop_count', 0)}")
    elif final_state.get("escalated_to_human"):
        print(f"🚨 Escalated to human review after {final_state.get('loop_count')} loops")
    print(f"{'='*60}\n")

    return final_state


if __name__ == "__main__":
    # Test with Pembrolizumab (Keytruda) — a real FDA-approved cancer drug
    # This will fetch REAL data from ClinicalTrials.gov!
    run_pharma_agent("Pembrolizumab")
