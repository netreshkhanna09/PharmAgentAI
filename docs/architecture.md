# PharmAgentAI — Architecture Documentation

## System Overview

PharmAgentAI is a graph-based multi-agent system built on LangGraph.

## Architecture Decision Records (ADRs)

### ADR-001: Why LangGraph over AutoGen/CrewAI?

**Decision:** Use LangGraph for agent orchestration.

**Reason:**
- LangGraph allows explicit state machine definition with typed state
- Control flow is deterministic and inspectable (not emergent/chaotic)
- Built-in support for cycles, conditional edges, and human-in-the-loop
- Production-ready streaming and checkpointing built-in

### ADR-002: Why Groq over OpenAI GPT-4?

**Decision:** Use Groq API with Llama 3.3 70B for inference.

**Reason:**
- Sub-500ms inference latency (critical for multi-step agent loops)
- Free tier sufficient for development
- Llama 3.3 70B matches GPT-4 quality on structured output tasks

### ADR-003: Why FAISS over Pinecone/ChromaDB?

**Decision:** Use local FAISS for the FDA guidelines vector store.

**Reason:**
- Zero infrastructure overhead (no managed service needed)
- FDA guidelines are a fixed, small dataset (~50MB)
- Sub-millisecond local search vs 50-200ms managed API latency

### ADR-004: Why Pydantic for structured outputs?

**Decision:** All LLM responses must conform to Pydantic models.

**Reason:**
- Type safety across the entire pipeline
- Automatic validation prevents corrupt state from propagating
- Enables deterministic routing based on typed fields (e.g., failure_type)

## Data Flow

```
User Request
    → Medical Affairs Agent (fetches real trial data)
    → Commercial Agent (drafts claim from data)
    → Regulatory Compliance Agent (validates against FDA rules)
    → [PASS] → PDF Generation → Email Output
    → [FAIL: factual] → Medical Affairs Agent (re-fetch)
    → [FAIL: tone] → Commercial Agent (re-draft)
    → [loop >= 3] → Human Review Queue
```
