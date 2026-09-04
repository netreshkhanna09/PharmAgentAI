# 🧬 PharmAgentAI

> **Production-Grade Multi-Agent AI System for Pharmaceutical Regulatory Compliance**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What is PharmAgentAI?

PharmAgentAI is a **multi-agent orchestration system** that automates the creation and regulatory compliance validation of pharmaceutical marketing claims. 

Pharmaceutical companies must ensure every product claim (e.g., *"Drug X improved Overall Survival by 34%"*) is:
- ✅ Backed by verifiable clinical trial data
- ✅ Compliant with FDA promotional guidelines
- ✅ Written in appropriate scientific tone
- ✅ Free from unsubstantiated or misleading language

PharmAgentAI uses a **graph-based multi-agent architecture** to autonomously draft, validate, and refine claims — escalating to human review only when automated validation fails repeatedly.

---

## 🏗️ System Architecture

```
                         ┌─────────────────────────────────┐
                         │         USER / API REQUEST       │
                         │   "Write a claim for Drug X"     │
                         └─────────────────┬───────────────┘
                                           │
                                           ▼
                         ┌─────────────────────────────────┐
                         │        ORCHESTRATOR              │
                         │     (LangGraph State Machine)    │
                         └──┬──────────┬──────────┬────────┘
                            │          │          │
                            ▼          │          ▼
              ┌─────────────────────┐  │  ┌──────────────────────┐
              │  MEDICAL AFFAIRS    │  │  │  COMMERCIAL AGENT     │
              │  AGENT              │  │  │                       │
              │  - Fetches trial    │  │  │  - Writes marketing   │
              │    data from        │  │  │    claim copy         │
              │    ClinicalTrials   │  │  │  - Adjusts tone       │
              │  - Validates        │  │  │  - Ensures clarity    │
              │    statistics       │  │  └──────────────────────┘
              └─────────────────────┘  │
                                       ▼
                         ┌─────────────────────────────────┐
                         │    REGULATORY COMPLIANCE AGENT   │
                         │                                  │
                         │  - Checks claim vs FDA rules     │
                         │  - Validates citations           │
                         │  - Returns: PASS / FAIL + type   │
                         └──────────────────┬──────────────┘
                                            │
                              ┌─────────────┴──────────────┐
                              │                            │
                         ┌────▼──────┐           ┌────────▼──────────┐
                         │   PASS    │           │      FAIL          │
                         │           │           │  factual → Medical │
                         │  Generate │           │  tone → Commercial │
                         │  PDF +    │           │  loop≥3 → Human   │
                         │  Send     │           │  Review            │
                         └───────────┘           └───────────────────┘
```

---

## 🤖 The 3 Specialized Agents

### 1. Medical Affairs Agent
**Role:** The "doctor" of the system. Responsible for factual accuracy.
- Fetches real clinical trial data from **ClinicalTrials.gov API**
- Validates statistical claims (p-values, hazard ratios, confidence intervals)
- Tools: `search_clinical_trials()`, `validate_statistics()`

### 2. Commercial Agent  
**Role:** The "copywriter" of the system. Responsible for marketing language.
- Drafts clear, compelling, scientifically-accurate claim copy
- Adjusts tone when flagged by compliance (never alters facts)
- Tools: `draft_claim()`, `adjust_tone()`

### 3. Regulatory Compliance Agent
**Role:** The "auditor" of the system. Acts as adversarial checker.
- Checks every claim against 8 FDA promotional guideline rules via **RAG + FAISS**
- Returns structured output: `{status, failure_type, violation_details}`
- Tools: `check_fda_compliance()`, `retrieve_guidelines()`

---

## 🧠 5-Component Agent Framework

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Reasoning Engine** | Groq API (Llama 3.3 70B) | Sub-second structured JSON reasoning |
| **Perception** | Regulatory Compliance Agent | Evaluates draft claims against FDA rules |
| **Tools / Actions** | ClinicalTrials.gov, FAISS, WeasyPrint, SMTP | External world interaction |
| **Memory** | LangGraph State + PostgreSQL | Short-term run state + long-term rejection history |
| **Orchestration** | LangGraph Graph | Dynamic routing based on failure_type |

---

## 💾 Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   MEMORY LAYERS                          │
│                                                          │
│  Working Memory    → LangGraph State (current run)       │
│  Episodic Memory   → PostgreSQL (past_rejections table)  │
│  Semantic Memory   → FAISS Vector Store (FDA guidelines) │
│  Procedural Memory → System Prompts + Graph Topology     │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Agent Orchestration** | LangGraph 0.2+ | Graph-based state machine for dynamic routing |
| **LLM Provider** | Groq API (Llama 3.3 70B) | Sub-500ms inference, free tier |
| **Vector Database** | FAISS | Local, fast, no infrastructure needed |
| **Embeddings** | OpenAI `text-embedding-3-small` | High quality, cheap |
| **Long-term Memory** | PostgreSQL + SQLAlchemy | ACID compliance for audit trails |
| **Structured Outputs** | Pydantic v2 | Type-safe LLM responses |
| **PDF Generation** | WeasyPrint | Compliance-ready PDF output |
| **External API** | ClinicalTrials.gov REST API | Real clinical trial data |

---

## 📁 Project Structure

```
PharmAgentAI/
├── README.md                     # You are here
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
├── config/
│   └── settings.py               # Central config management
├── src/
│   ├── agents/
│   │   ├── medical_agent.py      # Clinical data retrieval + validation
│   │   ├── commercial_agent.py   # Claim drafting + tone adjustment
│   │   └── regulatory_agent.py  # FDA compliance checking
│   ├── orchestrator/
│   │   └── graph.py              # LangGraph state machine
│   ├── tools/
│   │   ├── clinical_trials.py    # ClinicalTrials.gov API wrapper
│   │   ├── fda_checker.py        # RAG-based compliance checker
│   │   └── pdf_generator.py      # PDF output generator
│   ├── memory/
│   │   ├── vector_store.py       # FAISS vector DB management
│   │   └── postgres_memory.py    # PostgreSQL episodic memory
│   └── models/
│       └── schemas.py            # Pydantic data models
├── data/
│   ├── fda_guidelines/           # FDA promotional guideline PDFs
│   └── sample_claims/            # Test claim inputs
├── tests/
│   ├── test_agents.py            # Unit tests for each agent
│   └── test_tools.py             # Tool integration tests
└── docs/
    └── architecture.md           # Detailed architecture documentation
```

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/netreshkhanna09/PharmAgentAI.git
cd PharmAgentAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the system
python -m src.orchestrator.graph
```

---

## 🗺️ Development Roadmap

- [x] **Day 1** — Project setup, architecture design, Git initialization
- [ ] **Day 2** — First working agent node (Medical Affairs Agent)
- [ ] **Day 3** — Long-term memory (PostgreSQL episodic memory)
- [ ] **Day 4** — LangGraph multi-agent orchestration graph
- [ ] **Day 5** — RAG pipeline (FAISS + FDA guidelines)
- [ ] **Day 6** — Human-in-the-Loop + production guardrails
- [ ] **Day 7** — Evaluation framework + full system demo

---

## 👤 Author

**Netresh Khanna** — AI Agent Engineer  
GitHub: [@netreshkhanna09](https://github.com/netreshkhanna09)

---

*Built as a portfolio project demonstrating production-grade Multi-Agent AI engineering skills.*
