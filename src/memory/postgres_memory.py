# ==============================================================
# Episodic Memory Manager — SQLite/PostgreSQL via SQLAlchemy
# ==============================================================
# PURPOSE: Persists every agent run's outcomes across sessions.
#          Enables the system to learn from past failures.
#
# DATABASE: SQLite by default (zero setup, file-based)
#           Switch to PostgreSQL by changing DATABASE_URL in .env
#           The SQLAlchemy code is IDENTICAL -- only the URL changes.
#           This is database-agnostic design.
#
# 3 TABLES:
#   agent_runs         -- One row per execution of the pipeline
#   claim_history      -- Every claim generated (pass or fail)
#   compliance_failures -- Only failed claims (retrieved for memory augmentation)
#
# KEY DESIGN PATTERNS:
#   - Connection pooling via SQLAlchemy engine (not per-call connections)
#   - Parameterized queries (never format user input into SQL strings)
#   - Context managers for automatic transaction commit/rollback
#   - Singleton engine (one engine, shared across all agents)
# ==============================================================

import uuid
from datetime import datetime, timezone
from typing import Optional, List
from pathlib import Path

from sqlalchemy import (
    create_engine, text, MetaData, Table, Column,
    String, Integer, Float, DateTime, Text
)
from sqlalchemy.engine import Engine


# ── Database URL ────────────────────────────────────────────────
# SQLite: zero setup, file stored at data/pharmagent.db
# To switch to PostgreSQL: change DATABASE_URL in .env to:
#   postgresql://user:password@localhost:5432/pharmagent_db
# Zero code changes needed -- SQLAlchemy handles both identically

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SQLITE_URL = f"sqlite:///{DATA_DIR}/pharmagent.db"


def _get_database_url() -> str:
    """
    Returns the database URL.
    Tries PostgreSQL from settings, falls back to SQLite.
    SQLite requires zero setup -- perfect for development.
    """
    try:
        from config.settings import settings
        db_url = settings.database_url

        # Only use PostgreSQL if explicitly configured (not default placeholder)
        if "postgresql" in db_url and "localhost" in db_url:
            # Try if PostgreSQL is actually available
            test_engine = create_engine(db_url, pool_pre_ping=True)
            with test_engine.connect():
                print("[Memory] PostgreSQL connection successful")
                return db_url
    except Exception:
        pass

    print("[Memory] Using SQLite (zero-setup local database)")
    return SQLITE_URL


# ── Engine (Connection Pool Singleton) ─────────────────────────
# WHY SINGLETON?
# Creating an engine is expensive (establishes connection pool).
# All agents share this ONE engine instance -- no per-call overhead.
# pool_pre_ping validates connections haven't timed out before use.
_db_url = _get_database_url()
_engine: Engine = create_engine(
    _db_url,
    connect_args={"check_same_thread": False} if "sqlite" in _db_url else {},
    pool_pre_ping=True,
)

metadata = MetaData()

# ── Table Definitions ───────────────────────────────────────────
# Using SQLAlchemy Core (not full ORM) for clarity and control.
# Core gives us: parameterized queries, portability, type safety
# without the complexity of mapped classes and sessions.

agent_runs = Table(
    "agent_runs", metadata,
    Column("id",           String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column("drug_name",    String(255), nullable=False),
    Column("status",       String(20),  nullable=False),  # completed | escalated | error
    Column("loop_count",   Integer,     nullable=False, default=0),
    Column("final_claim",  Text,        nullable=True),
    Column("created_at",   DateTime,    default=lambda: datetime.now(timezone.utc)),
    Column("completed_at", DateTime,    nullable=True),
)

claim_history = Table(
    "claim_history", metadata,
    Column("id",                String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column("run_id",            String(36), nullable=False),   # FK to agent_runs.id
    Column("drug_name",         String(255), nullable=False),
    Column("claim_text",        Text,        nullable=False),
    Column("tone",              String(50),  nullable=True),
    Column("compliance_status", String(20),  nullable=True),   # PASS | FAIL
    Column("loop_number",       Integer,     nullable=True),
    Column("created_at",        DateTime,    default=lambda: datetime.now(timezone.utc)),
)

compliance_failures = Table(
    "compliance_failures", metadata,
    Column("id",               String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column("run_id",           String(36), nullable=False),
    Column("drug_name",        String(255), nullable=False),   # Indexed for fast retrieval
    Column("failure_type",     String(50),  nullable=True),    # factual | tone | citation
    Column("violation_details",Text,        nullable=True),
    Column("fda_rule",         String(50),  nullable=True),
    Column("claim_text",       Text,        nullable=True),
    Column("confidence_score", Float,       nullable=True),
    Column("created_at",       DateTime,    default=lambda: datetime.now(timezone.utc)),
)


def init_db():
    """
    Creates all tables if they don't exist.
    Safe to call on every startup -- skips existing tables.
    """
    metadata.create_all(_engine)
    print(f"[Memory] Database initialized: {_db_url}")


# ── Write Functions ─────────────────────────────────────────────

def save_run(
    drug_name: str,
    status: str,
    loop_count: int,
    final_claim: Optional[str] = None,
    run_id: Optional[str] = None,      # ← NEW: accept pre-generated ID from API
) -> str:
    """
    Saves a completed agent run to the database.
    Returns the run_id for linking child records.

    WHY optional run_id?
    When the API pre-generates a UUID for the client to poll against,
    it MUST be the same UUID stored in the DB.
    Without this, the API UUID and DB UUID are different → polling breaks.
    """
    run_id = run_id or str(uuid.uuid4())   # use provided or generate new
    now = datetime.now(timezone.utc)

    with _engine.begin() as conn:
        conn.execute(
            agent_runs.insert().values(
                id=run_id,
                drug_name=drug_name,
                status=status,
                loop_count=loop_count,
                final_claim=final_claim,
                created_at=now,
                completed_at=now,
            )
        )

    return run_id



def save_claim(
    run_id: str,
    drug_name: str,
    claim_text: str,
    tone: str,
    compliance_status: str,
    loop_number: int,
):
    """Saves a claim (pass or fail) to claim_history."""
    with _engine.begin() as conn:
        conn.execute(
            claim_history.insert().values(
                id=str(uuid.uuid4()),
                run_id=run_id,
                drug_name=drug_name,
                claim_text=claim_text,
                tone=tone,
                compliance_status=compliance_status,
                loop_number=loop_number,
                created_at=datetime.now(timezone.utc),
            )
        )


def save_failure(
    run_id: str,
    drug_name: str,
    failure_type: Optional[str],
    violation_details: Optional[str],
    fda_rule: Optional[str],
    claim_text: str,
    confidence_score: float,
):
    """
    Saves a compliance failure for future memory retrieval.
    These records power the memory-augmented generation on future runs.
    """
    with _engine.begin() as conn:
        conn.execute(
            compliance_failures.insert().values(
                id=str(uuid.uuid4()),
                run_id=run_id,
                drug_name=drug_name,
                failure_type=failure_type,
                violation_details=violation_details,
                fda_rule=fda_rule,
                claim_text=claim_text,
                confidence_score=confidence_score,
                created_at=datetime.now(timezone.utc),
            )
        )


# ── Read Functions ──────────────────────────────────────────────

def get_past_failures(drug_name: str, limit: int = 3) -> List[dict]:
    """
    Retrieves the most recent compliance failures for a drug.
    Used by Commercial Agent to avoid repeating past mistakes.

    WHY TOP 3?
    - More than 3 failures would overwhelm the prompt context
    - Most recent failures are most relevant
    - Ordered by created_at DESC ensures freshest knowledge

    This is the READ side of episodic memory:
    "What went wrong the last 3 times we processed this drug?"
    """
    with _engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT failure_type, violation_details, fda_rule, claim_text, created_at
                FROM compliance_failures
                WHERE drug_name = :drug_name
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"drug_name": drug_name, "limit": limit}
        )
        rows = result.fetchall()

    return [
        {
            "failure_type": row[0],
            "violation_details": row[1],
            "fda_rule": row[2],
            "claim_text": row[3],
            "created_at": str(row[4]),
        }
        for row in rows
    ]


def get_run_stats(drug_name: str) -> dict:
    """
    Returns summary statistics for a drug's run history.
    Useful for monitoring and dashboards.
    """
    with _engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    COUNT(*) as total_runs,
                    AVG(loop_count) as avg_loops,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs
                FROM agent_runs
                WHERE drug_name = :drug_name
            """),
            {"drug_name": drug_name}
        )
        row = result.fetchone()

    return {
        "drug_name": drug_name,
        "total_runs": row[0] or 0,
        "avg_loops": round(float(row[1] or 0), 2),
        "successful_runs": row[2] or 0,
    }


# Initialize tables on module load
init_db()
