# ==============================================================
# PharmAgentAI -- Central Configuration
# ==============================================================
# Uses pydantic-settings to load from .env file automatically.
# Import the `settings` singleton anywhere in the codebase.
# ==============================================================

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Central configuration for PharmAgentAI.
    All values auto-loaded from .env file.
    """

    # --- LLM Provider (Required) ---
    groq_api_key: str = Field(..., description="Groq API key for LLM inference")

    # --- Embeddings (Optional -- needed only for RAG on Day 5) ---
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for embeddings (needed from Day 5 onward)"
    )

    # --- LLM Settings ---
    llm_model: str = Field(default="qwen/qwen3.8-27b")
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=4096)

    # --- Database (needed from Day 4 onward) ---
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/pharmagent_db"
    )

    # --- Agent Behaviour ---
    max_retry_loops: int = Field(
        default=3,
        description="Max retry loops before Human-in-the-Loop escalation"
    )
    log_level: str = Field(default="INFO")

    # --- External APIs ---
    clinical_trials_base_url: str = Field(
        default="https://clinicaltrials.gov/api/v2"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton -- import this object everywhere
settings = Settings()
