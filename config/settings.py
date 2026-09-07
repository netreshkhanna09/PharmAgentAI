# ==============================================================
# PharmAgentAI — Central Configuration
# ==============================================================
# Uses pydantic-settings to load from .env file automatically
# ==============================================================

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Central configuration for PharmAgentAI.
    All values are loaded from the .env file automatically.
    """

    # --- LLM Provider ---
    groq_api_key: str = Field(..., description="Groq API key for LLM inference")
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")

    # --- LLM Settings ---
    llm_model: str = Field(default="llama3-70b-8192")
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=4096)

    # --- Database ---
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/pharmagent_db"
    )

    # --- Agent Behaviour ---
    max_retry_loops: int = Field(
        default=3,
        description="Max agent retry loops before Human-in-the-Loop escalation"
    )
    log_level: str = Field(default="INFO")

    # --- External APIs ---
    clinical_trials_base_url: str = Field(
        default="https://clinicaltrials.gov/api/v2"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this object everywhere in the codebase
settings = Settings()
