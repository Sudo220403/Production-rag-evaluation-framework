"""
Configuration & OpenRouter API settings manager (`src/config.py`).
Configured by default for OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)
and `meta-llama/llama-3.3-70b-instruct`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _load_env_file_fallback() -> None:
    """Lightweight .env reader in case python-dotenv is not installed."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.split("#", 1)[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_env_file_fallback()


@dataclass
class RouterAPIConfig:
    """Configuration for OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)."""

    api_key: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_API_KEY",
            os.getenv("ROUTER_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        )
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "ROUTER_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    )
    chat_completions_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_CHAT_URL",
            "https://openrouter.ai/api/v1/chat/completions",
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "ROUTER_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"
        )
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv(
            "ROUTER_JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "ROUTER_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    use_router_embeddings: bool = field(
        default_factory=lambda: os.getenv("USE_ROUTER_EMBEDDINGS", "false").lower()
        in ("true", "1", "yes")
    )
    http_referer: str = field(
        default_factory=lambda: os.getenv(
            "ROUTER_HTTP_REFERER", "http://localhost:8501"
        )
    )
    x_openrouter_title: str = field(
        default_factory=lambda: os.getenv(
            "ROUTER_X_OPENROUTER_TITLE",
            os.getenv("ROUTER_X_TITLE", "Production-RAG-Evaluation-Framework"),
        )
    )
    timeout_seconds: float = 45.0
    temperature: float = 0.0

    @property
    def is_configured(self) -> bool:
        """Returns True if a real OpenRouter API key has been provided."""
        placeholder_keys = {
            "",
            "<openrouter_api_key>",
            "your_router_api_key_here",
            "sk-placeholder",
            "none",
            "null",
        }
        return bool(self.api_key and self.api_key.strip().lower() not in placeholder_keys)

    @property
    def default_headers(self) -> Dict[str, str]:
        """Standard extra headers for OpenRouter (`HTTP-Referer` & `X-OpenRouter-Title`)."""
        headers: Dict[str, str] = {}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_openrouter_title:
            headers["X-OpenRouter-Title"] = self.x_openrouter_title
        return headers

    @property
    def request_headers(self) -> Dict[str, str]:
        """Full HTTP headers dictionary used for `requests.post` to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.default_headers)
        return headers

    def masked_key(self) -> str:
        """Returns a safely masked string representation of the API key for logging."""
        if not self.is_configured:
            return "<NOT_SET>"
        key = self.api_key.strip()
        if len(key) <= 8:
            return "****"
        return f"{key[:4]}...{key[-4:]}"


@dataclass
class EvaluationExperimentConfig:
    """Configuration for a full multi-dimensional RAG ablation experiment."""

    router: RouterAPIConfig = field(default_factory=RouterAPIConfig)
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVAL", "3")))
    evaluation_mode: str = field(
        default_factory=lambda: os.getenv("EVALUATION_MODE", "auto").lower()
    )
    use_ragas_library: bool = True
    random_seed: int = 42

    def resolve_mode(self) -> str:
        """
        Determines whether to run in 'live' (OpenRouter API active) or 'calibrated_hybrid'
        mode when the user has not yet replaced `<OPENROUTER_API_KEY>`.
        """
        if self.evaluation_mode == "live":
            return "live"
        if self.evaluation_mode == "offline":
            return "calibrated_hybrid"
        return "live" if self.router.is_configured else "calibrated_hybrid"
