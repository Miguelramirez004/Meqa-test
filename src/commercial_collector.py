"""Commercial model collector for Layer 1: Drug Mention Audit.

Sends queries directly to commercial AI models (ChatGPT, Gemini, Perplexity)
WITHOUT any RAG context. The model responds purely from its parametric
knowledge — this is the core test for corpus density and brand/generic
mention asymmetry.

This is distinct from the RAG pipeline (rag_engine.py) which feeds
prospecto context to the model. The commercial collector measures what
the model "knows" on its own.

Supported models:
  - OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
  - Google: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash
  - Perplexity: llama-3.1-sonar-small-128k-online, etc.
"""

import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import RESPONSES_DIR

logger = logging.getLogger(__name__)


def _create_openai_client(api_key: str):
    """Create OpenAI client."""
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def _create_gemini_client(api_key: str):
    """Create Google Gemini client via OpenAI-compatible endpoint."""
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def _create_perplexity_client(api_key: str):
    """Create Perplexity client via OpenAI-compatible endpoint."""
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://api.perplexity.ai",
    )


# Provider detection from model name
PROVIDER_PREFIXES = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "gemini-": "gemini",
    "llama-": "perplexity",
    "sonar": "perplexity",
}

PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    "perplexity": [
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-online",
    ],
}


def detect_provider(model: str) -> str:
    """Detect the API provider from the model name."""
    model_lower = model.lower()
    for prefix, provider in PROVIDER_PREFIXES.items():
        if model_lower.startswith(prefix):
            return provider
    return "openai"  # default


def get_all_models() -> list[str]:
    """Return flat list of all available models across providers."""
    models = []
    for provider_models in PROVIDER_MODELS.values():
        models.extend(provider_models)
    return models


def _build_client(provider: str, api_key: str):
    """Build the appropriate client for the provider."""
    if provider == "gemini":
        return _create_gemini_client(api_key)
    elif provider == "perplexity":
        return _create_perplexity_client(api_key)
    else:
        return _create_openai_client(api_key)


# System prompt: neutral, no bias toward brand or generic
_SYSTEM_PROMPT = """\
Eres un asistente de información farmacéutica. Responde las preguntas \
sobre medicamentos de forma precisa, objetiva y profesional. \
Responde siempre en español."""


def query_commercial_model(
    query_text: str,
    model: str,
    client,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    """Send a single query to a commercial model and return the response text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Model query failed: %s", e)
        return ""


def collect_commercial_responses(
    queries: list[dict],
    model: str,
    api_key: str,
    responses_dir: Path = RESPONSES_DIR,
    delay: float = 0.5,
    temperature: float = 0.3,
    skip_existing: bool = True,
    progress_callback=None,
) -> list[dict]:
    """Collect responses from a commercial model for all queries.

    No RAG context is provided — the model answers purely from
    its parametric knowledge. This is the core Layer 1 test.

    Args:
        queries: List of query dicts from query_battery.json.
        model: Model identifier (e.g. 'gpt-4o', 'gemini-2.0-flash').
        api_key: API key for the provider.
        responses_dir: Directory to save response JSONs.
        delay: Seconds between API calls.
        temperature: Sampling temperature.
        skip_existing: Skip queries with existing response files.
        progress_callback: Optional callable(current, total).

    Returns:
        List of newly collected response dicts.
    """
    responses_dir.mkdir(parents=True, exist_ok=True)

    provider = detect_provider(model)
    client = _build_client(provider, api_key)

    collected = []
    total = len(queries)

    for i, query in enumerate(queries):
        qid = query.get("query_id", f"Q{i+1:04d}")
        fpath = responses_dir / f"{qid}.json"

        if skip_existing and fpath.exists():
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        query_text = query.get("query_text", "")
        response_text = query_commercial_model(query_text, model, client, temperature)

        response_data = {
            "query_id": qid,
            "pair_id": query.get("pair_id", ""),
            "principio_activo": query.get("principio_activo", ""),
            "grupo_terapeutico": query.get("grupo_terapeutico", ""),
            "drug_name": query.get("drug_name", ""),
            "brand_name": query.get("brand_name", ""),
            "generic_names": query.get("generic_names", []),
            "is_generic": query.get("is_generic", ""),
            "query_type": query.get("query_type", ""),
            "query_category": query.get("query_category", ""),
            "query_text": query_text,
            "condition": query.get("condition", ""),
            "symptom": query.get("symptom", ""),
            "response_text": response_text,
            "response_word_count": len(response_text.split()) if response_text else 0,
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "collection_method": "commercial_direct",
            "model": model,
            "provider": provider,
            "notes": f"Commercial model ({provider}/{model}), no RAG context",
        }

        if not response_text:
            response_data["error"] = "Empty response from model"

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)

        collected.append(response_data)

        if progress_callback:
            progress_callback(i + 1, total)

        if delay > 0 and i < total - 1:
            time.sleep(delay)

    return collected
