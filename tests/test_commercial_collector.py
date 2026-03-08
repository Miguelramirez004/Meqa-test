"""Tests for commercial model collector (Layer 1: Drug Mention Audit).

Validates:
- Provider detection from model names
- Client creation logic
- Response collection with mocked API calls
- Model listing
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.commercial_collector import (
    detect_provider,
    get_all_models,
    _build_client,
    query_commercial_model,
    collect_commercial_responses,
    PROVIDER_MODELS,
    PROVIDER_PREFIXES,
)


# ═════════════════════════════════════════════════════════════════════════
# 1. Provider Detection
# ═════════════════════════════════════════════════════════════════════════

class TestProviderDetection:

    @pytest.mark.parametrize("model,expected", [
        ("gpt-4o", "openai"),
        ("gpt-4o-mini", "openai"),
        ("gpt-3.5-turbo", "openai"),
        ("o1-preview", "openai"),
        ("o3-mini", "openai"),
        ("gemini-2.0-flash", "gemini"),
        ("gemini-1.5-pro", "gemini"),
        ("gemini-1.5-flash", "gemini"),
        ("llama-3.1-sonar-small-128k-online", "perplexity"),
        ("sonar-large", "perplexity"),
    ])
    def test_detect_known_providers(self, model, expected):
        assert detect_provider(model) == expected

    def test_unknown_model_defaults_to_openai(self):
        assert detect_provider("unknown-model-xyz") == "openai"

    def test_case_insensitive(self):
        assert detect_provider("GPT-4o") == "openai"
        assert detect_provider("Gemini-2.0-flash") == "gemini"


# ═════════════════════════════════════════════════════════════════════════
# 2. Model Listing
# ═════════════════════════════════════════════════════════════════════════

class TestModelListing:

    def test_get_all_models_nonempty(self):
        models = get_all_models()
        assert len(models) > 0

    def test_all_models_are_strings(self):
        for m in get_all_models():
            assert isinstance(m, str)

    def test_contains_known_models(self):
        models = get_all_models()
        assert "gpt-4o" in models
        assert "gemini-2.0-flash" in models

    def test_provider_models_consistent(self):
        """All models in get_all_models should map back to their provider."""
        for provider, models in PROVIDER_MODELS.items():
            for model in models:
                detected = detect_provider(model)
                assert detected == provider, (
                    f"Model {model} detected as {detected}, expected {provider}"
                )


# ═════════════════════════════════════════════════════════════════════════
# 3. Client Building
# ═════════════════════════════════════════════════════════════════════════

class TestClientBuilding:

    @patch("src.commercial_collector._create_openai_client")
    def test_build_openai_client(self, mock_create):
        mock_create.return_value = MagicMock()
        client = _build_client("openai", "test-key")
        mock_create.assert_called_once_with("test-key")

    @patch("src.commercial_collector._create_gemini_client")
    def test_build_gemini_client(self, mock_create):
        mock_create.return_value = MagicMock()
        client = _build_client("gemini", "test-key")
        mock_create.assert_called_once_with("test-key")

    @patch("src.commercial_collector._create_perplexity_client")
    def test_build_perplexity_client(self, mock_create):
        mock_create.return_value = MagicMock()
        client = _build_client("perplexity", "test-key")
        mock_create.assert_called_once_with("test-key")


# ═════════════════════════════════════════════════════════════════════════
# 4. Query Commercial Model
# ═════════════════════════════════════════════════════════════════════════

class TestQueryCommercialModel:

    def _mock_client(self, response_text="Losec contiene omeprazol."):
        client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = response_text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        client.chat.completions.create.return_value = mock_response
        return client

    def test_returns_response_text(self):
        client = self._mock_client("Respuesta de prueba.")
        result = query_commercial_model("¿Qué es Losec?", "gpt-4o", client)
        assert result == "Respuesta de prueba."

    def test_passes_model_and_query(self):
        client = self._mock_client()
        query_commercial_model("¿Qué es Losec?", "gpt-4o", client, temperature=0.5)
        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"
        assert call_kwargs.kwargs["temperature"] == 0.5
        messages = call_kwargs.kwargs["messages"]
        assert messages[1]["content"] == "¿Qué es Losec?"

    def test_returns_empty_on_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        result = query_commercial_model("test", "gpt-4o", client)
        assert result == ""


# ═════════════════════════════════════════════════════════════════════════
# 5. Collect Commercial Responses
# ═════════════════════════════════════════════════════════════════════════

class TestCollectCommercialResponses:

    def _mock_query_fn(self, response_text="Respuesta de prueba."):
        """Return a patch for query_commercial_model."""
        return patch(
            "src.commercial_collector.query_commercial_model",
            return_value=response_text,
        )

    def test_collects_responses(self, tmp_path):
        queries = [
            {"query_id": "Q0001", "query_text": "¿Qué es Losec?",
             "pair_id": "P01", "principio_activo": "omeprazol",
             "grupo_terapeutico": "IBP", "drug_name": "LOSEC",
             "brand_name": "LOSEC", "generic_names": ["OMEPRAZOL CINFA"],
             "is_generic": False, "query_type": "single",
             "query_category": "what_is", "condition": "", "symptom": ""},
        ]

        with self._mock_query_fn():
            with patch("src.commercial_collector._build_client", return_value=MagicMock()):
                results = collect_commercial_responses(
                    queries, "gpt-4o", "fake-key",
                    responses_dir=tmp_path, delay=0,
                )

        assert len(results) == 1
        assert results[0]["query_id"] == "Q0001"
        assert results[0]["collection_method"] == "commercial_direct"
        assert results[0]["provider"] == "openai"
        assert results[0]["model"] == "gpt-4o"

        # Check file was saved
        saved = tmp_path / "Q0001.json"
        assert saved.exists()
        with open(saved, encoding="utf-8") as f:
            data = json.load(f)
        assert data["response_text"] == "Respuesta de prueba."

    def test_skip_existing(self, tmp_path):
        # Pre-create a response file
        (tmp_path / "Q0001.json").write_text("{}", encoding="utf-8")

        queries = [
            {"query_id": "Q0001", "query_text": "test",
             "pair_id": "P01", "principio_activo": "", "grupo_terapeutico": "",
             "drug_name": "", "brand_name": "", "generic_names": [],
             "is_generic": False, "query_type": "single",
             "query_category": "what_is", "condition": "", "symptom": ""},
        ]

        with self._mock_query_fn():
            with patch("src.commercial_collector._build_client", return_value=MagicMock()):
                results = collect_commercial_responses(
                    queries, "gpt-4o", "fake-key",
                    responses_dir=tmp_path, delay=0, skip_existing=True,
                )

        assert len(results) == 0

    def test_empty_response_flagged(self, tmp_path):
        queries = [
            {"query_id": "Q0001", "query_text": "test",
             "pair_id": "P01", "principio_activo": "", "grupo_terapeutico": "",
             "drug_name": "", "brand_name": "", "generic_names": [],
             "is_generic": False, "query_type": "single",
             "query_category": "what_is", "condition": "", "symptom": ""},
        ]

        with self._mock_query_fn(response_text=""):
            with patch("src.commercial_collector._build_client", return_value=MagicMock()):
                results = collect_commercial_responses(
                    queries, "gpt-4o", "fake-key",
                    responses_dir=tmp_path, delay=0,
                )

        assert results[0].get("error") == "Empty response from model"

    def test_progress_callback(self, tmp_path):
        queries = [
            {"query_id": f"Q{i:04d}", "query_text": "test",
             "pair_id": "P01", "principio_activo": "", "grupo_terapeutico": "",
             "drug_name": "", "brand_name": "", "generic_names": [],
             "is_generic": False, "query_type": "single",
             "query_category": "what_is", "condition": "", "symptom": ""}
            for i in range(3)
        ]

        callback = MagicMock()

        with self._mock_query_fn():
            with patch("src.commercial_collector._build_client", return_value=MagicMock()):
                collect_commercial_responses(
                    queries, "gpt-4o", "fake-key",
                    responses_dir=tmp_path, delay=0,
                    progress_callback=callback,
                )

        assert callback.call_count == 3
        callback.assert_any_call(1, 3)
        callback.assert_any_call(2, 3)
        callback.assert_any_call(3, 3)

    def test_gemini_provider(self, tmp_path):
        queries = [
            {"query_id": "Q0001", "query_text": "test",
             "pair_id": "P01", "principio_activo": "", "grupo_terapeutico": "",
             "drug_name": "", "brand_name": "", "generic_names": [],
             "is_generic": False, "query_type": "single",
             "query_category": "what_is", "condition": "", "symptom": ""},
        ]

        with self._mock_query_fn():
            with patch("src.commercial_collector._build_client", return_value=MagicMock()):
                results = collect_commercial_responses(
                    queries, "gemini-2.0-flash", "fake-key",
                    responses_dir=tmp_path, delay=0,
                )

        assert results[0]["provider"] == "gemini"
        assert results[0]["model"] == "gemini-2.0-flash"
