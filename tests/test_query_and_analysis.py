"""Tests for Layer 1 query generator and mention-based response analyzer.

Validates:
- Query battery generation with Block A/B/C structure
- Drug name extraction and mention detection
- Response analysis with mention metrics
- Asymmetry score computation
- Incomplete response filtering
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drug_pairs import get_offline_pairs
from src.query_generator import generate_queries, CONDITION_QUERIES, SINGLE_DRUG_QUERIES, COMPARATIVE_QUERIES
from src.response_analyzer import (
    analyze_response,
    is_incomplete_response,
    compute_asymmetry_scores,
    _extract_short_name,
    _find_mentions,
    _sentences_containing,
    INCOMPLETE_RESPONSE_INDICATORS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def offline_pairs():
    return get_offline_pairs()


@pytest.fixture
def queries(offline_pairs):
    return generate_queries(offline_pairs)


@pytest.fixture
def sample_query():
    return {
        "query_id": "Q0001",
        "pair_id": "P01_OMEPRAZOL",
        "brand_name": "LOSEC 20 MG CAPSULAS DURAS GASTRORRESISTENTES",
        "generic_names": [
            "OMEPRAZOL CINFA 20 MG CAPSULAS DURAS GASTRORRESISTENTES EFG",
            "OMEPRAZOL NORMON 20 MG CAPSULAS DURAS GASTRORRESISTENTES EFG",
        ],
        "drug_name": "LOSEC 20 MG CAPSULAS DURAS GASTRORRESISTENTES",
        "is_generic": False,
        "query_type": "single",
        "query_category": "what_is",
    }


# ═════════════════════════════════════════════════════════════════════════
# 1. Query Battery Generation
# ═════════════════════════════════════════════════════════════════════════

class TestQueryGeneration:

    def test_generates_nonempty_battery(self, queries):
        assert len(queries) > 0

    def test_has_all_three_blocks(self, queries):
        types = {q["query_type"] for q in queries}
        assert "condition" in types
        assert "single" in types
        assert "comparative" in types

    def test_condition_queries_have_no_drug_name(self, queries):
        for q in queries:
            if q["query_type"] == "condition":
                assert q["drug_name"] == "", f"Condition query should have empty drug_name: {q['query_id']}"

    def test_single_queries_have_drug_name(self, queries):
        for q in queries:
            if q["query_type"] == "single":
                assert q["drug_name"] != "", f"Single query missing drug_name: {q['query_id']}"

    def test_comparative_queries_have_vs(self, queries):
        for q in queries:
            if q["query_type"] == "comparative":
                assert "vs" in q["drug_name"], f"Comparative should have 'vs': {q['query_id']}"

    def test_all_queries_have_required_fields(self, queries):
        required = {"query_id", "pair_id", "query_type", "query_category",
                     "query_text", "brand_name"}
        for q in queries:
            for field in required:
                assert field in q, f"Missing field '{field}' in {q.get('query_id', '?')}"

    def test_query_ids_unique(self, queries):
        ids = [q["query_id"] for q in queries]
        assert len(ids) == len(set(ids))

    def test_condition_queries_dedup_by_condition(self, queries):
        """Condition queries should run once per unique condition, not per pair."""
        condition_queries = [q for q in queries if q["query_type"] == "condition"]
        # P02 and P08 share 'hipercolesterolemia' — should not be duplicated
        conditions = [q["query_text"] for q in condition_queries]
        assert len(conditions) == len(set(conditions)), "Duplicate condition queries found"

    def test_single_queries_include_brand_and_generic(self, queries):
        single = [q for q in queries if q["query_type"] == "single"]
        has_brand = any(q["is_generic"] is False for q in single)
        has_generic = any(q["is_generic"] is True for q in single)
        assert has_brand
        assert has_generic

    def test_each_pair_has_condition_and_symptom(self, offline_pairs):
        for pair in offline_pairs:
            assert pair.get("condition"), f"Missing condition for {pair['pair_id']}"
            assert pair.get("symptom"), f"Missing symptom for {pair['pair_id']}"

    def test_single_queries_have_paired_drugs(self, queries):
        """Single-drug queries should include paired_drugs for cross-ref analysis."""
        single = [q for q in queries if q["query_type"] == "single"]
        for q in single:
            assert "paired_drugs" in q, f"Missing paired_drugs in {q['query_id']}"


# ═════════════════════════════════════════════════════════════════════════
# 2. Drug Name Extraction
# ═════════════════════════════════════════════════════════════════════════

class TestDrugNameExtraction:

    @pytest.mark.parametrize("full_name,expected", [
        ("LOSEC 20 MG CAPSULAS DURAS GASTRORRESISTENTES", "losec"),
        ("OMEPRAZOL CINFA 20 MG CAPSULAS DURAS EFG", "omeprazol cinfa"),
        ("CARDYL 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA", "cardyl"),
        ("NEOBRUFEN 600 MG COMPRIMIDOS", "neobrufen"),
        ("EUTIROX 100 MICROGRAMOS COMPRIMIDOS", "eutirox"),
    ])
    def test_extract_short_name(self, full_name, expected):
        result = _extract_short_name(full_name)
        assert result == expected

    def test_extract_short_name_empty(self):
        result = _extract_short_name("")
        assert result == ""


# ═════════════════════════════════════════════════════════════════════════
# 3. Mention Detection
# ═════════════════════════════════════════════════════════════════════════

class TestMentionDetection:

    def test_finds_brand_mention(self):
        text = "Losec es un medicamento que contiene omeprazol."
        positions = _find_mentions(text, "LOSEC 20 MG CAPSULAS")
        assert len(positions) > 0

    def test_finds_generic_mention(self):
        text = "Omeprazol Cinfa es un genérico del mismo principio activo."
        positions = _find_mentions(text, "OMEPRAZOL CINFA 20 MG CAPSULAS EFG")
        assert len(positions) > 0

    def test_no_mention_returns_empty(self):
        text = "Este medicamento se usa para la acidez."
        positions = _find_mentions(text, "LOSEC 20 MG CAPSULAS")
        assert len(positions) == 0

    def test_multiple_mentions(self):
        text = "Losec se toma una vez al día. Losec pertenece a los IBP."
        positions = _find_mentions(text, "LOSEC 20 MG")
        assert len(positions) >= 2

    def test_case_insensitive(self):
        text = "LOSEC es un inhibidor de bomba de protones."
        positions = _find_mentions(text, "losec 20 mg")
        assert len(positions) > 0

    def test_sentences_containing_drug(self):
        text = "Losec contiene omeprazol. Se usa para la acidez. Losec reduce el ácido."
        sentences = _sentences_containing(text, "LOSEC 20 MG")
        assert len(sentences) == 2
        assert all("osec" in s.lower() for s in sentences)


# ═════════════════════════════════════════════════════════════════════════
# 4. Response Analysis
# ═════════════════════════════════════════════════════════════════════════

class TestResponseAnalysis:

    def test_analyze_detects_brand(self, sample_query):
        text = "Losec es un medicamento que contiene omeprazol como principio activo."
        metrics = analyze_response(text, sample_query)
        assert metrics["brand_mentioned"] is True
        assert metrics["brand_mention_count"] >= 1

    def test_analyze_detects_generic(self, sample_query):
        text = "Omeprazol Cinfa es un genérico equivalente a Losec."
        metrics = analyze_response(text, sample_query)
        assert metrics["generic_mentioned"] is True

    def test_analyze_detects_first_mentioned(self, sample_query):
        text = "Losec es un IBP. Omeprazol Cinfa es su genérico."
        metrics = analyze_response(text, sample_query)
        assert metrics["first_drug_mentioned"] == "brand"

    def test_analyze_neither_mentioned(self, sample_query):
        text = "Los inhibidores de la bomba de protones reducen el ácido."
        metrics = analyze_response(text, sample_query)
        assert metrics["first_drug_mentioned"] == "neither"

    def test_analyze_has_required_fields(self, sample_query):
        text = "Losec contiene omeprazol."
        metrics = analyze_response(text, sample_query)
        required = {
            "query_id", "pair_id", "brand_mentioned", "generic_mentioned",
            "brand_mention_count", "generic_mention_count",
            "brand_first_position", "generic_first_position",
            "first_drug_mentioned", "response_word_count",
        }
        for field in required:
            assert field in metrics, f"Missing field: {field}"

    def test_analyze_word_count(self, sample_query):
        text = "one two three four five"
        metrics = analyze_response(text, sample_query)
        assert metrics["response_word_count"] == 5

    def test_generic_names_from_string(self):
        """generic_names as semicolon-separated string (from CSV)."""
        query = {
            "query_id": "Q0001",
            "pair_id": "P01",
            "brand_name": "LOSEC 20 MG",
            "generic_names": "OMEPRAZOL CINFA 20 MG; OMEPRAZOL NORMON 20 MG",
            "drug_name": "LOSEC 20 MG",
            "is_generic": False,
            "query_type": "single",
            "query_category": "what_is",
        }
        text = "Losec y Omeprazol Cinfa son equivalentes."
        metrics = analyze_response(text, query)
        assert metrics["brand_mentioned"] is True
        assert metrics["generic_mentioned"] is True


# ═════════════════════════════════════════════════════════════════════════
# 5. Incomplete Response Filtering
# ═════════════════════════════════════════════════════════════════════════

class TestIncompleteFiltering:

    def test_context_not_available(self):
        data = {
            "response_text": "Something",
            "meqa_metadata": {"context_available": False, "chunks_retrieved": 0},
        }
        assert is_incomplete_response("Something", data) is True

    def test_zero_chunks(self):
        data = {
            "response_text": "Something",
            "meqa_metadata": {"context_available": True, "chunks_retrieved": 0},
        }
        assert is_incomplete_response("Something", data) is True

    def test_text_indicator(self):
        text = "Esta información no se encuentra en el prospecto consultado."
        assert is_incomplete_response(text, {}) is True

    def test_complete_response(self):
        text = "Losec contiene omeprazol y se usa para la acidez."
        data = {"meqa_metadata": {"context_available": True, "chunks_retrieved": 5}}
        assert is_incomplete_response(text, data) is False

    def test_no_metadata_complete_text(self):
        text = "Este medicamento se usa para tratar la acidez."
        assert is_incomplete_response(text, {}) is False


# ═════════════════════════════════════════════════════════════════════════
# 6. Asymmetry Score Computation
# ═════════════════════════════════════════════════════════════════════════

class TestAsymmetryScores:

    def test_basic_asymmetry(self):
        metrics = [
            {"pair_id": "P01", "brand_mention_count": 5, "generic_mention_count": 2,
             "brand_first_position": 3, "generic_first_position": 10,
             "first_drug_mentioned": "brand", "query_type": "single",
             "is_generic": False, "brand_mentioned": True, "generic_mentioned": False},
            {"pair_id": "P01", "brand_mention_count": 3, "generic_mention_count": 1,
             "brand_first_position": 5, "generic_first_position": 15,
             "first_drug_mentioned": "brand", "query_type": "single",
             "is_generic": True, "brand_mentioned": True, "generic_mentioned": False},
        ]
        scores = compute_asymmetry_scores(metrics)
        assert len(scores) == 1
        assert scores[0]["pair_id"] == "P01"
        assert scores[0]["mention_asymmetry"] > 0  # brand dominates

    def test_symmetric_mentions(self):
        metrics = [
            {"pair_id": "P01", "brand_mention_count": 3, "generic_mention_count": 3,
             "brand_first_position": 5, "generic_first_position": 5,
             "first_drug_mentioned": "brand", "query_type": "single",
             "is_generic": False, "brand_mentioned": True, "generic_mentioned": True},
            {"pair_id": "P01", "brand_mention_count": 3, "generic_mention_count": 3,
             "brand_first_position": 5, "generic_first_position": 5,
             "first_drug_mentioned": "generic", "query_type": "single",
             "is_generic": True, "brand_mentioned": True, "generic_mentioned": True},
        ]
        scores = compute_asymmetry_scores(metrics)
        assert scores[0]["mention_asymmetry"] == 0.0

    def test_multiple_pairs(self):
        metrics = [
            {"pair_id": "P01", "brand_mention_count": 5, "generic_mention_count": 2,
             "brand_first_position": None, "generic_first_position": None,
             "first_drug_mentioned": "brand", "query_type": "condition",
             "is_generic": "condition", "brand_mentioned": True, "generic_mentioned": False},
            {"pair_id": "P02", "brand_mention_count": 1, "generic_mention_count": 4,
             "brand_first_position": None, "generic_first_position": None,
             "first_drug_mentioned": "generic", "query_type": "condition",
             "is_generic": "condition", "brand_mentioned": False, "generic_mentioned": True},
        ]
        scores = compute_asymmetry_scores(metrics)
        assert len(scores) == 2

    def test_composite_score_bounded(self):
        metrics = [
            {"pair_id": "P01", "brand_mention_count": 10, "generic_mention_count": 0,
             "brand_first_position": 0, "generic_first_position": None,
             "first_drug_mentioned": "brand", "query_type": "single",
             "is_generic": False, "brand_mentioned": True, "generic_mentioned": False},
        ]
        scores = compute_asymmetry_scores(metrics)
        cas = scores[0]["composite_asymmetry_score"]
        assert 0 <= cas <= 1, f"CAS out of bounds: {cas}"
