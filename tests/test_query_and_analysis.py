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
    _extract_inn,
    _find_mentions,
    _find_all_brand_mentions,
    _find_inn_mentions,
    _find_generic_product_mentions,
    _sentences_containing,
    INCOMPLETE_RESPONSE_INDICATORS,
    BRAND_NAMES_BY_INN,
    GENERIC_LABS,
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
        "principio_activo": "Omeprazol 20mg",
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
# 2b. INN Extraction
# ═════════════════════════════════════════════════════════════════════════

class TestINNExtraction:

    @pytest.mark.parametrize("pa,expected", [
        ("Omeprazol 20mg", "omeprazol"),
        ("Atorvastatina 20mg", "atorvastatina"),
        ("Levotiroxina 100mcg", "levotiroxina"),
        ("Ibuprofeno 600mg", "ibuprofeno"),
        ("", ""),
    ])
    def test_extract_inn(self, pa, expected):
        assert _extract_inn(pa) == expected


# ═════════════════════════════════════════════════════════════════════════
# 2c. Brand Knowledge Base
# ═════════════════════════════════════════════════════════════════════════

class TestBrandKnowledgeBase:

    def test_all_inns_have_brands(self):
        """Every INN in the knowledge base must have at least one brand."""
        for inn, brands in BRAND_NAMES_BY_INN.items():
            assert len(brands) > 0, f"No brands for {inn}"

    def test_known_brand_for_omeprazol(self):
        brands = BRAND_NAMES_BY_INN["omeprazol"]
        assert "losec" in brands
        assert "mepral" in brands
        assert "prilosec" in brands

    def test_known_brand_for_atorvastatina(self):
        brands = BRAND_NAMES_BY_INN["atorvastatina"]
        assert "cardyl" in brands
        assert "lipitor" in brands

    def test_generic_labs_nonempty(self):
        assert len(GENERIC_LABS) > 10

    def test_generic_labs_contains_spanish_labs(self):
        assert "cinfa" in GENERIC_LABS
        assert "normon" in GENERIC_LABS
        assert "kern pharma" in GENERIC_LABS
        assert "teva" in GENERIC_LABS
        assert "stada" in GENERIC_LABS


# ═════════════════════════════════════════════════════════════════════════
# 2d. Broad Brand Detection
# ═════════════════════════════════════════════════════════════════════════

class TestBroadBrandDetection:

    def test_detects_primary_brand(self):
        text = "Losec es un inhibidor de bomba de protones."
        positions, found = _find_all_brand_mentions(text, "omeprazol", "LOSEC 20 MG")
        assert len(positions) > 0
        assert "losec" in found

    def test_detects_alternative_brand(self):
        text = "Mepral también contiene omeprazol como principio activo."
        positions, found = _find_all_brand_mentions(text, "omeprazol")
        assert len(positions) > 0
        assert "mepral" in found

    def test_detects_international_brand(self):
        text = "Prilosec es la marca comercial en Estados Unidos."
        positions, found = _find_all_brand_mentions(text, "omeprazol")
        assert len(positions) > 0
        assert "prilosec" in found

    def test_detects_multiple_brands(self):
        text = "Losec y Mepral son marcas comerciales de omeprazol."
        positions, found = _find_all_brand_mentions(text, "omeprazol")
        assert "losec" in found
        assert "mepral" in found

    def test_no_brand_returns_empty(self):
        text = "El omeprazol es un medicamento genérico."
        positions, found = _find_all_brand_mentions(text, "omeprazol")
        assert len(positions) == 0
        assert len(found) == 0

    def test_lipitor_detected_for_atorvastatina(self):
        text = "Lipitor es una de las estatinas más vendidas."
        positions, found = _find_all_brand_mentions(text, "atorvastatina")
        assert "lipitor" in found

    def test_zoloft_detected_for_sertralina(self):
        text = "Zoloft se usa para tratar la depresión."
        positions, found = _find_all_brand_mentions(text, "sertralina")
        assert "zoloft" in found


# ═════════════════════════════════════════════════════════════════════════
# 2e. INN Mention Detection
# ═════════════════════════════════════════════════════════════════════════

class TestINNMentionDetection:

    def test_finds_inn(self):
        text = "El omeprazol reduce la producción de ácido gástrico."
        positions = _find_inn_mentions(text, "omeprazol")
        assert len(positions) > 0

    def test_finds_inn_multiple(self):
        text = "El omeprazol es un IBP. Omeprazol se toma una vez al día."
        positions = _find_inn_mentions(text, "omeprazol")
        assert len(positions) >= 2

    def test_no_inn_returns_empty(self):
        text = "Este medicamento se usa para tratar la acidez."
        positions = _find_inn_mentions(text, "omeprazol")
        assert len(positions) == 0


# ═════════════════════════════════════════════════════════════════════════
# 2f. Generic Product Detection (INN + Lab)
# ═════════════════════════════════════════════════════════════════════════

class TestGenericProductDetection:

    def test_detects_cinfa_product(self):
        text = "Omeprazol Cinfa es un genérico bioequivalente."
        positions, labs = _find_generic_product_mentions(text, "omeprazol")
        assert len(positions) > 0
        assert "cinfa" in labs

    def test_detects_normon_product(self):
        text = "Omeprazol Normon también está disponible en farmacias."
        positions, labs = _find_generic_product_mentions(text, "omeprazol")
        assert "normon" in labs

    def test_detects_teva_product(self):
        text = "Atorvastatina Teva es una alternativa genérica."
        positions, labs = _find_generic_product_mentions(text, "atorvastatina")
        assert "teva" in labs

    def test_no_lab_name_returns_empty(self):
        text = "El omeprazol es un principio activo muy utilizado."
        positions, labs = _find_generic_product_mentions(text, "omeprazol")
        assert len(labs) == 0

    def test_detects_multiple_labs(self):
        text = "Ibuprofeno Cinfa e ibuprofeno Kern Pharma son genéricos."
        positions, labs = _find_generic_product_mentions(text, "ibuprofeno")
        assert "cinfa" in labs
        assert "kern pharma" in labs


# ═════════════════════════════════════════════════════════════════════════
# 3. Mention Detection (legacy)
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

    def test_analyze_detects_inn(self, sample_query):
        text = "El omeprazol es un inhibidor de la bomba de protones."
        metrics = analyze_response(text, sample_query)
        assert metrics["inn_mentioned"] is True
        assert metrics["inn_mention_count"] >= 1
        assert metrics["generic_mentioned"] is True  # INN counts as generic

    def test_analyze_detects_alternative_brand(self, sample_query):
        """Should detect Mepral as a brand for omeprazol, not just Losec."""
        text = "Mepral es otra marca comercial de omeprazol."
        metrics = analyze_response(text, sample_query)
        assert metrics["brand_mentioned"] is True
        assert "mepral" in metrics["brand_names_found"]

    def test_analyze_detects_generic_product(self, sample_query):
        text = "Omeprazol Cinfa es un genérico bioequivalente."
        metrics = analyze_response(text, sample_query)
        assert metrics["generic_product_mentioned"] is True
        assert "cinfa" in metrics["generic_labs_found"]

    def test_analyze_detects_first_mentioned_brand(self, sample_query):
        text = "Losec es un IBP. El omeprazol es su principio activo."
        metrics = analyze_response(text, sample_query)
        assert metrics["first_drug_mentioned"] == "brand"

    def test_analyze_detects_first_mentioned_generic(self, sample_query):
        text = "El omeprazol es un IBP. Losec es una marca de omeprazol."
        metrics = analyze_response(text, sample_query)
        assert metrics["first_drug_mentioned"] == "generic"

    def test_analyze_neither_mentioned(self, sample_query):
        text = "Los inhibidores de la bomba de protones reducen el ácido."
        metrics = analyze_response(text, sample_query)
        assert metrics["first_drug_mentioned"] == "neither"
        assert metrics["brand_mentioned"] is False
        assert metrics["generic_mentioned"] is False
        assert metrics["inn_mentioned"] is False

    def test_analyze_has_required_fields(self, sample_query):
        text = "Losec contiene omeprazol."
        metrics = analyze_response(text, sample_query)
        required = {
            "query_id", "pair_id", "brand_mentioned", "generic_mentioned",
            "brand_mention_count", "generic_mention_count",
            "brand_first_position", "generic_first_position",
            "first_drug_mentioned", "response_word_count",
            # New broad detection fields
            "inn_mentioned", "inn_mention_count", "inn_first_position",
            "brand_names_found", "generic_product_mentioned",
            "generic_product_mention_count", "generic_labs_found",
            "principio_activo",
        }
        for field in required:
            assert field in metrics, f"Missing field: {field}"

    def test_analyze_word_count(self, sample_query):
        text = "one two three four five"
        metrics = analyze_response(text, sample_query)
        assert metrics["response_word_count"] == 5

    def test_only_inn_mentioned_counts_as_generic(self, sample_query):
        """When only the INN is mentioned (no brand, no specific generic product),
        it should still count as generic_mentioned=True."""
        text = "El omeprazol es un fármaco ampliamente utilizado en gastroenterología."
        metrics = analyze_response(text, sample_query)
        assert metrics["inn_mentioned"] is True
        assert metrics["generic_mentioned"] is True
        assert metrics["brand_mentioned"] is False
        assert metrics["first_drug_mentioned"] == "generic"

    def test_brand_and_inn_both_detected(self, sample_query):
        """Both brand and INN in same response."""
        text = "Losec contiene omeprazol como principio activo."
        metrics = analyze_response(text, sample_query)
        assert metrics["brand_mentioned"] is True
        assert metrics["inn_mentioned"] is True
        assert metrics["generic_mentioned"] is True

    def test_multiple_brands_detected(self, sample_query):
        """Multiple brand names for same INN detected."""
        text = "Losec y Mepral son marcas comerciales de omeprazol."
        metrics = analyze_response(text, sample_query)
        assert metrics["brand_mentioned"] is True
        assert "losec" in metrics["brand_names_found"]
        assert "mepral" in metrics["brand_names_found"]

    def test_atorvastatina_broad_detection(self):
        """Test broad detection for a different INN."""
        query = {
            "query_id": "Q0010",
            "pair_id": "P02_ATORVAST",
            "principio_activo": "Atorvastatina 20mg",
            "brand_name": "CARDYL 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
            "generic_names": ["ATORVASTATINA CINFA 20 MG COMPRIMIDOS EFG"],
            "drug_name": "CARDYL 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
            "is_generic": False,
            "query_type": "single",
            "query_category": "what_is",
        }
        text = "Lipitor (atorvastatina) es una de las estatinas más vendidas. Atorvastatina Teva es una alternativa genérica."
        metrics = analyze_response(text, query)
        assert metrics["brand_mentioned"] is True
        assert "lipitor" in metrics["brand_names_found"]
        assert metrics["inn_mentioned"] is True
        assert metrics["generic_product_mentioned"] is True
        assert "teva" in metrics["generic_labs_found"]


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
