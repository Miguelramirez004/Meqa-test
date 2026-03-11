"""Tests for PubMed client — unit tests with mocked HTTP."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.pubmed_client import (
    PubMedClient,
    PubMedSearchResult,
    DrugPairPubMedData,
    BrandSearchResult,
    GenericLabSearchResult,
    BRAND_NAMES_BY_INN,
    GENERIC_LABS,
    collect_pubmed_data,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _mock_esearch_response(count=42, ids=None):
    """Build a fake esearch JSON response."""
    if ids is None:
        ids = ["111", "222", "333"]
    return {
        "esearchresult": {
            "count": str(count),
            "idlist": ids,
        }
    }


def _mock_esummary_response(ids):
    """Build a fake esummary JSON response."""
    result = {"uids": ids}
    for aid in ids:
        result[str(aid)] = {
            "uid": str(aid),
            "title": f"Test article {aid}",
            "pubdate": "2024 Jan",
            "source": "Test Journal",
            "authors": [{"name": "Smith A"}, {"name": "Jones B"}],
        }
    return {"result": result}


# ── INN extraction ───────────────────────────────────────────────────


class TestExtractINN:
    def test_basic(self):
        assert PubMedClient._extract_inn("Paracetamol 1g") == "paracetamol"

    def test_with_dose(self):
        assert PubMedClient._extract_inn("Omeprazol 20mg") == "omeprazol"

    def test_single_word(self):
        assert PubMedClient._extract_inn("Ibuprofeno") == "ibuprofeno"

    def test_empty(self):
        assert PubMedClient._extract_inn("") == ""

    def test_complex(self):
        assert PubMedClient._extract_inn("Atorvastatina 20mg") == "atorvastatina"


# ── Brand & generic knowledge base ──────────────────────────────────


class TestKnowledgeBase:
    def test_brand_names_covers_all_20_inns(self):
        expected_inns = [
            "paracetamol", "ibuprofeno", "amoxicilina", "omeprazol",
            "atorvastatina", "enalapril", "metformina", "lorazepam",
            "sertralina", "salbutamol", "amlodipino", "azitromicina",
            "simvastatina", "fluoxetina", "ramipril", "diclofenaco",
            "pantoprazol", "levotiroxina", "alprazolam", "ciprofloxacino",
        ]
        for inn in expected_inns:
            assert inn in BRAND_NAMES_BY_INN, f"Missing INN: {inn}"
            assert len(BRAND_NAMES_BY_INN[inn]) >= 3, (
                f"INN '{inn}' has only {len(BRAND_NAMES_BY_INN[inn])} brands"
            )

    def test_brand_names_include_international(self):
        assert "tylenol" in BRAND_NAMES_BY_INN["paracetamol"]
        assert "advil" in BRAND_NAMES_BY_INN["ibuprofeno"]
        assert "prilosec" in BRAND_NAMES_BY_INN["omeprazol"]
        assert "xanax" in BRAND_NAMES_BY_INN["alprazolam"]
        assert "synthroid" in BRAND_NAMES_BY_INN["levotiroxina"]

    def test_generic_labs_has_major_spanish_labs(self):
        for lab in ["cinfa", "normon", "kern pharma", "teva", "stada", "sandoz"]:
            assert lab in GENERIC_LABS, f"Missing Spanish lab: {lab}"

    def test_generic_labs_has_international_labs(self):
        for lab in ["cipla", "lupin", "aurobindo", "hikma", "fresenius kabi"]:
            assert lab in GENERIC_LABS, f"Missing international lab: {lab}"

    def test_generic_labs_minimum_count(self):
        assert len(GENERIC_LABS) >= 40


# ── _get_all_brands ─────────────────────────────────────────────────


class TestGetAllBrands:
    def test_merges_pair_brands_and_knowledge_base(self):
        pair = {
            "principio_activo": "Paracetamol 1g",
            "brand": "Gelocatil",
            "brands": ["Gelocatil", "Termalgin"],
        }
        brands = PubMedClient._get_all_brands(pair)
        # Should include pair brands + KB brands
        brand_lower = [b.lower() for b in brands]
        assert "gelocatil" in brand_lower
        assert "termalgin" in brand_lower
        assert "tylenol" in brand_lower
        assert "panadol" in brand_lower

    def test_deduplicates(self):
        pair = {
            "principio_activo": "Paracetamol 1g",
            "brand": "Gelocatil",
            "brands": ["Gelocatil", "Termalgin", "Efferalgan"],
        }
        brands = PubMedClient._get_all_brands(pair)
        brand_lower = [b.lower() for b in brands]
        # No duplicates
        assert len(brand_lower) == len(set(brand_lower))

    def test_unknown_inn_uses_pair_brands_only(self):
        pair = {
            "principio_activo": "UnknownDrug 5mg",
            "brand": "BrandX",
            "brands": ["BrandX", "BrandY"],
        }
        brands = PubMedClient._get_all_brands(pair)
        assert len(brands) == 2
        assert brands[0] == "BrandX"

    def test_fallback_to_primary_brand(self):
        pair = {
            "principio_activo": "UnknownDrug 5mg",
            "brand": "BrandX",
        }
        brands = PubMedClient._get_all_brands(pair)
        assert "BrandX" in brands


# ── _get_generic_labs ────────────────────────────────────────────────


class TestGetGenericLabs:
    def test_includes_pair_generics(self):
        pair = {
            "principio_activo": "Omeprazol 20mg",
            "brand": "Losec",
            "generics": ["Omeprazol Cinfa", "Omeprazol Normon"],
        }
        labs = PubMedClient._get_generic_labs(pair)
        lab_names = [name.lower() for name, _ in labs]
        assert "cinfa" in lab_names
        assert "normon" in lab_names

    def test_adds_extended_labs(self):
        pair = {
            "principio_activo": "Omeprazol 20mg",
            "brand": "Losec",
            "generics": ["Omeprazol Cinfa"],
        }
        labs = PubMedClient._get_generic_labs(pair)
        lab_names = [name.lower() for name, _ in labs]
        # Should include labs beyond just Cinfa
        assert "teva" in lab_names
        assert "sandoz" in lab_names

    def test_product_names_include_inn(self):
        pair = {
            "principio_activo": "Omeprazol 20mg",
            "brand": "Losec",
            "generics": ["Omeprazol Cinfa"],
        }
        labs = PubMedClient._get_generic_labs(pair)
        for _, product in labs:
            assert "omeprazol" in product.lower() or "cinfa" in product.lower()

    def test_deduplicates_labs(self):
        pair = {
            "principio_activo": "Paracetamol 1g",
            "brand": "Gelocatil",
            "generics": [
                "Paracetamol Cinfa", "Paracetamol Normon",
                "Paracetamol Kern Pharma",
            ],
        }
        labs = PubMedClient._get_generic_labs(pair)
        lab_names = [name.lower() for name, _ in labs]
        assert len(lab_names) == len(set(lab_names))


# ── esearch ──────────────────────────────────────────────────────────


class TestEsearch:
    @patch("src.pubmed_client.time.sleep")
    def test_esearch_success(self, mock_sleep):
        client = PubMedClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_esearch_response(100, ["1", "2"])
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_resp):
            result = client.esearch("paracetamol")

        assert result["esearchresult"]["count"] == "100"
        assert result["esearchresult"]["idlist"] == ["1", "2"]

    @patch("src.pubmed_client.time.sleep")
    def test_esearch_includes_api_key(self, mock_sleep):
        client = PubMedClient(api_key="test_key_123")
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_esearch_response()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            client.esearch("test")
            call_params = mock_get.call_args[1]["params"]
            assert call_params["api_key"] == "test_key_123"

    @patch("src.pubmed_client.time.sleep")
    def test_esearch_network_error(self, mock_sleep):
        import requests as req
        client = PubMedClient()
        with patch.object(client.session, "get",
                          side_effect=req.exceptions.ConnectionError("Network error")):
            result = client.esearch("test")
        assert result is None


# ── esummary ─────────────────────────────────────────────────────────


class TestEsummary:
    @patch("src.pubmed_client.time.sleep")
    def test_esummary_success(self, mock_sleep):
        client = PubMedClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_esummary_response(["111", "222"])
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_resp):
            result = client.esummary(["111", "222"])

        assert "result" in result
        assert result["result"]["111"]["title"] == "Test article 111"

    @patch("src.pubmed_client.time.sleep")
    def test_esummary_empty_ids(self, mock_sleep):
        client = PubMedClient()
        assert client.esummary([]) is None


# ── search_and_summarise ─────────────────────────────────────────────


class TestSearchAndSummarise:
    @patch("src.pubmed_client.time.sleep")
    def test_full_flow(self, mock_sleep):
        client = PubMedClient()
        ids = ["111", "222"]

        mock_search = MagicMock()
        mock_search.json.return_value = _mock_esearch_response(50, ids)
        mock_search.raise_for_status = MagicMock()

        mock_summary = MagicMock()
        mock_summary.json.return_value = _mock_esummary_response(ids)
        mock_summary.raise_for_status = MagicMock()

        with patch.object(client.session, "get",
                          side_effect=[mock_search, mock_summary]):
            result = client.search_and_summarise("paracetamol", top_n=5)

        assert result.total_count == 50
        assert len(result.articles) == 2
        assert result.articles[0]["pmid"] == "111"

    @patch("src.pubmed_client.time.sleep")
    def test_no_results(self, mock_sleep):
        client = PubMedClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_esearch_response(0, [])
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_resp):
            result = client.search_and_summarise("nonexistent_drug_xyz")

        assert result.total_count == 0
        assert result.articles == []


# ── collect_for_pair ─────────────────────────────────────────────────


class TestCollectForPair:
    @patch("src.pubmed_client.time.sleep")
    def test_searches_all_brands_and_generics(self, mock_sleep):
        client = PubMedClient()
        pair = {
            "pair_id": "P01_PARACET",
            "principio_activo": "Paracetamol 1g",
            "brand": "Gelocatil",
            "brands": ["Gelocatil", "Termalgin"],
            "generics": ["Paracetamol Cinfa", "Paracetamol Normon"],
        }

        # Mock search_and_summarise to avoid counting exact HTTP calls
        mock_result = PubMedSearchResult(query="test", total_count=10)
        with patch.object(client, "search_and_summarise", return_value=mock_result):
            data = client.collect_for_pair(pair, top_n=2)

        assert data.pair_id == "P01_PARACET"
        assert data.inn_search is not None
        assert data.brand_vs_generic_search is not None
        assert data.bioequivalence_search is not None

        # Should have searched all brands (pair brands + KB brands)
        assert len(data.brand_searches) > 2  # at least pair brands + some KB
        assert data.total_brand_pubs > 0

        # Should have searched all generic labs (pair generics + extended)
        assert len(data.generic_lab_searches) > 2  # at least pair labs + extended
        assert data.total_generic_pubs > 0

        # all_brands should be populated
        assert "Gelocatil" in data.all_brands
        assert "Termalgin" in data.all_brands

    @patch("src.pubmed_client.time.sleep")
    def test_brand_searches_include_kb_brands(self, mock_sleep):
        client = PubMedClient()
        pair = {
            "pair_id": "P04_OMEPRAZ",
            "principio_activo": "Omeprazol 20mg",
            "brand": "Losec",
            "brands": ["Losec", "Belfa"],
            "generics": ["Omeprazol Cinfa"],
        }

        mock_result = PubMedSearchResult(query="test", total_count=5)
        with patch.object(client, "search_and_summarise", return_value=mock_result):
            data = client.collect_for_pair(pair, top_n=1)

        brand_names = [bs["brand_name"].lower() for bs in data.brand_searches]
        assert "losec" in brand_names
        assert "prilosec" in brand_names  # from KB
        assert "mepral" in brand_names   # from KB

    @patch("src.pubmed_client.time.sleep")
    def test_generic_searches_include_lab_name(self, mock_sleep):
        client = PubMedClient()
        pair = {
            "pair_id": "P04_OMEPRAZ",
            "principio_activo": "Omeprazol 20mg",
            "brand": "Losec",
            "brands": ["Losec"],
            "generics": ["Omeprazol Cinfa", "Omeprazol Teva"],
        }

        queries_used = []

        def mock_search(query, top_n=5):
            queries_used.append(query)
            return PubMedSearchResult(query=query, total_count=3)

        with patch.object(client, "search_and_summarise", side_effect=mock_search):
            data = client.collect_for_pair(pair, top_n=1)

        # Generic searches should contain both INN and lab name
        generic_queries = [
            gs["search"]["query"]
            for gs in data.generic_lab_searches
            if gs.get("search")
        ]
        # Check that Cinfa and Teva appear in queries
        cinfa_queries = [q for q in generic_queries if "Cinfa" in q or "cinfa" in q]
        assert len(cinfa_queries) >= 1
        teva_queries = [q for q in generic_queries if "Teva" in q or "teva" in q]
        assert len(teva_queries) >= 1


# ── collect_pubmed_data ──────────────────────────────────────────────


class TestCollectPubMedData:
    @patch("src.pubmed_client.time.sleep")
    def test_saves_files(self, mock_sleep, tmp_path):
        pairs = [
            {
                "pair_id": "P01_TEST",
                "principio_activo": "TestDrug 10mg",
                "brand": "BrandX",
                "brands": ["BrandX"],
                "generics": ["TestDrug Cinfa"],
            },
        ]

        with patch("src.pubmed_client.PubMedClient.esearch",
                    return_value=_mock_esearch_response(5, ["1"])):
            with patch("src.pubmed_client.PubMedClient.esummary",
                       return_value=_mock_esummary_response(["1"])):
                results = collect_pubmed_data(
                    pairs, output_dir=tmp_path, top_n=2)

        assert len(results) == 1
        assert results[0]["pair_id"] == "P01_TEST"

        # Check files were saved
        assert (tmp_path / "P01_TEST_pubmed.json").exists()
        assert (tmp_path / "pubmed_all_pairs.json").exists()

        # Verify JSON is valid and has new structure
        with open(tmp_path / "pubmed_all_pairs.json") as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert "brand_searches" in loaded[0]
        assert "generic_lab_searches" in loaded[0]
        assert "total_brand_pubs" in loaded[0]
        assert "total_generic_pubs" in loaded[0]
