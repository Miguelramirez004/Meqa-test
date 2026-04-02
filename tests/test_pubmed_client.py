"""Tests for PubMed client — unit tests with mocked HTTP."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.pubmed_client import (
    PubMedClient,
    PubMedSearchResult,
    DrugPairPubMedData,
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
    def test_collects_four_searches(self, mock_sleep):
        client = PubMedClient()
        pair = {
            "pair_id": "P01_PARACET",
            "principio_activo": "Paracetamol 1g",
            "brand": "Gelocatil",
            "generics": [
                "Paracetamol Cinfa",
                "Paracetamol Normon",
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_esearch_response(10, ["1"])
        mock_resp.raise_for_status = MagicMock()

        mock_summary = MagicMock()
        mock_summary.json.return_value = _mock_esummary_response(["1"])
        mock_summary.raise_for_status = MagicMock()

        # 4 searches × 2 calls each (esearch + esummary) = 8 HTTP calls
        with patch.object(client.session, "get",
                          side_effect=[mock_resp, mock_summary] * 4):
            data = client.collect_for_pair(pair)

        assert data.pair_id == "P01_PARACET"
        assert data.inn_search is not None
        assert data.generic_search is not None
        assert data.brand_search is not None
        assert data.bioequivalence_search is not None
        assert data.inn_search.total_count == 10


# ── collect_pubmed_data ──────────────────────────────────────────────


class TestCollectPubMedData:
    @patch("src.pubmed_client.time.sleep")
    def test_saves_files(self, mock_sleep, tmp_path):
        pairs = [
            {
                "pair_id": "P01_TEST",
                "principio_activo": "TestDrug 10mg",
                "brand": "BrandX",
            },
        ]

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_esearch_response(5, ["1"])
        mock_resp.raise_for_status = MagicMock()

        mock_summary = MagicMock()
        mock_summary.json.return_value = _mock_esummary_response(["1"])
        mock_summary.raise_for_status = MagicMock()

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

        # Verify JSON is valid
        with open(tmp_path / "pubmed_all_pairs.json") as f:
            loaded = json.load(f)
        assert len(loaded) == 1
