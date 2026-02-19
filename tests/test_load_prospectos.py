"""Tests for prospecto download, structure validation, and section extraction.

Validates that:
- CIMA API returns well-formed prospecto data
- Downloaded JSON files have the required fields
- Section extraction produces non-empty text chunks
- Section classification maps to correct leaflet numbers (1-6)
- Incomplete / empty prospectos are detected and flagged
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow running from repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cima_client import CIMAClient
from src.drug_pairs import Medicamento, DrugPair, get_offline_pairs
from src.rag_engine import (
    _walk_and_extract,
    _strip_html,
    classify_section,
    load_prospecto_documents,
)
from scripts.download_prospectos import download_prospectos


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CIMA_DOC_SEGMENTADO = {
    "titulo": "PROSPECTO: INFORMACIÓN PARA EL USUARIO",
    "secciones": [
        {
            "titulo": "1. Qué es Omeprazol Cinfa y para qué se utiliza",
            "contenido": "<p>Omeprazol pertenece a un grupo de medicamentos denominados "
                         "«inhibidores de la bomba de protones». Actúan reduciendo la "
                         "cantidad de ácido que produce el estómago.</p>",
        },
        {
            "titulo": "2. Qué necesita saber antes de empezar a tomar Omeprazol Cinfa",
            "contenido": "<p>No tome Omeprazol Cinfa si es alérgico al omeprazol o a "
                         "cualquiera de los demás componentes.</p>",
        },
        {
            "titulo": "3. Cómo tomar Omeprazol Cinfa",
            "contenido": "<p>Siga exactamente las instrucciones de administración de "
                         "este medicamento indicadas por su médico. La dosis "
                         "recomendada es de 20 mg una vez al día.</p>",
        },
        {
            "titulo": "4. Posibles efectos adversos",
            "contenido": "<p>Al igual que todos los medicamentos, este medicamento "
                         "puede producir efectos adversos, como dolor de cabeza, "
                         "diarrea, náuseas y dolor abdominal.</p>",
        },
        {
            "titulo": "5. Conservación de Omeprazol Cinfa",
            "contenido": "<p>No conservar a temperatura superior a 25°C. "
                         "Conservar en el embalaje original.</p>",
        },
        {
            "titulo": "6. Contenido del envase e información adicional",
            "contenido": "<p>Titular de la autorización de comercialización: "
                         "Laboratorios Cinfa, S.A.</p>",
        },
    ],
}

SAMPLE_PROSPECTO_JSON = {
    "nregistro": "68889",
    "nombre": "OMEPRAZOL CINFA 20 MG CAPSULAS DURAS GASTRORRESISTENTES EFG",
    "labtitular": "CINFA",
    "es_generico": True,
    "prospecto_sections": SAMPLE_CIMA_DOC_SEGMENTADO,
}

SAMPLE_EMPTY_PROSPECTO = {
    "nregistro": "99999",
    "nombre": "FAKE DRUG 100 MG",
    "labtitular": "FAKE LAB",
    "es_generico": False,
    "prospecto_sections": None,
}

SAMPLE_BRAND_PROSPECTO_JSON = {
    "nregistro": "59386",
    "nombre": "LOSEC 20 MG CAPSULAS DURAS GASTRORRESISTENTES",
    "labtitular": "ASTRAZENECA",
    "es_generico": False,
    "prospecto_sections": {
        "titulo": "PROSPECTO: INFORMACIÓN PARA EL USUARIO",
        "secciones": [
            {
                "titulo": "1. Qué es Losec y para qué se utiliza",
                "contenido": "<p>Losec contiene el principio activo omeprazol.</p>",
            },
            {
                "titulo": "4. Posibles efectos adversos",
                "contenido": "<p>Efectos frecuentes: dolor de cabeza, diarrea.</p>",
            },
        ],
    },
}


@pytest.fixture
def tmp_prospectos_dir(tmp_path):
    """Create a temp directory with sample prospecto JSON files."""
    d = tmp_path / "prospectos"
    d.mkdir()
    with open(d / "68889_OMEPRAZOL_CINFA.json", "w", encoding="utf-8") as f:
        json.dump(SAMPLE_PROSPECTO_JSON, f, ensure_ascii=False)
    with open(d / "59386_LOSEC.json", "w", encoding="utf-8") as f:
        json.dump(SAMPLE_BRAND_PROSPECTO_JSON, f, ensure_ascii=False)
    return d


@pytest.fixture
def tmp_empty_prospectos_dir(tmp_path):
    """Create a temp directory with an empty/invalid prospecto."""
    d = tmp_path / "prospectos_empty"
    d.mkdir()
    with open(d / "99999_FAKE_DRUG.json", "w", encoding="utf-8") as f:
        json.dump(SAMPLE_EMPTY_PROSPECTO, f, ensure_ascii=False)
    return d


@pytest.fixture
def mock_cima_client():
    """Create a mock CIMA client that returns sample prospecto data."""
    client = MagicMock(spec=CIMAClient)
    client.get_doc_segmentado.return_value = SAMPLE_CIMA_DOC_SEGMENTADO
    return client


@pytest.fixture
def sample_drug_pair():
    """Create a sample DrugPair for testing download logic."""
    brand = Medicamento(
        nregistro="59386",
        nombre="LOSEC 20 MG CAPSULAS DURAS GASTRORRESISTENTES",
        labtitular="ASTRAZENECA",
        pactivos="OMEPRAZOL",
        comerc=True,
        es_generico=False,
    )
    generic = Medicamento(
        nregistro="68889",
        nombre="OMEPRAZOL CINFA 20 MG CAPSULAS DURAS GASTRORRESISTENTES EFG",
        labtitular="CINFA",
        pactivos="OMEPRAZOL",
        comerc=True,
        es_generico=True,
    )
    return DrugPair(
        pair_id="P01_OMEPRAZOL",
        principio_activo="Omeprazol 20mg",
        dosis="20 mg",
        grupo_terapeutico="Gastrointestinal",
        atc_code="A02BC01",
        brand=brand,
        generics=[generic],
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. CIMA API Response Structure
# ═════════════════════════════════════════════════════════════════════════════

class TestCIMAResponseStructure:
    """Verify the expected CIMA API response structure."""

    def test_doc_segmentado_has_secciones(self):
        """CIMA doc_segmentado must contain a 'secciones' list."""
        doc = SAMPLE_CIMA_DOC_SEGMENTADO
        assert "secciones" in doc
        assert isinstance(doc["secciones"], list)
        assert len(doc["secciones"]) > 0

    def test_each_section_has_titulo_and_contenido(self):
        """Each section must have 'titulo' and 'contenido' fields."""
        for sec in SAMPLE_CIMA_DOC_SEGMENTADO["secciones"]:
            assert "titulo" in sec, f"Section missing 'titulo': {sec}"
            assert "contenido" in sec, f"Section missing 'contenido': {sec}"
            assert len(sec["contenido"]) > 0, f"Empty contenido in: {sec['titulo']}"

    def test_all_six_standard_sections_present(self):
        """A complete prospecto should have all 6 standard leaflet sections."""
        titles = [s["titulo"] for s in SAMPLE_CIMA_DOC_SEGMENTADO["secciones"]]
        # Check numbered sections 1-6
        for i in range(1, 7):
            found = any(t.startswith(f"{i}.") for t in titles)
            assert found, f"Missing standard section {i}"

    def test_contenido_is_html_string(self):
        """Section content should be HTML-formatted string."""
        for sec in SAMPLE_CIMA_DOC_SEGMENTADO["secciones"]:
            assert isinstance(sec["contenido"], str)
            # CIMA returns HTML content with tags
            assert "<p>" in sec["contenido"] or len(sec["contenido"]) > 20


# ═════════════════════════════════════════════════════════════════════════════
# 2. Prospecto Download & File Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestProspectoDownload:
    """Verify download_prospectos creates valid files."""

    def test_download_creates_json_files(self, mock_cima_client, sample_drug_pair, tmp_path):
        """download_prospectos should create JSON files for each drug."""
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(mock_cima_client, [sample_drug_pair])

        json_files = list(prospectos_dir.glob("*.json"))
        assert len(json_files) == 2, f"Expected 2 files (brand + generic), got {len(json_files)}"

    def test_downloaded_json_has_required_fields(self, mock_cima_client, sample_drug_pair, tmp_path):
        """Each downloaded JSON must contain nregistro, nombre, es_generico, prospecto_sections."""
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(mock_cima_client, [sample_drug_pair])

        required_fields = {"nregistro", "nombre", "labtitular", "es_generico", "prospecto_sections"}
        for fp in prospectos_dir.glob("*.json"):
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            for field in required_fields:
                assert field in data, f"Missing field '{field}' in {fp.name}"

    def test_downloaded_prospecto_sections_not_empty(self, mock_cima_client, sample_drug_pair, tmp_path):
        """prospecto_sections must not be None or empty."""
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(mock_cima_client, [sample_drug_pair])

        for fp in prospectos_dir.glob("*.json"):
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            assert data["prospecto_sections"] is not None, (
                f"prospecto_sections is None in {fp.name}"
            )
            assert data["prospecto_sections"], f"prospecto_sections is empty in {fp.name}"

    def test_skips_already_downloaded(self, mock_cima_client, sample_drug_pair, tmp_path):
        """Should skip files that already exist."""
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(mock_cima_client, [sample_drug_pair])
            # Reset call count and download again
            mock_cima_client.get_doc_segmentado.reset_mock()
            download_prospectos(mock_cima_client, [sample_drug_pair])

        # API should NOT be called again since files exist
        mock_cima_client.get_doc_segmentado.assert_not_called()

    def test_handles_failed_api_response(self, sample_drug_pair, tmp_path):
        """Should handle None response from CIMA API gracefully."""
        client = MagicMock(spec=CIMAClient)
        client.get_doc_segmentado.return_value = None

        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(client, [sample_drug_pair])

        json_files = list(prospectos_dir.glob("*.json"))
        assert len(json_files) == 0, "Should not create files when API returns None"

    def test_filename_sanitization(self, mock_cima_client, tmp_path):
        """Drug names with / and spaces should be sanitized in filenames."""
        med = Medicamento(
            nregistro="12345",
            nombre="DRUG/NAME WITH SPACES 20 MG",
            labtitular="LAB",
            pactivos="DRUG",
            comerc=True,
            es_generico=False,
        )
        pair = DrugPair(
            pair_id="TEST",
            principio_activo="Drug",
            dosis="20mg",
            grupo_terapeutico="Test",
            atc_code="X00XX00",
            brand=med,
            generics=[],
        )
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(mock_cima_client, [pair])

        for fp in prospectos_dir.glob("*.json"):
            assert "/" not in fp.name or os.sep == "/", f"Unsanitized filename: {fp.name}"
            assert " " not in fp.stem, f"Spaces in filename: {fp.name}"


# ═════════════════════════════════════════════════════════════════════════════
# 3. Section Extraction from Prospecto JSON
# ═════════════════════════════════════════════════════════════════════════════

class TestSectionExtraction:
    """Verify _walk_and_extract produces correct text from CIMA JSON."""

    def test_extracts_all_sections(self):
        """Should extract all 6 sections from a complete prospecto."""
        extracted = _walk_and_extract(SAMPLE_CIMA_DOC_SEGMENTADO)
        assert len(extracted) >= 6, f"Expected >= 6 sections, got {len(extracted)}"

    def test_extracted_content_not_empty(self):
        """Every extracted section must have non-empty content."""
        extracted = _walk_and_extract(SAMPLE_CIMA_DOC_SEGMENTADO)
        for sec in extracted:
            assert sec["content"].strip(), f"Empty content in section: {sec['title']}"

    def test_html_stripped_from_content(self):
        """Extracted content should not contain HTML tags."""
        extracted = _walk_and_extract(SAMPLE_CIMA_DOC_SEGMENTADO)
        for sec in extracted:
            assert "<p>" not in sec["content"], f"HTML not stripped: {sec['content'][:50]}"
            assert "</" not in sec["content"], f"HTML closing tag found: {sec['content'][:50]}"

    def test_titles_preserved(self):
        """Section titles should be extracted and preserved."""
        extracted = _walk_and_extract(SAMPLE_CIMA_DOC_SEGMENTADO)
        titles = [sec["title"] for sec in extracted if sec["title"]]
        assert len(titles) >= 5, "Too few titled sections extracted"

    def test_handles_none_input(self):
        """Should handle None gracefully."""
        result = _walk_and_extract(None)
        assert result == []

    def test_handles_empty_dict(self):
        """Should return empty list for empty dict."""
        result = _walk_and_extract({})
        assert result == []

    def test_handles_plain_string(self):
        """Should wrap a plain string into a section."""
        result = _walk_and_extract("Simple text content")
        assert len(result) == 1
        assert result[0]["content"] == "Simple text content"

    def test_nested_subsections(self):
        """Should handle nested subsecciones."""
        nested = {
            "titulo": "Parent Section",
            "secciones": [
                {
                    "titulo": "Child 1",
                    "contenido": "<p>Child 1 content text.</p>",
                },
                {
                    "titulo": "Child 2",
                    "contenido": "<p>Child 2 content text.</p>",
                },
            ],
        }
        extracted = _walk_and_extract(nested)
        assert len(extracted) >= 2
        contents = [s["content"] for s in extracted]
        assert any("Child 1" in c for c in contents)
        assert any("Child 2" in c for c in contents)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Section Classification
# ═════════════════════════════════════════════════════════════════════════════

class TestSectionClassification:
    """Verify classify_section maps text to the correct leaflet section number."""

    @pytest.mark.parametrize("title,expected", [
        ("1. Qué es Omeprazol y para qué se utiliza", 1),
        ("2. Qué necesita saber antes de empezar a tomar", 2),
        ("3. Cómo tomar Omeprazol", 3),
        ("4. Posibles efectos adversos", 4),
        ("5. Conservación de Omeprazol", 5),
        ("6. Contenido del envase e información adicional", 6),
    ])
    def test_numbered_titles(self, title, expected):
        """Numbered titles (1.-6.) should map to correct section."""
        result = classify_section(title, "")
        assert result == expected, f"'{title}' → {result}, expected {expected}"

    @pytest.mark.parametrize("title,expected", [
        ("Qué es y para qué se utiliza", 1),
        ("Indicaciones terapéuticas", 1),
        ("Antes de tomar este medicamento", 2),
        ("No tome este medicamento", 2),
        ("Cómo tomar este medicamento", 3),
        ("Posología y forma de administración", 3),
        ("Posibles efectos adversos", 4),
        ("Efectos secundarios", 4),
        ("Conservación", 5),
        ("Cómo conservar", 5),
        ("Contenido del envase", 6),
        ("Información adicional", 6),
    ])
    def test_keyword_based_titles(self, title, expected):
        result = classify_section(title, "")
        assert result == expected, f"'{title}' → {result}, expected {expected}"

    def test_empty_title_falls_back_to_content(self):
        """When title is empty, should classify based on content."""
        content = "Este medicamento puede producir efectos adversos frecuentes como dolor de cabeza."
        result = classify_section("", content)
        assert result == 4

    def test_unknown_section_returns_zero(self):
        """Completely unrecognizable text should return 0."""
        result = classify_section("", "xyz abc 123")
        assert result == 0


# ═════════════════════════════════════════════════════════════════════════════
# 5. Load Prospecto Documents (LangChain integration)
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadProspectoDocuments:
    """Verify load_prospecto_documents produces valid LangChain Documents."""

    def test_loads_documents_from_directory(self, tmp_prospectos_dir):
        """Should load Documents from prospecto JSON files."""
        docs = load_prospecto_documents(tmp_prospectos_dir)
        assert len(docs) > 0, "Should produce at least one Document chunk"

    def test_documents_have_required_metadata(self, tmp_prospectos_dir):
        """Each Document must have drug_name, nregistro, es_generico, section_number."""
        docs = load_prospecto_documents(tmp_prospectos_dir)
        required_meta = {"drug_name", "nregistro", "es_generico", "section_number"}
        for doc in docs:
            for key in required_meta:
                assert key in doc.metadata, f"Missing metadata '{key}' in doc"

    def test_documents_have_nonempty_content(self, tmp_prospectos_dir):
        """Every Document chunk must have non-empty page_content."""
        docs = load_prospecto_documents(tmp_prospectos_dir)
        for doc in docs:
            assert doc.page_content.strip(), "Document has empty page_content"

    def test_section_numbers_in_valid_range(self, tmp_prospectos_dir):
        """Section numbers should be 0-6 (0 = unclassified)."""
        docs = load_prospecto_documents(tmp_prospectos_dir)
        for doc in docs:
            sec = doc.metadata["section_number"]
            assert 0 <= sec <= 6, f"Invalid section_number: {sec}"

    def test_empty_directory_returns_empty(self, tmp_path):
        """Empty prospectos directory should return empty list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        docs = load_prospecto_documents(empty_dir)
        assert docs == []

    def test_null_prospecto_sections_handled(self, tmp_empty_prospectos_dir):
        """Files with None prospecto_sections should not crash."""
        docs = load_prospecto_documents(tmp_empty_prospectos_dir)
        # May or may not produce docs, but should not raise
        assert isinstance(docs, list)

    def test_brand_and_generic_both_loaded(self, tmp_prospectos_dir):
        """Both brand and generic drug documents should be loaded."""
        docs = load_prospecto_documents(tmp_prospectos_dir)
        drug_names = {doc.metadata["drug_name"] for doc in docs}
        assert len(drug_names) >= 2, f"Expected docs from 2+ drugs, got: {drug_names}"

    def test_chunk_size_respected(self, tmp_prospectos_dir):
        """Chunks should not exceed chunk_size + overlap."""
        chunk_size = 300
        docs = load_prospecto_documents(tmp_prospectos_dir, chunk_size=chunk_size)
        for doc in docs:
            # Allow some tolerance for the splitter
            assert len(doc.page_content) <= chunk_size * 2, (
                f"Chunk too large: {len(doc.page_content)} chars"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 6. HTML Stripping
# ═════════════════════════════════════════════════════════════════════════════

class TestHTMLStripping:
    """Verify _strip_html removes tags correctly."""

    def test_strips_paragraph_tags(self):
        assert _strip_html("<p>Hello world</p>") == "Hello world"

    def test_strips_nested_tags(self):
        result = _strip_html("<div><p><b>Bold</b> text</p></div>")
        assert "<" not in result
        assert "Bold" in result and "text" in result

    def test_preserves_text_content(self):
        result = _strip_html("<p>Omeprazol 20 mg cápsulas</p>")
        assert "Omeprazol 20 mg cápsulas" in result

    def test_collapses_whitespace(self):
        result = _strip_html("<p>word1</p>  <p>word2</p>")
        assert "  " not in result

    def test_empty_input(self):
        assert _strip_html("") == ""

    def test_no_tags(self):
        assert _strip_html("plain text") == "plain text"
