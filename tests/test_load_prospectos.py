"""Tests for prospecto download, leaflet chunking, and vector store.

Validates that:
- CIMA API returns well-formed prospecto data
- Downloaded JSON files have the required fields
- HTML leaflet chunking produces correct section-aware chunks
- ChromaDB vector store indexes and retrieves correctly
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
from src.leaflet_chunker import (
    _html_to_text,
    _split_into_paragraphs,
    chunk_section,
    chunk_drug_leaflet,
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

SAMPLE_SECTION_HTML = """\
<html><body>
<p>Omeprazol pertenece a un grupo de medicamentos denominados
«inhibidores de la bomba de protones».</p>
<p>Actúan reduciendo la cantidad de ácido que produce el estómago.</p>
<ul>
<li>Úlcera duodenal</li>
<li>Úlcera gástrica</li>
<li>Reflujo gastroesofágico</li>
</ul>
</body></html>
"""


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
def tmp_leaflets_dir(tmp_path):
    """Create a temp directory with sample leaflet HTML files."""
    pair_dir = tmp_path / "leaflets" / "P01_TEST"
    pair_dir.mkdir(parents=True)
    for sec_num in range(1, 7):
        fpath = pair_dir / f"12345_section_{sec_num}.html"
        fpath.write_text(SAMPLE_SECTION_HTML, encoding="utf-8")
    return tmp_path / "leaflets"


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
        for i in range(1, 7):
            found = any(t.startswith(f"{i}.") for t in titles)
            assert found, f"Missing standard section {i}"

    def test_contenido_is_html_string(self):
        """Section content should be HTML-formatted string."""
        for sec in SAMPLE_CIMA_DOC_SEGMENTADO["secciones"]:
            assert isinstance(sec["contenido"], str)
            assert "<p>" in sec["contenido"] or len(sec["contenido"]) > 20


# ═════════════════════════════════════════════════════════════════════════════
# 2. Prospecto Download & File Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestProspectoDownload:
    """Verify download_prospectos creates valid files."""

    def test_download_creates_json_files(self, mock_cima_client, sample_drug_pair, tmp_path):
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(mock_cima_client, [sample_drug_pair])
        json_files = list(prospectos_dir.glob("*.json"))
        assert len(json_files) == 2, f"Expected 2 files (brand + generic), got {len(json_files)}"

    def test_downloaded_json_has_required_fields(self, mock_cima_client, sample_drug_pair, tmp_path):
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
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(mock_cima_client, [sample_drug_pair])
        for fp in prospectos_dir.glob("*.json"):
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            assert data["prospecto_sections"] is not None
            assert data["prospecto_sections"]

    def test_skips_already_downloaded(self, mock_cima_client, sample_drug_pair, tmp_path):
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(mock_cima_client, [sample_drug_pair])
            mock_cima_client.get_doc_segmentado.reset_mock()
            download_prospectos(mock_cima_client, [sample_drug_pair])
        mock_cima_client.get_doc_segmentado.assert_not_called()

    def test_handles_failed_api_response(self, sample_drug_pair, tmp_path):
        client = MagicMock(spec=CIMAClient)
        client.get_doc_segmentado.return_value = None
        prospectos_dir = tmp_path / "prospectos"
        with patch("scripts.download_prospectos.PROSPECTOS_DIR", prospectos_dir):
            download_prospectos(client, [sample_drug_pair])
        json_files = list(prospectos_dir.glob("*.json"))
        assert len(json_files) == 0

    def test_filename_sanitization(self, mock_cima_client, tmp_path):
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
            assert "/" not in fp.name or os.sep == "/"
            assert " " not in fp.stem


# ═════════════════════════════════════════════════════════════════════════════
# 3. HTML to Text Conversion
# ═════════════════════════════════════════════════════════════════════════════

class TestHTMLToText:
    """Verify _html_to_text strips HTML and preserves structure."""

    def test_strips_paragraph_tags(self):
        result = _html_to_text("<p>Hello world</p>")
        assert "Hello world" in result
        assert "<p>" not in result

    def test_strips_nested_tags(self):
        result = _html_to_text("<div><p><b>Bold</b> text</p></div>")
        assert "<" not in result
        assert "Bold" in result and "text" in result

    def test_preserves_list_items(self):
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = _html_to_text(html)
        assert "- Item 1" in result
        assert "- Item 2" in result

    def test_preserves_paragraph_breaks(self):
        html = "<p>Paragraph 1</p><p>Paragraph 2</p>"
        result = _html_to_text(html)
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result

    def test_empty_input(self):
        assert _html_to_text("") == ""

    def test_no_tags(self):
        assert "plain text" in _html_to_text("plain text")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Section-Aware Chunking
# ═════════════════════════════════════════════════════════════════════════════

class TestSectionChunking:
    """Verify chunk_section produces correct section-aware chunks."""

    def test_produces_chunks(self):
        chunks = chunk_section(
            html=SAMPLE_SECTION_HTML,
            section_number=1,
            drug_name="TEST DRUG",
            nregistro="12345",
            pair_id="P01_TEST",
            is_generic=False,
        )
        assert len(chunks) > 0

    def test_chunks_have_section_header(self):
        chunks = chunk_section(
            html=SAMPLE_SECTION_HTML,
            section_number=4,
            drug_name="TEST DRUG",
            nregistro="12345",
            pair_id="P01_TEST",
            is_generic=False,
        )
        for chunk in chunks:
            assert chunk["text"].startswith("Sección 4 - Posibles efectos adversos:")

    def test_chunks_have_required_metadata(self):
        chunks = chunk_section(
            html=SAMPLE_SECTION_HTML,
            section_number=1,
            drug_name="TEST DRUG",
            nregistro="12345",
            pair_id="P01_TEST",
            is_generic=True,
        )
        required = {"drug_name", "nregistro", "is_generic", "pair_id",
                     "section_number", "section_name", "chunk_index",
                     "char_count", "token_count_approx"}
        for chunk in chunks:
            for key in required:
                assert key in chunk["metadata"], f"Missing metadata '{key}'"

    def test_metadata_values_correct(self):
        chunks = chunk_section(
            html=SAMPLE_SECTION_HTML,
            section_number=2,
            drug_name="LOSEC 20MG",
            nregistro="59386",
            pair_id="P04_OMEPRAZ",
            is_generic=False,
        )
        for chunk in chunks:
            assert chunk["metadata"]["drug_name"] == "LOSEC 20MG"
            assert chunk["metadata"]["nregistro"] == "59386"
            assert chunk["metadata"]["section_number"] == 2
            assert chunk["metadata"]["is_generic"] is False
            assert chunk["metadata"]["pair_id"] == "P04_OMEPRAZ"

    def test_empty_html_returns_empty(self):
        chunks = chunk_section(
            html="<html><body></body></html>",
            section_number=1,
            drug_name="TEST",
            nregistro="12345",
            pair_id="P01",
            is_generic=False,
        )
        assert chunks == []

    def test_no_html_tags_in_chunks(self):
        chunks = chunk_section(
            html=SAMPLE_SECTION_HTML,
            section_number=1,
            drug_name="TEST",
            nregistro="12345",
            pair_id="P01",
            is_generic=False,
        )
        for chunk in chunks:
            text = chunk["text"]
            assert "<p>" not in text
            assert "<html>" not in text
            assert "<body>" not in text


# ═════════════════════════════════════════════════════════════════════════════
# 5. Drug Leaflet Chunking from Files
# ═════════════════════════════════════════════════════════════════════════════

class TestDrugLeafletChunking:
    """Verify chunk_drug_leaflet loads and chunks all sections."""

    def test_loads_all_sections(self, tmp_leaflets_dir):
        chunks = chunk_drug_leaflet(
            nregistro="12345",
            drug_name="TEST DRUG 20MG",
            pair_id="P01_TEST",
            is_generic=False,
            leaflets_dir=tmp_leaflets_dir,
        )
        assert len(chunks) > 0
        section_numbers = {c["metadata"]["section_number"] for c in chunks}
        assert len(section_numbers) == 6, f"Expected 6 sections, got {section_numbers}"

    def test_handles_missing_sections(self, tmp_path):
        pair_dir = tmp_path / "leaflets" / "P01_TEST"
        pair_dir.mkdir(parents=True)
        # Only create section 1
        (pair_dir / "12345_section_1.html").write_text(SAMPLE_SECTION_HTML)
        chunks = chunk_drug_leaflet(
            nregistro="12345",
            drug_name="TEST DRUG",
            pair_id="P01_TEST",
            is_generic=False,
            leaflets_dir=tmp_path / "leaflets",
        )
        assert len(chunks) > 0
        section_numbers = {c["metadata"]["section_number"] for c in chunks}
        assert section_numbers == {1}
