"""CIMA REST API v1.23 client for AEMPS medication database."""

import time
import requests
from .config import CIMA_BASE_URL, CIMA_DOC_BASE, API_DELAY, API_TIMEOUT


class CIMAClient:
    """Client for the CIMA REST API (Agencia Española de Medicamentos y Productos Sanitarios).

    API docs: https://www.aemps.gob.es/apps/cima/docs/CIMA_REST_API.pdf
    Base URL: https://cima.aemps.es/cima/rest/
    """

    def __init__(self, delay: float = API_DELAY):
        self.base_url = CIMA_BASE_URL
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "MeQA-Research/1.0 (Academic Thesis)",
        })

    def _get(self, endpoint: str, params: dict = None) -> dict | None:
        url = f"{self.base_url}/{endpoint}"
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] {url}: {e}")
            return None

    def _post(self, endpoint: str, json_body: list | dict) -> dict | None:
        url = f"{self.base_url}/{endpoint}"
        time.sleep(self.delay)
        try:
            resp = self.session.post(url, json=json_body, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] {url}: {e}")
            return None

    # ── Medications ──────────────────────────────────────────────────────

    def search_medicamentos(self, nombre: str = None, practiv1: str = None,
                            comerc: int = 1, autorizados: int = 1,
                            pagina: int = 1) -> dict | None:
        """Search medications. Returns paginated list."""
        params = {"comerc": comerc, "autorizados": autorizados, "pagina": pagina}
        if nombre:
            params["nombre"] = nombre
        if practiv1:
            params["practiv1"] = practiv1
        return self._get("medicamentos", params)

    def get_medicamento(self, nregistro: str = None, cn: str = None) -> dict | None:
        """Get full medication details by registration number or national code."""
        params = {}
        if nregistro:
            params["nregistro"] = nregistro
        if cn:
            params["cn"] = cn
        return self._get("medicamento", params)

    def search_by_active_ingredient(self, principio_activo: str,
                                     comerc: int = 1) -> dict | None:
        """Find all commercialized meds for a given active ingredient."""
        return self.search_medicamentos(practiv1=principio_activo, comerc=comerc)

    # ── Documents ────────────────────────────────────────────────────────

    def get_doc_segmentado(self, nregistro: str, tipo_doc: int = 2,
                           seccion: str = None) -> dict | None:
        """Get segmented document. tipo_doc: 1=Ficha Técnica, 2=Prospecto."""
        params = {"nregistro": nregistro}
        if seccion:
            params["seccion"] = seccion
        return self._get(f"docSegmentado/contenido/{tipo_doc}", params)

    def get_doc_secciones(self, nregistro: str, tipo_doc: int = 2) -> dict | None:
        """Get document section index (without content)."""
        return self._get(f"docSegmentado/secciones/{tipo_doc}",
                         {"nregistro": nregistro})

    def buscar_en_ficha_tecnica(self, seccion: str, texto: str,
                                contiene: int = 1) -> dict | None:
        """Search within fichas técnicas by section and text."""
        body = [{"seccion": seccion, "texto": texto, "contiene": contiene}]
        return self._post("buscarEnFichaTecnica", body)

    # ── URL builders ─────────────────────────────────────────────────────

    @staticmethod
    def prospecto_url(nregistro: str) -> str:
        return f"{CIMA_DOC_BASE}/p/{nregistro}/Prospecto.html"

    @staticmethod
    def ficha_tecnica_url(nregistro: str) -> str:
        return f"{CIMA_DOC_BASE}/ft/{nregistro}/FichaTecnica.html"

    # ── Master data ──────────────────────────────────────────────────────

    def get_maestras(self, maestra_id: int, nombre: str = None) -> dict | None:
        """Get master data. 1=P.Activos, 6=Labs, 7=ATC."""
        params = {"maestra": maestra_id}
        if nombre:
            params["nombre"] = nombre
        return self._get("maestras", params)

    # ── Safety & supply ──────────────────────────────────────────────────

    def get_notas_seguridad(self, nregistro: str) -> dict | None:
        return self._get(f"notas/{nregistro}")

    def get_problemas_suministro(self, cn: str = None) -> dict | None:
        if cn:
            return self._get(f"psuministro/{cn}")
        return self._get("psuministro")
