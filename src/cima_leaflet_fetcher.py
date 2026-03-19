"""Fetch leaflet sections from CIMA HTML API.

Downloads the 6 standard patient leaflet (prospecto) sections for a drug
using its nregistro, saving each section as a separate HTML file.

CIMA API endpoints:
  Full leaflet:   GET https://cima.aemps.es/cima/dochtml/p/{nregistro}/Prospecto.html
  Single section: GET https://cima.aemps.es/cima/dochtml/p/{nregistro}/{section}/Prospecto.html
"""

import logging
import time
from pathlib import Path

import requests

from .config import CIMA_DOC_BASE, API_TIMEOUT, LEAFLETS_DIR

logger = logging.getLogger(__name__)

SECTION_NAMES = {
    1: "Qué es y para qué se utiliza",
    2: "Qué necesita saber antes de tomar",
    3: "Cómo tomar",
    4: "Posibles efectos adversos",
    5: "Conservación",
    6: "Contenido del envase e información adicional",
}

# Minimum delay between CIMA requests (seconds)
CIMA_FETCH_DELAY = 0.5


def fetch_section_html(nregistro: str, section: int,
                       session: requests.Session | None = None,
                       timeout: int = API_TIMEOUT) -> str | None:
    """Fetch a single leaflet section as HTML from CIMA.

    Args:
        nregistro: CIMA registration number.
        section: Section number (1-6).
        session: Optional requests session for connection reuse.
        timeout: Request timeout in seconds.

    Returns:
        HTML string or None if fetch failed.
    """
    url = f"{CIMA_DOC_BASE}/p/{nregistro}/{section}/Prospecto.html"
    requester = session or requests
    try:
        resp = requester.get(url, timeout=timeout, headers={
            "User-Agent": "MeQA-Research/1.0 (Academic Thesis)",
        })
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch section %d for nregistro=%s: %s",
                       section, nregistro, e)
        return None


def fetch_full_leaflet_html(nregistro: str,
                            session: requests.Session | None = None,
                            timeout: int = API_TIMEOUT) -> str | None:
    """Fetch the complete leaflet as HTML."""
    url = f"{CIMA_DOC_BASE}/p/{nregistro}/Prospecto.html"
    requester = session or requests
    try:
        resp = requester.get(url, timeout=timeout, headers={
            "User-Agent": "MeQA-Research/1.0 (Academic Thesis)",
        })
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch full leaflet for nregistro=%s: %s",
                       nregistro, e)
        return None


def fetch_and_save_leaflet(
    nregistro: str,
    drug_name: str,
    pair_id: str,
    leaflets_dir: Path = LEAFLETS_DIR,
    delay: float = CIMA_FETCH_DELAY,
    skip_existing: bool = True,
) -> dict:
    """Fetch all 6 sections for a drug and save as HTML files.

    Saves to: leaflets_dir/{pair_id}/{nregistro}_section_{n}.html
    Also saves the combined full leaflet.

    Returns:
        Dict with fetch results: {section_number: {"path": Path, "status": str}}
    """
    pair_dir = leaflets_dir / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    results = {}

    # Fetch each section individually
    for sec_num in range(1, 7):
        fpath = pair_dir / f"{nregistro}_section_{sec_num}.html"

        if skip_existing and fpath.exists() and fpath.stat().st_size > 0:
            results[sec_num] = {"path": fpath, "status": "cached"}
            logger.info("  Section %d: cached (%s)", sec_num, fpath.name)
            continue

        time.sleep(delay)
        html = fetch_section_html(nregistro, sec_num, session=session)

        if html and len(html.strip()) > 50:
            fpath.write_text(html, encoding="utf-8")
            results[sec_num] = {"path": fpath, "status": "fetched"}
            logger.info("  Section %d: fetched (%d bytes)", sec_num, len(html))
        else:
            results[sec_num] = {"path": None, "status": "empty"}
            logger.warning("  Section %d: empty or failed", sec_num)

    # Also fetch and save the full leaflet for reference
    full_path = pair_dir / f"{nregistro}_full.html"
    if not (skip_existing and full_path.exists() and full_path.stat().st_size > 0):
        time.sleep(delay)
        full_html = fetch_full_leaflet_html(nregistro, session=session)
        if full_html:
            full_path.write_text(full_html, encoding="utf-8")

    session.close()
    return results


def fetch_all_leaflets(
    drug_pairs: list[dict],
    leaflets_dir: Path = LEAFLETS_DIR,
    delay: float = CIMA_FETCH_DELAY,
    skip_existing: bool = True,
    progress_callback=None,
) -> dict:
    """Fetch leaflets for all drugs in all pairs.

    Args:
        drug_pairs: List of pair dicts, each with 'pair_id', 'branded', 'generic' keys.
            Each drug dict must have 'nregistro' and 'name'.
        leaflets_dir: Base directory for saving leaflets.
        delay: Delay between CIMA API requests.
        skip_existing: Skip already-downloaded sections.
        progress_callback: Optional callable(current, total).

    Returns:
        Dict mapping nregistro -> fetch results.
    """
    all_results = {}
    drugs = []

    for pair in drug_pairs:
        pair_id = pair["pair_id"]
        for role in ("branded", "generic"):
            drug = pair.get(role)
            if drug and drug.get("nregistro"):
                drugs.append((pair_id, drug))

    total = len(drugs)
    for i, (pair_id, drug) in enumerate(drugs):
        nregistro = drug["nregistro"]
        name = drug.get("name", nregistro)
        logger.info("Fetching leaflet: %s (nregistro=%s)", name[:40], nregistro)

        results = fetch_and_save_leaflet(
            nregistro=nregistro,
            drug_name=name,
            pair_id=pair_id,
            leaflets_dir=leaflets_dir,
            delay=delay,
            skip_existing=skip_existing,
        )
        all_results[nregistro] = results

        if progress_callback:
            progress_callback(i + 1, total)

    return all_results
