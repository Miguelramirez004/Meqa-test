"""PubMed / NCBI E-utilities client for collecting literature on drug pairs.

Queries PubMed for each drug pair's active ingredient to gather published
evidence about brand vs generic drug studies, bioequivalence, and
pharmacological profiles.

NCBI E-utilities docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

import json
import time
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree

import requests

from .config import PUBMED_BASE_URL, PUBMED_DELAY, PUBMED_MAX_RESULTS, PUBMED_DIR


class PubMedClient:
    """Client for NCBI E-utilities (PubMed search and fetch)."""

    def __init__(self, api_key: str = "", email: str = "",
                 delay: float = PUBMED_DELAY):
        self.base_url = PUBMED_BASE_URL
        self.delay = delay if not api_key else 0.11  # 10/s with key
        self.session = requests.Session()
        self.params = {}
        if api_key:
            self.params["api_key"] = api_key
        if email:
            self.params["email"] = email

    def _get(self, endpoint: str, params: dict) -> requests.Response | None:
        url = f"{self.base_url}/{endpoint}"
        merged = {**self.params, **params}
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, params=merged, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] {url}: {e}")
            return None

    # ── ESearch: find PMIDs ───────────────────────────────────────────────

    def search(self, query: str, max_results: int = PUBMED_MAX_RESULTS,
               sort: str = "relevance") -> list[str]:
        """Search PubMed and return list of PMIDs."""
        resp = self._get("esearch.fcgi", {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "retmode": "json",
        })
        if resp is None:
            return []
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    # ── EFetch: get article details ───────────────────────────────────────

    def fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        """Fetch article metadata and abstracts for given PMIDs."""
        if not pmids:
            return []
        resp = self._get("efetch.fcgi", {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        })
        if resp is None:
            return []
        return self._parse_articles_xml(resp.text)

    @staticmethod
    def _parse_articles_xml(xml_text: str) -> list[dict]:
        """Parse PubMed XML into list of article dicts."""
        articles = []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return []

        for article_el in root.findall(".//PubmedArticle"):
            medline = article_el.find("MedlineCitation")
            if medline is None:
                continue

            pmid_el = medline.find("PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            art = medline.find("Article")
            if art is None:
                continue

            # Title
            title_el = art.find("ArticleTitle")
            title = title_el.text or "" if title_el is not None else ""

            # Abstract
            abstract_parts = []
            abstract_el = art.find("Abstract")
            if abstract_el is not None:
                for abs_text in abstract_el.findall("AbstractText"):
                    label = abs_text.get("Label", "")
                    text = "".join(abs_text.itertext())
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            abstract = "\n".join(abstract_parts)

            # Journal
            journal_el = art.find("Journal/Title")
            journal = journal_el.text or "" if journal_el is not None else ""

            # Year
            year = ""
            pub_date = art.find("Journal/JournalIssue/PubDate")
            if pub_date is not None:
                year_el = pub_date.find("Year")
                if year_el is not None:
                    year = year_el.text or ""

            # Authors
            authors = []
            author_list = art.find("AuthorList")
            if author_list is not None:
                for author_el in author_list.findall("Author"):
                    last = author_el.find("LastName")
                    fore = author_el.find("ForeName")
                    if last is not None:
                        name = last.text or ""
                        if fore is not None and fore.text:
                            name += f" {fore.text}"
                        authors.append(name)

            # MeSH terms
            mesh_terms = []
            mesh_list = medline.find("MeshHeadingList")
            if mesh_list is not None:
                for heading in mesh_list.findall("MeshHeading"):
                    desc = heading.find("DescriptorName")
                    if desc is not None and desc.text:
                        mesh_terms.append(desc.text)

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": authors,
                "mesh_terms": mesh_terms,
            })

        return articles

    # ── High-level: search + fetch for a drug pair ────────────────────────

    def collect_for_pair(self, principio_activo: str, condition: str = "",
                         max_results: int = PUBMED_MAX_RESULTS) -> dict:
        """Search PubMed for a drug pair and return structured results.

        Builds multiple search queries to capture:
        1. General efficacy/safety literature
        2. Brand vs generic / bioequivalence studies
        3. Condition-specific treatment evidence
        """
        # Strip dosage for cleaner search (e.g., "Paracetamol 1g" -> "Paracetamol")
        active = principio_activo.split()[0] if principio_activo else ""
        if not active:
            return {"queries": [], "articles": [], "total_found": 0}

        queries_results = []

        # Query 1: General drug literature
        q1 = f"{active}[Title/Abstract]"
        pmids_1 = self.search(q1, max_results=max_results)

        # Query 2: Brand vs generic / bioequivalence
        q2 = f"({active}[Title/Abstract]) AND (generic OR bioequivalence OR brand)[Title/Abstract]"
        pmids_2 = self.search(q2, max_results=max_results)

        # Query 3: Condition-specific (if provided)
        pmids_3 = []
        q3 = ""
        if condition:
            q3 = f"({active}[Title/Abstract]) AND ({condition}[Title/Abstract])"
            pmids_3 = self.search(q3, max_results=max_results)

        # Deduplicate PMIDs preserving order
        seen = set()
        all_pmids = []
        for pmid in pmids_1 + pmids_2 + pmids_3:
            if pmid not in seen:
                seen.add(pmid)
                all_pmids.append(pmid)

        # Fetch abstracts for all unique PMIDs
        articles = self.fetch_abstracts(all_pmids)

        # Tag articles with which query found them
        pmid_set_1 = set(pmids_1)
        pmid_set_2 = set(pmids_2)
        pmid_set_3 = set(pmids_3)
        for art in articles:
            tags = []
            if art["pmid"] in pmid_set_1:
                tags.append("general")
            if art["pmid"] in pmid_set_2:
                tags.append("bioequivalence")
            if art["pmid"] in pmid_set_3:
                tags.append("condition_specific")
            art["search_tags"] = tags

        queries_used = [
            {"label": "general", "query": q1, "results": len(pmids_1)},
            {"label": "bioequivalence", "query": q2, "results": len(pmids_2)},
        ]
        if q3:
            queries_used.append(
                {"label": "condition_specific", "query": q3, "results": len(pmids_3)}
            )

        return {
            "queries": queries_used,
            "articles": articles,
            "total_found": len(articles),
        }


def collect_pubmed_for_all_pairs(
    drug_pairs: list[dict],
    output_dir: Path = PUBMED_DIR,
    api_key: str = "",
    email: str = "",
    max_results: int = PUBMED_MAX_RESULTS,
    skip_existing: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Collect PubMed literature for all drug pairs.

    Returns list of result dicts, one per pair.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = PubMedClient(api_key=api_key, email=email)

    results = []
    total = len(drug_pairs)

    for i, pair in enumerate(drug_pairs):
        pair_id = pair["pair_id"]
        out_file = output_dir / f"{pair_id}_pubmed.json"

        if skip_existing and out_file.exists():
            with open(out_file, encoding="utf-8") as f:
                results.append(json.load(f))
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        principio = pair.get("principio_activo", "")
        condition = pair.get("condition", "")

        pair_data = client.collect_for_pair(
            principio_activo=principio,
            condition=condition,
            max_results=max_results,
        )

        result = {
            "pair_id": pair_id,
            "principio_activo": principio,
            "condition": condition,
            **pair_data,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    return results
