"""PubMed E-utilities client for collecting drug-pair literature data.

Uses NCBI's E-utilities (esearch + esummary) to query PubMed for each
drug pair (brand name vs INN/generic) and collect publication counts,
recent article metadata, and comparative metrics.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
Base URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
"""

import time
import json
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from .config import DATA_DIR, API_DELAY


PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_DELAY = 0.4  # NCBI allows ~3 req/s without API key; be conservative


@dataclass
class PubMedSearchResult:
    """Result of a single PubMed search query."""
    query: str
    total_count: int = 0
    article_ids: list = field(default_factory=list)
    articles: list = field(default_factory=list)  # summaries for top N


@dataclass
class DrugPairPubMedData:
    """PubMed data collected for one drug pair."""
    pair_id: str
    principio_activo: str
    brand_name: str
    brand_search: Optional[PubMedSearchResult] = None
    inn_search: Optional[PubMedSearchResult] = None
    brand_vs_generic_search: Optional[PubMedSearchResult] = None
    bioequivalence_search: Optional[PubMedSearchResult] = None


class PubMedClient:
    """Client for NCBI PubMed E-utilities API.

    Provides methods to search PubMed and retrieve article summaries.
    Rate-limited to respect NCBI's usage policies.
    """

    def __init__(self, api_key: str = None, delay: float = PUBMED_DELAY,
                 timeout: int = 30):
        """Initialise the client.

        Args:
            api_key: Optional NCBI API key (raises rate limit to 10 req/s).
            delay: Seconds between requests.
            timeout: HTTP timeout in seconds.
        """
        self.base_url = PUBMED_BASE_URL
        self.api_key = api_key
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MeQA-Research/1.0 (Academic Thesis)",
        })

    def _base_params(self) -> dict:
        """Common params for all E-utility requests."""
        params = {"retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def esearch(self, query: str, db: str = "pubmed",
                retmax: int = 20) -> dict | None:
        """Search PubMed and return matching article IDs.

        Args:
            query: PubMed search query string.
            db: NCBI database (default: pubmed).
            retmax: Maximum number of IDs to return.

        Returns:
            Parsed JSON response or None on error.
        """
        url = f"{self.base_url}/esearch.fcgi"
        params = {**self._base_params(), "db": db, "term": query,
                  "retmax": retmax, "usehistory": "y"}
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] esearch '{query}': {e}")
            return None

    def esummary(self, ids: list[str], db: str = "pubmed") -> dict | None:
        """Fetch article summaries for a list of PubMed IDs.

        Args:
            ids: List of PubMed article IDs.
            db: NCBI database.

        Returns:
            Parsed JSON response or None on error.
        """
        if not ids:
            return None
        url = f"{self.base_url}/esummary.fcgi"
        params = {**self._base_params(), "db": db,
                  "id": ",".join(str(i) for i in ids)}
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] esummary: {e}")
            return None

    def search_and_summarise(self, query: str,
                             top_n: int = 5) -> PubMedSearchResult:
        """Search PubMed and return count + top N article summaries.

        Args:
            query: PubMed search string.
            top_n: How many article summaries to fetch.

        Returns:
            PubMedSearchResult with counts and article metadata.
        """
        result = PubMedSearchResult(query=query)

        # Step 1: search
        search_data = self.esearch(query, retmax=top_n)
        if not search_data or "esearchresult" not in search_data:
            return result

        esearch = search_data["esearchresult"]
        result.total_count = int(esearch.get("count", 0))
        result.article_ids = esearch.get("idlist", [])

        # Step 2: fetch summaries for top N
        if result.article_ids:
            summary_data = self.esummary(result.article_ids[:top_n])
            if summary_data and "result" in summary_data:
                for aid in result.article_ids[:top_n]:
                    article = summary_data["result"].get(str(aid))
                    if article:
                        result.articles.append({
                            "pmid": aid,
                            "title": article.get("title", ""),
                            "pubdate": article.get("pubdate", ""),
                            "source": article.get("source", ""),
                            "authors": [
                                a.get("name", "")
                                for a in article.get("authors", [])[:3]
                            ],
                        })

        return result

    # ── Drug-pair queries ──────────────────────────────────────────────

    @staticmethod
    def _extract_inn(principio_activo: str) -> str:
        """Extract bare INN from principio_activo string.

        'Paracetamol 1g' -> 'paracetamol'
        'Omeprazol 20mg' -> 'omeprazol'
        """
        parts = principio_activo.strip().split()
        return parts[0].lower() if parts else principio_activo.lower()

    def collect_for_pair(self, pair: dict,
                         top_n: int = 5) -> DrugPairPubMedData:
        """Collect PubMed data for a single drug pair.

        Runs four searches per pair:
        1. Brand name search (e.g. "Losec")
        2. INN search (e.g. "omeprazol")
        3. Brand vs generic comparison search
        4. Bioequivalence search

        Args:
            pair: Drug pair dict from OFFLINE_PAIRS.
            top_n: Number of article summaries per search.

        Returns:
            DrugPairPubMedData with all search results.
        """
        inn = self._extract_inn(pair["principio_activo"])
        brand = pair["brand"]
        pair_id = pair["pair_id"]

        data = DrugPairPubMedData(
            pair_id=pair_id,
            principio_activo=pair["principio_activo"],
            brand_name=brand,
        )

        print(f"  [{pair_id}] Searching PubMed for '{brand}' / '{inn}'...")

        # 1. Brand name
        data.brand_search = self.search_and_summarise(
            f'"{brand}"[Title/Abstract] AND drug',
            top_n=top_n,
        )
        print(f"    Brand '{brand}': {data.brand_search.total_count} results")

        # 2. INN (active ingredient)
        data.inn_search = self.search_and_summarise(
            f'"{inn}"[Title/Abstract] AND drug',
            top_n=top_n,
        )
        print(f"    INN '{inn}': {data.inn_search.total_count} results")

        # 3. Brand vs generic
        data.brand_vs_generic_search = self.search_and_summarise(
            f'"{inn}"[Title/Abstract] AND (generic OR brand OR branded) '
            f'AND (comparison OR equivalence)',
            top_n=top_n,
        )
        print(f"    Brand-vs-generic: "
              f"{data.brand_vs_generic_search.total_count} results")

        # 4. Bioequivalence
        data.bioequivalence_search = self.search_and_summarise(
            f'"{inn}"[Title/Abstract] AND bioequivalence',
            top_n=top_n,
        )
        print(f"    Bioequivalence: "
              f"{data.bioequivalence_search.total_count} results")

        return data


def collect_pubmed_data(pairs: list[dict], api_key: str = None,
                        top_n: int = 5,
                        output_dir: Path = None) -> list[dict]:
    """Collect PubMed data for all drug pairs and save to JSON.

    Args:
        pairs: List of drug pair dicts (from OFFLINE_PAIRS).
        api_key: Optional NCBI API key.
        top_n: Article summaries per search.
        output_dir: Where to save output (default: data/pubmed/).

    Returns:
        List of serialised DrugPairPubMedData dicts.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "pubmed"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = PubMedClient(api_key=api_key)
    results = []

    print(f"Collecting PubMed data for {len(pairs)} drug pairs...")
    for pair in pairs:
        pair_data = client.collect_for_pair(pair, top_n=top_n)
        serialised = asdict(pair_data)
        results.append(serialised)

        # Save individual pair file
        pair_file = output_dir / f"{pair['pair_id']}_pubmed.json"
        with open(pair_file, "w", encoding="utf-8") as f:
            json.dump(serialised, f, ensure_ascii=False, indent=2)

    # Save combined file
    combined_file = output_dir / "pubmed_all_pairs.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved {len(results)} pair results to {output_dir}/")
    return results
