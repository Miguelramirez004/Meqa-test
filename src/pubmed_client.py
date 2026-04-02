"""PubMed E-utilities client for collecting drug-pair literature data.

Uses NCBI's E-utilities (esearch + esummary) to query PubMed for each
drug pair (brand name vs INN/generic) and collect publication counts,
recent article metadata, and comparative metrics.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
Base URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
"""

import logging
import time
import json
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from .config import DATA_DIR, PUBMED_DIR, API_DELAY


log = logging.getLogger(__name__)

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_DELAY = 0.4  # NCBI allows ~3 req/s without API key; be conservative

# NCBI requires tool + email for identification (may block without them)
# See: https://www.ncbi.nlm.nih.gov/books/NBK25497/
NCBI_TOOL = "MeQA-Research"
NCBI_EMAIL = "meqa.research@example.com"


@dataclass
class PubMedSearchResult:
    """Result of a single PubMed search query."""
    query: str
    total_count: int = 0
    article_ids: list = field(default_factory=list)
    articles: list = field(default_factory=list)  # summaries for top N
    error: str = ""


@dataclass
class DrugPairPubMedData:
    """PubMed data collected for one drug pair."""
    pair_id: str
    principio_activo: str
    brand_name: str
    inn_search: Optional[PubMedSearchResult] = None
    generic_search: Optional[PubMedSearchResult] = None
    brand_search: Optional[PubMedSearchResult] = None
    bioequivalence_search: Optional[PubMedSearchResult] = None


class PubMedClient:
    """Client for NCBI PubMed E-utilities API.

    Provides methods to search PubMed and retrieve article summaries.
    Rate-limited to respect NCBI's usage policies.
    """

    def __init__(self, api_key: str = None, email: str = NCBI_EMAIL,
                 tool: str = NCBI_TOOL, delay: float = PUBMED_DELAY,
                 timeout: int = 30, retries: int = 2):
        self.base_url = PUBMED_BASE_URL
        self.api_key = api_key.strip() if api_key else None
        self.email = email
        self.tool = tool
        self.delay = delay if not api_key else 0.12  # 10/s with key
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"{tool}/1.0 ({email})",
        })

    def _base_params(self) -> dict:
        """Common params for all E-utility requests (NCBI-required)."""
        params = {
            "retmode": "json",
            "tool": self.tool,
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get_with_retry(self, url: str, params: dict) -> requests.Response:
        """GET with retry and rate-limit delay."""
        last_exc = None
        for attempt in range(self.retries + 1):
            time.sleep(self.delay)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                last_exc = e
                log.warning("Attempt %d/%d failed for %s: %s",
                            attempt + 1, self.retries + 1, url, e)
                if attempt < self.retries:
                    time.sleep(2 ** attempt)  # exponential backoff
        raise last_exc

    def esearch(self, query: str, db: str = "pubmed",
                retmax: int = 20) -> dict | None:
        """Search PubMed and return matching article IDs."""
        url = f"{self.base_url}/esearch.fcgi"
        params = {**self._base_params(), "db": db, "term": query,
                  "retmax": retmax}
        try:
            resp = self._get_with_retry(url, params)
            data = resp.json()
            # Check for NCBI-level errors
            if "esearchresult" in data:
                err = data["esearchresult"].get("ERROR")
                if err:
                    log.error("NCBI esearch error for '%s': %s", query, err)
                    return None
            return data
        except requests.exceptions.RequestException as e:
            log.error("esearch failed for '%s': %s", query, e)
            return None
        except ValueError as e:
            log.error("esearch JSON parse error for '%s': %s", query, e)
            return None

    def esummary(self, ids: list[str], db: str = "pubmed") -> dict | None:
        """Fetch article summaries for a list of PubMed IDs."""
        if not ids:
            return None
        url = f"{self.base_url}/esummary.fcgi"
        params = {**self._base_params(), "db": db,
                  "id": ",".join(str(i) for i in ids),
                  "version": "2.0"}
        try:
            resp = self._get_with_retry(url, params)
            return resp.json()
        except requests.exceptions.RequestException as e:
            log.error("esummary failed: %s", e)
            return None
        except ValueError as e:
            log.error("esummary JSON parse error: %s", e)
            return None

    def search_and_summarise(self, query: str,
                             top_n: int = 5) -> PubMedSearchResult:
        """Search PubMed and return count + top N article summaries."""
        result = PubMedSearchResult(query=query)

        # Step 1: search
        search_data = self.esearch(query, retmax=top_n)
        if not search_data or "esearchresult" not in search_data:
            result.error = "esearch returned no data"
            return result

        esearch = search_data["esearchresult"]
        result.total_count = int(esearch.get("count", 0))
        result.article_ids = esearch.get("idlist", [])

        if not result.article_ids:
            return result

        # Step 2: fetch summaries for top N
        summary_data = self.esummary(result.article_ids[:top_n])
        if not summary_data or "result" not in summary_data:
            result.error = "esummary returned no data"
            return result

        for aid in result.article_ids[:top_n]:
            article = summary_data["result"].get(str(aid))
            if not article or "error" in article:
                continue
            result.articles.append({
                "pmid": aid,
                "title": article.get("title", ""),
                "pubdate": article.get("pubdate",
                           article.get("sortpubdate", "")),
                "source": article.get("source",
                          article.get("fulljournalname", "")),
                "authors": [
                    a.get("name", "")
                    for a in article.get("authors", [])[:5]
                ],
            })

        return result

    # ── Drug-pair queries ──────────────────────────────────────────────

    # Spanish INN → English INN for PubMed (English-language database).
    # Only includes names that differ between languages.
    _INN_ES_TO_EN: dict[str, str] = {
        "ibuprofeno": "ibuprofen",
        "amoxicilina": "amoxicillin",
        "omeprazol": "omeprazole",
        "atorvastatina": "atorvastatin",
        "metformina": "metformin",
        "sertralina": "sertraline",
        "amlodipino": "amlodipine",
        "azitromicina": "azithromycin",
        "simvastatina": "simvastatin",
        "fluoxetina": "fluoxetine",
        "diclofenaco": "diclofenac",
        "pantoprazol": "pantoprazole",
        "levotiroxina": "levothyroxine",
        "ciprofloxacino": "ciprofloxacin",
    }

    @staticmethod
    def _extract_inn(principio_activo: str) -> str:
        """Extract bare INN from principio_activo string, in English.

        'Paracetamol 1g' -> 'paracetamol'
        'Omeprazol 20mg' -> 'omeprazole'
        """
        parts = principio_activo.strip().split()
        inn_es = parts[0].lower() if parts else principio_activo.lower()
        return PubMedClient._INN_ES_TO_EN.get(inn_es, inn_es)

    def collect_for_pair(self, pair: dict,
                         top_n: int = 5) -> DrugPairPubMedData:
        """Collect PubMed data for a single drug pair.

        Runs four searches per pair:
        1. Active principle (INN) — no dosage (e.g. "omeprazol")
        2. Generic lab-specific — INN + lab names (e.g. "omeprazol AND (Cinfa OR Normon ...)")
        3. Branded drug — brand name (e.g. "Losec")
        4. Bioequivalence & comparison — INN + bioequivalence/comparison terms
        """
        inn = self._extract_inn(pair["principio_activo"])
        brand = pair["brand"]
        pair_id = pair["pair_id"]

        data = DrugPairPubMedData(
            pair_id=pair_id,
            principio_activo=pair["principio_activo"],
            brand_name=brand,
        )

        log.info("[%s] Searching PubMed for '%s' / '%s'...", pair_id, brand, inn)

        # 1. Active principle (INN) — plain search, no dosage
        data.inn_search = self.search_and_summarise(
            f'{inn}[Title/Abstract]',
            top_n=top_n,
        )
        log.info("  INN '%s': %d results", inn, data.inn_search.total_count)

        # 2. Generic lab-specific — search INN + laboratory names from generics list
        generics = pair.get("generics", [])
        if generics:
            # Extract lab names (second word onwards, e.g. "Cinfa" from "Paracetamol Cinfa")
            lab_names = []
            for g in generics:
                parts = g.strip().split()
                if len(parts) >= 2:
                    lab_names.append(" ".join(parts[1:]))
            if lab_names:
                labs_or = " OR ".join(f'"{lab}"' for lab in lab_names)
                data.generic_search = self.search_and_summarise(
                    f'{inn}[Title/Abstract] AND ({labs_or})',
                    top_n=top_n,
                )
            else:
                data.generic_search = self.search_and_summarise(
                    f'{inn}[Title/Abstract] AND generic',
                    top_n=top_n,
                )
        else:
            data.generic_search = self.search_and_summarise(
                f'{inn}[Title/Abstract] AND generic',
                top_n=top_n,
            )
        log.info("  Generic lab-specific: %d results",
                 data.generic_search.total_count)

        # 3. Branded drug — search all known brand names for this INN
        all_brands = pair.get("brands", [brand])
        if not all_brands:
            all_brands = [brand]
        brands_or = " OR ".join(f'"{b}"' for b in all_brands)
        data.brand_search = self.search_and_summarise(
            f'({brands_or})[Title/Abstract] AND drug',
            top_n=top_n,
        )
        log.info("  Brands %s: %d results", all_brands, data.brand_search.total_count)

        # 4. Bioequivalence & comparison articles
        data.bioequivalence_search = self.search_and_summarise(
            f'{inn}[Title/Abstract] AND (bioequivalence OR bioequivalent '
            f'OR "therapeutic equivalence" OR "generic comparison" '
            f'OR "brand vs generic" OR "branded vs generic")',
            top_n=top_n,
        )
        log.info("  Bioequivalence & comparison: %d results",
                 data.bioequivalence_search.total_count)

        return data


def collect_pubmed_data(pairs: list[dict], api_key: str = None,
                        top_n: int = 5,
                        output_dir: Path = None) -> list[dict]:
    """Collect PubMed data for all drug pairs and save to JSON."""
    if output_dir is None:
        output_dir = DATA_DIR / "pubmed"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = PubMedClient(api_key=api_key)
    results = []

    log.info("Collecting PubMed data for %d drug pairs...", len(pairs))
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

    log.info("Done. Saved %d pair results to %s/", len(results), output_dir)
    return results


def collect_pubmed_for_all_pairs(
    drug_pairs: list[dict],
    output_dir: Path = PUBMED_DIR,
    api_key: str = "",
    email: str = NCBI_EMAIL,
    max_results: int = 5,
    skip_existing: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Collect PubMed literature for all drug pairs (Streamlit-compatible wrapper).

    Wraps PubMedClient with skip_existing, progress_callback, and a return
    format compatible with the Streamlit UI (keys: total_found, queries, articles).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = PubMedClient(api_key=api_key.strip() or None, email=email)

    # Logging visible in Streamlit logs
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    results = []
    total = len(drug_pairs)
    errors = []

    for i, pair in enumerate(drug_pairs):
        pair_id = pair["pair_id"]
        out_file = output_dir / f"{pair_id}_pubmed.json"

        # Skip if already collected
        if skip_existing and out_file.exists():
            with open(out_file, encoding="utf-8") as f:
                results.append(json.load(f))
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        try:
            # Collect via the existing client method
            pair_data = client.collect_for_pair(pair, top_n=max_results)
            serialised = asdict(pair_data)

            # Build the UI-friendly dict with total_found / queries / articles
            all_articles = []
            queries_used = []
            for search_key, label in [
                ("inn_search", "inn"),
                ("generic_search", "generic_lab"),
                ("brand_search", "brand"),
                ("bioequivalence_search", "bioequivalence"),
            ]:
                search = serialised.get(search_key)
                if search:
                    queries_used.append({
                        "label": label,
                        "query": search.get("query", ""),
                        "results": search.get("total_count", 0),
                        "error": search.get("error", ""),
                    })
                    for art in search.get("articles", []):
                        art.setdefault("search_tags", []).append(label)
                        if not any(a["pmid"] == art["pmid"] for a in all_articles):
                            all_articles.append(art)

            result = {
                **serialised,
                "condition": pair.get("condition", ""),
                "queries": queries_used,
                "articles": all_articles,
                "total_found": len(all_articles),
            }

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            results.append(result)

        except Exception as e:
            log.error("Failed to collect for %s: %s", pair_id, e)
            errors.append(f"{pair_id}: {e}")
            results.append({
                "pair_id": pair_id,
                "principio_activo": pair.get("principio_activo", ""),
                "condition": pair.get("condition", ""),
                "queries": [],
                "articles": [],
                "total_found": 0,
                "error": str(e),
            })

        if progress_callback:
            progress_callback(i + 1, total)

    if errors:
        log.warning("Errors during collection: %s", "; ".join(errors))

    return results
