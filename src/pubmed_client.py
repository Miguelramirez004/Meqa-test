"""PubMed E-utilities client for collecting drug-pair literature data.

Uses NCBI's E-utilities (esearch + esummary) to query PubMed for each
drug pair (brand name vs INN/generic) and collect publication counts,
recent article metadata, and comparative metrics.

Searches ALL known brand names and generic laboratory products for each
pair, not just the primary brand.

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


# ── Extended brand & generic knowledge base ──────────────────────────────────
# International brand names per INN (superset of drug_pairs.py brands +
# response_analyzer.py BRAND_NAMES_BY_INN, plus additional worldwide names).

BRAND_NAMES_BY_INN = {
    "paracetamol": [
        "gelocatil", "termalgin", "efferalgan", "apiretal", "dolocatil",
        "panadol", "tylenol", "calpol", "ben-u-ron", "doliprane",
        "dafalgan", "perfalgan", "panodil", "biogesic", "tempra",
        "acamol", "febrectal", "lixidol",
    ],
    "ibuprofeno": [
        "nurofen", "espidifen", "dalsy", "dobupal", "bexistar",
        "neobrufen", "advil", "motrin", "ibufen", "algidol",
        "brufen", "genpril", "midol", "caldolor", "ibumetin",
        "spidifen", "junifen", "calprofen",
    ],
    "amoxicilina": [
        "clamoxyl", "amoxicilina beecham", "amoxil", "augmentine",
        "trimox", "wymox", "moxatag", "larotid", "polymox",
        "biomox", "dispermox",
    ],
    "omeprazol": [
        "losec", "belfa", "mopral", "mepral", "prilosec",
        "zegerid", "omepral", "antra", "logastric", "lomac",
        "gastrimut", "pepticum", "ulceral",
    ],
    "atorvastatina": [
        "lipitor", "cardyl", "zarator", "sortis", "torvast", "totalip",
        "atoris", "caduet", "tulip", "torvacard", "liprimar",
    ],
    "enalapril": [
        "renitec", "cardiovasil", "dilvas", "acetensil", "naprilene",
        "vasotec", "enaladex", "envas", "enapren", "baripril",
    ],
    "metformina": [
        "dianben", "glucophage", "fortamet", "glumetza", "riomet",
        "glycon", "diabex", "glucomin", "metfogamma",
    ],
    "lorazepam": [
        "orfidal", "idalprem", "placinoral",
        "ativan", "temesta", "tavor", "lorabenz", "anxira",
    ],
    "sertralina": [
        "besitran", "aremis", "zoloft",
        "lustral", "gladem", "sealdin", "altruline",
    ],
    "salbutamol": [
        "ventolin", "buto asma", "salbulair", "ventolin accuhaler",
        "proventil", "proair", "airomir", "salamol", "asthalin",
        "combivent",
    ],
    "amlodipino": [
        "norvasc", "astudal", "neotensin", "istin",
        "amlopin", "amlor", "amlong", "stamlo",
    ],
    "azitromicina": [
        "zithromax", "vinzam", "toraseptol", "sumamed",
        "azithral", "zmax", "zitromax", "azibiot",
    ],
    "simvastatina": [
        "zocor", "pantok", "algoren",
        "simvacor", "simlup", "simvador", "simvor",
    ],
    "fluoxetina": [
        "prozac", "adofen", "reneuron",
        "sarafem", "rapiflux", "selfemra", "fluoxac",
    ],
    "ramipril": [
        "altace", "acovil", "ramicor", "triatec", "tritace",
        "ramipro", "cardace", "delix", "pramace", "unipril",
    ],
    "diclofenaco": [
        "voltaren", "artrotec",
        "cataflam", "zipsor", "zorvolex", "dicloflex", "diclon",
        "dyloject", "solaraze", "voltarol",
    ],
    "pantoprazol": [
        "pantoc", "anagastra", "pantoloc", "pantecta", "zurcal",
        "protonix", "controloc", "somac", "pantopan",
    ],
    "levotiroxina": [
        "eutirox", "levothroid", "synthroid", "levothyrox",
        "eltroxin", "euthyrox", "tirosint", "levoxyl", "unithroid",
        "oroxine",
    ],
    "alprazolam": [
        "trankimazin", "tranquimazin retard", "xanax",
        "niravam", "kalma", "alprax", "restyl", "tafil",
    ],
    "ciprofloxacino": [
        "baycip", "rigoran", "ciflosin", "ciproxin", "cetraxal",
        "cipro", "ciloxan", "ciprobay", "ciplox", "ciproxina",
    ],
}

# Extended list of generic pharmaceutical laboratories worldwide
GENERIC_LABS = [
    # Major Spanish labs
    "cinfa", "normon", "kern pharma", "teva", "stada", "mylan",
    "sandoz", "ratiopharm", "pensa", "alter", "vir", "aurovitas",
    "zentiva", "aristo", "bluefish", "accord", "sanofi", "ranbaxy",
    "actavis", "apotex", "biogaran", "davur", "edigen", "farmalider",
    "gedeon richter", "korhispana", "ratio", "sun pharma", "tecnigen",
    "winthrop",
    # Major international labs
    "dr. reddy's", "dr reddy", "cipla", "lupin", "aurobindo",
    "torrent", "zydus", "hetero", "glenmark", "macleods",
    "alkem", "ipca", "biocon", "jubilant", "laurus",
    "hikma", "endo", "par pharmaceutical", "amneal", "impax",
    "watson", "barr", "perrigo", "lannett", "cambrex",
    "fresenius kabi", "hospira", "baxter", "b. braun",
    "hexal", "1a pharma", "al inde", "basics", "dura",
    "mepha", "helvepharm", "spirig", "labatec",
    "eg", "biogaran", "arrow", "cristers", "evolupharm",
    "rosemont", "tillomed", "milpharm", "kent pharma",
]


@dataclass
class PubMedSearchResult:
    """Result of a single PubMed search query."""
    query: str
    total_count: int = 0
    article_ids: list = field(default_factory=list)
    articles: list = field(default_factory=list)  # summaries for top N


@dataclass
class BrandSearchResult:
    """PubMed search results for a single brand name."""
    brand_name: str
    search: Optional[PubMedSearchResult] = None


@dataclass
class GenericLabSearchResult:
    """PubMed search results for an INN + laboratory combination."""
    lab_name: str
    generic_product: str  # e.g. "Omeprazol Cinfa"
    search: Optional[PubMedSearchResult] = None


@dataclass
class DrugPairPubMedData:
    """PubMed data collected for one drug pair."""
    pair_id: str
    principio_activo: str
    brand_name: str
    # Per-brand searches (all known brands for this INN)
    all_brands: list = field(default_factory=list)
    brand_searches: list = field(default_factory=list)  # list of BrandSearchResult
    total_brand_pubs: int = 0
    # INN search (active ingredient)
    inn_search: Optional[PubMedSearchResult] = None
    # Per-generic-lab searches
    generic_labs_searched: list = field(default_factory=list)
    generic_lab_searches: list = field(default_factory=list)  # list of GenericLabSearchResult
    total_generic_pubs: int = 0
    # Comparison & bioequivalence
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

    @staticmethod
    def _get_all_brands(pair: dict) -> list[str]:
        """Get all known brand names for a pair.

        Merges pair['brands'] with the extended BRAND_NAMES_BY_INN
        knowledge base, deduplicating by lowercase.
        """
        inn = PubMedClient._extract_inn(pair["principio_activo"])
        seen = set()
        brands = []

        # Start with pair-defined brands
        for b in pair.get("brands", [pair["brand"]]):
            key = b.lower().strip()
            if key and key not in seen:
                seen.add(key)
                brands.append(b)

        # Add from extended knowledge base
        for b in BRAND_NAMES_BY_INN.get(inn, []):
            key = b.lower().strip()
            if key and key not in seen:
                seen.add(key)
                brands.append(b.title())

        return brands

    @staticmethod
    def _get_generic_labs(pair: dict) -> list[tuple[str, str]]:
        """Get generic lab names and product names for a pair.

        Returns list of (lab_name, product_name) tuples.
        Merges pair['generics'] with the extended GENERIC_LABS list.
        """
        inn = PubMedClient._extract_inn(pair["principio_activo"])
        inn_title = inn.title()
        seen_labs = set()
        results = []

        # Start with pair-defined generics (extract lab name from product)
        for g in pair.get("generics", []):
            # "Paracetamol Cinfa" -> lab = "cinfa"
            parts = g.split()
            if len(parts) >= 2:
                lab = " ".join(parts[1:])
            else:
                lab = g
            key = lab.lower().strip()
            if key and key not in seen_labs:
                seen_labs.add(key)
                results.append((lab, g))

        # Add major labs not already covered
        for lab in GENERIC_LABS:
            key = lab.lower().strip()
            if key not in seen_labs:
                seen_labs.add(key)
                product = f"{inn_title} {lab.title()}"
                results.append((lab.title(), product))

        return results

    def collect_for_pair(self, pair: dict,
                         top_n: int = 5) -> DrugPairPubMedData:
        """Collect PubMed data for a single drug pair.

        Runs searches for:
        1. Each known brand name (all brands, not just primary)
        2. INN (active ingredient)
        3. Each generic laboratory product (INN + lab name)
        4. Brand vs generic comparison studies
        5. Bioequivalence studies

        Args:
            pair: Drug pair dict from OFFLINE_PAIRS.
            top_n: Number of article summaries per search.

        Returns:
            DrugPairPubMedData with all search results.
        """
        inn = self._extract_inn(pair["principio_activo"])
        pair_id = pair["pair_id"]

        all_brands = self._get_all_brands(pair)
        generic_labs = self._get_generic_labs(pair)

        data = DrugPairPubMedData(
            pair_id=pair_id,
            principio_activo=pair["principio_activo"],
            brand_name=pair["brand"],
            all_brands=[b for b in all_brands],
            generic_labs_searched=[lab for lab, _ in generic_labs],
        )

        print(f"  [{pair_id}] Searching PubMed for '{inn}'...")
        print(f"    Brands to search: {len(all_brands)} — "
              f"Generic labs: {len(generic_labs)}")

        # 1. Per-brand searches
        total_brand = 0
        for brand in all_brands:
            result = self.search_and_summarise(
                f'"{brand}"[Title/Abstract]',
                top_n=top_n,
            )
            data.brand_searches.append(asdict(
                BrandSearchResult(brand_name=brand, search=result)
            ))
            total_brand += result.total_count
            print(f"    Brand '{brand}': {result.total_count} results")
        data.total_brand_pubs = total_brand

        # 2. INN (active ingredient)
        data.inn_search = self.search_and_summarise(
            f'"{inn}"[Title/Abstract]',
            top_n=top_n,
        )
        print(f"    INN '{inn}': {data.inn_search.total_count} results")

        # 3. Per-generic-lab searches
        total_generic = 0
        for lab, product in generic_labs:
            result = self.search_and_summarise(
                f'"{inn}"[Title/Abstract] AND "{lab}"[Title/Abstract]',
                top_n=top_n,
            )
            data.generic_lab_searches.append(asdict(
                GenericLabSearchResult(
                    lab_name=lab, generic_product=product, search=result
                )
            ))
            total_generic += result.total_count
            print(f"    Generic '{product}': {result.total_count} results")
        data.total_generic_pubs = total_generic

        # 4. Brand vs generic
        data.brand_vs_generic_search = self.search_and_summarise(
            f'"{inn}"[Title/Abstract] AND (generic OR brand OR branded) '
            f'AND (comparison OR equivalence)',
            top_n=top_n,
        )
        print(f"    Brand-vs-generic: "
              f"{data.brand_vs_generic_search.total_count} results")

        # 5. Bioequivalence
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
