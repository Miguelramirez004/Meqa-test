"""Configuration: paths, API URLs, and constants."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAIRS_DIR = DATA_DIR / "drug_pairs"
PROSPECTOS_DIR = DATA_DIR / "prospectos"
QUERIES_DIR = DATA_DIR / "queries"
RESPONSES_DIR = DATA_DIR / "responses"
ANALYSIS_DIR = DATA_DIR / "analysis"
PUBMED_DIR = DATA_DIR / "pubmed"

# ── CIMA API ─────────────────────────────────────────────────────────────────
CIMA_BASE_URL = "https://cima.aemps.es/cima/rest"
CIMA_DOC_BASE = "https://cima.aemps.es/cima/dochtml"
API_DELAY = 1.0  # seconds between API requests (be respectful)
API_TIMEOUT = 30  # seconds

# ── MeQA ─────────────────────────────────────────────────────────────────────
MEQA_URL = "https://cima.aemps.es/cima/publico/meqa.html"
MEQA_QUERY_DELAY = 5  # seconds between MeQA queries

# ── Experiment ───────────────────────────────────────────────────────────────
MAX_GENERICS_PER_PAIR = 3
ALPHA = 0.05  # significance level for statistical tests
