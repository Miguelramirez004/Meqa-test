# MeQA Asymmetry Test

**Testing for systematic information asymmetries between brand-name and generic medications in commercial AI models — with a RAG replication of Spain's MeQA system as baseline.**

> Part of doctoral research on **Generative Engine Optimization (GEO) opportunity gap in the pharmaceutical industry**.

## Background

[MeQA](https://cima.aemps.es/cima/publico/meqa.html) (Medicines Questions & Answers) is a RAG-based AI tool launched by Spain's drug regulator ([AEMPS](https://www.aemps.gob.es/)) in May 2025. It answers natural language questions about medications using official prospectus data from the [CIMA database](https://cima.aemps.es/).

**Research question:** Do commercial AI models (ChatGPT, Gemini, Perplexity) exhibit systematic brand/generic mention asymmetries when answering drug-related questions — and does the same asymmetry appear in a RAG system grounded in official regulatory data?

The experiment has two parallel tracks:

1. **Layer 1 — Commercial model audit:** Send fully open queries (no drug names in the prompt) to commercial models and measure which drug names they mention unprompted, using only their parametric knowledge.
2. **Layer 2 — MeQA RAG replication:** Run the same queries through a faithful reproduction of MeQA's pipeline, feeding official CIMA prospectos as context, to test whether asymmetries persist even in grounded retrieval.

## Drug Pairs

The sample covers **20 high-volume drug pairs** across 6 therapeutic groups, each pairing one brand-name drug against its bioequivalent Spanish generics (EFG). After deduplication of shared clinical conditions, the 20 pairs yield **15 unique conditions** that drive query generation — because several pairs share the same therapeutic condition or pharmacological class.

Five pharmacological classes are represented by two pairs each:

| Shared Class | Pairs |
|---|---|
| NSAIDs (AINEs) | Ibuprofeno 600mg · Diclofenaco 50mg |
| Proton-pump inhibitors | Omeprazol 20mg · Pantoprazol 20mg |
| Statins | Atorvastatina 20mg · Simvastatina 20mg |
| ACE inhibitors (IECAs) | Enalapril 20mg · Ramipril 5mg |
| Benzodiazepines | Lorazepam 1mg · Alprazolam 0.5mg |
| SSRIs | Sertralina 50mg · Fluoxetina 20mg |

Full sample:

| # | Active Ingredient | Brand | Therapeutic Group |
|---|---|---|---|
| P01 | Paracetamol 1g | Gelocatil | Analgesic |
| P02 | Ibuprofeno 600mg | Nurofen | Anti-inflammatory |
| P03 | Amoxicilina 500mg | Clamoxyl | Anti-infective |
| P04 | Omeprazol 20mg | Losec | Gastrointestinal |
| P05 | Atorvastatina 20mg | Lipitor | Cardiovascular |
| P06 | Enalapril 20mg | Renitec | Cardiovascular |
| P07 | Metformina 850mg | Dianben | Antidiabetic |
| P08 | Lorazepam 1mg | Orfidal | CNS — Anxiolytic |
| P09 | Sertralina 50mg | Besitran | CNS — Antidepressant |
| P10 | Salbutamol 100mcg | Ventolin | Respiratory |
| P11 | Amlodipino 5mg | Norvasc | Cardiovascular |
| P12 | Azitromicina 500mg | Zithromax | Anti-infective |
| P13 | Simvastatina 20mg | Zocor | Cardiovascular |
| P14 | Fluoxetina 20mg | Prozac | CNS — Antidepressant |
| P15 | Ramipril 5mg | Acovil | Cardiovascular |
| P16 | Diclofenaco 50mg | Voltaren | Anti-inflammatory |
| P17 | Pantoprazol 20mg | Pantoc | Gastrointestinal |
| P18 | Levotiroxina 100mcg | Eutirox | Endocrinology |
| P19 | Alprazolam 0.5mg | Trankimazin | CNS — Anxiolytic |
| P20 | Ciprofloxacino 500mg | Baycip | Anti-infective |

## Query Battery

### Design rationale

The core methodological constraint is that **no drug name appears in any query**. Every prompt is fully open — the model (or RAG system) must choose which names to use from its own knowledge or from retrieved context. This isolates the signal: any brand/generic mention asymmetry reflects the model's internal bias or retrieval behavior, not a prompt artifact.

Queries simulate six realistic user personas covering how real users actually interact with AI for health information:

| Persona | Example template | Dedup key |
|---|---|---|
| `symptom_casual` — everyday self-medication | "Tengo {symptom}, ¿qué me puedo tomar?" | symptom × category |
| `pharmacy_advice` — asking at the counter | "¿Qué me puede dar el farmacéutico para {symptom}?" | symptom × category |
| `worried_patient` — safety concerns | "¿Es seguro tomar {drug_class_short} durante mucho tiempo?" | drug class × category |
| `search_engine` — short keyword-style | "mejor medicamento {symptom}" | symptom × category |
| `condition` — clinical treatment | "¿Qué medicamentos se usan para tratar {condition}?" | condition × category |
| `drug_class` — pharmacological group | "¿Cuáles son los principales {drug_class} disponibles en España?" | drug class × category |

### Deduplication: 20 pairs → 15 unique conditions

Queries are deduplicated by `(key_field, template_category)` so each question is asked only once, even when multiple pairs share the same symptom, condition, or pharmacological class. Starting from 20 pairs:

- Condition templates deduplicate on the condition string → **15 unique conditions**
- Symptom-based templates deduplicate on the symptom string → **13 unique symptoms**
- Drug-class templates deduplicate on the class string → **14 unique drug classes**

This produces **257 queries total** across all six personas:

| Persona | Queries |
|---|---|
| `symptom_casual` | 52 |
| `worried_patient` | 54 |
| `condition` | 45 |
| `pharmacy_advice` | 39 |
| `search_engine` | 39 |
| `drug_class` | 28 |
| **Total** | **257** |

Brand names, INN names, and generic product names are stored per query in metadata fields (`brand_names`, `generic_names`) for post-hoc mention detection — but never injected into the query text itself.

## Layer 1: Commercial Model Data Collection

Queries are sent directly to commercial AI APIs with **no retrieval context**. The model answers purely from its parametric knowledge — this is the core test for brand/generic mention asymmetry in the training corpus.

Supported providers and models:

| Provider | Models |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` |
| Google | `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| Perplexity | `llama-3.1-sonar-small-128k-online`, `llama-3.1-sonar-large-128k-online` |

All requests use the same neutral Spanish system prompt (no brand/generic framing) at `temperature=0.3` for reproducibility. Each response is saved as a JSON file recording the query, raw response text, word count, timestamp, model, and provider. The Gemini 2.5 thinking models use `max_completion_tokens=8192` to avoid truncated responses caused by reasoning token overhead.

## PubMed Document Retrieval

For each drug pair the system queries NCBI PubMed via the E-utilities API (esearch + esummary) with four complementary search strategies:

1. **INN search** — the international nonproprietary name alone (e.g., `omeprazole[Title/Abstract]`)
2. **Brand search** — all known brand names (e.g., `"Losec"[Title/Abstract] OR "Mopral"[Title/Abstract]`)
3. **Generic-lab search** — INN combined with Spanish generic manufacturer names (e.g., Cinfa, Normon, Kern Pharma)
4. **Bioequivalence search** — INN plus terms: `bioequivalence`, `bioequivalent`, `therapeutic equivalence`, `generic comparison`, `brand vs generic`

This produces publication counts and article metadata (title, authors, journal, year, PMID) for each search. The data quantifies corpus density asymmetry: how much more scientific literature exists for brand names versus INNs in PubMed, providing external validation of the GEO hypothesis independent of the model responses.

The client respects NCBI rate limits (0.4 s between requests, ~3 req/s without API key) and identifies itself with a tool name and contact email per NCBI policy.

## Layer 2: RAG Replication of MeQA

The RAG pipeline is a faithful adaptation of the three-block MeQA architecture described in Santamaría (2021, arXiv:2111.02760):

```
┌─────────────────────────────────────────────────────────┐
│  BLOCK 1 — Question Processing                          │
│  • Text normalisation (lowercase, accent strip)         │
│  • Simplified NER: drug name, dose, pharmaceutical form │
│  • Section prediction: maps query category → prospecto  │
│    sections, simulating MeQA's bi-LSTM trained on       │
│    13,989 Doctoralia questions                          │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  BLOCK 2 — Leaflet & Section Extraction                 │
│  • Fetches official prospectos from CIMA REST API       │
│  • Parses HTML into 6-section structure                 │
│  • Chunks by section → paragraph → sentence             │
│    (~400 tokens per chunk, 50-token overlap)            │
│  • Embeds with text-embedding-3-small (1536-dim)        │
│  • In-memory numpy vector store, filtered by pair_id   │
│  • Replaces MeQA's TF-IDF + VSM/LSI with dense retrieval│
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  BLOCK 3 — Answer Generation                            │
│  • Fixed system prompt (identical for brand and generic)│
│  • Top-5 retrieved chunks injected as grounded context  │
│  • gpt-4o-mini, temperature=0.0 for reproducibility    │
│  • Response grounded exclusively in prospecto content   │
│  • Replaces MeQA's rule-based sentence extraction       │
└─────────────────────────────────────────────────────────┘
```

The section prediction step (Block 1) maps each query category to the leaflet sections most likely to contain the answer, mirroring MeQA's bi-LSTM classifier:

| Query category | Predicted sections |
|---|---|
| indication | 1 — Qué es y para qué se utiliza |
| adverse_effects | 4 — Posibles efectos adversos |
| dosage | 3 — Cómo tomar |
| contraindications | 2 — Qué necesita saber antes de tomar |
| pregnancy / alcohol / food | 2, 3 |
| equivalence / switching | 1, 6 |

The identical pipeline runs for brand and generic drugs — any response asymmetry therefore originates from differences in the prospecto content retrieved from CIMA, not from model or pipeline differences.

The vector store auto-bootstraps: if empty it fetches, chunks, and indexes all prospectos from CIMA automatically before the first query.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/miguelramirez004/meqa-test.git
cd meqa-test
pip install -r requirements.txt
```

### 2. Generate Query Battery

```bash
python scripts/generate_queries.py
# Output: data/queries/query_battery.csv (257 queries)
```

### 3. Download Ground Truth (Prospectos from CIMA)

```bash
python scripts/download_prospectos.py
# Downloads official prospectus HTML for all 20 drug pairs
```

### 4. Collect PubMed Literature

```bash
python scripts/collect_pubmed.py
# Queries NCBI for each pair: INN, brand, generic, bioequivalence
```

### 5. Collect Commercial Model Responses (Layer 1)

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export PERPLEXITY_API_KEY=...

python -c "
import os
from src.drug_pairs import get_offline_pairs
from src.query_generator import generate_queries
from src.commercial_collector import collect_commercial_responses

queries = generate_queries(get_offline_pairs())
collect_commercial_responses(queries, model='gpt-4o', api_key=os.environ['OPENAI_API_KEY'])
"
```

### 6. Run RAG Pipeline (Layer 2)

```bash
# Requires OPENAI_API_KEY for embeddings + generation
python -c "
from src.rag_engine import collect_all_responses
collect_all_responses()
"
```

### 7. Launch Dashboard

```bash
streamlit run app.py
```

## Project Structure

```
meqa-asymmetry-test/
├── README.md
├── requirements.txt
├── app.py                          # Streamlit dashboard (7-stage pipeline UI)
├── src/
│   ├── config.py                   # Paths, constants, API config
│   ├── drug_pairs.py               # 20 OFFLINE_PAIRS definitions
│   ├── query_generator.py          # 257-query battery (6 personas, deduped)
│   ├── commercial_collector.py     # Layer 1: commercial model API calls
│   ├── pubmed_client.py            # PubMed E-utilities (esearch + esummary)
│   ├── cima_client.py              # CIMA REST API client
│   ├── cima_leaflet_fetcher.py     # Prospecto HTML downloader
│   ├── leaflet_chunker.py          # HTML → section-aware text chunks
│   ├── vector_store.py             # In-memory numpy vector store
│   ├── rag_engine.py               # Layer 2: MeQA RAG replication (3 blocks)
│   └── response_analyzer.py        # NLP-based response coding
├── scripts/
│   ├── generate_queries.py
│   ├── download_prospectos.py
│   ├── collect_pubmed.py
│   ├── meqa_scraper.py             # Playwright-based MeQA web scraper
│   ├── analyze_responses.py
│   └── statistical_analysis.py
├── data/
│   ├── drug_pairs/                 # JSON pair definitions
│   ├── queries/                    # query_battery.csv / .json
│   ├── prospectos/                 # CIMA prospecto HTML (ground truth)
│   ├── leaflets/                   # Chunked leaflet text
│   ├── pubmed/                     # PubMed search results per pair
│   ├── responses/                  # Model responses (JSON per query)
│   └── analysis/                   # Metrics, reports
└── tests/
    └── test_cima_client.py
```

## CIMA REST API

Base URL: `https://cima.aemps.es/cima/rest/`

| Endpoint | Description |
|---|---|
| `GET medicamentos?practiv1=X&comerc=1` | Find drugs by active ingredient |
| `GET medicamento?nregistro=X` | Drug details by registration number |
| `GET docSegmentado/contenido/2?nregistro=X` | Prospecto sections (ground truth) |
| `POST buscarEnFichaTecnica` | Search within fichas técnicas |

Full documentation: [CIMA REST API v1.23 (PDF)](https://www.aemps.gob.es/apps/cima/docs/CIMA_REST_API.pdf)

## Key References

1. Santamaría, J. (2021). "Medicines Question Answering System, MeQA" — AEMPS, arXiv:2111.02760
2. Aggarwal et al. (2024). "GEO: Generative Engine Optimization" — *KDD '24*, arXiv:2311.09735
3. AEMPS (2025). "La AEMPS lanza MeQA" — Nota informativa, 13 May 2025
4. Kang et al. (2025). "RAG for 10 LLMs in Assessing Medical Fitness" — *npj Digital Medicine*, 8:195

## License

MIT — See [LICENSE](LICENSE).

## Disclaimer

This project is for academic research purposes only. It is not affiliated with AEMPS. All medication information is publicly available through CIMA. API calls are made at respectful intervals to avoid server impact.
