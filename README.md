# MeQA Asymmetry Test

**Testing for systematic information asymmetries between brand-name and generic medications in AEMPS MeQA — Spain's official AI-powered pharmaceutical Q&A system.**

> 📚 Part of doctoral research on **Generative Engine Optimization (GEO) opportunity gap in the pharmaceutical industry**.

## Background

[MeQA](https://cima.aemps.es/cima/publico/meqa.html) (Medicines Questions & Answers) is a RAG-based AI tool launched by Spain's drug regulator ([AEMPS](https://www.aemps.gob.es/)) in May 2025. It answers natural language questions about medications using official prospectus data from the [CIMA database](https://cima.aemps.es/).

**Research question:** Does MeQA produce systematically different responses when queried about brand-name drugs vs. their bioequivalent generic counterparts?

If asymmetries exist in a government RAG system grounded in official data, this demonstrates that GEO dynamics operate even within closed, curated retrieval systems — a finding with implications for pharmaceutical information equity and regulatory oversight.

## Experimental Design

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  CIMA REST API   │────▶│  Drug Pairs   │────▶│  Query Battery   │
│  (ground truth)  │     │  Brand vs EFG │     │  318 queries     │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                       │
                         ┌──────────────┐              │
                         │    MeQA      │◀─────────────┘
                         │  (web UI)    │
                         └──────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │  Response Analysis     │
                    │  • Word count          │
                    │  • Completeness score  │
                    │  • Safety warnings     │
                    │  • Adverse effects     │
                    │  • Professional refs   │
                    │  • Bioequivalence      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Statistical Tests     │
                    │  • Wilcoxon signed-rank│
                    │  • McNemar's test      │
                    │  • Cohen's kappa       │
                    │  • Kruskal-Wallis      │
                    └───────────────────────┘
```

### Hypotheses

| ID | Hypothesis | Test |
|----|-----------|------|
| H1 | MeQA provides more complete responses for brand-name drugs than generics | Wilcoxon signed-rank |
| H2 | MeQA shows differential accuracy rates between brand and generic queries | McNemar's test |
| H3 | Comparative queries ("¿es lo mismo X que Y?") show systematic framing bias | Chi-square |
| H4 | Asymmetry magnitude varies by therapeutic area | Kruskal-Wallis |

### Sample

| # | Active Ingredient | Brand | Generics | Therapeutic Group |
|---|------------------|-------|----------|-------------------|
| 1 | Omeprazol 20mg | Losec | Cinfa, Normon | Gastrointestinal |
| 2 | Atorvastatina 20mg | Cardyl | Cinfa, Teva | Cardiovascular |
| 3 | Escitalopram 10mg | Cipralex | Cinfa, Stada | Antidepressants |
| 4 | Amoxicilina 500mg | Clamoxyl | Cinfa, Normon | Anti-infectives |
| 5 | Ibuprofeno 600mg | Neobrufen | Cinfa, Kern | Anti-inflammatory |
| 6 | Metformina 850mg | Dianben | Cinfa, Kern | Metabolism |
| 7 | Sertralina 50mg | Besitran | Cinfa, Normon | Antidepressants |
| 8 | Simvastatina 20mg | Zocor | Cinfa, Normon | Cardiovascular |
| 9 | Amlodipino 5mg | Norvasc | Cinfa, Normon | Cardiovascular |
| 10 | Levotiroxina 100mcg | Eutirox | Sanofi | Thyroid hormones |

### Query Battery

- **90** brand-name queries (9 question types × 10 brands)
- **171** generic queries (9 question types × 19 generics)
- **57** comparative queries (3 question types × 19 brand-generic pairs)
- **318 total queries**

## Quick Start

### 1. Setup

```bash
git clone https://github.com/YOUR_USER/meqa-asymmetry-test.git
cd meqa-asymmetry-test
pip install -r requirements.txt
```

### 2. Generate Query Battery

```bash
python scripts/generate_queries.py
# Output: data/queries/query_battery.csv (318 queries ready to use)
```

### 3. Download Ground Truth (Prospectos from CIMA API)

```bash
python scripts/download_prospectos.py
# Downloads official prospectus data for all drugs in sample
```

### 4. Collect MeQA Responses

**Option A — Semi-automated (Playwright):**
```bash
playwright install chromium
python scripts/meqa_scraper.py
```

**Option B — Manual collection:**
Open [MeQA](https://cima.aemps.es/cima/publico/meqa.html), use the CSV as checklist, paste responses into `data/responses/`.

### 5. Analyze Results

```bash
python scripts/analyze_responses.py    # Code response metrics
python scripts/statistical_analysis.py # Run hypothesis tests
```

## Project Structure

```
meqa-asymmetry-test/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── cima_client.py          # CIMA REST API client
│   ├── drug_pairs.py           # Drug pair definitions & builder
│   ├── query_generator.py      # Standardized query battery
│   ├── response_analyzer.py    # NLP-based response coding
│   └── config.py               # Paths, constants, API config
├── scripts/
│   ├── generate_queries.py     # Generate query battery CSV
│   ├── download_prospectos.py  # Download ground truth from CIMA
│   ├── meqa_scraper.py         # Playwright-based MeQA scraper
│   ├── analyze_responses.py    # Run response analysis
│   └── statistical_analysis.py # Hypothesis testing
├── data/
│   ├── drug_pairs/             # Drug pair definitions (JSON)
│   ├── queries/                # Generated query battery (CSV/JSON)
│   ├── prospectos/             # Ground truth prospectos (JSON)
│   ├── responses/              # MeQA responses (JSON per query)
│   └── analysis/               # Results, metrics, reports
├── docs/
│   ├── methodology.md          # Detailed experimental methodology
│   └── cima_api_reference.md   # CIMA REST API v1.23 reference
└── tests/
    └── test_cima_client.py     # API client tests
```

## CIMA REST API

Base URL: `https://cima.aemps.es/cima/rest/`

| Endpoint | Description |
|----------|------------|
| `GET medicamentos?practiv1=X&comerc=1` | Find drugs by active ingredient |
| `GET medicamento?nregistro=X` | Drug details by registration number |
| `GET docSegmentado/contenido/2?nregistro=X` | Prospecto sections (ground truth) |
| `POST buscarEnFichaTecnica` | Search within fichas técnicas |
| `GET presentaciones?nregistro=X` | Drug presentations |

Full documentation: [CIMA REST API v1.23 (PDF)](https://www.aemps.gob.es/apps/cima/docs/CIMA_REST_API.pdf)

## Key References

1. Aggarwal et al. (2024) "GEO: Generative Engine Optimization" — *KDD '24*, arXiv:2311.09735
2. AEMPS (2025) "La AEMPS lanza MeQA" — Nota informativa, 13 May 2025
3. Kang et al. (2025) "RAG for 10 LLMs in Assessing Medical Fitness" — *npj Digital Medicine*, 8:195
4. McKinsey (2025) "Scaling Gen AI in Life Sciences" — Only 5% achieved competitive differentiation

## License

MIT — See [LICENSE](LICENSE).

## Disclaimer

This project is for academic research purposes. It is not affiliated with AEMPS. All medication information used is publicly available through CIMA. MeQA queries are conducted at respectful intervals to avoid server impact.
