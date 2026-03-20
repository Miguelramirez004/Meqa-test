"""MeQA Asymmetry Test — Streamlit UI

Cloud-deployable dashboard for running the full MeQA asymmetry experiment:
query generation, response analysis, and statistical testing.
"""

import sys
import json
import io
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATA_DIR, QUERIES_DIR, RESPONSES_DIR, ANALYSIS_DIR, PROSPECTOS_DIR, PAIRS_DIR,
    PUBMED_DIR, LEAFLETS_DIR,
)
from src.drug_pairs import get_offline_pairs, OFFLINE_PAIRS
from src.query_generator import generate_queries, save_queries
from src.response_analyzer import analyze_response, analyze_all_responses, compute_asymmetry_scores
from src.cima_client import CIMAClient
from src.rag_engine import collect_all_responses as rag_collect
from src.commercial_collector import (
    collect_commercial_responses, detect_provider,
    PROVIDER_MODELS, get_all_models,
)
from src.pubmed_client import collect_pubmed_for_all_pairs

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MeQA Asymmetry Test",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────────────────

st.sidebar.title("MeQA Asymmetry Test")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Pipeline Stage",
    [
        "1. Drug Pairs",
        "2. Generate Queries",
        "3. Download Prospectos",
        "4. PubMed Literature",
        "5. Collect Responses",
        "6. Analyze Responses",
        "7. Statistical Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Doctoral research — GEO asymmetry in MeQA (AEMPS)")


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [DATA_DIR, QUERIES_DIR, RESPONSES_DIR, ANALYSIS_DIR, PROSPECTOS_DIR, PAIRS_DIR, PUBMED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()


def load_queries_df() -> pd.DataFrame | None:
    csv_path = QUERIES_DIR / "query_battery.csv"
    json_path = QUERIES_DIR / "query_battery.json"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        return pd.read_json(json_path)
    return None


def load_responses_count() -> int:
    return len(list(RESPONSES_DIR.glob("Q*.json")))


def load_metrics_df() -> pd.DataFrame | None:
    f = ANALYSIS_DIR / "response_metrics.csv"
    return pd.read_csv(f) if f.exists() else None


# ── Page 1: Drug Pairs ──────────────────────────────────────────────────────

if page == "1. Drug Pairs":
    st.header("Drug Pairs (Brand vs Generic)")
    st.markdown(
        "10 pre-defined pairs across 6 therapeutic groups. "
        "Each pair has one brand-name drug and 1–3 bioequivalent generics."
    )

    pairs = get_offline_pairs()

    for pair in pairs:
        with st.expander(f"**{pair['pair_id']}** — {pair['principio_activo']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Therapeutic group:** {pair['grupo']}")
                st.markdown(f"**ATC:** {pair['atc']}")
            with col2:
                st.markdown(f"**Brand:** {pair['brand']}")
                st.markdown("**Generics:**")
                for g in pair["generics"]:
                    st.markdown(f"- {g}")

    st.markdown("---")
    st.subheader("Summary Table")

    rows = []
    for p in pairs:
        rows.append({
            "Pair ID": p["pair_id"],
            "Active Ingredient": p["principio_activo"],
            "Group": p["grupo"],
            "ATC": p["atc"],
            "Brand": p["brand"].split(" ")[0],
            "Generics": len(p["generics"]),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ── Page 2: Generate Queries ─────────────────────────────────────────────────

elif page == "2. Generate Queries":
    st.header("Generate Query Battery")
    st.markdown(
        "Creates Layer 1 Drug Mention Audit queries in three blocks:\n\n"
        "**Block A** — Condition-first (no drug name, tests corpus density)\n\n"
        "**Block B** — Single-drug (tests cross-referencing)\n\n"
        "**Block C** — Direct comparison (tests information balance)"
    )

    existing_df = load_queries_df()
    if existing_df is not None:
        st.success(f"Query battery already exists: **{len(existing_df)} queries**")
        st.dataframe(existing_df.head(20), width="stretch", hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            csv_data = existing_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV", csv_data, "query_battery.csv", "text/csv"
            )
        with col2:
            json_data = existing_df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "Download JSON", json_data, "query_battery.json", "application/json"
            )

    st.markdown("---")

    if st.button("Generate / Regenerate Query Battery", type="primary"):
        with st.spinner("Generating queries..."):
            pairs = get_offline_pairs()
            queries = generate_queries(pairs)
            save_queries(queries)

        condition_q = sum(1 for q in queries if q["query_type"] == "condition")
        single_q = sum(1 for q in queries if q["query_type"] == "single")
        comp_q = sum(1 for q in queries if q["query_type"] == "comparative")

        st.success(f"Generated **{len(queries)}** queries")
        col1, col2, col3 = st.columns(3)
        col1.metric("Block A: Condition", condition_q)
        col2.metric("Block B: Single-drug", single_q)
        col3.metric("Block C: Comparative", comp_q)
        st.rerun()


# ── Page 3: Download Prospectos ──────────────────────────────────────────────

elif page == "3. Download Prospectos":
    st.header("Download Prospectos (Ground Truth)")
    st.markdown(
        "Fetches official drug prospectuses from the **CIMA REST API** "
        "to serve as ground truth for accuracy evaluation."
    )

    existing = list(PROSPECTOS_DIR.glob("*.json"))
    if existing:
        st.info(f"**{len(existing)}** prospectos already downloaded.")
        names = [f.stem for f in sorted(existing)]
        st.dataframe(pd.DataFrame({"Prospecto": names}), width="stretch", hide_index=True)

    st.markdown("---")

    st.warning(
        "This step queries the live CIMA API and may take several minutes. "
        "Requires internet access."
    )

    if st.button("Download from CIMA API", type="primary"):
        progress_area = st.empty()
        status_area = st.empty()

        try:
            from src.drug_pairs import build_pairs_from_cima, save_pairs

            cima = CIMAClient()

            status_area.info("Building drug pairs from CIMA API...")
            pairs = build_pairs_from_cima(cima)

            if not pairs:
                st.error("No valid pairs found. Check internet connection or CIMA API availability.")
            else:
                save_pairs(pairs)
                status_area.info(f"Found {len(pairs)} pairs. Downloading prospectos...")

                PROSPECTOS_DIR.mkdir(parents=True, exist_ok=True)
                downloaded = 0
                total_meds = sum(1 + len(p.generics) for p in pairs)
                progress = st.progress(0)

                # Also build nregistro config for RAG pipeline
                nregistro_pairs = []

                for pair in pairs:
                    all_meds = [pair.brand] + pair.generics
                    for med in all_meds:
                        safe_name = med.nombre[:40].replace("/", "_").replace(" ", "_")
                        filename = PROSPECTOS_DIR / f"{med.nregistro}_{safe_name}.json"

                        if not filename.exists():
                            doc = cima.get_doc_segmentado(med.nregistro, tipo_doc=2)
                            if doc:
                                with open(filename, "w", encoding="utf-8") as f:
                                    json.dump({
                                        "nregistro": med.nregistro,
                                        "nombre": med.nombre,
                                        "labtitular": med.labtitular,
                                        "es_generico": med.es_generico,
                                        "prospecto_sections": doc,
                                    }, f, ensure_ascii=False, indent=2)

                        downloaded += 1
                        progress.progress(downloaded / total_meds)

                    # Build nregistro config entry
                    nregistro_pairs.append({
                        "pair_id": pair.pair_id,
                        "principio_activo": pair.principio_activo,
                        "grupo_terapeutico": pair.grupo_terapeutico,
                        "atc_code": pair.atc_code,
                        "branded": {
                            "name": pair.brand.nombre if pair.brand else "",
                            "nregistro": pair.brand.nregistro if pair.brand else "",
                            "is_generic": False,
                        },
                        "generic": {
                            "name": pair.generics[0].nombre if pair.generics else "",
                            "nregistro": pair.generics[0].nregistro if pair.generics else "",
                            "is_generic": True,
                        },
                    })

                # Save nregistro config for RAG pipeline
                nregistro_path = PAIRS_DIR / "drug_pairs_nregistro.json"
                PAIRS_DIR.mkdir(parents=True, exist_ok=True)
                with open(nregistro_path, "w", encoding="utf-8") as f:
                    json.dump(nregistro_pairs, f, ensure_ascii=False, indent=2)

                progress.empty()
                st.success(
                    f"Downloaded prospectos for {total_meds} medications. "
                    f"nregistro config saved for RAG pipeline."
                )
                st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")


# ── Page 4: PubMed Literature ────────────────────────────────────────────────

elif page == "4. PubMed Literature":
    st.header("PubMed Literature Collection")
    st.markdown(
        "Collects published literature from **PubMed** (NCBI) for each drug pair. "
        "Searches for:\n\n"
        "1. **General literature** on the active ingredient\n"
        "2. **Bioequivalence / brand vs generic** studies\n"
        "3. **Condition-specific** treatment evidence"
    )

    # Show existing results
    existing_pubmed = sorted(PUBMED_DIR.glob("*_pubmed.json"))
    pairs = get_offline_pairs()

    col1, col2 = st.columns(2)
    col1.metric("Drug Pairs", len(pairs))
    col2.metric("PubMed Files Collected", len(existing_pubmed))

    if existing_pubmed:
        st.success(f"**{len(existing_pubmed)}/{len(pairs)}** pairs have PubMed data.")

        # Summary table
        summary_rows = []
        for pf in existing_pubmed:
            with open(pf, encoding="utf-8") as f:
                pdata = json.load(f)
            summary_rows.append({
                "Pair ID": pdata.get("pair_id", pf.stem),
                "Active Ingredient": pdata.get("principio_activo", ""),
                "Articles Found": pdata.get("total_found", 0),
                "Queries Used": len(pdata.get("queries", [])),
            })
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        # Expandable detail per pair
        st.markdown("---")
        st.subheader("Article Details")
        for pf in existing_pubmed:
            with open(pf, encoding="utf-8") as f:
                pdata = json.load(f)
            pair_id = pdata.get("pair_id", pf.stem)
            n_articles = pdata.get("total_found", 0)
            with st.expander(f"**{pair_id}** — {pdata.get('principio_activo', '')} ({n_articles} articles)"):
                # Show queries used
                for q in pdata.get("queries", []):
                    st.caption(f"**{q['label']}**: `{q['query']}` → {q['results']} results")

                # Show articles
                for art in pdata.get("articles", [])[:10]:
                    tags = ", ".join(art.get("search_tags", []))
                    st.markdown(
                        f"**[PMID {art['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{art['pmid']}/)** "
                        f"— {art.get('title', 'No title')}"
                    )
                    st.caption(
                        f"{art.get('journal', '')} ({art.get('year', 'N/A')}) "
                        f"| Tags: {tags}"
                    )
                    if art.get("abstract"):
                        st.text(art["abstract"][:300] + ("..." if len(art.get("abstract", "")) > 300 else ""))
                    st.markdown("---")

                if n_articles > 10:
                    st.caption(f"Showing first 10 of {n_articles} articles.")

        # Download all
        st.markdown("---")
        all_pubmed_data = []
        for pf in existing_pubmed:
            with open(pf, encoding="utf-8") as f:
                all_pubmed_data.append(json.load(f))
        json_bytes = json.dumps(all_pubmed_data, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "Download All PubMed Data (JSON)",
            json_bytes,
            "pubmed_all_pairs.json",
            "application/json",
        )

    st.markdown("---")

    # Collection controls
    st.subheader("Collect from PubMed")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        pubmed_api_key = st.text_input(
            "NCBI API Key (optional)",
            type="password",
            help="Get one at https://www.ncbi.nlm.nih.gov/account/settings/ — "
                 "increases rate limit from 3/s to 10/s. Leave blank for unauthenticated access.",
        )
    with col_cfg2:
        pubmed_max = st.slider(
            "Max articles per query", 5, 50, 20,
            help="Number of articles to retrieve per search query.",
        )

    skip_existing_pubmed = st.checkbox("Skip pairs with existing data", value=True)

    if st.button("Collect PubMed Literature", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_pubmed_progress(current, total):
            progress_bar.progress(current / total)
            pair_name = pairs[current - 1]["principio_activo"] if current <= len(pairs) else ""
            status_text.text(f"Pair {current}/{total} — {pair_name}")

        try:
            status_text.text("Connecting to PubMed (NCBI E-utilities)...")
            results = collect_pubmed_for_all_pairs(
                drug_pairs=pairs,
                output_dir=PUBMED_DIR,
                api_key=pubmed_api_key,
                max_results=pubmed_max,
                skip_existing=skip_existing_pubmed,
                progress_callback=update_pubmed_progress,
            )
            progress_bar.empty()
            status_text.empty()

            total_articles = sum(r.get("total_found", 0) for r in results)
            pair_errors = [r for r in results if r.get("error")]
            st.success(
                f"Collected literature for **{len(results)}** drug pairs. "
                f"Total articles: **{total_articles}**"
            )
            if pair_errors:
                st.warning(
                    f"**{len(pair_errors)}** pairs had errors: "
                    + ", ".join(f"{r['pair_id']}: {r['error']}" for r in pair_errors[:5])
                )
            if total_articles == 0 and not pair_errors:
                st.info(
                    "No articles found. If you have an NCBI API key, enter it above. "
                    "Also check your Streamlit Cloud logs for detailed error messages."
                )
            st.rerun()
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error during PubMed collection: {e}")


# ── Page 5: Collect Responses ────────────────────────────────────────────────

elif page == "5. Collect Responses":
    st.header("Collect Responses")
    st.markdown(
        "Send queries to AI models and collect responses. Choose between "
        "**Commercial Models** (direct, no RAG) for the Drug Mention Audit, "
        "or the **MeQA RAG Pipeline** (prospectos + LLM)."
    )

    resp_count = load_responses_count()
    queries_df = load_queries_df()
    total_queries = len(queries_df) if queries_df is not None else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Responses Collected", resp_count)
    col2.metric("Total Queries", total_queries)
    col3.metric("Completion", f"{resp_count/total_queries*100:.0f}%" if total_queries else "N/A")

    if resp_count > 0 and total_queries > 0:
        st.progress(min(resp_count / total_queries, 1.0))

    st.markdown("---")

    # ── Collection mode selector ──
    collection_mode = st.radio(
        "Collection Method",
        ["Commercial Models (ChatGPT, Gemini, Perplexity)", "MeQA RAG Pipeline"],
        index=0,
        help="Commercial: queries models directly without RAG context (Layer 1 audit). "
             "RAG: uses prospectos as retrieval context.",
    )

    skip_existing = st.checkbox("Skip already-collected responses", value=True)

    # ══════════════════════════════════════════════════════════════
    # COMMERCIAL MODELS — Direct queries, no RAG context
    # ══════════════════════════════════════════════════════════════
    if collection_mode.startswith("Commercial"):
        st.subheader("Commercial Model Collection (No RAG)")
        st.markdown(
            "Sends queries directly to commercial AI models **without any RAG context**. "
            "The model responds purely from its parametric knowledge — "
            "this is the core test for the **Layer 1: Drug Mention Audit**."
        )

        # Provider selection
        provider = st.selectbox(
            "Provider",
            ["openai", "gemini", "perplexity"],
            format_func=lambda x: {"openai": "OpenAI (ChatGPT)", "gemini": "Google Gemini", "perplexity": "Perplexity"}[x],
        )

        # Model selection based on provider
        available_models = PROVIDER_MODELS[provider]
        model = st.selectbox("Model", available_models, index=0)

        # Provider-specific API key
        key_labels = {
            "openai": ("OpenAI API Key", "OPENAI_API_KEY"),
            "gemini": ("Google Gemini API Key", "GEMINI_API_KEY"),
            "perplexity": ("Perplexity API Key", "PERPLEXITY_API_KEY"),
        }
        key_label, key_env = key_labels[provider]

        api_key = st.secrets.get(key_env, "") if hasattr(st, "secrets") else ""
        if not api_key:
            api_key = st.text_input(
                key_label,
                type="password",
                help=f"Set `{key_env}` in .streamlit/secrets.toml or enter here.",
            )

        # Temperature
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1,
                                help="Lower = more deterministic. 0.3 recommended for audit.")

        if not api_key:
            st.info(f"Enter your {key_label} above to enable collection.")
        elif total_queries == 0:
            st.warning("No query battery found. Run **Step 2** first.")
        else:
            if st.button("Run Commercial Collection", type="primary"):
                queries_json = QUERIES_DIR / "query_battery.json"
                with open(queries_json, encoding="utf-8") as f:
                    queries_list = json.load(f)

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"Query {current}/{total} — {provider}/{model}")

                try:
                    status_text.text(f"Connecting to {provider}/{model}...")
                    collected = collect_commercial_responses(
                        queries=queries_list,
                        model=model,
                        api_key=api_key,
                        responses_dir=RESPONSES_DIR,
                        delay=0.5,
                        temperature=temperature,
                        skip_existing=skip_existing,
                        progress_callback=update_progress,
                    )
                    progress_bar.empty()
                    status_text.empty()
                    new_count = len(collected)
                    total_now = load_responses_count()
                    st.success(
                        f"Collected **{new_count}** new responses via {provider}/{model}. "
                        f"Total: **{total_now}/{total_queries}**"
                    )
                    st.rerun()
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Error during commercial collection: {e}")

    # ══════════════════════════════════════════════════════════════
    # RAG PIPELINE — CIMA Leaflets + OpenAI Embeddings + LLM
    # ══════════════════════════════════════════════════════════════
    else:
        st.subheader("MeQA RAG Pipeline Collection")
        st.markdown(
            "Adapts the [MeQA architecture](https://arxiv.org/abs/2111.02760) "
            "(AEMPS, Santamaría 2021) using modern dense retrieval:\n\n"
            "**Block 1** — Question Processing: normalisation + NER + section prediction\n\n"
            "**Block 2** — Dense Retrieval: OpenAI `text-embedding-3-small` "
            "embeddings + cosine similarity with metadata filtering\n\n"
            "**Block 3** — Answer Generation: LLM grounded in retrieved chunks with "
            "section context (MEQa's 'context addition')\n\n"
            "**Self-bootstrapping**: The pipeline auto-indexes all available "
            "prospectos and leaflets before processing queries."
        )

        api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
        if not api_key:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Set `OPENAI_API_KEY` in .streamlit/secrets.toml or enter here.",
            )

        # Vector store is in-memory — index is built at collection time
        chroma_count = 0  # Will be populated during RAG collection

        # Check data sources
        prospecto_count = len([
            f for f in PROSPECTOS_DIR.glob("*.json") if f.name != ".gitkeep"
        ]) if PROSPECTOS_DIR.exists() else 0
        leaflet_count = len(list(LEAFLETS_DIR.glob("**/*.html"))) if LEAFLETS_DIR.exists() else 0

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            model = st.selectbox(
                "LLM Model (generation)",
                ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
                index=0,
                help="gpt-4o-mini is recommended (fast and cost-effective).",
            )
            top_k = st.slider(
                "Chunks to retrieve (top-k)", min_value=1, max_value=20, value=5,
                help="Number of prospecto chunks fed as context to the LLM.",
            )
        with col_cfg2:
            st.metric("Data Sources", f"{prospecto_count} JSON / {leaflet_count} HTML")

        # Status indicators
        if prospecto_count > 0 or leaflet_count > 0:
            st.info(
                f"Data available ({prospecto_count} prospectos, {leaflet_count} leaflets). "
                "Will be indexed in-memory on collection run."
            )
        else:
            st.info(
                "No local data yet. The pipeline will **auto-fetch leaflets from CIMA** "
                "on first run (may take a few minutes for the initial bootstrap)."
            )

        st.markdown("---")

        # ── Run RAG Collection ──
        if not api_key:
            st.info("Enter your OpenAI API key above to enable automated collection.")
        elif total_queries == 0:
            st.warning("No query battery found. Run **Step 2** first.")
        else:
            if st.button("Run MeQA RAG Collection", type="primary"):
                queries_json = QUERIES_DIR / "query_battery.json"
                with open(queries_json, encoding="utf-8") as f:
                    queries_list = json.load(f)

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    if total > 0:
                        progress_bar.progress(min(current / total, 1.0))
                    status_text.text(
                        f"Processing query {current}/{total}..."
                        if current > 0 else "Auto-bootstrapping: fetching leaflets from CIMA..."
                    )

                try:
                    status_text.text("Initialising RAG pipeline (auto-bootstrap if needed)...")
                    collected = rag_collect(
                        queries=queries_list,
                        api_key=api_key,
                        model=model,
                        prospectos_dir=PROSPECTOS_DIR,
                        responses_dir=RESPONSES_DIR,
                        top_k=top_k,
                        delay=0.3,
                        progress_callback=update_progress,
                        skip_existing=skip_existing,
                    )
                    progress_bar.empty()
                    status_text.empty()
                    new_count = len(collected)
                    total_now = load_responses_count()
                    st.success(
                        f"Collected **{new_count}** new responses. "
                        f"Total: **{total_now}/{total_queries}**"
                    )
                    st.rerun()
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Error during RAG collection: {e}")

    st.markdown("---")

    # ── Manual upload: CSV ──
    st.subheader("Manual Upload (CSV)")
    st.markdown(
        "Upload a CSV with columns `query_id` and `response_text`. "
        "Each row will be saved as an individual response file."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        try:
            resp_df = pd.read_csv(uploaded)
            st.dataframe(resp_df.head(10), width="stretch", hide_index=True)

            if "query_id" not in resp_df.columns or "response_text" not in resp_df.columns:
                st.error("CSV must contain `query_id` and `response_text` columns.")
            elif st.button("Save Responses", type="primary"):
                RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
                saved = 0

                # Merge with query metadata if available
                if queries_df is not None:
                    meta_cols = [c for c in queries_df.columns if c != "response_text" and c != "response_word_count"]
                    merged = resp_df.merge(
                        queries_df[meta_cols], on="query_id", how="left"
                    )
                else:
                    merged = resp_df

                for _, row in merged.iterrows():
                    qid = row["query_id"]
                    data = row.to_dict()
                    # Clean NaN values for JSON
                    data = {k: (v if pd.notna(v) else "") for k, v in data.items()}
                    fpath = RESPONSES_DIR / f"{qid}.json"
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    saved += 1

                st.success(f"Saved **{saved}** response files to `data/responses/`")
                st.rerun()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    st.markdown("---")

    # ── Manual upload: JSON ──
    st.subheader("Upload Individual Responses (JSON)")
    json_files = st.file_uploader(
        "Upload Q*.json files", type=["json"], accept_multiple_files=True
    )
    if json_files:
        if st.button("Save JSON Responses"):
            RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
            saved = 0
            for jf in json_files:
                data = json.load(jf)
                qid = data.get("query_id", jf.name.replace(".json", ""))
                fpath = RESPONSES_DIR / f"{qid}.json"
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                saved += 1
            st.success(f"Saved **{saved}** response files.")
            st.rerun()

    # ── Show existing responses ──
    if resp_count > 0:
        st.markdown("---")
        st.subheader("Existing Responses")

        # ── Download All Responses ──
        all_resp_files = sorted(RESPONSES_DIR.glob("Q*.json"))
        all_resp_data = []
        for rf in all_resp_files:
            with open(rf, encoding="utf-8") as f:
                all_resp_data.append(json.load(f))

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            resp_df_export = pd.json_normalize(all_resp_data, sep="_")
            csv_bytes = resp_df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download All Responses (CSV)",
                csv_bytes,
                "meqa_responses.csv",
                "text/csv",
                key="dl_resp_csv",
            )
        with col_dl2:
            json_bytes = json.dumps(
                all_resp_data, ensure_ascii=False, indent=2
            ).encode("utf-8")
            st.download_button(
                "Download All Responses (JSON)",
                json_bytes,
                "meqa_responses.json",
                "application/json",
                key="dl_resp_json",
            )

        st.markdown("---")

        # Preview table
        resp_files = all_resp_files[:20]
        rows = []
        for rf_data in all_resp_data[:20]:
            text = rf_data.get("response_text", "")
            rows.append({
                "Query ID": rf_data.get("query_id", ""),
                "Drug": rf_data.get("drug_name", ""),
                "Category": rf_data.get("query_category", ""),
                "Words": len(text.split()) if text else 0,
                "Preview": text[:100] + "..." if len(text) > 100 else text,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if resp_count > 20:
            st.caption(f"Showing first 20 of {resp_count} responses.")


# ── Page 6: Analyze Responses ────────────────────────────────────────────────

elif page == "6. Analyze Responses":
    st.header("Analyze Responses (Drug Mention Audit)")
    st.markdown(
        "Codes each response with **broad drug name recognition**:\n\n"
        "- **INN detection**: active ingredient mentions (e.g., *omeprazol*)\n"
        "- **Brand detection**: all known commercial names (e.g., *Losec*, *Mepral*, *Prilosec*)\n"
        "- **Generic product detection**: INN + laboratory (e.g., *Omeprazol Cinfa*)\n\n"
        "Computes mention-based asymmetry scores per drug pair."
    )

    resp_count = load_responses_count()
    metrics_df = load_metrics_df()

    if resp_count == 0:
        st.warning("No responses found. Go to **Step 5** to upload MeQA responses first.")
    else:
        st.info(f"**{resp_count}** responses available for analysis.")

        if metrics_df is not None:
            st.success(f"Analysis complete: **{len(metrics_df)}** responses coded.")

        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing responses..."):
                all_metrics = analyze_all_responses()

            if all_metrics:
                st.success(f"Coded **{len(all_metrics)}** responses.")
                metrics_df = load_metrics_df()
                st.rerun()
            else:
                st.error("No valid responses found to analyze.")

    if metrics_df is not None:
        st.markdown("---")
        st.subheader("Coded Metrics")
        st.dataframe(metrics_df, width="stretch", hide_index=True)

        # Key stats
        st.markdown("---")
        st.subheader("Overview")

        condition = metrics_df[metrics_df["query_type"] == "condition"]
        single = metrics_df[metrics_df["query_type"] == "single"]
        comp = metrics_df[metrics_df["query_type"] == "comparative"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Block A: Condition", len(condition))
        col2.metric("Block B: Single-drug", len(single))
        col3.metric("Block C: Comparative", len(comp))

        st.markdown("#### Drug Mention Summary")
        col1, col2, col3 = st.columns(3)
        inn_col = "inn_mentioned" if "inn_mentioned" in metrics_df.columns else None
        if inn_col:
            col1.metric("INN (Active Ingredient)",
                         f"{metrics_df['inn_mentioned'].sum()}/{len(metrics_df)}")
        col2.metric("Brand Name Detected",
                     f"{metrics_df['brand_mentioned'].sum()}/{len(metrics_df)}")
        col3.metric("Generic (INN+Product)",
                     f"{metrics_df['generic_mentioned'].sum()}/{len(metrics_df)}")

        col1, col2, col3, col4 = st.columns(4)
        total_brand = metrics_df["brand_mention_count"].sum()
        total_generic = metrics_df["generic_mention_count"].sum()
        col1.metric("Total Brand Mentions", int(total_brand))
        col2.metric("Total Generic Mentions", int(total_generic))
        if "inn_mention_count" in metrics_df.columns:
            col3.metric("Total INN Mentions", int(metrics_df["inn_mention_count"].sum()))
        if "generic_product_mention_count" in metrics_df.columns:
            col4.metric("Generic Products (INN+Lab)",
                         int(metrics_df["generic_product_mention_count"].sum()))

        # Brand names found breakdown
        if "brand_names_found" in metrics_df.columns:
            st.markdown("#### Brand Names Detected")
            brand_found = metrics_df["brand_names_found"].dropna()
            brand_found = brand_found[brand_found != ""]
            if not brand_found.empty:
                all_brands = []
                for entry in brand_found:
                    all_brands.extend(b.strip() for b in entry.split(";") if b.strip())
                if all_brands:
                    brand_counts = pd.Series(all_brands).value_counts()
                    st.bar_chart(brand_counts)

        # Generic labs found breakdown
        if "generic_labs_found" in metrics_df.columns:
            st.markdown("#### Generic Laboratories Detected")
            labs_found = metrics_df["generic_labs_found"].dropna()
            labs_found = labs_found[labs_found != ""]
            if not labs_found.empty:
                all_labs = []
                for entry in labs_found:
                    all_labs.extend(l.strip() for l in entry.split(";") if l.strip())
                if all_labs:
                    lab_counts = pd.Series(all_labs).value_counts()
                    st.bar_chart(lab_counts)

        st.markdown("#### First-Mentioned Advantage")
        brand_first = (metrics_df["first_drug_mentioned"] == "brand").sum()
        generic_first = (metrics_df["first_drug_mentioned"] == "generic").sum()
        neither = (metrics_df["first_drug_mentioned"] == "neither").sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Brand First", brand_first)
        col2.metric("Generic First", generic_first)
        col3.metric("Neither", neither)

        # Mention counts by query type
        st.markdown("#### Mentions by Query Type")
        agg_dict = {
            "n": ("query_id", "count"),
            "brand_mentions": ("brand_mention_count", "sum"),
            "generic_mentions": ("generic_mention_count", "sum"),
            "brand_pct": ("brand_mentioned", "mean"),
            "generic_pct": ("generic_mentioned", "mean"),
        }
        if "inn_mention_count" in metrics_df.columns:
            agg_dict["inn_mentions"] = ("inn_mention_count", "sum")
        mention_by_type = metrics_df.groupby("query_type").agg(**agg_dict).reset_index()
        mention_by_type["brand_pct"] = (mention_by_type["brand_pct"] * 100).round(0).astype(int).astype(str) + "%"
        mention_by_type["generic_pct"] = (mention_by_type["generic_pct"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(mention_by_type, width="stretch", hide_index=True)

        # Asymmetry scores
        asym_file = ANALYSIS_DIR / "asymmetry_scores.csv"
        if asym_file.exists():
            st.markdown("---")
            st.subheader("Asymmetry Scores (per drug pair)")
            asym_df = pd.read_csv(asym_file)
            st.dataframe(asym_df, width="stretch", hide_index=True)

            if not asym_df.empty:
                st.bar_chart(asym_df.set_index("pair_id")["composite_asymmetry_score"])

        # Download
        csv_data = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Metrics CSV", csv_data, "response_metrics.csv", "text/csv")


# ── Page 7: Statistical Analysis ─────────────────────────────────────────────

elif page == "7. Statistical Analysis":
    st.header("Statistical Hypothesis Testing")
    st.markdown(
        "Non-parametric tests for drug mention asymmetry between "
        "brand-name and generic drugs."
    )

    metrics_df = load_metrics_df()

    if metrics_df is None:
        st.warning("No coded metrics found. Run **Step 6** first.")
    else:
        st.info(f"Loaded **{len(metrics_df)}** coded responses.")

        import numpy as np
        from scipy import stats

        tabs = st.tabs(["H1: Mention Asymmetry", "H2: Position Advantage",
                         "H3: Cross-Reference", "H4: Condition Visibility",
                         "H5: Therapeutic Variation", "Summary"])

        # ── H1: Mention Asymmetry ──
        with tabs[0]:
            st.subheader("H1: Brand drugs mentioned more often than generics")
            st.markdown("**Test:** Wilcoxon signed-rank on per-pair mention counts")

            pairs = metrics_df["pair_id"].unique()
            rows = []
            for pid in pairs:
                p = metrics_df[metrics_df["pair_id"] == pid]
                rows.append({
                    "Pair ID": pid,
                    "Brand Mentions": int(p["brand_mention_count"].sum()),
                    "Generic Mentions": int(p["generic_mention_count"].sum()),
                    "Difference": int(p["brand_mention_count"].sum() - p["generic_mention_count"].sum()),
                })

            if rows:
                agg = pd.DataFrame(rows)
                st.dataframe(agg, width="stretch", hide_index=True)

                if len(agg) >= 5:
                    w, p = stats.wilcoxon(agg["Brand Mentions"].values, agg["Generic Mentions"].values)
                    st.markdown(
                        f"**Wilcoxon signed-rank:** W={w:.4f}, p={p:.6f} — "
                        f"{'**Significant**' if p < 0.05 else 'Not significant'} (alpha=0.05)"
                    )

        # ── H2: Position Advantage ──
        with tabs[1]:
            st.subheader("H2: Brand drugs appear earlier in responses")
            st.markdown("**Test:** Binomial test on first-mention proportion")

            brand_first = (metrics_df["first_drug_mentioned"] == "brand").sum()
            generic_first = (metrics_df["first_drug_mentioned"] == "generic").sum()
            neither = (metrics_df["first_drug_mentioned"] == "neither").sum()
            total = brand_first + generic_first

            col1, col2, col3 = st.columns(3)
            col1.metric("Brand First", brand_first)
            col2.metric("Generic First", generic_first)
            col3.metric("Neither", neither)

            if total > 0:
                ratio = brand_first / total
                st.metric("Brand-First Ratio", f"{ratio:.4f}", help="0.5 = no advantage")
                result = stats.binomtest(brand_first, total, p=0.5, alternative="greater")
                st.markdown(
                    f"**Binomial test** (H0: ratio=0.5): p={result.pvalue:.6f} — "
                    f"{'**Significant**' if result.pvalue < 0.05 else 'Not significant'} (alpha=0.05)"
                )

        # ── H3: Cross-Reference ──
        with tabs[2]:
            st.subheader("H3: Cross-Reference Asymmetry")
            st.markdown(
                "When querying about one drug, does the paired drug get mentioned? "
                "Measures directional cross-reference probability."
            )

            single = metrics_df[metrics_df["query_type"] == "single"]
            brand_queried = single[single["is_generic"].astype(str) == "False"]
            generic_queried = single[single["is_generic"].astype(str) == "True"]

            if not brand_queried.empty or not generic_queried.empty:
                col1, col2 = st.columns(2)
                if not generic_queried.empty:
                    p_bg = generic_queried["brand_mentioned"].mean()
                    col1.metric("P(brand | generic queried)", f"{p_bg:.4f}")
                if not brand_queried.empty:
                    p_gb = brand_queried["generic_mentioned"].mean()
                    col2.metric("P(generic | brand queried)", f"{p_gb:.4f}")

                a = int(generic_queried["brand_mentioned"].sum()) if not generic_queried.empty else 0
                b = len(generic_queried) - a
                c = int(brand_queried["generic_mentioned"].sum()) if not brand_queried.empty else 0
                d = len(brand_queried) - c

                if a + b > 0 and c + d > 0:
                    odds_ratio, p = stats.fisher_exact([[a, b], [c, d]])
                    st.markdown(
                        f"**Fisher's exact test:** OR={odds_ratio:.4f}, p={p:.6f} — "
                        f"{'**Significant**' if p < 0.05 else 'Not significant'} (alpha=0.05)"
                    )
            else:
                st.warning("No single-drug queries found.")

        # ── H4: Condition Visibility ──
        with tabs[3]:
            st.subheader("H4: Condition-First Query Visibility")
            st.markdown(
                "When no drug name is in the prompt, which drugs does the model mention?"
            )

            condition = metrics_df[metrics_df["query_type"] == "condition"]
            if condition.empty:
                st.warning("No condition-first responses found.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Responses", len(condition))
                col2.metric("Brand Mentioned",
                            f"{condition['brand_mentioned'].sum()} ({condition['brand_mentioned'].mean()*100:.0f}%)")
                col3.metric("Generic Mentioned",
                            f"{condition['generic_mentioned'].sum()} ({condition['generic_mentioned'].mean()*100:.0f}%)")

        # ── H5: Therapeutic Variation ──
        with tabs[4]:
            st.subheader("H5: Asymmetry Varies by Therapeutic Area")

            asym_file = ANALYSIS_DIR / "asymmetry_scores.csv"
            if asym_file.exists():
                asym_df = pd.read_csv(asym_file)
                st.dataframe(asym_df, width="stretch", hide_index=True)

                cas = asym_df["composite_asymmetry_score"].values
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean CAS", f"{np.mean(cas):.4f}")
                col2.metric("Median CAS", f"{np.median(cas):.4f}")
                col3.metric("Std CAS", f"{np.std(cas):.4f}")

                if len(cas) >= 5:
                    w, p = stats.wilcoxon(cas - 0.5)
                    st.markdown(
                        f"**Wilcoxon** (H0: median CAS=0.5): W={w:.4f}, p={p:.6f} — "
                        f"{'**Significant**' if p < 0.05 else 'Not significant'}"
                    )

                st.bar_chart(asym_df.set_index("pair_id")["composite_asymmetry_score"])
            else:
                st.warning("No asymmetry scores found. Run **Step 6** first.")

        # ── Summary ──
        with tabs[5]:
            st.subheader("Overall Summary")

            for label, filt in [("Condition", "condition"), ("Single-drug", "single"), ("Comparative", "comparative")]:
                sub = metrics_df[metrics_df["query_type"] == filt]
                if sub.empty:
                    continue

                st.markdown(f"#### {label} (n={len(sub)})")
                stats_rows = []
                for col_name in ["brand_mention_count", "generic_mention_count",
                                  "response_word_count", "safety_warnings_count"]:
                    if col_name in sub.columns:
                        stats_rows.append({
                            "Metric": col_name,
                            "Mean": f"{sub[col_name].mean():.2f}",
                            "Std": f"{sub[col_name].std():.2f}",
                            "Median": f"{sub[col_name].median():.1f}",
                        })
                st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)

            summary_data = metrics_df.groupby("query_type").agg({
                "brand_mentioned": ["sum", "mean"],
                "generic_mentioned": ["sum", "mean"],
                "brand_mention_count": ["mean", "std"],
                "generic_mention_count": ["mean", "std"],
                "response_word_count": ["mean", "std", "median"],
            })
            csv_summary = summary_data.to_csv().encode("utf-8")
            st.download_button("Download Summary CSV", csv_summary, "summary_report.csv", "text/csv")
