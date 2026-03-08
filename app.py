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
)
from src.drug_pairs import get_offline_pairs, OFFLINE_PAIRS
from src.query_generator import generate_queries, save_queries
from src.response_analyzer import analyze_response, analyze_all_responses, compute_asymmetry_scores
from src.cima_client import CIMAClient
from src.rag_engine import collect_all_responses as rag_collect

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
        "4. Collect Responses",
        "5. Analyze Responses",
        "6. Statistical Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Doctoral research — GEO asymmetry in MeQA (AEMPS)")


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [DATA_DIR, QUERIES_DIR, RESPONSES_DIR, ANALYSIS_DIR, PROSPECTOS_DIR, PAIRS_DIR]:
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
        st.dataframe(existing_df.head(20), use_container_width=True, hide_index=True)

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
        st.dataframe(pd.DataFrame({"Prospecto": names}), use_container_width=True, hide_index=True)

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

                progress.empty()
                st.success(f"Downloaded prospectos for {total_meds} medications.")
                st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")


# ── Page 4: Collect Responses ────────────────────────────────────────────────

elif page == "4. Collect Responses":
    st.header("Collect MeQA Responses")
    st.markdown(
        "Generate responses automatically using a RAG pipeline "
        "(prospectos + LLM), or upload manually collected responses."
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

    # ── Automated RAG collection ──
    st.subheader("Automated Collection (MeQA RAG Pipeline)")
    st.markdown(
        "Simulates the [MeQA architecture](https://arxiv.org/abs/2111.02760) "
        "(AEMPS, Santamaría 2021) using LangChain:\n\n"
        "**Block 1** — Question Processing: normalisation + NER + section prediction\n\n"
        "**Block 2** — Hybrid Retrieval: BM25 (sparse, like TF-IDF) + FAISS (dense) "
        "with section-aware filtering\n\n"
        "**Block 3** — Answer Extraction: context assembly + LLM generation"
    )

    # Check for API key
    api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Set `OPENAI_API_KEY` in .streamlit/secrets.toml or enter here.",
        )

    prospecto_count = len(list(PROSPECTOS_DIR.glob("*.json"))) - (
        1 if (PROSPECTOS_DIR / ".gitkeep").exists() else 0
    )

    # Pipeline configuration
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        model = st.selectbox(
            "LLM Model (generation)",
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
            index=0,
            help="gpt-4o-mini is recommended (fast and cost-effective).",
        )
        top_k = st.slider(
            "Chunks to retrieve (top-k)", min_value=1, max_value=20, value=12,
            help="Number of prospecto chunks fed as context to the LLM.",
        )
    with col_cfg2:
        chunk_size = st.selectbox(
            "Chunk size (chars)", [300, 500, 600, 750, 1000], index=2,
            help="Smaller chunks = more precise retrieval; larger = more context.",
        )
        bm25_weight = st.slider(
            "BM25 weight (sparse vs dense)", 0.0, 1.0, 0.4, 0.1,
            help="0.0 = dense only (FAISS), 1.0 = sparse only (BM25). "
                 "0.4 recommended (like MeQA's TF-IDF + VSM/LSI blend).",
        )

    skip_existing = st.checkbox("Skip already-collected responses", value=True)

    if prospecto_count > 0:
        st.info(f"**{prospecto_count}** prospectos available as RAG context.")
    else:
        st.warning(
            "No prospectos downloaded yet. The model will answer from general "
            "pharmaceutical knowledge. For better results, run **Step 3** first."
        )

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
                progress_bar.progress(current / total)
                status_text.text(f"Processing query {current}/{total}...")

            try:
                status_text.text("Building hybrid retriever (BM25 + FAISS)...")
                collected = rag_collect(
                    queries=queries_list,
                    api_key=api_key,
                    model=model,
                    prospectos_dir=PROSPECTOS_DIR,
                    responses_dir=RESPONSES_DIR,
                    chunk_size=chunk_size,
                    chunk_overlap=80,
                    top_k=top_k,
                    bm25_weight=bm25_weight,
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
            st.dataframe(resp_df.head(10), use_container_width=True, hide_index=True)

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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if resp_count > 20:
            st.caption(f"Showing first 20 of {resp_count} responses.")


# ── Page 5: Analyze Responses ────────────────────────────────────────────────

elif page == "5. Analyze Responses":
    st.header("Analyze Responses (Drug Mention Audit)")
    st.markdown(
        "Codes each response into mention-based metrics: "
        "brand/generic mention counts, first-mention position, "
        "cross-reference detection, and computes asymmetry scores per pair."
    )

    resp_count = load_responses_count()
    metrics_df = load_metrics_df()

    if resp_count == 0:
        st.warning("No responses found. Go to **Step 4** to upload MeQA responses first.")
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
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Brand Mentioned",
                     f"{metrics_df['brand_mentioned'].sum()}/{len(metrics_df)}")
        col2.metric("Generic Mentioned",
                     f"{metrics_df['generic_mentioned'].sum()}/{len(metrics_df)}")
        total_brand = metrics_df["brand_mention_count"].sum()
        total_generic = metrics_df["generic_mention_count"].sum()
        col3.metric("Total Brand Mentions", int(total_brand))
        col4.metric("Total Generic Mentions", int(total_generic))

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
        mention_by_type = metrics_df.groupby("query_type").agg(
            n=("query_id", "count"),
            brand_mentions=("brand_mention_count", "sum"),
            generic_mentions=("generic_mention_count", "sum"),
            brand_pct=("brand_mentioned", "mean"),
            generic_pct=("generic_mentioned", "mean"),
        ).reset_index()
        mention_by_type["brand_pct"] = (mention_by_type["brand_pct"] * 100).round(0).astype(int).astype(str) + "%"
        mention_by_type["generic_pct"] = (mention_by_type["generic_pct"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(mention_by_type, use_container_width=True, hide_index=True)

        # Asymmetry scores
        asym_file = ANALYSIS_DIR / "asymmetry_scores.csv"
        if asym_file.exists():
            st.markdown("---")
            st.subheader("Asymmetry Scores (per drug pair)")
            asym_df = pd.read_csv(asym_file)
            st.dataframe(asym_df, use_container_width=True, hide_index=True)

            if not asym_df.empty:
                st.bar_chart(asym_df.set_index("pair_id")["composite_asymmetry_score"])

        # Download
        csv_data = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Metrics CSV", csv_data, "response_metrics.csv", "text/csv")


# ── Page 6: Statistical Analysis ─────────────────────────────────────────────

elif page == "6. Statistical Analysis":
    st.header("Statistical Hypothesis Testing")
    st.markdown(
        "Non-parametric tests for drug mention asymmetry between "
        "brand-name and generic drugs."
    )

    metrics_df = load_metrics_df()

    if metrics_df is None:
        st.warning("No coded metrics found. Run **Step 5** first.")
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
                st.dataframe(agg, use_container_width=True, hide_index=True)

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
                st.dataframe(asym_df, use_container_width=True, hide_index=True)

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
                st.warning("No asymmetry scores found. Run **Step 5** first.")

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
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

            summary_data = metrics_df.groupby("query_type").agg({
                "brand_mentioned": ["sum", "mean"],
                "generic_mentioned": ["sum", "mean"],
                "brand_mention_count": ["mean", "std"],
                "generic_mention_count": ["mean", "std"],
                "response_word_count": ["mean", "std", "median"],
            })
            csv_summary = summary_data.to_csv().encode("utf-8")
            st.download_button("Download Summary CSV", csv_summary, "summary_report.csv", "text/csv")
