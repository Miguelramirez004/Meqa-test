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
from src.response_analyzer import analyze_response, analyze_all_responses
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
        "Creates standardized queries for each drug: "
        "9 factual/interaction types + 3 comparative types per pair."
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

        brand_q = sum(1 for q in queries if q["is_generic"] is False)
        generic_q = sum(1 for q in queries if q["is_generic"] is True)
        comp_q = sum(1 for q in queries if q["is_generic"] == "comparative")

        st.success(f"Generated **{len(queries)}** queries")
        col1, col2, col3 = st.columns(3)
        col1.metric("Brand", brand_q)
        col2.metric("Generic", generic_q)
        col3.metric("Comparative", comp_q)
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
            "Chunks to retrieve (top-k)", min_value=1, max_value=12, value=8,
            help="Number of prospecto chunks fed as context to the LLM.",
        )
    with col_cfg2:
        chunk_size = st.selectbox(
            "Chunk size (chars)", [300, 500, 750, 1000], index=1,
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
        resp_files = sorted(RESPONSES_DIR.glob("Q*.json"))[:20]
        rows = []
        for rf in resp_files:
            with open(rf, encoding="utf-8") as f:
                d = json.load(f)
            text = d.get("response_text", "")
            rows.append({
                "Query ID": d.get("query_id", rf.stem),
                "Drug": d.get("drug_name", ""),
                "Category": d.get("query_category", ""),
                "Words": len(text.split()) if text else 0,
                "Preview": text[:100] + "..." if len(text) > 100 else text,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if resp_count > 20:
            st.caption(f"Showing first 20 of {resp_count} responses.")


# ── Page 5: Analyze Responses ────────────────────────────────────────────────

elif page == "5. Analyze Responses":
    st.header("Analyze Responses (NLP Coding)")
    st.markdown(
        "Codes each MeQA response into quantitative metrics: "
        "word count, completeness, safety signals, professional referrals, "
        "and bioequivalence awareness."
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

        brand = metrics_df[metrics_df["is_generic"].astype(str) == "False"]
        generic = metrics_df[metrics_df["is_generic"].astype(str) == "True"]
        comp = metrics_df[metrics_df["is_generic"].astype(str) == "comparative"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Brand Responses", len(brand))
        col2.metric("Generic Responses", len(generic))
        col3.metric("Comparative Responses", len(comp))

        if not brand.empty and not generic.empty:
            st.markdown("#### Word Count Comparison")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Brand Words", f"{brand['word_count'].mean():.0f}")
            col2.metric("Avg Generic Words", f"{generic['word_count'].mean():.0f}")
            diff = brand['word_count'].mean() - generic['word_count'].mean()
            col3.metric("Difference", f"{diff:+.0f}")

            st.markdown("#### Completeness Score (0-6)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Brand", f"{brand['completeness_score'].mean():.2f}")
            col2.metric("Avg Generic", f"{generic['completeness_score'].mean():.2f}")
            diff_c = brand['completeness_score'].mean() - generic['completeness_score'].mean()
            col3.metric("Difference", f"{diff_c:+.2f}")

            # Chart: word count by drug type
            st.markdown("#### Word Count Distribution")
            chart_data = pd.DataFrame({
                "Brand": brand["word_count"].values,
            })
            chart_data2 = pd.DataFrame({
                "Generic": generic["word_count"].values,
            })

            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(brand.groupby("query_category")["word_count"].mean(), horizontal=True)
                st.caption("Brand — Avg words by category")
            with col2:
                st.bar_chart(generic.groupby("query_category")["word_count"].mean(), horizontal=True)
                st.caption("Generic — Avg words by category")

        # Download
        csv_data = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Metrics CSV", csv_data, "response_metrics.csv", "text/csv")


# ── Page 6: Statistical Analysis ─────────────────────────────────────────────

elif page == "6. Statistical Analysis":
    st.header("Statistical Hypothesis Testing")
    st.markdown(
        "Non-parametric tests for information asymmetry between "
        "brand-name and generic drug responses."
    )

    metrics_df = load_metrics_df()

    if metrics_df is None:
        st.warning("No coded metrics found. Run **Step 5** first.")
    else:
        st.info(f"Loaded **{len(metrics_df)}** coded responses.")

        import numpy as np
        from scipy import stats

        tabs = st.tabs(["H1: Completeness", "H2: Accuracy", "H3: Comparative Bias", "H4: Therapeutic Variation", "Summary"])

        # ── H1 ──
        with tabs[0]:
            st.subheader("H1: Brand responses are more complete than generic")
            st.markdown("**Test:** Wilcoxon signed-rank (paired, non-parametric)")

            factual = metrics_df[metrics_df["query_type"] == "factual"].copy()
            brand = factual[factual["is_generic"].astype(str) == "False"]
            generic = factual[factual["is_generic"].astype(str) == "True"]

            brand_agg = brand.groupby(["pair_id", "query_category"]).agg(
                word_count=("word_count", "mean"),
                completeness=("completeness_score", "mean"),
                ae_named=("adverse_effects_named", "mean"),
            ).reset_index()

            generic_agg = generic.groupby(["pair_id", "query_category"]).agg(
                word_count=("word_count", "mean"),
                completeness=("completeness_score", "mean"),
                ae_named=("adverse_effects_named", "mean"),
            ).reset_index()

            merged = brand_agg.merge(
                generic_agg, on=["pair_id", "query_category"], suffixes=("_brand", "_generic")
            )

            results = []
            for metric in ["word_count", "completeness", "ae_named"]:
                b = merged[f"{metric}_brand"].values
                g = merged[f"{metric}_generic"].values
                if len(b) >= 5:
                    w, p = stats.wilcoxon(b, g)
                    d = np.median(b - g)
                    results.append({
                        "Metric": metric,
                        "Brand Mean": f"{np.mean(b):.2f}",
                        "Generic Mean": f"{np.mean(g):.2f}",
                        "Wilcoxon W": f"{w:.4f}",
                        "p-value": f"{p:.6f}",
                        "Median Diff": f"{d:+.2f}",
                        "Significant": "Yes" if p < 0.05 else "No",
                    })

            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            else:
                st.warning("Insufficient paired data for Wilcoxon test (need >= 5 pairs).")

        # ── H2 ──
        with tabs[1]:
            st.subheader("H2: Differential Accuracy")
            st.markdown(
                "Accuracy requires manual coding against ground truth prospectos. "
                "Download the template, fill in scores, and re-upload."
            )

            template = metrics_df[["query_id", "pair_id", "drug_name", "is_generic",
                                    "query_category"]].copy()
            template["accuracy_score"] = ""
            template["error_type"] = ""
            template["coder"] = ""
            template["notes"] = ""

            csv = template.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Accuracy Coding Template",
                csv, "accuracy_coding_template.csv", "text/csv"
            )

            st.markdown("---")
            st.markdown(
                "**Scoring guide:**\n"
                "- 1 = Accurate\n"
                "- 2 = Partially accurate\n"
                "- 3 = Omission\n"
                "- 4 = Error/hallucination"
            )

        # ── H3 ──
        with tabs[2]:
            st.subheader("H3: Framing Bias in Comparative Queries")

            comp = metrics_df[metrics_df["query_type"] == "comparative"].copy()
            if comp.empty:
                st.warning("No comparative responses found.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Comparative", len(comp))
                col2.metric("Mentions Bioequivalence",
                            f"{comp['mentions_bioequivalence'].sum()} ({comp['mentions_bioequivalence'].mean()*100:.0f}%)")
                col3.metric("Avg Word Count", f"{comp['word_count'].mean():.0f}")

                st.markdown("#### By Category")
                cat_stats = comp.groupby("query_category").agg(
                    n=("query_id", "count"),
                    avg_words=("word_count", "mean"),
                    bioequiv_pct=("mentions_bioequivalence", "mean"),
                    professional_pct=("mentions_professional", "mean"),
                ).reset_index()
                cat_stats["bioequiv_pct"] = (cat_stats["bioequiv_pct"] * 100).round(0).astype(int).astype(str) + "%"
                cat_stats["professional_pct"] = (cat_stats["professional_pct"] * 100).round(0).astype(int).astype(str) + "%"
                st.dataframe(cat_stats, use_container_width=True, hide_index=True)

        # ── H4 ──
        with tabs[3]:
            st.subheader("H4: Asymmetry Varies by Therapeutic Area")

            factual = metrics_df[metrics_df["query_type"] == "factual"].copy()
            pair_ids = factual["pair_id"].unique()
            rows = []

            for pid in pair_ids:
                p = factual[factual["pair_id"] == pid]
                b_wc = p[p["is_generic"].astype(str) == "False"]["word_count"].mean()
                g_wc = p[p["is_generic"].astype(str) == "True"]["word_count"].mean()
                if not (np.isnan(b_wc) or np.isnan(g_wc)):
                    rows.append({
                        "Pair ID": pid,
                        "Brand Avg Words": round(b_wc, 1),
                        "Generic Avg Words": round(g_wc, 1),
                        "Asymmetry (B-G)": round(b_wc - g_wc, 1),
                    })

            if rows:
                asym_df = pd.DataFrame(rows)
                st.dataframe(asym_df, use_container_width=True, hide_index=True)

                asym_vals = asym_df["Asymmetry (B-G)"].values
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Asymmetry", f"{np.mean(asym_vals):+.1f}")
                col2.metric("Median Asymmetry", f"{np.median(asym_vals):+.1f}")
                col3.metric("Std Dev", f"{np.std(asym_vals):.1f}")

                if len(asym_vals) >= 5:
                    w, p = stats.wilcoxon(asym_vals)
                    st.markdown(
                        f"**One-sample Wilcoxon** (H0: median=0): "
                        f"W={w:.4f}, p={p:.6f} — "
                        f"{'**Significant**' if p < 0.05 else 'Not significant'} (alpha=0.05)"
                    )

                st.bar_chart(asym_df.set_index("Pair ID")["Asymmetry (B-G)"])
            else:
                st.warning("Insufficient data to compute asymmetry by pair.")

        # ── Summary ──
        with tabs[4]:
            st.subheader("Overall Summary")

            for label, filt in [("Brand", "False"), ("Generic", "True"), ("Comparative", "comparative")]:
                sub = metrics_df[metrics_df["is_generic"].astype(str) == filt]
                if sub.empty:
                    continue

                st.markdown(f"#### {label} (n={len(sub)})")
                stats_rows = []
                for col in ["word_count", "completeness_score", "safety_warnings_count", "adverse_effects_named"]:
                    stats_rows.append({
                        "Metric": col,
                        "Mean": f"{sub[col].mean():.2f}",
                        "Std": f"{sub[col].std():.2f}",
                        "Median": f"{sub[col].median():.1f}",
                        "Min": f"{sub[col].min():.0f}",
                        "Max": f"{sub[col].max():.0f}",
                    })
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

            # Download full summary
            summary_data = metrics_df.groupby("is_generic").agg({
                "word_count": ["mean", "std", "median"],
                "completeness_score": ["mean", "std"],
                "safety_warnings_count": ["mean", "std"],
                "adverse_effects_named": ["mean", "std"],
            })
            csv = summary_data.to_csv().encode("utf-8")
            st.download_button("Download Summary CSV", csv, "summary_report.csv", "text/csv")
