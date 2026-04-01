"""Prompt Experiment — Compare system prompts for drug mention behaviour.

This page runs an A/B experiment between the original (citation-heavy) system
prompt and a configurable loose prompt.  Retrieval is shared so chunk counts
are identical between conditions.

Added on the experiment branch only — does NOT exist on main.
"""

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, RESPONSES_DIR, PROSPECTOS_DIR, LEAFLETS_DIR
from src.rag_engine import (
    _SYSTEM_PROMPT as ORIGINAL_SYSTEM_PROMPT,
    _USER_TEMPLATE,
    _USER_TEMPLATE_NO_CONTEXT,
    _format_context,
    _bootstrap_index,
    retrieve_chunks,
    normalise_query,
    extract_entities,
    predict_sections,
    LEAFLET_SECTIONS,
    collect_all_responses,
)
from src.vector_store import MeQAVectorStore
from src.response_analyzer import (
    BRAND_NAMES_BY_INN,
    GENERIC_LABS,
    _extract_inn,
    _get_brand_names,
)

import re
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Prompt Experiment", page_icon="🔬", layout="wide")
st.title("🔬 System Prompt Experiment")
st.caption("Compare drug mention rates between original and loose system prompts")

# ── Paths ────────────────────────────────────────────────────────────────────

QUERIES_PATH = DATA_DIR / "queries" / "query_battery.json"
RESPONSES_LOOSE_DIR = DATA_DIR / "responses_loose"

# ── Default loose prompts ────────────────────────────────────────────────────

LOOSE_PRESETS = {
    "Pharmacist (moderate)": (
        "Eres un farmacéutico comunitario experimentado. Responde a la pregunta del "
        "paciente utilizando ÚNICAMENTE la información proporcionada en los fragmentos "
        "del prospecto oficial. Explica de forma clara y cercana, como lo haría un "
        "farmacéutico en la oficina de farmacia. Si la información no aparece en los "
        "fragmentos proporcionados, indícalo claramente. Responde de forma concisa "
        "en español."
    ),
    "Pharmacist (very loose)": (
        "Eres un farmacéutico comunitario. Responde a la pregunta del paciente "
        "basándote en los fragmentos del prospecto proporcionados. Usa un lenguaje "
        "sencillo y natural, sin necesidad de citar nombres comerciales ni secciones "
        "específicas del prospecto. Simplemente explica la información relevante "
        "de forma clara. Si no encuentras la respuesta, dilo. Responde en español."
    ),
    "Neutral assistant": (
        "Eres un asistente de información farmacéutica. Responde a la pregunta "
        "utilizando únicamente los fragmentos proporcionados. No menciones nombres "
        "comerciales a menos que sea estrictamente necesario para responder la pregunta. "
        "Usa el nombre del principio activo cuando te refieras al medicamento. "
        "Responde de forma concisa en español."
    ),
    "Custom": "",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.header("Experiment Config")

# API key
api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Set OPENAI_API_KEY in .streamlit/secrets.toml or enter here.",
    )

# Query selection
if not QUERIES_PATH.exists():
    st.error(f"Query battery not found at {QUERIES_PATH}")
    st.stop()

with open(QUERIES_PATH, encoding="utf-8") as f:
    all_queries = json.load(f)

# Group by pair_id
pair_ids = sorted(set(q.get("pair_id", "") for q in all_queries))
query_types = sorted(set(q.get("is_generic", "") for q in all_queries if q.get("is_generic") != "comparative"),
                     key=lambda x: str(x))

st.sidebar.subheader("Query Selection")
selected_pairs = st.sidebar.multiselect(
    "Pair IDs", pair_ids, default=pair_ids,
    help="Select which drug pairs to include",
)

include_comparative = st.sidebar.checkbox("Include comparative queries", value=False)

# Filter queries
filtered = [
    q for q in all_queries
    if q.get("pair_id", "") in selected_pairs
    and (include_comparative or q.get("is_generic") != "comparative")
]

max_queries = len(filtered)
n_queries = st.sidebar.slider(
    "Number of queries",
    min_value=10, max_value=max_queries, value=min(max_queries, 261),
    step=1,
    help=f"Available: {max_queries} queries matching your filters",
)

seed = st.sidebar.number_input("Random seed", value=42, step=1)

# Prompt selection
st.sidebar.subheader("Loose Prompt")
preset = st.sidebar.selectbox("Preset", list(LOOSE_PRESETS.keys()))

if preset == "Custom":
    loose_prompt = st.sidebar.text_area(
        "Custom system prompt",
        value=LOOSE_PRESETS["Pharmacist (moderate)"],
        height=200,
    )
else:
    loose_prompt = LOOSE_PRESETS[preset]
    st.sidebar.text_area("Prompt preview", value=loose_prompt, height=150, disabled=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original prompt")
    st.code(ORIGINAL_SYSTEM_PROMPT, language=None)
with col2:
    st.subheader("Loose prompt")
    st.code(loose_prompt, language=None)

st.markdown("---")

# Sample queries
rng = random.Random(seed)
sampled = rng.sample(filtered, min(n_queries, len(filtered)))
st.info(f"**{len(sampled)} queries** selected across **{len(set(q['pair_id'] for q in sampled))} pair_ids**")


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION HELPERS (same as compare_results.py)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_mentions(response_text: str, query: dict) -> dict:
    principio = query.get("principio_activo", "")
    inn = _extract_inn(principio)
    drug_name = query.get("drug_name", "")
    brand_names = _get_brand_names(inn, pair_brand=drug_name)

    brand_found = False
    for b in brand_names:
        if len(b) < 3:
            continue
        if re.search(r"\b" + re.escape(b) + r"\b", response_text, re.IGNORECASE):
            brand_found = True
            break

    generic_found = False
    for lab in GENERIC_LABS:
        pattern = f"{inn} {lab}"
        if len(pattern) < 5:
            continue
        if re.search(r"\b" + re.escape(pattern) + r"\b", response_text, re.IGNORECASE):
            generic_found = True
            break

    inn_found = bool(inn and len(inn) >= 3 and
                     re.search(r"\b" + re.escape(inn) + r"\b", response_text, re.IGNORECASE))

    return {
        "brand": brand_found,
        "generic": generic_found,
        "inn": inn_found,
        "both": brand_found and generic_found,
        "neither": not brand_found and not generic_found and not inn_found,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

if st.button("▶ Run Experiment", type="primary", disabled=not api_key):
    if not api_key:
        st.error("Please provide an OpenAI API key.")
        st.stop()

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_LOOSE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise vector store
    with st.spinner("Initialising vector store..."):
        store = MeQAVectorStore(openai_api_key=api_key)
        if store.count == 0:
            st.info("Vector store empty — bootstrapping from prospectos...")
            _bootstrap_index(
                queries=all_queries,
                store=store,
                prospectos_dir=PROSPECTOS_DIR,
                leaflets_dir=LEAFLETS_DIR,
            )
        st.success(f"Vector store ready: {store.count} chunks")

    client = OpenAI(api_key=api_key)
    model = "gpt-4o-mini"
    top_k = 5

    progress = st.progress(0, text="Running experiment...")
    results_original = []
    results_loose = []

    for i, query in enumerate(sampled):
        qid = query.get("query_id", f"Q{i+1:04d}")
        query_text = query.get("query_text", "")
        drug_name = query.get("drug_name", "")
        pair_id = query.get("pair_id", "")
        is_comparative = query.get("is_generic") == "comparative"
        entities = extract_entities(query)

        # ── SHARED RETRIEVAL ──
        filter_drug = None if is_comparative else drug_name
        retrieved = retrieve_chunks(
            query_text=query_text,
            store=store,
            pair_id=pair_id,
            drug_name=filter_drug,
            top_k=top_k,
        )
        context_str = _format_context(retrieved)
        chunk_count = len(retrieved)

        # Build retrieval details
        retrieval_details = []
        sections_retrieved = set()
        for r in retrieved:
            meta = r.get("metadata", {})
            sections_retrieved.add(meta.get("section_number", 0))
            retrieval_details.append({
                "section_number": meta.get("section_number", 0),
                "section_name": meta.get("section_name", ""),
                "similarity": round(r.get("similarity", 0.0), 4),
                "chunk_index": meta.get("chunk_index", 0),
                "drug_name": meta.get("drug_name", ""),
            })

        # ── Helper to build user message ──
        if context_str.strip():
            user_msg = _USER_TEMPLATE.format(context=context_str, query=query_text)
        else:
            user_msg = _USER_TEMPLATE_NO_CONTEXT.format(
                drug=drug_name,
                ingredient=entities.get("active_ingredient", ""),
                query=query_text,
            )

        # ── CONDITION A: Original prompt ──
        fpath_orig = RESPONSES_DIR / f"{qid}.json"
        if fpath_orig.exists():
            with open(fpath_orig, encoding="utf-8") as f:
                resp_orig_text = json.load(f).get("response_text", "")
        else:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=1000,
                messages=[
                    {"role": "system", "content": ORIGINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            resp_orig_text = resp.choices[0].message.content.strip()
            _save_response(fpath_orig, qid, query, resp_orig_text,
                           retrieval_details, sections_retrieved, context_str,
                           model, top_k, "original", store.count)

        # ── CONDITION B: Loose prompt ──
        fpath_loose = RESPONSES_LOOSE_DIR / f"{qid}.json"
        if fpath_loose.exists():
            with open(fpath_loose, encoding="utf-8") as f:
                resp_loose_text = json.load(f).get("response_text", "")
        else:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=1000,
                messages=[
                    {"role": "system", "content": loose_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            resp_loose_text = resp.choices[0].message.content.strip()
            _save_response(fpath_loose, qid, query, resp_loose_text,
                           retrieval_details, sections_retrieved, context_str,
                           model, top_k, "loose", store.count)

        # Detect mentions
        m_orig = detect_mentions(resp_orig_text, query)
        m_loose = detect_mentions(resp_loose_text, query)

        results_original.append({
            "query_id": qid, "pair_id": pair_id, "drug_name": drug_name,
            "is_generic": query.get("is_generic", ""), "chunks": chunk_count,
            **{f"orig_{k}": v for k, v in m_orig.items()},
        })
        results_loose.append({
            "query_id": qid, "pair_id": pair_id, "drug_name": drug_name,
            "is_generic": query.get("is_generic", ""), "chunks": chunk_count,
            **{f"loose_{k}": v for k, v in m_loose.items()},
        })

        pct = (i + 1) / len(sampled)
        progress.progress(pct, text=f"[{i+1}/{len(sampled)}] {qid} — {chunk_count} chunks")

        if i < len(sampled) - 1:
            time.sleep(0.3)

    progress.progress(1.0, text="Done!")

    # ── Merge results ──
    df_orig = pd.DataFrame(results_original)
    df_loose = pd.DataFrame(results_loose)
    df = df_orig.merge(df_loose[["query_id"] + [c for c in df_loose.columns if c.startswith("loose_")]],
                       on="query_id")

    st.session_state["experiment_df"] = df
    st.session_state["experiment_done"] = True


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("experiment_done"):
    df = st.session_state["experiment_df"]
    n = len(df)

    st.markdown("---")
    st.subheader("Results")

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)

    orig_brand = df["orig_brand"].sum()
    loose_brand = df["loose_brand"].sum()
    col1.metric("Brand mention rate",
                f"{loose_brand/n:.1%}",
                f"{(loose_brand - orig_brand)/n:+.1%} vs original",
                help=f"Original: {orig_brand}/{n} ({orig_brand/n:.1%})")

    orig_inn = df["orig_inn"].sum()
    loose_inn = df["loose_inn"].sum()
    col2.metric("INN mention rate",
                f"{loose_inn/n:.1%}",
                f"{(loose_inn - orig_inn)/n:+.1%} vs original",
                help=f"Original: {orig_inn}/{n} ({orig_inn/n:.1%})")

    orig_generic = df["orig_generic"].sum()
    loose_generic = df["loose_generic"].sum()
    col3.metric("Generic mention rate",
                f"{loose_generic/n:.1%}",
                f"{(loose_generic - orig_generic)/n:+.1%} vs original",
                help=f"Original: {orig_generic}/{n} ({orig_generic/n:.1%})")

    orig_neither = df["orig_neither"].sum()
    loose_neither = df["loose_neither"].sum()
    col4.metric("Neither mentioned",
                f"{loose_neither/n:.1%}",
                f"{(loose_neither - orig_neither)/n:+.1%} vs original",
                delta_color="inverse",
                help=f"Original: {orig_neither}/{n} ({orig_neither/n:.1%})")

    # ── Summary table ──
    st.markdown("### Overall Comparison")
    summary = pd.DataFrame({
        "Metric": ["Brand mention", "INN mention", "Generic product mention",
                    "Both (brand+generic)", "Neither"],
        "Original": [
            f"{df['orig_brand'].sum()}/{n} ({df['orig_brand'].mean():.1%})",
            f"{df['orig_inn'].sum()}/{n} ({df['orig_inn'].mean():.1%})",
            f"{df['orig_generic'].sum()}/{n} ({df['orig_generic'].mean():.1%})",
            f"{df['orig_both'].sum()}/{n} ({df['orig_both'].mean():.1%})",
            f"{df['orig_neither'].sum()}/{n} ({df['orig_neither'].mean():.1%})",
        ],
        "Loose": [
            f"{df['loose_brand'].sum()}/{n} ({df['loose_brand'].mean():.1%})",
            f"{df['loose_inn'].sum()}/{n} ({df['loose_inn'].mean():.1%})",
            f"{df['loose_generic'].sum()}/{n} ({df['loose_generic'].mean():.1%})",
            f"{df['loose_both'].sum()}/{n} ({df['loose_both'].mean():.1%})",
            f"{df['loose_neither'].sum()}/{n} ({df['loose_neither'].mean():.1%})",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Breakdown by pair_id ──
    st.markdown("### Breakdown by Pair ID")
    pair_summary = []
    for pid in sorted(df["pair_id"].unique()):
        sub = df[df["pair_id"] == pid]
        m = len(sub)
        pair_summary.append({
            "pair_id": pid,
            "n": m,
            "O.brand": f"{sub['orig_brand'].sum()}/{m}",
            "O.inn": f"{sub['orig_inn'].sum()}/{m}",
            "O.generic": f"{sub['orig_generic'].sum()}/{m}",
            "L.brand": f"{sub['loose_brand'].sum()}/{m}",
            "L.inn": f"{sub['loose_inn'].sum()}/{m}",
            "L.generic": f"{sub['loose_generic'].sum()}/{m}",
            "O.neither": f"{sub['orig_neither'].sum()}/{m}",
            "L.neither": f"{sub['loose_neither'].sum()}/{m}",
        })
    st.dataframe(pd.DataFrame(pair_summary), use_container_width=True, hide_index=True)

    # ── Breakdown by is_generic ──
    st.markdown("### Breakdown by Query Drug Type")
    type_summary = []
    for gtype in sorted(df["is_generic"].unique(), key=str):
        sub = df[df["is_generic"] == gtype]
        m = len(sub)
        label = {True: "Generic drug", False: "Brand drug", "comparative": "Comparative"}.get(gtype, str(gtype))
        type_summary.append({
            "query_type": label,
            "n": m,
            "O.brand": f"{sub['orig_brand'].sum()}/{m} ({sub['orig_brand'].mean():.0%})",
            "L.brand": f"{sub['loose_brand'].sum()}/{m} ({sub['loose_brand'].mean():.0%})",
            "O.inn": f"{sub['orig_inn'].sum()}/{m} ({sub['orig_inn'].mean():.0%})",
            "L.inn": f"{sub['loose_inn'].sum()}/{m} ({sub['loose_inn'].mean():.0%})",
        })
    st.dataframe(pd.DataFrame(type_summary), use_container_width=True, hide_index=True)

    # ── Download ──
    st.markdown("### Export")
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download results CSV",
        csv, "prompt_experiment_results.csv", "text/csv",
    )

    # ── Raw data expander ──
    with st.expander("View raw data"):
        st.dataframe(df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER (identical to experiments/loose_prompt_test.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _save_response(
    fpath, qid, query, response_text, retrieval_details,
    sections_retrieved, context_str, model, top_k, condition, doc_count,
):
    normalised = normalise_query(query.get("query_text", ""))
    entities = extract_entities(query)
    predicted_secs = predict_sections(query.get("query_category", ""))

    prompt_text = ORIGINAL_SYSTEM_PROMPT if condition == "original" else loose_prompt
    full = f"System: {prompt_text}\n\nUser: {context_str[:200]}"
    prompt_hash = hashlib.md5(full.encode()).hexdigest()
    response_hash = hashlib.md5(response_text.encode()).hexdigest() if response_text else ""

    data = {
        "query_id": qid,
        "pair_id": query.get("pair_id", ""),
        "principio_activo": query.get("principio_activo", ""),
        "grupo_terapeutico": query.get("grupo_terapeutico", ""),
        "drug_name": query.get("drug_name", ""),
        "is_generic": query.get("is_generic", ""),
        "query_type": query.get("query_type", ""),
        "query_category": query.get("query_category", ""),
        "query_text": query.get("query_text", ""),
        "response_text": response_text,
        "response_word_count": len(response_text.split()) if response_text else 0,
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "meqa_metadata": {
            "model": model,
            "temperature": 0.0,
            "normalised_query": normalised,
            "entities": entities,
            "predicted_sections": predicted_secs,
            "predicted_section_names": [LEAFLET_SECTIONS.get(s, "") for s in predicted_secs],
            "chunks_retrieved": len(retrieval_details),
            "total_doc_chunks": doc_count,
            "context_available": bool(context_str),
            "context_length": len(context_str),
            "retrieval_method": "openai_embedding_cosine_numpy",
            "retrieval_details": retrieval_details,
            "sections_retrieved": sorted(sections_retrieved),
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "experiment_condition": condition,
        },
        "notes": f"MeQA-RAG experiment ({condition}) (model={model}, temp=0.0, top_k={top_k})",
    }
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
