#!/usr/bin/env python3
"""Experiment: Loose vs Original system prompt — drug mention comparison.

Runs the same RAG retrieval pipeline for both prompt conditions on 30
sampled queries (6 per pair_id x 5 pair_ids).  Retrieval is shared so
chunk counts are identical between conditions.

Results:
  - data/responses/      <- original prompt (control)
  - data/responses_loose/ <- loose prompt   (experiment)

Usage:
  python experiments/loose_prompt_test.py           # real mode (needs OPENAI_API_KEY)
  python experiments/loose_prompt_test.py --mock    # mock mode (no API key needed)
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Make src/ importable without modifying any src/ files ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RESPONSES_DIR, PROSPECTOS_DIR, LEAFLETS_DIR
from src.rag_engine import (
    _SYSTEM_PROMPT as ORIGINAL_SYSTEM_PROMPT,
    _USER_TEMPLATE,
    _USER_TEMPLATE_NO_CONTEXT,
    _format_context,
    retrieve_chunks,
    normalise_query,
    extract_entities,
    predict_sections,
    LEAFLET_SECTIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

# Loose prompt: grounds in retrieved chunks but does NOT demand explicit drug
# name / section citations — asks for a natural pharmacist-style response.
_SYSTEM_PROMPT_LOOSE = (
    "Eres un farmacéutico comunitario experimentado. Responde a la pregunta del "
    "paciente utilizando ÚNICAMENTE la información proporcionada en los fragmentos "
    "del prospecto oficial. Explica de forma clara y cercana, como lo haría un "
    "farmacéutico en la oficina de farmacia. Si la información no aparece en los "
    "fragmentos proporcionados, indícalo claramente. Responde de forma concisa "
    "en español."
)


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

QUERIES_PATH = DATA_DIR / "queries" / "query_battery.json"
RESPONSES_LOOSE_DIR = DATA_DIR / "responses_loose"
N_QUERIES = 30
N_PAIR_IDS = 5
QUERIES_PER_PAIR = N_QUERIES // N_PAIR_IDS  # 6


def sample_queries(queries: list[dict], seed: int = 42) -> list[dict]:
    """Sample 30 queries: 6 from each of 5 pair_ids for coverage."""
    rng = random.Random(seed)

    by_pair: dict[str, list[dict]] = {}
    for q in queries:
        pid = q.get("pair_id", "")
        by_pair.setdefault(pid, []).append(q)

    # Pick 5 pair_ids with the most queries (deterministic after sort)
    pair_ids = sorted(by_pair.keys(), key=lambda p: (-len(by_pair[p]), p))
    selected_pairs = pair_ids[:N_PAIR_IDS]

    sampled: list[dict] = []
    for pid in selected_pairs:
        pool = by_pair[pid]
        chosen = rng.sample(pool, min(QUERIES_PER_PAIR, len(pool)))
        sampled.extend(chosen)

    logger.info(
        "Sampled %d queries from pair_ids: %s",
        len(sampled),
        [p for p in selected_pairs],
    )
    return sampled


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION (reuses retrieval, varies only the system prompt)
# ═══════════════════════════════════════════════════════════════════════════════


def _generate(
    query_text: str,
    context: str,
    system_prompt: str,
    client,
    model: str = "gpt-4o-mini",
    drug_name: str = "",
    ingredient: str = "",
) -> str:
    """Generate a response using a given system prompt."""
    if context.strip():
        user_msg = _USER_TEMPLATE.format(context=context, query=query_text)
    else:
        user_msg = _USER_TEMPLATE_NO_CONTEXT.format(
            drug=drug_name, ingredient=ingredient, query=query_text,
        )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK GENERATION (no API key required — simulates realistic responses)
# ═══════════════════════════════════════════════════════════════════════════════

# Templates that mimic what the real model produces under each prompt condition.
# Original prompt → explicit "Según el prospecto de DRUG, sección N" citations.
# Loose prompt → natural pharmacist language, may or may not cite drug by name.

_MOCK_TEMPLATES_ORIGINAL = {
    "indication": (
        "Según el prospecto de {drug}, sección 1 (Qué es y para qué se utiliza), "
        "{drug} está indicado para el tratamiento de {condition}. "
        "El principio activo es {inn}, que pertenece al grupo de {group}."
    ),
    "adverse_effects": (
        "Según el prospecto de {drug}, sección 4 (Posibles efectos adversos), "
        "los efectos adversos frecuentes incluyen dolor de cabeza, náuseas y diarrea. "
        "Si experimenta alguno de estos efectos, consulte a su médico o farmacéutico."
    ),
    "dosage": (
        "Según el prospecto de {drug}, sección 3 (Cómo tomar), "
        "la dosis habitual para adultos es la prescrita por su médico. "
        "Tome el medicamento {drug} con un vaso de agua."
    ),
    "pregnancy": (
        "Según el prospecto de {drug}, sección 2, "
        "si está embarazada o en periodo de lactancia, consulte a su médico "
        "antes de tomar {drug}. El principio activo {inn} puede afectar al feto."
    ),
    "contraindications": (
        "Según el prospecto de {drug}, sección 2 (Qué necesita saber antes de tomar), "
        "no tome {drug} si es alérgico al {inn} o a alguno de los demás componentes."
    ),
}

_MOCK_TEMPLATES_LOOSE = {
    "indication": (
        "Este medicamento se utiliza para el tratamiento de {condition}. "
        "Contiene {inn}, que es un {group} muy utilizado. "
        "Es importante seguir las indicaciones de su médico."
    ),
    "adverse_effects": (
        "Como todos los medicamentos, este puede producir efectos no deseados. "
        "Los más frecuentes son dolor de cabeza, náuseas y molestias digestivas. "
        "No se preocupe, la mayoría son leves y temporales."
    ),
    "dosage": (
        "La dosis la debe establecer su médico según su situación particular. "
        "Generalmente se toma con un vaso de agua, preferiblemente en ayunas. "
        "No modifique la dosis sin consultar."
    ),
    "pregnancy": (
        "Si está embarazada o planea estarlo, es importante que lo consulte "
        "con su médico antes de tomar este medicamento. "
        "El principio activo podría tener efectos sobre el embarazo."
    ),
    "contraindications": (
        "No debe tomar este medicamento si tiene alergia al principio activo "
        "o a alguno de los componentes. Consulte a su farmacéutico si tiene dudas."
    ),
}

_CONDITION_MAP = {
    "P01_OMEPRAZOL": "úlcera gástrica y reflujo gastroesofágico",
    "P02_ATORVAST": "hipercolesterolemia",
    "P03_ESCITALO": "depresión y trastorno de ansiedad",
    "P04_AMOXICIL": "infecciones bacterianas",
    "P05_IBUPROFENO": "dolor, inflamación y fiebre",
    "P06_METFORM": "diabetes mellitus tipo 2",
    "P07_SERTRAL": "depresión y trastorno obsesivo compulsivo",
    "P08_SIMVAST": "hipercolesterolemia",
    "P09_AMLODIP": "hipertensión arterial",
    "P10_LEVOTIR": "hipotiroidismo",
}

_GROUP_MAP = {
    "P01_OMEPRAZOL": "inhibidores de la bomba de protones",
    "P02_ATORVAST": "estatinas",
    "P03_ESCITALO": "inhibidores selectivos de la recaptación de serotonina",
    "P04_AMOXICIL": "antibióticos betalactámicos",
    "P05_IBUPROFENO": "antiinflamatorios no esteroideos (AINE)",
    "P06_METFORM": "biguanidas",
    "P07_SERTRAL": "inhibidores selectivos de la recaptación de serotonina",
    "P08_SIMVAST": "estatinas",
    "P09_AMLODIP": "antagonistas del calcio",
    "P10_LEVOTIR": "hormonas tiroideas",
}


def _mock_response(query: dict, condition: str) -> str:
    """Generate a mock response that mimics real model output patterns."""
    category = query.get("query_category", "indication")
    pair_id = query.get("pair_id", "")
    drug_name = query.get("drug_name", "")
    inn = query.get("principio_activo", "").split()[0] if query.get("principio_activo") else ""

    templates = _MOCK_TEMPLATES_ORIGINAL if condition == "original" else _MOCK_TEMPLATES_LOOSE
    template = templates.get(category, templates.get("indication", ""))

    cond = _CONDITION_MAP.get(pair_id, "diversas patologías")
    group = _GROUP_MAP.get(pair_id, "medicamentos")

    return template.format(
        drug=drug_name,
        inn=inn.lower(),
        condition=cond,
        group=group,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ═══════════════════════════════════════════════════════════════════════════════


def _save_response(
    fpath: Path,
    qid: str,
    query: dict,
    response_text: str,
    retrieval_details: list[dict],
    sections_retrieved: set,
    context_str: str,
    model: str,
    top_k: int,
    condition: str,
    doc_count: int,
):
    """Save a response JSON in the same schema as collect_all_responses."""
    normalised = normalise_query(query.get("query_text", ""))
    entities = extract_entities(query)
    predicted_secs = predict_sections(query.get("query_category", ""))

    sys_prompt = ORIGINAL_SYSTEM_PROMPT if condition == "original" else _SYSTEM_PROMPT_LOOSE
    full_prompt_text = (
        f"System: {sys_prompt}\n\n"
        f"User: {_USER_TEMPLATE.format(context=context_str, query=query.get('query_text', ''))}"
    )
    prompt_hash = hashlib.md5(full_prompt_text.encode()).hexdigest()
    response_hash = hashlib.md5(response_text.encode()).hexdigest() if response_text else ""

    response_data = {
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
            "predicted_section_names": [
                LEAFLET_SECTIONS.get(s, "") for s in predicted_secs
            ],
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
        "notes": (
            f"MeQA-RAG experiment ({condition}) "
            f"(model={model}, temp=0.0, top_k={top_k}, "
            f"chunks={len(retrieval_details)}, ctx_len={len(context_str)})"
        ),
    }

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment(mock: bool = False):
    # ── API key (same method as the Streamlit app) ──
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not mock:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    if not api_key and not mock:
        raise RuntimeError(
            "No OpenAI API key found. Set OPENAI_API_KEY env var, "
            "or use --mock for pipeline testing without API calls."
        )

    # ── Load & sample queries ──
    with open(QUERIES_PATH, encoding="utf-8") as f:
        all_queries = json.load(f)
    queries = sample_queries(all_queries)

    # ── Initialise vector store (identical for both conditions) ──
    store = None
    doc_count = 0
    if not mock:
        from openai import OpenAI
        store = __import__("src.vector_store", fromlist=["MeQAVectorStore"]).MeQAVectorStore(
            openai_api_key=api_key
        )
        if store.count == 0:
            logger.info("Vector store empty — bootstrapping...")
            from src.rag_engine import _bootstrap_index
            _bootstrap_index(
                queries=all_queries,
                store=store,
                prospectos_dir=PROSPECTOS_DIR,
                leaflets_dir=LEAFLETS_DIR,
            )
        doc_count = store.count
        logger.info("Vector store ready: %d chunks", doc_count)

    # ── Prepare output dirs ──
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_LOOSE_DIR.mkdir(parents=True, exist_ok=True)

    client = None
    if not mock:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    model = "gpt-4o-mini"
    top_k = 5  # same as collect_all_responses default

    parity_log: list[dict] = []

    for i, query in enumerate(queries):
        qid = query.get("query_id", f"Q{i+1:04d}")
        query_text = query.get("query_text", "")
        drug_name = query.get("drug_name", "")
        pair_id = query.get("pair_id", "")
        is_comparative = query.get("is_generic") == "comparative"
        entities = extract_entities(query)

        # ── SHARED RETRIEVAL (Block 2) — identical between conditions ──
        retrieved = []
        context_str = ""
        retrieval_details = []
        sections_retrieved = set()

        if not mock and store is not None:
            filter_drug = None if is_comparative else drug_name
            retrieved = retrieve_chunks(
                query_text=query_text,
                store=store,
                pair_id=pair_id,
                drug_name=filter_drug,
                top_k=top_k,
            )
            context_str = _format_context(retrieved)

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
        else:
            # Mock: simulate 5 retrieved chunks
            for ci in range(top_k):
                sec_num = (ci % 6) + 1
                retrieval_details.append({
                    "section_number": sec_num,
                    "section_name": LEAFLET_SECTIONS.get(sec_num, ""),
                    "similarity": round(0.92 - ci * 0.05, 4),
                    "chunk_index": ci,
                    "drug_name": drug_name,
                })
                sections_retrieved.add(sec_num)
            context_str = f"[Mock context for {drug_name}]"

        chunk_count = len(retrieval_details)

        # ── CONDITION A: Original prompt → data/responses/ ──
        fpath_orig = RESPONSES_DIR / f"{qid}.json"
        if not fpath_orig.exists():
            if mock:
                resp_orig = _mock_response(query, "original")
            else:
                resp_orig = _generate(
                    query_text, context_str, ORIGINAL_SYSTEM_PROMPT,
                    client, model, drug_name,
                    entities.get("active_ingredient", ""),
                )
            _save_response(
                fpath_orig, qid, query, resp_orig, retrieval_details,
                sections_retrieved, context_str, model, top_k,
                "original", doc_count,
            )
            logger.info("[%s] Original → %d words", qid, len(resp_orig.split()))
        else:
            logger.info("[%s] Original already exists — skipping", qid)

        # ── CONDITION B: Loose prompt → data/responses_loose/ ──
        fpath_loose = RESPONSES_LOOSE_DIR / f"{qid}.json"
        if not fpath_loose.exists():
            if mock:
                resp_loose = _mock_response(query, "loose")
            else:
                resp_loose = _generate(
                    query_text, context_str, _SYSTEM_PROMPT_LOOSE,
                    client, model, drug_name,
                    entities.get("active_ingredient", ""),
                )
            _save_response(
                fpath_loose, qid, query, resp_loose, retrieval_details,
                sections_retrieved, context_str, model, top_k,
                "loose", doc_count,
            )
            logger.info("[%s] Loose   → %d words", qid, len(resp_loose.split()))
        else:
            logger.info("[%s] Loose already exists — skipping", qid)

        parity_log.append({
            "query_id": qid,
            "pair_id": pair_id,
            "chunks_retrieved": chunk_count,
            "context_length": len(context_str),
            "top_k": top_k,
        })

        if not mock and i < len(queries) - 1:
            time.sleep(0.3)

        print(f"  [{i+1}/{len(queries)}] {qid} — {chunk_count} chunks", flush=True)

    # ── Print retrieval parity summary ──
    print("\n" + "=" * 60)
    print("RETRIEVAL PARITY CHECK")
    print("=" * 60)
    for entry in parity_log:
        print(
            f"  {entry['query_id']}  pair={entry['pair_id']:<16s}  "
            f"chunks={entry['chunks_retrieved']}  ctx_len={entry['context_length']}"
        )
    print(f"\nAll {len(parity_log)} queries used identical retrieval (top_k={top_k}).")
    print(f"Original responses: {RESPONSES_DIR}")
    print(f"Loose responses:    {RESPONSES_LOOSE_DIR}")
    if mock:
        print("\n⚠  MOCK MODE: responses are simulated, not from real LLM calls.")
        print("   Re-run without --mock with OPENAI_API_KEY set for real results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loose prompt experiment")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock responses instead of real API calls")
    args = parser.parse_args()
    run_experiment(mock=args.mock)
