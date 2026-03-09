"""Drug pair definitions and CIMA-based pair builder.

Defines brand vs generic pairs across 6 therapeutic groups,
and provides functions to validate them against the live CIMA API.
"""

import json
from dataclasses import dataclass, field, asdict
from .config import PAIRS_DIR, MAX_GENERICS_PER_PAIR


@dataclass
class Medicamento:
    nregistro: str
    nombre: str
    labtitular: str
    pactivos: str
    comerc: bool
    es_generico: bool
    dosis: str = ""
    forma_farmaceutica: str = ""
    prospecto_url: str = ""
    ficha_tecnica_url: str = ""


@dataclass
class DrugPair:
    pair_id: str
    principio_activo: str
    dosis: str
    grupo_terapeutico: str
    atc_code: str
    brand: Medicamento = None
    generics: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.brand is not None and len(self.generics) >= 1


# ── Pre-defined pairs (offline mode) ────────────────────────────────────────
# Each pair includes condition/symptom for Layer 1 condition-first queries.

OFFLINE_PAIRS = [
    {
        "pair_id": "P01_OMEPRAZOL",
        "principio_activo": "Omeprazol 20mg",
        "grupo": "Gastrointestinal",
        "atc": "A02BC01",
        "drug_class": "inhibidores de la bomba de protones",
        "drug_class_short": "IBP",
        "condition": "acidez gástrica y reflujo gastroesofágico",
        "symptom": "la acidez de estómago",
        "brand": "LOSEC 20 MG CAPSULAS DURAS GASTRORRESISTENTES",
        "generics": [
            "OMEPRAZOL CINFA 20 MG CAPSULAS DURAS GASTRORRESISTENTES EFG",
            "OMEPRAZOL NORMON 20 MG CAPSULAS DURAS GASTRORRESISTENTES EFG",
        ],
    },
    {
        "pair_id": "P02_ATORVAST",
        "principio_activo": "Atorvastatina 20mg",
        "grupo": "Cardiovascular",
        "atc": "C10AA05",
        "drug_class": "estatinas",
        "drug_class_short": "estatinas",
        "condition": "hipercolesterolemia",
        "symptom": "el colesterol alto",
        "brand": "CARDYL 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "ATORVASTATINA CINFA 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "ATORVASTATINA TEVA 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P03_ESCITALO",
        "principio_activo": "Escitalopram 10mg",
        "grupo": "SNC - Antidepresivos",
        "atc": "N06AB10",
        "drug_class": "inhibidores selectivos de la recaptación de serotonina",
        "drug_class_short": "ISRS",
        "condition": "depresión y trastorno de ansiedad generalizada",
        "symptom": "la depresión y la ansiedad",
        "brand": "CIPRALEX 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "ESCITALOPRAM CINFA 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "ESCITALOPRAM STADA 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P04_AMOXICIL",
        "principio_activo": "Amoxicilina 500mg",
        "grupo": "Antiinfecciosos",
        "atc": "J01CA04",
        "drug_class": "antibióticos betalactámicos",
        "drug_class_short": "penicilinas",
        "condition": "infecciones bacterianas",
        "symptom": "una infección bacteriana",
        "brand": "CLAMOXYL 500 MG CAPSULAS DURAS",
        "generics": [
            "AMOXICILINA CINFA 500 MG CAPSULAS DURAS EFG",
            "AMOXICILINA NORMON 500 MG CAPSULAS DURAS EFG",
        ],
    },
    {
        "pair_id": "P05_IBUPROFENO",
        "principio_activo": "Ibuprofeno 600mg",
        "grupo": "Antiinflamatorio",
        "atc": "M01AE01",
        "drug_class": "antiinflamatorios no esteroideos",
        "drug_class_short": "AINE",
        "condition": "dolor e inflamación",
        "symptom": "el dolor y la inflamación",
        "brand": "NEOBRUFEN 600 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "IBUPROFENO CINFA 600 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "IBUPROFENO KERN PHARMA 600 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P06_METFORM",
        "principio_activo": "Metformina 850mg",
        "grupo": "Metabolismo",
        "atc": "A10BA02",
        "drug_class": "antidiabéticos orales",
        "drug_class_short": "biguanidas",
        "condition": "diabetes mellitus tipo 2",
        "symptom": "la diabetes tipo 2",
        "brand": "DIANBEN 850 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "METFORMINA CINFA 850 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "METFORMINA KERN PHARMA 850 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P07_SERTRAL",
        "principio_activo": "Sertralina 50mg",
        "grupo": "SNC - Antidepresivos",
        "atc": "N06AB06",
        "drug_class": "inhibidores selectivos de la recaptación de serotonina",
        "drug_class_short": "ISRS",
        "condition": "depresión y trastorno obsesivo-compulsivo",
        "symptom": "la depresión",
        "brand": "BESITRAN 50 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "SERTRALINA CINFA 50 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "SERTRALINA NORMON 50 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P08_SIMVAST",
        "principio_activo": "Simvastatina 20mg",
        "grupo": "Cardiovascular",
        "atc": "C10AA01",
        "drug_class": "estatinas",
        "drug_class_short": "estatinas",
        "condition": "hipercolesterolemia",
        "symptom": "el colesterol alto",
        "brand": "ZOCOR 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "SIMVASTATINA CINFA 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "SIMVASTATINA NORMON 20 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P09_AMLODIP",
        "principio_activo": "Amlodipino 5mg",
        "grupo": "Cardiovascular",
        "atc": "C08CA01",
        "drug_class": "antagonistas del calcio",
        "drug_class_short": "calcioantagonistas",
        "condition": "hipertensión arterial",
        "symptom": "la tensión alta",
        "brand": "NORVASC 5 MG COMPRIMIDOS",
        "generics": [
            "AMLODIPINO CINFA 5 MG COMPRIMIDOS EFG",
            "AMLODIPINO NORMON 5 MG COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P10_LEVOTIR",
        "principio_activo": "Levotiroxina 100mcg",
        "grupo": "Hormonas tiroideas",
        "atc": "H03AA01",
        "drug_class": "hormonas tiroideas",
        "drug_class_short": "levotiroxina",
        "condition": "hipotiroidismo",
        "symptom": "el tiroides bajo",
        "brand": "EUTIROX 100 MICROGRAMOS COMPRIMIDOS",
        "generics": [
            "LEVOTIROXINA SANOFI 100 MICROGRAMOS COMPRIMIDOS EFG",
        ],
    },
]


# ── CIMA API pair candidates (for online validation) ────────────────────────

CIMA_CANDIDATES = [
    {"principio_activo": "OMEPRAZOL", "dosis": "20 mg", "grupo": "Gastrointestinal",
     "atc": "A02BC01", "brand_keywords": ["LOSEC", "MEPRAL"]},
    {"principio_activo": "ATORVASTATINA", "dosis": "20 mg", "grupo": "Cardiovascular",
     "atc": "C10AA05", "brand_keywords": ["LIPITOR", "CARDYL", "PREVENCOR", "ZARATOR"]},
    {"principio_activo": "ESCITALOPRAM", "dosis": "10 mg", "grupo": "SNC - Antidepresivos",
     "atc": "N06AB10", "brand_keywords": ["CIPRALEX", "ESERTIA"]},
    {"principio_activo": "AMOXICILINA", "dosis": "500 mg", "grupo": "Antiinfecciosos",
     "atc": "J01CA04", "brand_keywords": ["CLAMOXYL"]},
    {"principio_activo": "IBUPROFENO", "dosis": "600 mg", "grupo": "Antiinflamatorio",
     "atc": "M01AE01", "brand_keywords": ["NEOBRUFEN", "ESPIDIFEN"]},
    {"principio_activo": "METFORMINA", "dosis": "850 mg", "grupo": "Metabolismo",
     "atc": "A10BA02", "brand_keywords": ["DIANBEN"]},
    {"principio_activo": "SERTRALINA", "dosis": "50 mg", "grupo": "SNC - Antidepresivos",
     "atc": "N06AB06", "brand_keywords": ["BESITRAN", "AREMIS"]},
    {"principio_activo": "SIMVASTATINA", "dosis": "20 mg", "grupo": "Cardiovascular",
     "atc": "C10AA01", "brand_keywords": ["ZOCOR"]},
    {"principio_activo": "AMLODIPINO", "dosis": "5 mg", "grupo": "Cardiovascular",
     "atc": "C08CA01", "brand_keywords": ["NORVASC"]},
    {"principio_activo": "LEVOTIROXINA", "dosis": "100 mcg", "grupo": "Hormonas tiroideas",
     "atc": "H03AA01", "brand_keywords": ["EUTIROX", "LEVOTHROID"]},
]


def get_offline_pairs() -> list[dict]:
    """Return pre-defined drug pairs (no API needed)."""
    return OFFLINE_PAIRS


def build_pairs_from_cima(cima_client) -> list[DrugPair]:
    """Query CIMA API to build validated drug pairs with registration numbers."""
    pairs = []

    for cand in CIMA_CANDIDATES:
        pa = cand["principio_activo"]
        print(f"\n{'='*60}\nSearching: {pa} {cand['dosis']}\n{'='*60}")

        result = cima_client.search_by_active_ingredient(pa)
        if not result or "resultados" not in result:
            print(f"  [WARN] No results for {pa}")
            continue

        meds = result["resultados"]
        print(f"  Found {len(meds)} results")

        pair = DrugPair(
            pair_id=f"P{len(pairs)+1:02d}_{pa[:10]}",
            principio_activo=pa,
            dosis=cand["dosis"],
            grupo_terapeutico=cand["grupo"],
            atc_code=cand["atc"],
        )

        for med in meds:
            nombre = med.get("nombre", "")
            es_generico = "EFG" in nombre.upper()

            docs = med.get("docs", [])
            prospecto_url = next(
                (d.get("url", "") for d in docs if d.get("tipo") == 2), "")
            ft_url = next(
                (d.get("url", "") for d in docs if d.get("tipo") == 1), "")

            medicamento = Medicamento(
                nregistro=med.get("nregistro", ""),
                nombre=nombre,
                labtitular=med.get("labtitular", ""),
                pactivos=med.get("pactivos", pa),
                comerc=med.get("comerc", False),
                es_generico=es_generico,
                dosis=med.get("dosis", ""),
                prospecto_url=prospecto_url,
                ficha_tecnica_url=ft_url,
            )

            if es_generico:
                pair.generics.append(medicamento)
            else:
                for kw in cand["brand_keywords"]:
                    if kw.upper() in nombre.upper():
                        pair.brand = medicamento
                        break

        if pair.is_valid:
            pair.generics = pair.generics[:MAX_GENERICS_PER_PAIR]
            pairs.append(pair)
            print(f"  ✓ BRAND: {pair.brand.nombre}")
            for g in pair.generics:
                print(f"  ✓ GENERIC: {g.nombre}")
        else:
            print(f"  [INCOMPLETE] Brand: {'YES' if pair.brand else 'NO'}, "
                  f"Generics: {len(pair.generics)}")

    return pairs


def save_pairs(pairs: list[DrugPair], filename: str = "drug_pairs.json"):
    """Save validated pairs to JSON."""
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PAIRS_DIR / filename
    data = [asdict(p) for p in pairs]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(pairs)} pairs → {filepath}")
