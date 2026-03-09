"""Drug pair definitions and CIMA-based pair builder.

Defines brand vs generic pairs focused on allergies and chronic diseases,
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
    # ── ALLERGIES ─────────────────────────────────────────────
    {
        "pair_id": "P01_CETIRIZ",
        "principio_activo": "Cetirizina 10mg",
        "grupo": "Alergias",
        "atc": "R06AE07",
        "drug_class": "antihistamínicos H1 de segunda generación",
        "drug_class_short": "antihistamínicos",
        "condition": "rinitis alérgica y urticaria",
        "symptom": "la alergia estacional",
        "brand": "ZYRTEC 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "CETIRIZINA CINFA 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "CETIRIZINA NORMON 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P02_LORATAD",
        "principio_activo": "Loratadina 10mg",
        "grupo": "Alergias",
        "atc": "R06AX13",
        "drug_class": "antihistamínicos H1 de segunda generación",
        "drug_class_short": "antihistamínicos",
        "condition": "rinitis alérgica y urticaria crónica",
        "symptom": "los síntomas de alergia",
        "brand": "CLARITYNE 10 MG COMPRIMIDOS",
        "generics": [
            "LORATADINA CINFA 10 MG COMPRIMIDOS EFG",
            "LORATADINA NORMON 10 MG COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P03_EBASTIN",
        "principio_activo": "Ebastina 10mg",
        "grupo": "Alergias",
        "atc": "R06AX22",
        "drug_class": "antihistamínicos H1 de segunda generación",
        "drug_class_short": "antihistamínicos",
        "condition": "rinitis alérgica estacional y perenne",
        "symptom": "la rinitis alérgica",
        "brand": "EBASTEL 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "EBASTINA CINFA 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "EBASTINA STADA 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P04_DESLORA",
        "principio_activo": "Desloratadina 5mg",
        "grupo": "Alergias",
        "atc": "R06AX27",
        "drug_class": "antihistamínicos H1 de segunda generación",
        "drug_class_short": "antihistamínicos",
        "condition": "rinitis alérgica y urticaria crónica idiopática",
        "symptom": "la congestión nasal alérgica",
        "brand": "AERIUS 5 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "DESLORATADINA CINFA 5 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "DESLORATADINA TEVA 5 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P05_MONTELUK",
        "principio_activo": "Montelukast 10mg",
        "grupo": "Alergias",
        "atc": "R03DC03",
        "drug_class": "antileucotrienos",
        "drug_class_short": "antileucotrienos",
        "condition": "asma alérgico y rinitis alérgica estacional",
        "symptom": "el asma alérgico",
        "brand": "SINGULAIR 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "MONTELUKAST CINFA 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "MONTELUKAST NORMON 10 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    # ── CHRONIC DISEASES ──────────────────────────────────────
    {
        "pair_id": "P06_METFORM",
        "principio_activo": "Metformina 850mg",
        "grupo": "Enfermedades crónicas - Diabetes",
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
        "pair_id": "P07_LOSARTAN",
        "principio_activo": "Losartán 50mg",
        "grupo": "Enfermedades crónicas - Hipertensión",
        "atc": "C09CA01",
        "drug_class": "antagonistas de los receptores de angiotensina II",
        "drug_class_short": "ARA-II",
        "condition": "hipertensión arterial",
        "symptom": "la tensión alta",
        "brand": "COZAAR 50 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "LOSARTAN CINFA 50 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "LOSARTAN NORMON 50 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P08_ENALAPRIL",
        "principio_activo": "Enalapril 20mg",
        "grupo": "Enfermedades crónicas - Hipertensión",
        "atc": "C09AA02",
        "drug_class": "inhibidores de la enzima convertidora de angiotensina",
        "drug_class_short": "IECA",
        "condition": "hipertensión arterial e insuficiencia cardíaca",
        "symptom": "la tensión alta",
        "brand": "RENITEC 20 MG COMPRIMIDOS",
        "generics": [
            "ENALAPRIL CINFA 20 MG COMPRIMIDOS EFG",
            "ENALAPRIL NORMON 20 MG COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P09_LEVOTIR",
        "principio_activo": "Levotiroxina 100mcg",
        "grupo": "Enfermedades crónicas - Tiroides",
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
    {
        "pair_id": "P10_SALBUTAM",
        "principio_activo": "Salbutamol 100mcg",
        "grupo": "Enfermedades crónicas - Asma",
        "atc": "R03AC02",
        "drug_class": "agonistas beta-2 adrenérgicos de acción corta",
        "drug_class_short": "SABA",
        "condition": "asma bronquial y broncoespasmo",
        "symptom": "el asma",
        "brand": "VENTOLIN 100 MICROGRAMOS SUSPENSION PARA INHALACION",
        "generics": [
            "SALBUTAMOL ALDO-UNION 100 MICROGRAMOS/DOSIS SUSPENSION PARA INHALACION EFG",
            "SALBUTAMOL SANDOZ 100 MICROGRAMOS/DOSIS SUSPENSION PARA INHALACION EFG",
        ],
    },
]


# ── CIMA API pair candidates (for online validation) ────────────────────────

CIMA_CANDIDATES = [
    # Allergies
    {"principio_activo": "CETIRIZINA", "dosis": "10 mg", "grupo": "Alergias",
     "atc": "R06AE07", "brand_keywords": ["ZYRTEC", "ALERLISIN"]},
    {"principio_activo": "LORATADINA", "dosis": "10 mg", "grupo": "Alergias",
     "atc": "R06AX13", "brand_keywords": ["CLARITYNE"]},
    {"principio_activo": "EBASTINA", "dosis": "10 mg", "grupo": "Alergias",
     "atc": "R06AX22", "brand_keywords": ["EBASTEL"]},
    {"principio_activo": "DESLORATADINA", "dosis": "5 mg", "grupo": "Alergias",
     "atc": "R06AX27", "brand_keywords": ["AERIUS"]},
    {"principio_activo": "MONTELUKAST", "dosis": "10 mg", "grupo": "Alergias",
     "atc": "R03DC03", "brand_keywords": ["SINGULAIR"]},
    # Chronic diseases
    {"principio_activo": "METFORMINA", "dosis": "850 mg", "grupo": "Enfermedades crónicas - Diabetes",
     "atc": "A10BA02", "brand_keywords": ["DIANBEN"]},
    {"principio_activo": "LOSARTAN", "dosis": "50 mg", "grupo": "Enfermedades crónicas - Hipertensión",
     "atc": "C09CA01", "brand_keywords": ["COZAAR"]},
    {"principio_activo": "ENALAPRIL", "dosis": "20 mg", "grupo": "Enfermedades crónicas - Hipertensión",
     "atc": "C09AA02", "brand_keywords": ["RENITEC"]},
    {"principio_activo": "LEVOTIROXINA", "dosis": "100 mcg", "grupo": "Enfermedades crónicas - Tiroides",
     "atc": "H03AA01", "brand_keywords": ["EUTIROX", "LEVOTHROID"]},
    {"principio_activo": "SALBUTAMOL", "dosis": "100 mcg", "grupo": "Enfermedades crónicas - Asma",
     "atc": "R03AC02", "brand_keywords": ["VENTOLIN"]},
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
