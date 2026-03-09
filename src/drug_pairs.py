"""Drug pair definitions and CIMA-based pair builder.

Defines brand vs generic pairs for 20 high-volume traditional drugs
commonly dispensed in Spanish pharmacies, and provides functions to
validate them against the live CIMA API.
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
        return self.brand is not None


# ── Pre-defined pairs (offline mode) ────────────────────────────────────────
# All 20 are traditional drugs with generics available in Spain.

OFFLINE_PAIRS = [
    {
        "pair_id": "P01_PARACET",
        "principio_activo": "Paracetamol 1g",
        "grupo": "Analgésico / Antipirético",
        "atc": "N02BE01",
        "drug_class": "analgésicos no opioides",
        "drug_class_short": "analgésicos",
        "condition": "dolor leve a moderado y fiebre",
        "symptom": "el dolor de cabeza y la fiebre",
        "brand": "EFFERALGAN 1 G COMPRIMIDOS EFERVESCENTES",
        "generics": [
            "PARACETAMOL CINFA 1 G COMPRIMIDOS EFG",
            "PARACETAMOL KERN PHARMA 1 G COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P02_IBUPRO",
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
        "pair_id": "P03_AMOXICI",
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
        "pair_id": "P04_OMEPRAZ",
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
        "pair_id": "P05_ATORVAS",
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
        "pair_id": "P06_ENALAPR",
        "principio_activo": "Enalapril 20mg",
        "grupo": "Cardiovascular",
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
        "pair_id": "P07_METFORM",
        "principio_activo": "Metformina 850mg",
        "grupo": "Antidiabéticos",
        "atc": "A10BA02",
        "drug_class": "biguanidas",
        "drug_class_short": "biguanidas",
        "condition": "diabetes mellitus tipo 2",
        "symptom": "la diabetes tipo 2",
        "brand": "DIANBEN 850 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "METFORMINA CINFA 850 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "METFORMINA TEVA 850 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P08_LORAZEP",
        "principio_activo": "Lorazepam 1mg",
        "grupo": "SNC - Ansiolíticos",
        "atc": "N05BA06",
        "drug_class": "benzodiazepinas ansiolíticas",
        "drug_class_short": "benzodiazepinas",
        "condition": "trastorno de ansiedad generalizada",
        "symptom": "la ansiedad",
        "brand": "ORFIDAL 1 MG COMPRIMIDOS",
        "generics": [
            "LORAZEPAM CINFA 1 MG COMPRIMIDOS EFG",
            "LORAZEPAM NORMON 1 MG COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P09_SERTRAL",
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
        "pair_id": "P10_SALBUT",
        "principio_activo": "Salbutamol 100mcg/dosis",
        "grupo": "Respiratorio",
        "atc": "R03AC02",
        "drug_class": "agonistas beta-2 adrenérgicos de acción corta",
        "drug_class_short": "SABA",
        "condition": "asma y broncoespasmo",
        "symptom": "el asma y la dificultad para respirar",
        "brand": "VENTOLIN 100 MICROGRAMOS/DOSIS SUSPENSIÓN PARA INHALACIÓN EN ENVASE A PRESIÓN",
        "generics": [
            "SALBUTAMOL CINFA 100 MICROGRAMOS/DOSIS SUSPENSIÓN PARA INHALACIÓN EFG",
            "SALBUTAMOL ALDO-UNION 100 MICROGRAMOS/DOSIS SUSPENSIÓN PARA INHALACIÓN EFG",
        ],
    },
    {
        "pair_id": "P11_AMLODIP",
        "principio_activo": "Amlodipino 5mg",
        "grupo": "Cardiovascular",
        "atc": "C08CA01",
        "drug_class": "antagonistas del calcio dihidropiridínicos",
        "drug_class_short": "antagonistas del calcio",
        "condition": "hipertensión arterial y angina de pecho",
        "symptom": "la tensión alta",
        "brand": "NORVASC 5 MG COMPRIMIDOS",
        "generics": [
            "AMLODIPINO CINFA 5 MG COMPRIMIDOS EFG",
            "AMLODIPINO NORMON 5 MG COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P12_AZITRO",
        "principio_activo": "Azitromicina 500mg",
        "grupo": "Antiinfecciosos",
        "atc": "J01FA10",
        "drug_class": "antibióticos macrólidos",
        "drug_class_short": "macrólidos",
        "condition": "infecciones respiratorias y de tejidos blandos",
        "symptom": "una infección respiratoria",
        "brand": "ZITROMAX 500 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "AZITROMICINA CINFA 500 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "AZITROMICINA NORMON 500 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
    {
        "pair_id": "P13_SIMVAS",
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
        "pair_id": "P14_FLUOXET",
        "principio_activo": "Fluoxetina 20mg",
        "grupo": "SNC - Antidepresivos",
        "atc": "N06AB03",
        "drug_class": "inhibidores selectivos de la recaptación de serotonina",
        "drug_class_short": "ISRS",
        "condition": "depresión mayor y trastorno obsesivo-compulsivo",
        "symptom": "la depresión",
        "brand": "PROZAC 20 MG CAPSULAS DURAS",
        "generics": [
            "FLUOXETINA CINFA 20 MG CAPSULAS DURAS EFG",
            "FLUOXETINA NORMON 20 MG CAPSULAS DURAS EFG",
        ],
    },
    {
        "pair_id": "P15_RAMIPR",
        "principio_activo": "Ramipril 5mg",
        "grupo": "Cardiovascular",
        "atc": "C09AA05",
        "drug_class": "inhibidores de la enzima convertidora de angiotensina",
        "drug_class_short": "IECA",
        "condition": "hipertensión arterial e insuficiencia cardíaca",
        "symptom": "la tensión alta",
        "brand": "ACOVIL 5 MG COMPRIMIDOS",
        "generics": [
            "RAMIPRIL CINFA 5 MG COMPRIMIDOS EFG",
            "RAMIPRIL NORMON 5 MG COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P16_DICLOF",
        "principio_activo": "Diclofenaco 50mg",
        "grupo": "Antiinflamatorio",
        "atc": "M01AB05",
        "drug_class": "antiinflamatorios no esteroideos",
        "drug_class_short": "AINE",
        "condition": "dolor e inflamación",
        "symptom": "el dolor y la inflamación",
        "brand": "VOLTAREN 50 MG COMPRIMIDOS GASTRORRESISTENTES",
        "generics": [
            "DICLOFENACO CINFA 50 MG COMPRIMIDOS GASTRORRESISTENTES EFG",
            "DICLOFENACO NORMON 50 MG COMPRIMIDOS GASTRORRESISTENTES EFG",
        ],
    },
    {
        "pair_id": "P17_PANTOP",
        "principio_activo": "Pantoprazol 20mg",
        "grupo": "Gastrointestinal",
        "atc": "A02BC02",
        "drug_class": "inhibidores de la bomba de protones",
        "drug_class_short": "IBP",
        "condition": "acidez gástrica y reflujo gastroesofágico",
        "symptom": "la acidez de estómago",
        "brand": "PANTECTA 20 MG COMPRIMIDOS GASTRORRESISTENTES",
        "generics": [
            "PANTOPRAZOL CINFA 20 MG COMPRIMIDOS GASTRORRESISTENTES EFG",
            "PANTOPRAZOL NORMON 20 MG COMPRIMIDOS GASTRORRESISTENTES EFG",
        ],
    },
    {
        "pair_id": "P18_LEVOTI",
        "principio_activo": "Levotiroxina 100mcg",
        "grupo": "Endocrinología",
        "atc": "H03AA01",
        "drug_class": "hormonas tiroideas",
        "drug_class_short": "hormonas tiroideas",
        "condition": "hipotiroidismo",
        "symptom": "el hipotiroidismo",
        "brand": "EUTIROX 100 MICROGRAMOS COMPRIMIDOS",
        "generics": [
            "LEVOTIROXINA SÓDICA TEVA 100 MICROGRAMOS COMPRIMIDOS EFG",
            "LEVOTIROXINA SÓDICA ACCORD 100 MICROGRAMOS COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P19_ALPRAZ",
        "principio_activo": "Alprazolam 0.5mg",
        "grupo": "SNC - Ansiolíticos",
        "atc": "N05BA12",
        "drug_class": "benzodiazepinas ansiolíticas",
        "drug_class_short": "benzodiazepinas",
        "condition": "trastorno de ansiedad generalizada",
        "symptom": "la ansiedad",
        "brand": "TRANKIMAZIN 0,5 MG COMPRIMIDOS",
        "generics": [
            "ALPRAZOLAM CINFA 0,5 MG COMPRIMIDOS EFG",
            "ALPRAZOLAM NORMON 0,5 MG COMPRIMIDOS EFG",
        ],
    },
    {
        "pair_id": "P20_CIPROFL",
        "principio_activo": "Ciprofloxacino 500mg",
        "grupo": "Antiinfecciosos",
        "atc": "J01MA02",
        "drug_class": "fluoroquinolonas",
        "drug_class_short": "fluoroquinolonas",
        "condition": "infecciones bacterianas del tracto urinario y respiratorio",
        "symptom": "una infección urinaria o respiratoria",
        "brand": "BAYCIP 500 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [
            "CIPROFLOXACINO CINFA 500 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
            "CIPROFLOXACINO NORMON 500 MG COMPRIMIDOS RECUBIERTOS CON PELICULA EFG",
        ],
    },
]


# ── CIMA API pair candidates (for online validation) ────────────────────────

CIMA_CANDIDATES = [
    {"principio_activo": "PARACETAMOL", "dosis": "1 g", "grupo": "Analgésico / Antipirético",
     "atc": "N02BE01", "brand_keywords": ["EFFERALGAN", "GELOCATIL"]},
    {"principio_activo": "IBUPROFENO", "dosis": "600 mg", "grupo": "Antiinflamatorio",
     "atc": "M01AE01", "brand_keywords": ["NEOBRUFEN", "ESPIDIFEN"]},
    {"principio_activo": "AMOXICILINA", "dosis": "500 mg", "grupo": "Antiinfecciosos",
     "atc": "J01CA04", "brand_keywords": ["CLAMOXYL"]},
    {"principio_activo": "OMEPRAZOL", "dosis": "20 mg", "grupo": "Gastrointestinal",
     "atc": "A02BC01", "brand_keywords": ["LOSEC", "MEPRAL"]},
    {"principio_activo": "ATORVASTATINA", "dosis": "20 mg", "grupo": "Cardiovascular",
     "atc": "C10AA05", "brand_keywords": ["CARDYL", "LIPITOR"]},
    {"principio_activo": "ENALAPRIL", "dosis": "20 mg", "grupo": "Cardiovascular",
     "atc": "C09AA02", "brand_keywords": ["RENITEC"]},
    {"principio_activo": "METFORMINA", "dosis": "850 mg", "grupo": "Antidiabéticos",
     "atc": "A10BA02", "brand_keywords": ["DIANBEN"]},
    {"principio_activo": "LORAZEPAM", "dosis": "1 mg", "grupo": "SNC - Ansiolíticos",
     "atc": "N05BA06", "brand_keywords": ["ORFIDAL"]},
    {"principio_activo": "SERTRALINA", "dosis": "50 mg", "grupo": "SNC - Antidepresivos",
     "atc": "N06AB06", "brand_keywords": ["BESITRAN", "AREMIS"]},
    {"principio_activo": "SALBUTAMOL", "dosis": "100 mcg/dosis", "grupo": "Respiratorio",
     "atc": "R03AC02", "brand_keywords": ["VENTOLIN"]},
    {"principio_activo": "AMLODIPINO", "dosis": "5 mg", "grupo": "Cardiovascular",
     "atc": "C08CA01", "brand_keywords": ["NORVASC"]},
    {"principio_activo": "AZITROMICINA", "dosis": "500 mg", "grupo": "Antiinfecciosos",
     "atc": "J01FA10", "brand_keywords": ["ZITROMAX"]},
    {"principio_activo": "SIMVASTATINA", "dosis": "20 mg", "grupo": "Cardiovascular",
     "atc": "C10AA01", "brand_keywords": ["ZOCOR"]},
    {"principio_activo": "FLUOXETINA", "dosis": "20 mg", "grupo": "SNC - Antidepresivos",
     "atc": "N06AB03", "brand_keywords": ["PROZAC"]},
    {"principio_activo": "RAMIPRIL", "dosis": "5 mg", "grupo": "Cardiovascular",
     "atc": "C09AA05", "brand_keywords": ["ACOVIL"]},
    {"principio_activo": "DICLOFENACO", "dosis": "50 mg", "grupo": "Antiinflamatorio",
     "atc": "M01AB05", "brand_keywords": ["VOLTAREN"]},
    {"principio_activo": "PANTOPRAZOL", "dosis": "20 mg", "grupo": "Gastrointestinal",
     "atc": "A02BC02", "brand_keywords": ["PANTECTA", "ANAGASTRA"]},
    {"principio_activo": "LEVOTIROXINA", "dosis": "100 mcg", "grupo": "Endocrinología",
     "atc": "H03AA01", "brand_keywords": ["EUTIROX"]},
    {"principio_activo": "ALPRAZOLAM", "dosis": "0,5 mg", "grupo": "SNC - Ansiolíticos",
     "atc": "N05BA12", "brand_keywords": ["TRANKIMAZIN"]},
    {"principio_activo": "CIPROFLOXACINO", "dosis": "500 mg", "grupo": "Antiinfecciosos",
     "atc": "J01MA02", "brand_keywords": ["BAYCIP", "CIPROXIN"]},
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
