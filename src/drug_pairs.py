"""Drug pair definitions and CIMA-based pair builder.

Defines brand vs generic/biosimilar pairs for 20 drugs spanning traditional
small-molecule medicines (with generics) and innovative biologics/gene
therapies (typically without generics), and provides functions to validate
them against the live CIMA API.
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
# Each pair includes condition/symptom for Layer 1 condition-first queries.

OFFLINE_PAIRS = [
    # ── INNOVATIVE / BIOLOGIC DRUGS (no generics available) ───
    {
        "pair_id": "P01_PEMBROLI",
        "principio_activo": "Pembrolizumab 25mg/mL",
        "grupo": "Oncología - Inmunoterapia",
        "atc": "L01FF02",
        "drug_class": "inhibidores de puntos de control inmunitario",
        "drug_class_short": "anti-PD-1",
        "condition": "melanoma avanzado y cáncer de pulmón no microcítico",
        "symptom": "el cáncer avanzado",
        "brand": "KEYTRUDA 25 MG/ML CONCENTRADO PARA SOLUCION PARA PERFUSION",
        "generics": [],
    },
    {
        "pair_id": "P03_EMICIZU",
        "principio_activo": "Emicizumab 150mg/mL",
        "grupo": "Hematología",
        "atc": "B02BX06",
        "drug_class": "anticuerpos monoclonales biespecíficos",
        "drug_class_short": "anticuerpos biespecíficos",
        "condition": "hemofilia A con inhibidores del factor VIII",
        "symptom": "la hemofilia A",
        "brand": "HEMLIBRA 150 MG/ML SOLUCION INYECTABLE",
        "generics": [],
    },
    {
        "pair_id": "P05_TISAGEN",
        "principio_activo": "Tisagenlecleucel",
        "grupo": "Oncología - Terapia celular",
        "atc": "L01XL04",
        "drug_class": "terapias de células T con receptor de antígeno quimérico",
        "drug_class_short": "CAR-T",
        "condition": "leucemia linfoblástica aguda de células B y linfoma difuso de células B grandes",
        "symptom": "la leucemia linfoblástica aguda",
        "brand": "KYMRIAH DISPERSION PARA PERFUSION",
        "generics": [],
    },
    {
        "pair_id": "P07_TOFERSE",
        "principio_activo": "Tofersen 100mg",
        "grupo": "Neurología - Enfermedades raras",
        "atc": "N07XX",
        "drug_class": "oligonucleótidos antisentido",
        "drug_class_short": "antisentido",
        "condition": "esclerosis lateral amiotrófica asociada a mutación SOD1",
        "symptom": "la ELA asociada a SOD1",
        "brand": "QALSODY 100 MG SOLUCION INYECTABLE",
        "generics": [],
    },
    {
        "pair_id": "P09_INCLISI",
        "principio_activo": "Inclisirán 284mg",
        "grupo": "Cardiovascular - Innovador",
        "atc": "C10AX16",
        "drug_class": "ARN de interferencia pequeño (siRNA)",
        "drug_class_short": "siRNA",
        "condition": "hipercolesterolemia primaria e hiperlipidemia mixta",
        "symptom": "el colesterol alto resistente a estatinas",
        "brand": "LEQVIO 284 MG SOLUCION INYECTABLE EN JERINGA PRECARGADA",
        "generics": [],
    },
    {
        "pair_id": "P11_BIMEKI",
        "principio_activo": "Bimekizumab 160mg",
        "grupo": "Dermatología - Innovador",
        "atc": "L04AC21",
        "drug_class": "anticuerpos monoclonales anti-IL-17A/F",
        "drug_class_short": "anti-IL-17",
        "condition": "psoriasis en placas de moderada a grave",
        "symptom": "la psoriasis en placas",
        "brand": "BIMZELX 160 MG SOLUCION INYECTABLE EN JERINGA PRECARGADA",
        "generics": [],
    },
    {
        "pair_id": "P13_ONASEMN",
        "principio_activo": "Onasemnogén abeparvovec",
        "grupo": "Neurología - Terapia génica",
        "atc": "M09AX09",
        "drug_class": "terapias génicas",
        "drug_class_short": "terapia génica",
        "condition": "atrofia muscular espinal tipo 1",
        "symptom": "la atrofia muscular espinal",
        "brand": "ZOLGENSMA 2 × 10^13 GENOMAS VECTORIALES/ML SOLUCION PARA PERFUSION",
        "generics": [],
    },
    {
        "pair_id": "P15_MAVACAM",
        "principio_activo": "Mavacamtén 5mg",
        "grupo": "Cardiovascular - Innovador",
        "atc": "C01EB24",
        "drug_class": "inhibidores de miosina cardíaca",
        "drug_class_short": "inhibidores de miosina",
        "condition": "miocardiopatía hipertrófica obstructiva",
        "symptom": "la miocardiopatía hipertrófica obstructiva",
        "brand": "CAMZYOS 5 MG CAPSULAS DURAS",
        "generics": [],
    },
    {
        "pair_id": "P17_ELEXACAF",
        "principio_activo": "Elexacaftor/Tezacaftor/Ivacaftor",
        "grupo": "Respiratorio - Innovador",
        "atc": "R07AX32",
        "drug_class": "moduladores de la proteína CFTR",
        "drug_class_short": "moduladores CFTR",
        "condition": "fibrosis quística con al menos una mutación F508del",
        "symptom": "la fibrosis quística",
        "brand": "KAFTRIO 75 MG/50 MG/100 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [],
    },
    {
        "pair_id": "P19_ADAGRAS",
        "principio_activo": "Adagrasib 200mg",
        "grupo": "Oncología - Terapia dirigida",
        "atc": "L01XX73",
        "drug_class": "inhibidores de KRAS G12C",
        "drug_class_short": "inhibidores KRAS",
        "condition": "cáncer de pulmón no microcítico con mutación KRAS G12C",
        "symptom": "el cáncer de pulmón con mutación KRAS",
        "brand": "KRAZATI 200 MG COMPRIMIDOS RECUBIERTOS CON PELICULA",
        "generics": [],
    },
    # ── TRADITIONAL DRUGS (generics available) ────────────────
    {
        "pair_id": "P02_OMEPRAZ",
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
        "pair_id": "P04_ATORVAS",
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
        "pair_id": "P06_AMOXICI",
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
        "pair_id": "P08_LORAZEP",
        "principio_activo": "Lorazepam 1mg",
        "grupo": "SNC - Ansiolíticos",
        "atc": "N05BA06",
        "drug_class": "benzodiazepinas",
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
        "pair_id": "P10_ENALAPR",
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
        "pair_id": "P12_FLUOXET",
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
        "pair_id": "P14_IBUPRO",
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
        "pair_id": "P16_SIMVAS",
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
        "pair_id": "P18_CIPROFL",
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
    {
        "pair_id": "P20_SERTRAL",
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
]


# ── CIMA API pair candidates (for online validation) ────────────────────────

CIMA_CANDIDATES = [
    # Innovative / Biologic drugs
    {"principio_activo": "PEMBROLIZUMAB", "dosis": "25 mg/mL", "grupo": "Oncología - Inmunoterapia",
     "atc": "L01FF02", "brand_keywords": ["KEYTRUDA"]},
    {"principio_activo": "EMICIZUMAB", "dosis": "150 mg/mL", "grupo": "Hematología",
     "atc": "B02BX06", "brand_keywords": ["HEMLIBRA"]},
    {"principio_activo": "TISAGENLECLEUCEL", "dosis": "", "grupo": "Oncología - Terapia celular",
     "atc": "L01XL04", "brand_keywords": ["KYMRIAH"]},
    {"principio_activo": "TOFERSEN", "dosis": "100 mg", "grupo": "Neurología - Enfermedades raras",
     "atc": "N07XX", "brand_keywords": ["QALSODY"]},
    {"principio_activo": "INCLISIRAN", "dosis": "284 mg", "grupo": "Cardiovascular - Innovador",
     "atc": "C10AX16", "brand_keywords": ["LEQVIO"]},
    {"principio_activo": "BIMEKIZUMAB", "dosis": "160 mg", "grupo": "Dermatología - Innovador",
     "atc": "L04AC21", "brand_keywords": ["BIMZELX"]},
    {"principio_activo": "ONASEMNOGENE ABEPARVOVEC", "dosis": "", "grupo": "Neurología - Terapia génica",
     "atc": "M09AX09", "brand_keywords": ["ZOLGENSMA"]},
    {"principio_activo": "MAVACAMTEN", "dosis": "5 mg", "grupo": "Cardiovascular - Innovador",
     "atc": "C01EB24", "brand_keywords": ["CAMZYOS"]},
    {"principio_activo": "ELEXACAFTOR/TEZACAFTOR/IVACAFTOR", "dosis": "75/50/100 mg",
     "grupo": "Respiratorio - Innovador",
     "atc": "R07AX32", "brand_keywords": ["KAFTRIO", "TRIKAFTA"]},
    {"principio_activo": "ADAGRASIB", "dosis": "200 mg", "grupo": "Oncología - Terapia dirigida",
     "atc": "L01XX73", "brand_keywords": ["KRAZATI"]},
    # Traditional drugs (with generics)
    {"principio_activo": "OMEPRAZOL", "dosis": "20 mg", "grupo": "Gastrointestinal",
     "atc": "A02BC01", "brand_keywords": ["LOSEC", "MEPRAL"]},
    {"principio_activo": "ATORVASTATINA", "dosis": "20 mg", "grupo": "Cardiovascular",
     "atc": "C10AA05", "brand_keywords": ["CARDYL", "LIPITOR"]},
    {"principio_activo": "AMOXICILINA", "dosis": "500 mg", "grupo": "Antiinfecciosos",
     "atc": "J01CA04", "brand_keywords": ["CLAMOXYL"]},
    {"principio_activo": "LORAZEPAM", "dosis": "1 mg", "grupo": "SNC - Ansiolíticos",
     "atc": "N05BA06", "brand_keywords": ["ORFIDAL"]},
    {"principio_activo": "ENALAPRIL", "dosis": "20 mg", "grupo": "Cardiovascular",
     "atc": "C09AA02", "brand_keywords": ["RENITEC"]},
    {"principio_activo": "FLUOXETINA", "dosis": "20 mg", "grupo": "SNC - Antidepresivos",
     "atc": "N06AB03", "brand_keywords": ["PROZAC"]},
    {"principio_activo": "IBUPROFENO", "dosis": "600 mg", "grupo": "Antiinflamatorio",
     "atc": "M01AE01", "brand_keywords": ["NEOBRUFEN", "ESPIDIFEN"]},
    {"principio_activo": "SIMVASTATINA", "dosis": "20 mg", "grupo": "Cardiovascular",
     "atc": "C10AA01", "brand_keywords": ["ZOCOR"]},
    {"principio_activo": "CIPROFLOXACINO", "dosis": "500 mg", "grupo": "Antiinfecciosos",
     "atc": "J01MA02", "brand_keywords": ["BAYCIP", "CIPROXIN"]},
    {"principio_activo": "SERTRALINA", "dosis": "50 mg", "grupo": "SNC - Antidepresivos",
     "atc": "N06AB06", "brand_keywords": ["BESITRAN", "AREMIS"]},
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
