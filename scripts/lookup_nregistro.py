#!/usr/bin/env python3
"""Helper utility to look up nregistro from CIMA by drug name.

Usage:
    python scripts/lookup_nregistro.py "Losec"
    python scripts/lookup_nregistro.py "Omeprazol Cinfa"
    python scripts/lookup_nregistro.py --all          # Look up all 20 pairs
    python scripts/lookup_nregistro.py --save         # Look up all and save config
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cima_client import CIMAClient
from src.drug_pairs import OFFLINE_PAIRS
from src.config import DATA_DIR, PAIRS_DIR


def lookup_nregistro(name: str, cima: CIMAClient) -> list[dict]:
    """Query CIMA for a drug name and return matching results with nregistro."""
    result = cima.search_medicamentos(nombre=name)
    if not result or "resultados" not in result:
        return []

    matches = []
    for med in result["resultados"]:
        matches.append({
            "nregistro": med.get("nregistro", ""),
            "nombre": med.get("nombre", ""),
            "labtitular": med.get("labtitular", ""),
            "comerc": med.get("comerc", False),
            "es_generico": "EFG" in med.get("nombre", "").upper(),
        })
    return matches


def build_nregistro_config(cima: CIMAClient) -> list[dict]:
    """Build drug_pairs config with nregistro for all 20 pairs.

    For each pair, finds the best matching brand and first matching generic
    from CIMA, using the pair's brand/generic names as search terms.
    """
    pairs_config = []

    for pair in OFFLINE_PAIRS:
        pair_id = pair["pair_id"]
        brand_name = pair["brand"]
        principio = pair["principio_activo"]
        generic_names = pair.get("generics", [])

        print(f"\n{'='*60}")
        print(f"Pair: {pair_id} — {principio}")
        print(f"{'='*60}")

        # Search for brand
        brand_result = None
        print(f"\n  Searching brand: {brand_name}")
        matches = lookup_nregistro(brand_name, cima)
        time.sleep(1)

        if matches:
            # Prefer marketed (comerc=True), non-generic
            for m in matches:
                if m["comerc"] and not m["es_generico"]:
                    brand_result = m
                    break
            if not brand_result:
                brand_result = matches[0]
            print(f"    → {brand_result['nregistro']}: {brand_result['nombre'][:60]}")
        else:
            print(f"    → NOT FOUND")

        # Search for first generic
        generic_result = None
        if generic_names:
            first_generic = generic_names[0]
            print(f"\n  Searching generic: {first_generic}")

            # Search by active ingredient to find generics
            pa_name = principio.split()[0]  # e.g., "Paracetamol" from "Paracetamol 1g"
            pa_matches = lookup_nregistro(pa_name, cima)
            time.sleep(1)

            if pa_matches:
                # Find first EFG match
                for m in pa_matches:
                    if m["es_generico"] and m["comerc"]:
                        lab_name = first_generic.split()[-1].upper()  # e.g., "CINFA"
                        if lab_name in m.get("labtitular", "").upper() or \
                           lab_name in m.get("nombre", "").upper():
                            generic_result = m
                            break
                # Fallback: any EFG
                if not generic_result:
                    for m in pa_matches:
                        if m["es_generico"] and m["comerc"]:
                            generic_result = m
                            break

            if generic_result:
                print(f"    → {generic_result['nregistro']}: {generic_result['nombre'][:60]}")
            else:
                print(f"    → NOT FOUND")

        pair_config = {
            "pair_id": pair_id,
            "principio_activo": principio,
            "grupo_terapeutico": pair.get("grupo", ""),
            "atc_code": pair.get("atc", ""),
            "branded": {
                "name": brand_result["nombre"] if brand_result else "",
                "nregistro": brand_result["nregistro"] if brand_result else "",
                "is_generic": False,
            } if brand_result else {"name": "", "nregistro": "", "is_generic": False},
            "generic": {
                "name": generic_result["nombre"] if generic_result else "",
                "nregistro": generic_result["nregistro"] if generic_result else "",
                "is_generic": True,
            } if generic_result else {"name": "", "nregistro": "", "is_generic": True},
        }
        pairs_config.append(pair_config)

    return pairs_config


def main():
    cima = CIMAClient(delay=0.5)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/lookup_nregistro.py 'drug name'")
        print("  python scripts/lookup_nregistro.py --all")
        print("  python scripts/lookup_nregistro.py --save")
        sys.exit(1)

    arg = sys.argv[1]

    if arg in ("--all", "--save"):
        config = build_nregistro_config(cima)

        print(f"\n\n{'='*60}")
        print(f"Results: {len(config)} pairs")
        print(f"{'='*60}")

        for p in config:
            brand_nr = p["branded"]["nregistro"] or "MISSING"
            generic_nr = p["generic"]["nregistro"] or "MISSING"
            print(f"  {p['pair_id']}: brand={brand_nr}, generic={generic_nr}")

        if arg == "--save":
            PAIRS_DIR.mkdir(parents=True, exist_ok=True)
            outpath = PAIRS_DIR / "drug_pairs_nregistro.json"
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"\n✓ Saved to {outpath}")

    else:
        print(f"\nSearching CIMA for: {arg}")
        matches = lookup_nregistro(arg, cima)
        if matches:
            print(f"\nFound {len(matches)} results:\n")
            for m in matches[:10]:
                efg = " [EFG]" if m["es_generico"] else ""
                comerc = " (marketed)" if m["comerc"] else " (not marketed)"
                print(f"  nregistro={m['nregistro']}: {m['nombre']}{efg}{comerc}")
                print(f"    Lab: {m['labtitular']}")
        else:
            print("  No results found.")


if __name__ == "__main__":
    main()
