#!/usr/bin/env python3
"""Download official prospectos from CIMA API as ground truth.

Usage: python scripts/download_prospectos.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cima_client import CIMAClient
from src.drug_pairs import build_pairs_from_cima, save_pairs
from src.config import PROSPECTOS_DIR


def download_prospectos(cima: CIMAClient, pairs):
    """Download segmented prospectos for all drugs in pairs."""
    PROSPECTOS_DIR.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        all_meds = [pair.brand] + pair.generics
        for med in all_meds:
            safe_name = med.nombre[:40].replace("/", "_").replace(" ", "_")
            filename = PROSPECTOS_DIR / f"{med.nregistro}_{safe_name}.json"
            if filename.exists():
                print(f"  [SKIP] {med.nombre[:40]}")
                continue

            print(f"  Downloading: {med.nombre[:40]}...")
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
                print(f"    ✓ Saved")
            else:
                print(f"    [WARN] Failed")


def main():
    print("="*60)
    print("CIMA Prospecto Downloader (Ground Truth)")
    print("="*60)

    cima = CIMAClient()

    print("\nStep 1: Building drug pairs from CIMA API...")
    pairs = build_pairs_from_cima(cima)
    print(f"\n✓ Found {len(pairs)} valid pairs")

    if pairs:
        save_pairs(pairs)
        print("\nStep 2: Downloading prospectos...")
        download_prospectos(cima, pairs)
        print("\n✓ Done.")
    else:
        print("[ERROR] No valid pairs. Check internet / CIMA API.")


if __name__ == "__main__":
    main()
