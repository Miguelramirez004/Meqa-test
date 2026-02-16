#!/usr/bin/env python3
"""Semi-automated MeQA query submitter using Playwright.

Opens a visible Chrome browser, submits queries one by one to MeQA,
and captures responses. Includes checkpoint/resume support.

Usage:
  pip install playwright && playwright install chromium
  python scripts/meqa_scraper.py

IMPORTANT: You will likely need to adapt the CSS selectors below
to match MeQA's actual DOM structure. On first run:
  1. Open MeQA in Chrome DevTools (F12)
  2. Inspect the input field, submit button, and response container
  3. Update SELECTORS dict below
"""
import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MEQA_URL, MEQA_QUERY_DELAY, QUERIES_DIR, RESPONSES_DIR

# ── Adapt these to MeQA's actual DOM ────────────────────────────────────────
SELECTORS = {
    "input": [
        'textarea[placeholder*="pregunta"]',
        'input[type="text"][placeholder*="pregunta"]',
        'textarea',
        'input[type="search"]',
    ],
    "submit": [
        'button:has-text("Preguntar")',
        'button:has-text("Enviar")',
        'button:has-text("Buscar")',
        'button[type="submit"]',
    ],
    "response": [
        '.meqa-response',
        '.meqa-answer',
        '#meqa-response',
        '[class*="respuesta"]',
        '[class*="answer"]',
    ],
}


async def find_element(page, selector_list, timeout=3000):
    """Try multiple selectors, return first match."""
    for sel in selector_list:
        try:
            el = await page.wait_for_selector(sel, timeout=timeout)
            if el:
                return el
        except Exception:
            continue
    return None


async def submit_query(page, query_text: str, timeout_ms=30000) -> dict:
    """Submit one query to MeQA and capture response."""
    try:
        input_el = await find_element(page, SELECTORS["input"])
        if not input_el:
            return {"error": "Input field not found"}

        await input_el.click()
        await input_el.fill("")
        await input_el.fill(query_text)
        await asyncio.sleep(0.5)

        submit_el = await find_element(page, SELECTORS["submit"], timeout=2000)
        if submit_el:
            await submit_el.click()
        else:
            await input_el.press("Enter")

        await asyncio.sleep(3)

        resp_el = await find_element(page, SELECTORS["response"], timeout=timeout_ms)
        response_text = await resp_el.inner_text() if resp_el else ""

        if not response_text:
            response_text = await page.inner_text("body")

        return {
            "response_text": response_text,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


async def run():
    from playwright.async_api import async_playwright

    # Load queries
    csv_path = QUERIES_DIR / "query_battery.csv"
    if not csv_path.exists():
        print(f"Query file not found: {csv_path}")
        print("Run: python scripts/generate_queries.py")
        return

    with open(csv_path, encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    # Load checkpoint
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = RESPONSES_DIR / "checkpoint.json"
    completed = set()
    if checkpoint.exists():
        with open(checkpoint) as f:
            completed = set(json.load(f))

    pending = [q for q in queries if q["query_id"] not in completed]
    print(f"Total: {len(queries)} | Completed: {len(completed)} | Pending: {len(pending)}")

    if not pending:
        print("All queries completed!")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(MEQA_URL)
        await asyncio.sleep(3)

        print("\n" + "="*60)
        print("Check the page loaded. Adapt SELECTORS in this script if needed.")
        print("="*60)
        input("Press Enter to start submitting queries...")

        for i, q in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}] {q['query_id']}: {q['query_text'][:60]}...")

            result = await submit_query(page, q["query_text"])
            record = {**q, **result}

            # Save individual response
            resp_file = RESPONSES_DIR / f"{q['query_id']}.json"
            with open(resp_file, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            # Update checkpoint
            completed.add(q["query_id"])
            with open(checkpoint, "w") as f:
                json.dump(list(completed), f)

            if "error" in result:
                print(f"  [ERROR] {result['error']}")
            else:
                wc = len(result.get("response_text", "").split())
                print(f"  ✓ {wc} words")

            await asyncio.sleep(MEQA_QUERY_DELAY)

        await browser.close()

    print(f"\n✓ Done. Responses in {RESPONSES_DIR}")


if __name__ == "__main__":
    asyncio.run(run())
