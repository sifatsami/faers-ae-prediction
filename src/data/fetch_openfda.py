import os
import time
import json
import requests
from pathlib import Path

BASE = "https://api.fda.gov/drug/event.json"
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_events(query: str, limit_per_call: int = 100, max_records: int = 500):
    fetched = 0
    skip = 0
    page = 0

    while fetched < max_records:
        params = {"limit": limit_per_call, "skip": skip, "search": query}
        r = requests.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results")
        if not results:
            print("No more results or empty page.")
            break

        out_path = OUT_DIR / f"openfda_page_{page:03d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

        print(f"Saved {len(results)} records to {out_path}")

        fetched += len(results)
        skip += limit_per_call
        page += 1
        time.sleep(0.5)

    print(f"Finished. Total fetched: {fetched}")

if __name__ == "__main__":
    q = 'patient.drug.medicinalproduct:"aspirin"'
    fetch_events(q)
