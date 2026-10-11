from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORY = "Cell_Phones_and_Accessories"
COLUMNS = ["asin", "user_id", "rating", "text", "timestamp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a bounded sample from Amazon Reviews 2023.")
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--max-scan", type=int, default=1_000_000,
                        help="Maximum source records read; prevents a full-category scan.")
    parser.add_argument("--target-products", type=int, default=1_000)
    parser.add_argument("--per-product-cap", type=int, default=100)
    parser.add_argument("--shuffle-buffer", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/interim/reviews_sample.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.max_scan, args.target_products, args.per_product_cap, args.shuffle_buffer) <= 0:
        raise SystemExit("Scan, product, per-product, and shuffle limits must be positive.")

    config = f"raw_review_{args.category}"
    revision = HfApi().dataset_info(DATASET_ID).sha
    data_url = f"hf://datasets/{DATASET_ID}@{revision}/raw/review_categories/{args.category}.jsonl"
    print(f"Opening {data_url} with datasets JSON streaming...")
    stream = load_dataset("json", data_files={"full": data_url}, split="full", streaming=True)
    if hasattr(stream, "shuffle"):
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    source_fields = list(stream.features.keys()) if stream.features else []
    counts: Counter[str] = Counter()
    candidates: list[dict] = []
    scanned = 0
    for row in stream:
        if not source_fields: source_fields = list(row.keys())
        scanned += 1
        if scanned % 100_000 == 0:
            print(f"Scanned {scanned:,} source rows; usable candidates={len(candidates):,}", flush=True)
        asin = row.get("asin")
        user_id = row.get("user_id")
        text = row.get("text")
        rating = row.get("rating")
        timestamp = row.get("timestamp")
        if not asin or not user_id or not isinstance(text, str) or not text.strip():
            if scanned >= args.max_scan:
                break
            continue
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            if scanned >= args.max_scan:
                break
            continue
        if not 1 <= rating_value <= 5:
            if scanned >= args.max_scan:
                break
            continue
        product = str(asin)
        counts[product] += 1
        candidates.append({
            "asin": product,
            "user_id": str(user_id),
            "rating": rating_value,
            "text": text,
            "timestamp": timestamp,
            "_sample_order": scanned,
        })
        if scanned >= args.max_scan:
            break

    selected_products = {asin for asin, _ in counts.most_common(args.target_products)}
    kept: list[dict] = []
    per_product: Counter[str] = Counter()
    for row in candidates:
        asin = row["asin"]
        if asin in selected_products and per_product[asin] < args.per_product_cap:
            per_product[asin] += 1
            kept.append({key: value for key, value in row.items() if key != "_sample_order"})

    if not kept:
        raise SystemExit("No usable records found. Check the dataset configuration and network access.")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kept, columns=COLUMNS).to_csv(output, index=False)

    metadata = {
        "dataset_id": DATASET_ID,
        "config": config,
        "split": "full",
        "streaming": True,
        "source_revision": revision,
        "category": args.category,
        "category_reference_estimates": {
            "reviews_or_ratings": 20_800_000,
            "users": 11_600_000,
            "products": 1_300_000,
            "source": "Amazon Reviews 2023 dataset card, rounded category totals",
        } if args.category == CATEGORY else {"reviews_or_ratings": None, "users": None, "products": None, "source": "unknown"},
        "source_fields": source_fields,
        "records_scanned": scanned,
        "valid_candidates": len(candidates),
        "sample_rows_written": len(kept),
        "sample_products_written": len(per_product),
        "sample_users_written": len({row["user_id"] for row in kept}),
        "sampling": {
            "seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
            "max_scan": args.max_scan,
            "target_products": args.target_products,
            "per_product_cap": args.per_product_cap,
            "note": "Bounded shuffled stream sample; observed sample counts are not full-category totals.",
        },
        "output": str(output),
    }
    metadata_path = output.with_name("sampling_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
