from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLUMNS = ["asin", "user_id", "rating", "text", "timestamp"]


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore sample size, ratings, text lengths, and repeated text.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/reviews.csv"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--metadata", type=Path, default=Path("data/interim/sampling_metadata.json"))
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}. Run download_amazon.py and preprocess.py first.")
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype={"asin": "string", "user_id": "string"})
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {sorted(missing)}")
    if df.empty:
        raise SystemExit("Processed data is empty.")
    df["text"] = df["text"].fillna("").astype(str)
    df["text_length_chars"] = df["text"].str.len()
    df["normalized_text"] = df["text"].map(normalized_text)

    product_counts = df.groupby("asin", dropna=False).size().rename("review_count").reset_index()
    product_counts = product_counts.sort_values(["review_count", "asin"], ascending=[False, True])
    product_counts.to_csv(args.tables / "product_review_counts.csv", index=False)
    product_summary = {
        "products": int(df["asin"].nunique()),
        "reviews": int(len(df)),
        "reviews_per_product_mean": float(product_counts["review_count"].mean()),
        "reviews_per_product_median": float(product_counts["review_count"].median()),
        "reviews_per_product_min": int(product_counts["review_count"].min()),
        "reviews_per_product_max": int(product_counts["review_count"].max()),
    }

    rating_counts = df["rating"].value_counts().sort_index().rename_axis("rating").rename("review_count").reset_index()
    rating_counts["share"] = rating_counts["review_count"] / len(df)
    rating_counts.to_csv(args.tables / "rating_distribution.csv", index=False)

    lengths = df["text_length_chars"]
    length_summary = {
        "unit": "characters after whitespace normalization",
        "mean": float(lengths.mean()),
        "std": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
        "min": int(lengths.min()),
        "p25": float(lengths.quantile(0.25)),
        "median": float(lengths.median()),
        "p75": float(lengths.quantile(0.75)),
        "p90": float(lengths.quantile(0.90)),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }

    text_counts = df.groupby("normalized_text", dropna=False).size()
    repeated_groups = text_counts[text_counts > 1]
    repeated_rows = int(repeated_groups.sum())
    repeated_extra_rows = int((repeated_groups - 1).sum())
    cross_user_groups = 0
    cross_user_rows = 0
    grouped = df.groupby("normalized_text", dropna=False)
    for _, group in grouped:
        if len(group) > 1 and group["user_id"].nunique() > 1:
            cross_user_groups += 1
            cross_user_rows += len(group)
    duplicate_summary = {
        "method": "exact match after trimming, whitespace collapse, and case-folding",
        "interpretation": "textual redundancy signal only; not a judgment about reviewer intent or authenticity",
        "unique_normalized_texts": int(len(text_counts)),
        "repeated_text_groups": int(len(repeated_groups)),
        "rows_in_repeated_text_groups": repeated_rows,
        "extra_repeated_rows_after_first_copy": repeated_extra_rows,
        "extra_repeated_row_share": repeated_extra_rows / len(df),
        "repeated_groups_seen_across_multiple_users": int(cross_user_groups),
        "rows_in_cross_user_repeated_groups": int(cross_user_rows),
    }

    fields = [
        ("rating", "float", "评分，1–5 星"),
        ("title", "string", "评论标题"),
        ("text", "string", "评论正文；本项目使用该字段"),
        ("images", "list", "评论附图"),
        ("asin", "string", "商品 ASIN"),
        ("parent_asin", "string", "父级商品 ID"),
        ("user_id", "string", "匿名评论者 ID"),
        ("timestamp", "integer", "Unix 毫秒时间戳"),
        ("helpful_vote", "integer", "有帮助票数"),
        ("verified_purchase", "boolean", "是否验证购买"),
    ]
    pd.DataFrame(fields, columns=["field", "source_type", "description"]).to_csv(args.tables / "source_fields.csv", index=False)
    if args.metadata.exists():
        sampling = json.loads(args.metadata.read_text(encoding="utf-8"))
        source_fields = sampling.get("source_fields", [])
        category_reference = sampling.get("category_reference_estimates", {})
    else:
        sampling, source_fields, category_reference = {}, [], {}

    summary = {
        "dataset_id": sampling.get("dataset_id", "McAuley-Lab/Amazon-Reviews-2023"),
        "category": sampling.get("config", "raw_review_Cell_Phones_and_Accessories").replace("raw_review_", ""),
        "processed_sample": {
            "reviews": int(len(df)),
            "products": int(df["asin"].nunique()),
            "users": int(df["user_id"].nunique()),
        },
        "official_category_reference_estimates": category_reference,
        "sampling": sampling.get("sampling", {}),
        "observed_source_fields": source_fields,
        "product_review_counts": product_summary,
        "rating_distribution": [
            {"rating": float(row.rating), "count": int(row.review_count), "share": float(row.share)}
            for row in rating_counts.itertuples(index=False)
        ],
        "review_length_characters": length_summary,
        "exact_repeated_text": duplicate_summary,
        "caveats": [
            "The local sample is bounded and shuffled; it is not a full-category census.",
            "Exact textual repetition identifies redundant wording, not fake reviews or coordinated behavior.",
            "Products are identified by asin to match the required output schema.",
        ],
    }
    (args.tables / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.bar(rating_counts["rating"].astype(str), rating_counts["review_count"], color="#3977a8")
    plt.xlabel("Star rating")
    plt.ylabel("Review count")
    plt.title("Star Rating Distribution")
    plt.tight_layout()
    plt.savefig(args.figures / "star_distribution.png", dpi=160)
    plt.close()

    upper = max(100, int(lengths.quantile(0.99)))
    plt.figure(figsize=(8, 5))
    plt.hist(lengths.clip(upper=upper), bins=50, color="#4c9a78", edgecolor="white")
    plt.xlabel("Review length (characters; clipped at p99)")
    plt.ylabel("Review count")
    plt.title("Review Length Distribution")
    plt.tight_layout()
    plt.savefig(args.figures / "review_length_distribution.png", dpi=160)
    plt.close()

    top = product_counts.head(20).sort_values("review_count")
    plt.figure(figsize=(9, 7))
    plt.barh(top["asin"].astype(str), top["review_count"], color="#b87a38")
    plt.xlabel("Review count")
    plt.ylabel("Product ASIN")
    plt.title("Reviews for the 20 Most Reviewed Products")
    plt.tight_layout()
    plt.savefig(args.figures / "top_product_review_counts.png", dpi=160)
    plt.close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Generated tables:", args.tables)
    print("Generated figures:", args.figures)


if __name__ == "__main__":
    main()
