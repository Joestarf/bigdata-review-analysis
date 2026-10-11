from __future__ import annotations
import argparse
from pathlib import Path
try:
    from .common import ROOT, clean_text, clean_frame, read_reviews, write_csv, write_json
except ImportError:
    from common import ROOT, clean_text, clean_frame, read_reviews, write_csv, write_json

def main():
    parser = argparse.ArgumentParser(description="Validate and clean reviews without silently replacing a frozen dataset.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/interim/reviews_sample.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/reviews.csv")
    parser.add_argument("--max-text-length", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing data and invalidate downstream versions.")
    args = parser.parse_args()
    df, excluded, report = clean_frame(read_reviews(args.input), args.max_text_length)
    if args.output.exists() and not args.overwrite:
        old = read_reviews(args.output)
        if not old.equals(df):
            raise SystemExit("Existing sample differs from improved cleaning. Use a new --output for inspection or explicitly --overwrite and rebuild versions.")
    write_csv(args.output, df)
    write_csv(args.output.with_name("excluded_rows.csv"), excluded)
    write_json(args.output.with_name("preprocess_summary.json"), report)
    print(report)
if __name__ == "__main__":
    main()
