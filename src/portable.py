from __future__ import annotations
"""Standalone, review-only inference from an explicitly packaged model bundle."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .common import pair_id
from .embedding import load_encoder, encode_texts
from .features import text_pair_features

METHODS = ["A", "B", "C", "D", "E"]
EXPECTED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_FEATURES = ["char_cosine", "word_jaccard", "length_ratio", "length_difference", "exact_match", "semantic_cosine"]
MAX_PAIRS = 1000
MAX_PRODUCT_COMMENTS = 1000
MAX_TEXT_CHARS = 10000
COLUMNS = ["pair_id", "asin", "review_id_a", "review_id_b", "method", "score", "decision", "reason"]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PortablePredictor:
    """Score pairs without loading the original corpus or annotation gates.

    Joblib files are executable pickle data; only load bundles from a trusted source.
    """

    def __init__(self, bundle_root):
        self.root = Path(bundle_root).resolve()
        manifest_path = self.root / "models/portable_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Portable manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("model_revision") != EXPECTED_REVISION:
            raise ValueError("Portable manifest uses an unexpected pretrained model revision.")
        if self.manifest.get("feature_order") != EXPECTED_FEATURES:
            raise ValueError("Portable manifest feature_order does not match the frozen feature implementation.")
        methods = self.manifest.get("methods")
        if not isinstance(methods, dict) or set(methods) != set(METHODS):
            raise ValueError("Portable manifest must define methods A through E.")
        for method in METHODS:
            config = methods[method]
            threshold = config.get("review_threshold")
            if (not isinstance(config.get("qualified"), bool) or
                    not isinstance(threshold, (int, float)) or not np.isfinite(threshold)):
                raise ValueError(f"Invalid portable method configuration for {method}.")
        expected_files = {"models/lexical.joblib", "models/C.joblib", "models/E.joblib"}
        checksums = self.manifest.get("files_sha256")
        if not isinstance(checksums, dict) or set(checksums) != expected_files:
            raise ValueError("Portable manifest must checksum lexical.joblib, C.joblib, and E.joblib.")
        for relative in sorted(expected_files):
            path = (self.root / relative).resolve()
            if self.root not in path.parents or not path.is_file():
                raise ValueError(f"Missing or invalid portable bundle file: {relative}")
            if _sha256(path) != checksums[relative]:
                raise ValueError(f"Portable bundle checksum mismatch: {relative}")
        self.vectorizer = joblib.load(self.root / "models/lexical.joblib")
        self.models = {method: joblib.load(self.root / f"models/{method}.joblib") for method in ("C", "E")}
        self.encoder = None

    def _encoder(self):
        if self.encoder is None:
            self.encoder = load_encoder(self.manifest["model_revision"])
        return self.encoder

    @staticmethod
    def _validate_text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Review texts must be nonempty strings.")
        if len(value) > MAX_TEXT_CHARS:
            raise ValueError(f"Review text exceeds {MAX_TEXT_CHARS} characters.")

    def _score(self, pairs, asin, ids):
        if not pairs:
            return pd.DataFrame(columns=COLUMNS)
        for a, b in pairs:
            self._validate_text(a)
            self._validate_text(b)
        unique = list(dict.fromkeys(text for pair in pairs for text in pair))
        encoder = self._encoder()
        vectors, _, _ = encode_texts(encoder, unique, 64)
        lookup = dict(zip(unique, vectors))
        semantic = [float(np.dot(lookup[a], lookup[b])) for a, b in pairs]
        x = text_pair_features(self.vectorizer, [a for a, _ in pairs], [b for _, b in pairs], semantic)
        if x.shape != (len(pairs), 6) or not np.isfinite(x).all():
            raise ValueError("Feature generation returned an invalid matrix.")
        scores = {
            "A": x[:, 4], "B": x[:, 0],
            "C": self.models["C"].predict_proba(x[:, :5])[:, 1],
            "D": x[:, 5], "E": self.models["E"].predict_proba(x)[:, 1],
        }
        token_lengths = encoder.tokenizer(
            unique, padding=False, truncation=False, return_length=True, verbose=False
        )["length"]
        truncated = {text: length > 256 for text, length in zip(unique, token_lengths)}
        rows = []
        for i, ((a, b), (id_a, id_b)) in enumerate(zip(pairs, ids)):
            diagnostic_flags = []
            if len(a.split()) < 5 or len(b.split()) < 5:
                diagnostic_flags.append("short_text")
            if truncated[a] or truncated[b]:
                diagnostic_flags.append("encoder_input_truncated")
            reason = "portable_un-calibrated_score; independent_review_required"
            if diagnostic_flags:
                reason += "; " + "; ".join(diagnostic_flags)
            for method in METHODS:
                score = float(scores[method][i])
                if not np.isfinite(score):
                    raise ValueError("Model produced a nonfinite score.")
                rows.append({"pair_id": pair_id(id_a, id_b), "asin": asin, "review_id_a": id_a,
                             "review_id_b": id_b, "method": method, "score": score,
                             "decision": "review", "reason": reason})
        return pd.DataFrame(rows, columns=COLUMNS)

    def predict_pairs(self, pairs):
        """Score up to 1,000 (text_a, text_b) pairs; return five rows per pair."""
        pairs = list(pairs)
        if len(pairs) > MAX_PAIRS:
            raise ValueError(f"At most {MAX_PAIRS} pairs are accepted per call.")
        if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
            raise ValueError("Each pair must contain exactly two texts.")
        pairs = [tuple(pair) for pair in pairs]
        for a, b in pairs:
            self._validate_text(a)
            self._validate_text(b)
        ids = [("input_" + hashlib.sha256(a.encode("utf-8")).hexdigest(),
                "input_" + hashlib.sha256(b.encode("utf-8")).hexdigest()) for a, b in pairs]
        return self._score(pairs, "", ids)

    def score_product(self, reviews):
        """Score all within-product pairs for a DataFrame with asin, review_id, text."""
        if not isinstance(reviews, pd.DataFrame) or not {"asin", "review_id", "text"}.issubset(reviews.columns):
            raise ValueError("Product input needs asin, review_id, and text columns.")
        if len(reviews) > MAX_PRODUCT_COMMENTS:
            raise ValueError(f"At most {MAX_PRODUCT_COMMENTS} comments are accepted per product.")
        if reviews.empty:
            return pd.DataFrame(columns=COLUMNS)
        data = reviews[["asin", "review_id", "text"]].copy()
        for field in ("asin", "review_id"):
            if not data[field].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"{field} values must be nonempty strings.")
        if data["asin"].nunique() != 1 or not data["review_id"].is_unique:
            raise ValueError("Use one asin and unique review_id values per product.")
        for value in data["text"]:
            self._validate_text(value)
        records = list(data.itertuples(index=False, name=None))
        combinations = list(itertools.combinations(records, 2))
        pairs = [(left[2], right[2]) for left, right in combinations]
        ids = [(left[1], right[1]) for left, right in combinations]
        return self._score(pairs, records[0][0], ids)


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Portable review-pair scoring; decisions always remain review.")
    parser.add_argument("--bundle", required=True, help="Directory containing models/portable_manifest.json")
    parser.add_argument("--input", required=True, help="CSV containing asin, review_id, text")
    parser.add_argument("--output", required=True, help="Output CSV")
    args = parser.parse_args(argv)
    reviews = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    required = {"asin", "review_id", "text"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    if not reviews.empty:
        for field in ("asin", "review_id", "text"):
            valid = reviews[field].map(lambda value: isinstance(value, str) and bool(value.strip()))
            if not valid.all():
                raise ValueError(f"Input CSV requires nonempty {field} values.")
        if not reviews["review_id"].is_unique:
            raise ValueError("Input CSV review_id values must be globally unique across products.")
        reviews["review_id"] = reviews["review_id"].astype(str)
    predictor = PortablePredictor(args.bundle)
    if reviews.empty:
        results = pd.DataFrame(columns=COLUMNS)
    else:
        results = pd.concat(
            [predictor.score_product(group) for _, group in reviews.groupby("asin", sort=False)],
            ignore_index=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Wrote {len(results)} method rows to {output}")


if __name__ == "__main__":
    _main()
