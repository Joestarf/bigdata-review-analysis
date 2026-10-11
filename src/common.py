from __future__ import annotations
import hashlib
import html
import io
import json
import os
import platform
import re
import tempfile
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["asin", "user_id", "rating", "text", "timestamp"]
LABELS = {"exact": 1, "near": 1, "distinct": 0, "uncertain": None}
SEED = 42
HTML_TAGS = re.compile(r"</?(?:br|p|div|span|b|i|em|strong|a|ul|ol|li|table|tr|td|img|hr|h[1-6])(?:\s[^<>]*)?\s*/?>", re.I)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ARTIFACT_VERSION = 1

def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()

def word_count(value: str) -> int:
    return len(str(value).split())

def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = CONTROL.sub(" ", value)
    value = HTML_TAGS.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()

def read_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype={"asin": str, "user_id": str, "text": str})
    missing = set(COLUMNS) - set(df)
    if missing:
        raise ValueError(f"Missing review columns: {sorted(missing)}")
    return df[COLUMNS].copy()

def clean_frame(raw: pd.DataFrame, max_length: int = 10000):
    df = raw[COLUMNS].copy()
    df["asin"] = df.asin.fillna("").astype(str).str.strip()
    df["user_id"] = df.user_id.fillna("").astype(str).str.strip()
    df["text"] = df.text.map(clean_text)
    rating = pd.to_numeric(df.rating, errors="coerce")
    stamp = pd.to_numeric(df.timestamp, errors="coerce")
    reasons = pd.DataFrame({
        "missing_asin": df.asin.eq(""),
        "missing_user_id": df.user_id.eq(""),
        "empty_text": df.text.eq(""),
        "placeholder_text": df.text.str.strip().isin(["N/A","NA","None"]),
        "invalid_rating": ~(np.isfinite(rating) & rating.between(1, 5) & rating.mod(1).eq(0)),
        "invalid_timestamp": ~(np.isfinite(stamp) & stamp.ge(0) & stamp.le(253402300799999) & stamp.mod(1).eq(0)),
        "text_too_long": df.text.str.len().gt(max_length),
    })
    invalid = reasons.any(axis=1)
    excluded = pd.DataFrame({
        "input_row": np.flatnonzero(invalid.to_numpy()),
        "reasons": ["|".join(row.index[row.to_numpy(dtype=bool)]) for _, row in reasons.loc[invalid].iterrows()],
        "text_length": df.loc[invalid, "text"].str.len().values,
    })
    df = df.loc[~invalid].copy()
    df["rating"] = rating.loc[~invalid].astype(float)
    df["timestamp"] = stamp.loc[~invalid].astype("int64")
    report = {
        "input_rows": len(raw), "processed_rows": len(df), "removed_rows": int(invalid.sum()),
        "reason_counts_overlapping": {k: int(v) for k, v in reasons.sum().items()},
        "max_text_length": max_length, "preserved_duplicate_rows": True,
        "long_text_range": None,
    }
    long = raw.loc[reasons.text_too_long, "text"].map(clean_text).str.len()
    if len(long):
        report["long_text_range"] = {"min": int(long.min()), "max": int(long.max())}
    return df.reset_index(drop=True), excluded, report

def atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def write_json(path: Path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_csv(path: Path, df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    atomic_bytes(path, buf.getvalue().encode("utf-8"))

def packages():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ["datasets", "numpy", "pandas", "scikit-learn", "sentence-transformers", "torch", "matplotlib", "psutil"]:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out

def pair_id(a: str, b: str) -> str:
    return sha_text("|".join(sorted([a, b])))

def load_study(root: Path = ROOT):
    manifest = read_json(root / "data/processed/dataset_manifest.json")
    if sha_file(root / "data/processed/reviews.csv") != manifest["dataset_version"]:
        raise ValueError("Frozen dataset changed. Run audit and rebuild downstream artifacts; existing labels must be revalidated.")
    for relative, digest in manifest.get("index_hashes", {}).items():
        if sha_file(root / relative) != digest:
            raise ValueError(f"Frozen index or split changed: {relative}")
    idx = pd.read_csv(root / "data/processed/review_index.csv", keep_default_na=False)
    reviews = read_reviews(root / "data/processed/reviews.csv")
    if len(reviews) != len(idx):
        raise ValueError("Review index row count mismatch.")
    df = pd.concat([idx.reset_index(drop=True), reviews.reset_index(drop=True)], axis=1)
    assignments = pd.read_csv(root / "data/processed/split_assignments.csv", keep_default_na=False)
    return df.merge(assignments, on="review_id", validate="one_to_one"), manifest
