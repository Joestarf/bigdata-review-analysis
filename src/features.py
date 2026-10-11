from __future__ import annotations
import re
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .common import ROOT, load_study, sha_text, read_json, write_json, normalize_text, sha_file

LEXICAL_FEATURES = ["char_cosine", "word_jaccard", "length_ratio", "length_difference", "exact_match"]
SEMANTIC_FEATURES = LEXICAL_FEATURES + ["semantic_cosine"]

def word_set(text):
    return set(re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", normalize_text(text)))

def structural_features(a, b):
    wa, wb = word_set(a), word_set(b)
    denom = len(wa | wb)
    la, lb = len(a), len(b)
    return [len(wa & wb)/denom if denom else 0.0,
            min(la, lb)/max(la, lb) if max(la, lb) else 1.0,
            abs(la-lb), float(normalize_text(a) == normalize_text(b))]

def fit_lexical(root: Path=ROOT):
    df, manifest = load_study(root)
    train = df[(df.split == "train") & df.supervised_eligible]
    fingerprint = sha_text("|".join(train.review_id))
    path = root / "models/lexical.joblib"
    metapath = root / "models/lexical_manifest.json"
    if path.exists() and metapath.exists():
        meta = read_json(metapath)
        if meta["dataset_version"] != manifest["dataset_version"] or meta["training_corpus_hash"] != fingerprint:
            raise ValueError("Stale TF-IDF model; remove/rebuild downstream artifacts after data change.")
        if "artifact_sha256" in meta and sha_file(path) != meta["artifact_sha256"]:
            raise ValueError("TF-IDF artifact checksum mismatch.")
        fitted = joblib.load(path)
        if fitted.analyzer != "char_wb" or fitted.ngram_range != (3,5) or fitted.min_df != 2 or fitted.max_features != 100000:
            raise ValueError("TF-IDF configuration mismatch.")
        if "artifact_sha256" not in meta:
            meta["artifact_sha256"] = sha_file(path)
            write_json(metapath,meta)
        return fitted
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100000,
                                dtype=np.float32)
    vectorizer.fit(train.text.tolist())
    path.parent.mkdir(exist_ok=True)
    joblib.dump(vectorizer, path)
    write_json(metapath, {"dataset_version": manifest["dataset_version"], "training_corpus_hash": fingerprint,
            "training_rows": len(train), "fitted_split": "train", "features": LEXICAL_FEATURES,
            "artifact_sha256": sha_file(path), "ngram_range": [3, 5], "max_features": 100000, "min_df": 2})
    return vectorizer

def text_pair_features(vectorizer, a, b, semantic=None):
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("Paired text lists must have equal lengths.")
    if not a:
        return np.empty((0, 6 if semantic is not None else 5), dtype=np.float32)
    xa, xb = vectorizer.transform(a), vectorizer.transform(b)
    cosine = np.asarray(xa.multiply(xb).sum(axis=1)).ravel()
    rows = [[float(cosine[i])] + structural_features(ta, tb) for i, (ta, tb) in enumerate(zip(a,b))]
    arr = np.asarray(rows, dtype=np.float32)
    if semantic is not None:
        arr = np.column_stack([arr, np.asarray(semantic, dtype=np.float32)])
    return arr

def indexed_pair_features(pairs, df, vectorizer, embeddings, semantic=True):
    positions = {r: i for i, r in enumerate(df.review_id)}
    a = np.array([positions[r] for r in pairs.review_id_a])
    b = np.array([positions[r] for r in pairs.review_id_b])
    sims = np.einsum("ij,ij->i", embeddings[a], embeddings[b]) if semantic else None
    return text_pair_features(vectorizer, df.iloc[a].text, df.iloc[b].text, sims)
