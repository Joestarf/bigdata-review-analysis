from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, sha_text, sha_file, load_study, read_json, write_json, write_csv

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256

def cache_key(dataset_version, text_hash, revision):
    return sha_text(json.dumps({"dataset_version": dataset_version, "texts_hash": text_hash,
            "model": MODEL_ID, "revision": revision, "max_tokens": MAX_TOKENS,
            "dimension": 384, "dtype": "float32", "normalize": True}, sort_keys=True))

def load_embedding_cache(root=ROOT):
    df, manifest = load_study(root)
    path = root / "models/embedding_manifest.json"
    if not path.exists():
        raise ValueError("No semantic cache. Run: python -m src.study embed")
    meta = read_json(path)
    thash = sha_text(json.dumps(df.text.tolist(), ensure_ascii=False, separators=(",", ":")))
    expected = cache_key(manifest["dataset_version"], thash, meta["model_revision"])
    if meta["cache_key"] != expected:
        raise ValueError("Embedding cache is stale. Dataset/model/token configuration differs.")
    arr = np.load(root / "models/review_embeddings.npy", mmap_mode="r")
    index = pd.read_csv(root / "models/embedding_index.csv", keep_default_na=False)
    if arr.shape != (len(df), 384) or arr.dtype != np.float32 or index.review_id.tolist() != df.review_id.tolist():
        raise ValueError("Embedding index alignment failed.")
    if sha_file(root / "models/embedding_index.csv") != meta["index_sha256"]:
        raise ValueError("Embedding index checksum mismatch.")
    if sha_file(root / "models/review_embeddings.npy") != meta["array_sha256"]:
        raise ValueError("Embedding cache file is corrupted.")
    return arr, index, meta

def load_encoder(revision, device=None):
    import torch
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_ID, revision=revision, device=device)
    model.max_seq_length = MAX_TOKENS
    return model

def encode_texts(model, texts, batch_size=64, progress=False):
    import torch
    current = batch_size
    while True:
        try:
            vectors = model.encode(texts, batch_size=current, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=progress, precision="float32")
            return np.asarray(vectors, dtype=np.float32), current, str(model.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current > 8:
                current = max(8, current // 2)
            else:
                model.to("cpu")
                current = 8

def run_embeddings(root: Path=ROOT):
    df, manifest = load_study(root)
    meta_path = root / "models/embedding_manifest.json"
    if meta_path.exists():
        arr, _, meta = load_embedding_cache(root)
        print(f"Using verified semantic cache: {arr.shape}; revision={meta['model_revision']}")
        return meta
    from huggingface_hub import HfApi
    import torch
    from .performance import Monitor
    started = time.perf_counter()
    revision = HfApi().model_info(MODEL_ID).sha
    model = load_encoder(revision)
    loading_seconds = time.perf_counter() - started
    unique = list(dict.fromkeys(df.text.tolist()))
    text_position = {t: i for i, t in enumerate(unique)}
    lengths = []
    for start in range(0, len(unique), 1000):
        tokens = model.tokenizer(unique[start:start + 1000], padding=False, truncation=False,
                                 return_length=True, verbose=False)
        lengths.extend(tokens["length"])
    with Monitor(gpu=True) as monitor:
        encoded, batch_size, device = encode_texts(model, unique, progress=True)
        vectors = encoded[np.array([text_position[t] for t in df.text], dtype=int)]
    out = root / "models"
    out.mkdir(exist_ok=True)
    temp = out / "review_embeddings.partial.npy"
    np.save(temp, vectors)
    os.replace(temp, out / "review_embeddings.npy")
    lengths = np.asarray(lengths)[np.array([text_position[t] for t in df.text], dtype=int)]
    index = pd.DataFrame({"review_id": df.review_id, "embedding_row": np.arange(len(df)),
                          "token_count": lengths, "truncated": lengths > MAX_TOKENS})
    write_csv(out / "embedding_index.csv", index)
    thash = sha_text(json.dumps(df.text.tolist(), ensure_ascii=False, separators=(",", ":")))
    meta = {"model_id": MODEL_ID, "model_revision": revision, "max_tokens": MAX_TOKENS,
            "dimension": 384, "dataset_version": manifest["dataset_version"], "texts_hash": thash,
            "cache_key": cache_key(manifest["dataset_version"], thash, revision),
            "array_sha256": sha_file(out / "review_embeddings.npy"),
            "index_sha256": sha_file(out / "embedding_index.csv"),
            "rows": len(df), "unique_texts_encoded": len(unique), "truncated_rows": int(index.truncated.sum()),
            "batch_size": batch_size, "device": device, "model_loading_or_download_seconds": loading_seconds,
            "encoding": monitor.result, "dtype": "float32", "normalized": True,
            "texts_sent_to_external_service": False}
    write_json(meta_path, meta)
    print(json.dumps(meta, indent=2))
    return meta
