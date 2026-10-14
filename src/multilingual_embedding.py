"""Versioned frozen multilingual encoding; long inputs use token windows."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
CONFIG = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "dimensions": 384,
          "window_tokens": 126, "overlap_tokens": 24,
          "aggregation": "coverage_weighted_mean_then_l2", "dtype": "float32",
          "schema": "multilingual-v2.1"}

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()

class MultilingualEncoder:
    def __init__(self, device=None, batch_size=64):
        if batch_size not in (8, 16, 32, 64):
            raise ValueError("batch_size must be 8, 16, 32 or 64")
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        started = time.perf_counter()
        self.model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION,
                                         device=self.device, trust_remote_code=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.model.get_embedding_dimension() != CONFIG["dimensions"]:
            raise ValueError("Unexpected encoder dimensions")
        if self.model.max_seq_length != 128:
            raise ValueError("Unexpected encoder maximum sequence length")
        self.load_seconds = time.perf_counter() - started

    def _windows(self, texts):
        """Build token-ID windows without decoding/re-tokenization or discarded tails."""
        tokenizer = self.model.tokenizer
        windows, owners, weights, counts = [], [], [], []
        for owner, text in enumerate(texts):
            tokens = tokenizer(str(text), add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            size, overlap = CONFIG["window_tokens"], CONFIG["overlap_tokens"]
            starts = list(range(0, max(1, len(tokens) - overlap), size - overlap))
            counts.append(len(starts))
            for index, start in enumerate(starts):
                segment = tokens[start:start + size]
                # This pinned BERT tokenizer adds CLS/SEP; never decode token windows.
                if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
                    raise ValueError("Pinned tokenizer requires CLS/SEP IDs")
                ids = [tokenizer.cls_token_id] + segment + [tokenizer.sep_token_id]
                prepared = {"input_ids": ids, "attention_mask": [1]*len(ids)}
                windows.append(prepared)
                owners.append(owner)
                weights.append(max(1, len(segment) - (overlap if index else 0)))
        return windows, np.asarray(owners), np.asarray(weights), counts

    def encode(self, texts, cache_dir=None):
        texts = list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() or len(t)>10000 for t in texts):
            raise ValueError("Provide 1 or more nonempty strings, each at most 10000 characters")
        key = digest({"texts": texts, "config": CONFIG,
                      "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            path, meta = cache_dir / (key + ".npy"), cache_dir / (key + ".json")
            if path.exists() and meta.exists():
                receipt = json.loads(meta.read_text())
                if receipt.get("key") != key or receipt.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("Multilingual vector cache mismatch")
                result = np.load(path, allow_pickle=False)
                if result.shape != (len(texts), 384) or result.dtype != np.float32 or not np.isfinite(result).all():
                    raise ValueError("Invalid cached vector shape/dtype/values")
                if not np.allclose(np.linalg.norm(result,axis=1),1.,atol=1e-5):
                    raise ValueError("Unnormalized cached vectors")
                return result, {**receipt, "cache_hit": True}
        start = time.perf_counter()
        windows, owners, weights, counts = self._windows(texts)
        result = np.zeros((len(texts), 384), dtype=np.float64)
        totals = np.zeros(len(texts), dtype=np.float64)
        batch_size = self.batch_size
        offset = 0
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        while offset < len(windows):
            batch = windows[offset:offset+batch_size]
            try:
                features = self.model.tokenizer.pad(batch, padding=True, return_tensors="pt")
                features = {k:v.to(self.device) for k,v in features.items()}
                with torch.inference_mode():
                    embeddings = self.model(features)["sentence_embedding"].cpu().numpy()
                for j, vector in enumerate(embeddings):
                    i = offset+j
                    result[owners[i]] += vector * weights[i]
                    totals[owners[i]] += weights[i]
                offset += len(batch)
            except torch.OutOfMemoryError:
                if batch_size > 8:
                    batch_size //= 2
                elif self.device.startswith("cuda"):
                    self.device = "cpu"
                    self.model.to("cpu")
                else:
                    raise
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        result /= totals[:, None]
        norm = np.linalg.norm(result, axis=1, keepdims=True)
        result = (result / np.maximum(norm, 1e-12)).astype(np.float32)
        receipt = {"key": key, "config": CONFIG, "input_count": len(texts),
                   "window_count": len(windows), "window_counts": counts,
                   "long_inputs": int(sum(n>1 for n in counts)), "discarded_tokens": 0,
                   "device": self.device, "batch_size": batch_size,
                   "encode_seconds": time.perf_counter()-start,
                   "load_seconds": self.load_seconds, "cache_hit": False,
                   "peak_gpu_mib": torch.cuda.max_memory_allocated()/1048576 if self.device.startswith("cuda") else 0}
        if cache_dir:
            np.save(path, result, allow_pickle=False)
            receipt["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            meta.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
        return result, receipt
