from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, SEED, pair_id, load_study, read_json, write_json, write_csv, sha_file, word_count
from .features import fit_lexical
from .embedding import load_embedding_cache

QUOTAS = {"train": 300, "validation": 150, "test": 150}
STRATA = ["lexical", "semantic_disagreement", "random", "short"]

def validate_pairs(pairs, df):
    if not pairs.pair_id.is_unique:
        raise ValueError("Duplicate pair IDs.")
    lookup = df.set_index("review_id")
    for row in pairs.itertuples():
        if row.review_id_a == row.review_id_b:
            raise ValueError("Self pair.")
        if pair_id(row.review_id_a, row.review_id_b) != row.pair_id:
            raise ValueError("Pair ID does not match endpoints.")
        a, b = lookup.loc[row.review_id_a], lookup.loc[row.review_id_b]
        if a.asin != b.asin or a.asin != row.asin:
            raise ValueError("Cross-product pair.")
        if a.split != b.split or a.split != row.split or row.split not in QUOTAS:
            raise ValueError("Cross-split pair.")
        if not a.supervised_eligible or not b.supervised_eligible or a.source_id == b.source_id:
            raise ValueError("Ineligible/same-source pair.")

def prepare_pairs(root: Path=ROOT):
    df, manifest = load_study(root)
    arr, _, emeta = load_embedding_cache(root)
    vectorizer = fit_lexical(root)
    out = root / "data/annotation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pair_manifest.json"
    if path.exists():
        meta = read_json(path)
        if meta["dataset_version"] != manifest["dataset_version"] or meta["embedding_cache_key"] != emeta["cache_key"]:
            raise ValueError("Pair tasks are stale. Preserve existing labels and rebuild under a new data version.")
        pairs = pd.read_csv(out / "pair_metadata.csv", keep_default_na=False)
        if sha_file(out / "pair_metadata.csv") != meta["pair_table_sha256"]:
            raise ValueError("Candidate table changed after manifest freeze.")
        validate_pairs(pairs, df)
        print("Using verified existing 600-pair annotation tasks.")
        return pairs
    rng = np.random.default_rng(SEED)
    pools = {s: {k: [] for k in STRATA} for s in QUOTAS}
    for asin, group in df[df.supervised_eligible].groupby("asin", sort=True):
        n = len(group)
        if n < 2:
            continue
        split = group.split.iloc[0]
        x = vectorizer.transform(group.text)
        lex = (x @ x.T).toarray()
        emb = np.asarray(arr[group.index.to_numpy()])
        sem = np.clip(emb @ emb.T, -1, 1)
        ia, ib = np.triu_indices(n, 1)
        short = np.array([word_count(t) < 5 for t in group.text])
        lp, sp = lex[ia,ib], sem[ia,ib]
        shortpair = short[ia] | short[ib]
        texts = group.text.tolist()
        ids = group.review_id.tolist()
        def row(k):
            a, b = ids[ia[k]], ids[ib[k]]
            return {"pair_id": pair_id(a,b), "asin": asin, "review_id_a": a, "review_id_b": b,
                    "split": split, "lexical_score": float(lp[k]), "semantic_score": float(sp[k]),
                    "short_pair": bool(shortpair[k])}
        def add(kind, indices):
            pools[split][kind].extend(row(int(k)) for k in indices)
        validlong = np.flatnonzero(~shortpair)
        add("lexical", sorted(validlong, key=lambda k: (-lp[k], -sp[k], int(k)))[:8])
        semindices = [k for k in validlong if lp[k] < .6]
        add("semantic_disagreement", sorted(semindices, key=lambda k: (-sp[k], lp[k], int(k)))[:8])
        shortindices = np.flatnonzero(shortpair)
        if len(shortindices):
            chosen = rng.choice(shortindices, size=min(8, len(shortindices)), replace=False)
            add("short", chosen)
        add("random", rng.choice(len(ia), size=min(10,len(ia)), replace=False))
    selected = []
    actual = []
    for split, total in QUOTAS.items():
        quotas = {"lexical": int(total*.4), "semantic_disagreement": round(total*.25),
                  "short": int(total*.15)}
        quotas["random"] = total - sum(quotas.values())
        used = set()
        rows = []
        # Short/disagreement pools are disjoint from long lexical mining.
        for kind in ["short", "semantic_disagreement", "lexical", "random"]:
            pool = pools[split][kind]
            if kind in ["lexical", "semantic_disagreement"]:
                score = "lexical_score" if kind == "lexical" else "semantic_score"
                pool = sorted(pool, key=lambda r: (-r[score], r["pair_id"]))[:max(quotas[kind]*4, quotas[kind])]
            order = rng.permutation(len(pool))
            count = 0
            for i in order:
                r = pool[int(i)]
                if r["pair_id"] in used:
                    continue
                rows.append(dict(r, sampling_stratum=kind))
                used.add(r["pair_id"])
                count += 1
                if count >= quotas[kind]:
                    break
            actual.append({"split": split, "sampling_stratum": kind, "requested": quotas[kind], "selected": count})
        # Record fallback explicitly; it never supplies a label.
        for r in pools[split]["random"]:
            if len(rows) >= total:
                break
            if r["pair_id"] not in used:
                rows.append(dict(r, sampling_stratum="random_fallback"))
                used.add(r["pair_id"])
        if len(rows) != total:
            raise ValueError(f"Not enough unique pairs for {split}: {len(rows)}/{total}")
        rows = [rows[int(i)] for i in rng.permutation(len(rows))]
        selected.extend(rows)
    pairs = pd.DataFrame(selected)
    validate_pairs(pairs, df)
    write_csv(out / "pair_metadata.csv", pairs)
    for split in QUOTAS:
        labels = pd.DataFrame({"pair_id": pairs.loc[pairs.split == split, "pair_id"],
                               "label": "", "reason": "", "annotation_version": ""})
        write_csv(out / f"labels_{split}.csv", labels)
    # Operator index contains texts only, no model scores or identity/rating fields.
    textlookup = df.set_index("review_id").text.to_dict()
    public = pd.DataFrame({"pair_id": pairs.pair_id, "text_a": pairs.review_id_a.map(textlookup),
                           "text_b": pairs.review_id_b.map(textlookup)})
    write_csv(out / "annotation_texts.csv", public)
    write_csv(root / "results/tables/annotation_sampling.csv", pd.DataFrame(actual))
    meta = {"dataset_version": manifest["dataset_version"], "embedding_cache_key": emeta["cache_key"],
            "seed": SEED, "quotas": QUOTAS, "pair_table_sha256": sha_file(out / "pair_metadata.csv"),
            "text_table_sha256": sha_file(out / "annotation_texts.csv"),
            "pilot_pair_ids": pairs.loc[pairs.split == "train", "pair_id"].head(60).tolist(),
            "note": "Sampling strata are enrichment choices, not labels or natural prevalence estimates."}
    write_json(path, meta)
    print(json.dumps({"pairs": len(pairs), "split_counts": pairs.split.value_counts().to_dict(),
                     "labels": "all blank; waiting for human annotation"}, indent=2))
    return pairs
