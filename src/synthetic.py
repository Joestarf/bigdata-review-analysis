from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, SEED, sha_text, load_study, normalize_text, word_count, write_csv, write_json, read_json, sha_file
from .annotations import require_frozen_guidelines, require_training_labels, GateError, labels_path, require_test_seal

TRANSFORMS = ["whitespace", "case", "punctuation", "typo"]
PROTECTED = {"never","cannot","without","nothing","barely","hardly","always","should","would","could","doesn't","don't","isn't","aren't","wasn't","weren't","can't","won't","not"}

TYPO_WORDS = {"comfortable","packaging","received","delivery","quality","purchase","excellent","perfectly","working","arrived","shipping","installation","installed","recommend","protects","protection","replacement","beautiful","durable","charging","connects","connected","experience","actually","plastic","pleased","returning","returned","unfortunately","disappointed","perfect","amazing","fantastic","wonderful","reliable","expensive","properly","worked"}

def transform_text(text, kind, rng):
    if kind == "whitespace":
        return "  " + text.replace(" ", "  ") + "  "
    if kind == "case":
        return text.swapcase()
    if kind == "punctuation":
        return text.replace("!", "!!") if "!" in text else text + "!"
    if kind == "typo":
        options=[m for m in re.finditer(r"(?<![A-Za-z0-9])([a-z]{6,})(?![A-Za-z0-9])",text)
                 if m.group(1).casefold() in TYPO_WORDS and m.group(1).casefold() not in PROTECTED]
        if not options:return None
        m=options[int(rng.integers(len(options)))]
        word=m.group(1)
        positions=[i for i in range(1,len(word)-2) if word[i] != word[i+1]]
        if not positions:return None
        i=positions[int(rng.integers(len(positions)))]
        swapped=word[:i]+word[i+1]+word[i]+word[i+2:]
        return text[:m.start(1)]+swapped+text[m.end(1):]
    raise ValueError("Unknown transformation.")

def generate_synthetic(df):
    rng=np.random.default_rng(SEED)
    eligible=df[(df.split == "train") & df.supervised_eligible & df.text.map(word_count).ge(10)]
    if len(eligible)<1000:raise GateError("Insufficient long training sources for synthetic examples.")
    source=eligible.iloc[rng.permutation(len(eligible))]
    rows=[]
    for r in source.head(1000).itertuples():
        rows.append({"pair_id":sha_text(r.review_id+":exact_copy"),"asin":r.asin,"split":"train",
            "source_review_id":r.review_id,"variant_id":sha_text(r.review_id+":variant:exact_copy"),
            "text_a":r.text,"text_b":r.text,"label":"exact","label_source":"synthetic","transform":"exact_copy","seed":SEED})
    for kind in TRANSFORMS:
        count=0
        for r in source.itertuples():
            b=transform_text(r.text,kind,rng)
            if b is None:continue
            rows.append({"pair_id":sha_text(r.review_id+":"+kind),"asin":r.asin,"split":"train",
                "source_review_id":r.review_id,"variant_id":sha_text(r.review_id+":variant:"+kind),
                "text_a":r.text,"text_b":b,"label":"exact" if normalize_text(r.text)==normalize_text(b) else "near",
                "label_source":"synthetic","transform":kind,"seed":SEED})
            count+=1
            if count>=250:break
        if count != 250:raise GateError(f"Insufficient sources for {kind}.")
    return pd.DataFrame(rows)

def prepare_synthetic(root: Path=ROOT):
    require_training_labels(root)
    require_test_seal(root) # Completion receipt only; no test label parsing.
    df,manifest=load_study(root)
    out=root / "data/annotation"
    mp=out / "synthetic_manifest.json"
    if mp.exists():
        meta=read_json(mp)
        if meta["dataset_version"] != manifest["dataset_version"] or sha_file(out/"synthetic_pairs.csv") != meta["sha256"]:
            raise GateError("Synthetic artifact is stale.")
        print("Using existing synthetic examples and transformation review tasks.")
        return
    syn=generate_synthetic(df)
    write_csv(out / "synthetic_pairs.csv",syn)
    checks=syn.groupby("transform",sort=True).head(20).copy()
    checks["label"]=""
    checks["reason"]=""
    checks["annotation_version"]=""
    write_csv(out / "synthetic_review.csv",checks)
    write_json(mp,{"dataset_version":manifest["dataset_version"],"seed":SEED,"rows":len(syn),
                   "sha256":sha_file(out/"synthetic_pairs.csv"),"check_rows":len(checks),
                   "label_source":"synthetic; AI-assisted transformation review authorized; no human gold standard"})
    print(f"Generated {len(syn)} synthetic positives; {len(checks)} transformation checks awaiting explicit transformation review.")

def approved_synthetic(root=ROOT):
    path=root / "data/annotation/synthetic_pairs.csv"
    if not path.exists():raise GateError("Run study synthesize after completing real annotation, then review its transformations.")
    meta=read_json(root/"data/annotation/synthetic_manifest.json")
    if sha_file(path) != meta["sha256"]:raise GateError("Synthetic training data changed after manifest.")
    checks=pd.read_csv(root/"data/annotation/synthetic_review.csv",keep_default_na=False)
    syn=pd.read_csv(path,keep_default_na=False)
    df,manifest=load_study(root)
    if meta["dataset_version"] != manifest["dataset_version"]:raise GateError("Synthetic dataset version mismatch.")
    eligible=df[(df.split=="train") & df.supervised_eligible].set_index("review_id")
    if not syn.split.eq("train").all() or not syn.label_source.eq("synthetic").all() or not syn.source_review_id.isin(eligible.index).all():
        raise GateError("Synthetic source leakage or lineage mismatch.")
    if syn.source_review_id.map(eligible.text).ne(syn.text_a).any():raise GateError("Synthetic source body mismatch.")
    expected=syn.groupby("transform",sort=True).head(20).pair_id.tolist()
    if checks.pair_id.tolist()!=expected:raise GateError("Synthetic review tasks do not match generated data.")
    fields=["text_a","text_b","transform","source_review_id"]
    expected_rows=syn.groupby("transform",sort=True).head(20)
    if not checks[fields].reset_index(drop=True).equals(expected_rows[fields].reset_index(drop=True)):
        raise GateError("Synthetic review bodies changed.")
    if checks.label.ne("approve").any() or checks.reason.str.strip().eq("").any():
        raise GateError("All transformation checks must be explicitly reviewed and approved. Rejected changes must be fixed and regenerated.")
    if checks.annotation_version.ne(meta["dataset_version"]).any():raise GateError("Synthetic approval version mismatch.")
    return syn
