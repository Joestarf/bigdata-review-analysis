"""Bounded, pinned Chinese adaptation corpus and product-disjoint label tasks."""
from __future__ import annotations
import argparse
import hashlib
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from datasets import load_dataset
from .common import ROOT, write_csv, write_json, sha_file, sha_text
from .multilingual_features import analyze_text, normalize_text
from .multilingual_embedding import MultilingualEncoder, digest

DATASET = "OpenModels/Chinese-Herbal-Medicine-Sentiment"
REVISION = "6b2b55d97dd3b99b917aa637da501357f7b8adc0"
LIMIT = 20000
CAP = 60
SEED = 42

def prepare(root=ROOT, limit=LIMIT):
    if not 1000 <= limit <= LIMIT:
        raise ValueError("Stream limit must be 1000..20000")
    directory = root/"data/multilingual_v2"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"reviews.csv"
    if path.exists():
        raise ValueError("Chinese sample already exists; use a new version directory instead of overwriting")
    rng = np.random.default_rng(SEED)
    counts, excluded, rows = Counter(), Counter(), []
    ds = load_dataset(DATASET, revision=REVISION, split="train", streaming=True)
    for index, record in enumerate(itertools.islice(ds, limit)):
        product_value=record.get("product_id")
        text_value=record.get("review_text")
        if product_value is None or not isinstance(text_value,str):
            excluded["missing_product_or_text"] += 1; continue
        product = str(product_value).strip()
        text = text_value.strip()
        if not {"user_id","review_time","rating"}<=record.keys():
            raise ValueError("Source identity schema changed; cannot identify complete record copies")
        if not product or not text:
            excluded["missing_product_or_text"] += 1; continue
        if text in ("此用户未填写评价内容", "此用户没有填写评价。", "用户未填写评价内容"):
            excluded["platform_placeholder"] += 1; continue
        if len(text)>10000:
            excluded["too_long"] += 1; continue
        if counts[product] >= CAP:
            excluded["product_cap"] += 1; continue
        counts[product] += 1
        identifier = sha_text(f"{DATASET}|{REVISION}|train|{index}")
        rows.append({"review_id":identifier, "asin":"zh:"+product, "text":text,
                     "source_id":digest([product,record.get("user_id"),record.get("review_time"),record.get("rating"),text]),
                     "text_hash":sha_text(normalize_text(text)),
                     "language":analyze_text(text)["language"]})
    if not rows:raise ValueError("No valid source reviews in bounded stream")
    reviews = pd.DataFrame(rows)
    products = sorted(reviews.asin.unique())
    rng.shuffle(products)
    n=len(products); a=int(.7*n); b=int(.85*n)
    assignment = {p:("train" if i<a else "validation" if i<b else "test") for i,p in enumerate(products)}
    reviews["product_split"]=reviews.asin.map(assignment)
    spread=reviews.groupby("text_hash").product_split.nunique()
    reviews["public_template"]=reviews.text_hash.isin(spread[spread.gt(1)].index)
    reviews["duplicate_within_product"]=reviews.duplicated(["asin","text_hash"])
    reviews["duplicate_record"]=reviews.duplicated("source_id")
    reviews["supervised_eligible"]=~(reviews.public_template | reviews.duplicate_record)
    # Same wording from different source records remains available for labeling.
    write_csv(path,reviews)
    write_json(directory/"source_manifest.json",{
        "dataset":DATASET,"revision":REVISION,"requested_stream_rows":limit,
        "observed_rows":index+1,"retained_rows":len(reviews),"products":len(products),
        "per_product_cap":CAP,"seed":SEED,"split_by_product":True,
        "sample_sha256":sha_file(path),"exclusions":dict(excluded),
        "public_template_rows":int(reviews.public_template.sum()),
        "duplicate_wording_within_product_rows":int(reviews.duplicate_within_product.sum()),
        "extra_identical_record_rows":int(reviews.duplicate_record.sum()),
        "language_counts":reviews.language.value_counts().to_dict(),
        "source_claim":"Card describes scraped Chinese e-commerce reviews; original platform and collection provenance not independently verified",
        "license_claim":"Dataset card declares MIT; review text is kept local, not redistributed",
        "scope":"Domain adaptation auxiliary corpus, not Cell Phones and Accessories",
        "original_dataset_sentiment_labels_used":False})
    return reviews

def candidates(root=ROOT):
    from .multilingual_features import MultilingualLexical
    directory=root/"data/multilingual_v2"
    reviews=pd.read_csv(directory/"reviews.csv",keep_default_na=False)
    eligible=reviews[reviews.supervised_eligible].copy()
    encoder=MultilingualEncoder()
    vectors,receipt=encoder.encode(eligible.text.tolist(),root/"models/multilingual_v2/cache")
    positions={rid:i for i,rid in enumerate(eligible.review_id)}
    lexical=MultilingualLexical().fit(eligible[eligible.product_split.eq("train")].text.tolist())
    # Candidate scores select tasks, never supply target labels.
    rng=np.random.default_rng(SEED); all_rows=[]
    for split, quota in (("train",300),("validation",150),("test",150)):
        pool=[]; seen=set()
        for asin, group in eligible[eligible.product_split.eq(split)].groupby("asin",sort=True):
            records=group.to_dict("records")
            if len(records)<2: continue
            comb=list(itertools.combinations(range(len(records)),2))
            aa=[records[i]["text"] for i,j in comb];bb=[records[j]["text"] for i,j in comb]
            lex=lexical.vectorizer.transform([normalize_text(r["text"]) for r in records])
            lex_sim=(lex @ lex.T).toarray()
            short=[analyze_text(r["text"])["short_evidence"] or analyze_text(r["text"])["generic_phrase"] for r in records]
            idx=[positions[r["review_id"]] for r in records]
            sim=vectors[idx]@vectors[idx].T
            for k,(i,j) in enumerate(comb):
                a,b=records[i],records[j]
                pair_id=sha_text("|".join(sorted([a["review_id"],b["review_id"]])))
                pool.append({"pair_id":pair_id,"asin":asin,"review_id_a":a["review_id"],
                    "review_id_b":b["review_id"],"split":split,"text_a":a["text"],"text_b":b["text"],
                    "char_score":float(lex_sim[i,j]),"semantic_score":float(sim[i,j]),
                    "short":bool(short[i] or short[j]),"language": "mixed" if "mixed" in (a["language"],b["language"]) else "zh"})
        table=pd.DataFrame(pool)
        if len(table)<quota:raise ValueError(f"{split}: only {len(table)} available pairs for {quota} tasks")
        chosen=[]
        pools=[
            ("lexical",int(quota*.4),table.sort_values(["char_score","pair_id"],ascending=[False,True])),
            ("semantic",int(quota*.25),table[table.char_score.lt(.35)].sort_values(["semantic_score","pair_id"],ascending=[False,True])),
            ("random",int(quota*.2),table.iloc[rng.permutation(len(table))]),
            ("short",quota-int(quota*.4)-int(quota*.25)-int(quota*.2),table[table.short].iloc[rng.permutation(int(table.short.sum()))])]
        for stratum, amount, ordered in pools:
            taken=0
            if amount == 0:continue
            for row in ordered.to_dict("records"):
                if row["pair_id"] in seen:continue
                seen.add(row["pair_id"]);row["sampling_stratum"]=stratum;chosen.append(row);taken+=1
                if taken>=amount:break
        for row in table.iloc[rng.permutation(len(table))].to_dict("records"):
            if len(chosen)>=quota:break
            if row["pair_id"] in seen:continue
            seen.add(row["pair_id"]);row["sampling_stratum"]="random_backfill";chosen.append(row)
        all_rows.extend(chosen)
    tasks=pd.DataFrame(all_rows)
    write_csv(directory/"pair_metadata.csv", tasks.drop(columns=["text_a","text_b"]))
    for split in ("train","validation","test"):
        task=tasks[tasks.split.eq(split)]
        # Blind annotation view excludes product, ratings, scores and sampling stratum.
        write_csv(directory/f"tasks_{split}.csv", task[["pair_id","text_a","text_b"]])
    write_json(directory/"candidate_manifest.json",{
        "sample_sha256":sha_file(directory/"reviews.csv"),
        "metadata_sha256":sha_file(directory/"pair_metadata.csv"),
        "task_sha256":{s:sha_file(directory/f"tasks_{s}.csv") for s in ("train","validation","test")},
        "quotas":tasks.groupby("split").size().to_dict(),
        "actual_strata":{s:g.sampling_stratum.value_counts().to_dict() for s,g in tasks.groupby("split")},
        "embedding_receipt":receipt,
        "labels":"Not yet provided. Sentiment/rating fields do not define repetition.",
        "label_provenance":"Automatically assisted judgments require review; not a human gold standard",
        "test_policy":"Test labels sealed separately; training only reads train and validation"})

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["sample","pairs"])
    args=parser.parse_args()
    if args.action=="sample":prepare()
    else:candidates()
