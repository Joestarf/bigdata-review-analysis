"""Executable acceptance checks; pending gates are explicitly reported."""
import numpy as np
import pandas as pd
from .common import ROOT,load_study,sha_file,write_json,read_json
from .audit import make_index,validate_splits
from .pairs import validate_pairs
from .annotations import validate_task_manifest,read_labels,require_training_labels,GateError,status
from .embedding import load_embedding_cache

def validate(root=ROOT):
    df,manifest=load_study(root)
    checks=[]
    def passed(name,details):checks.append({"check":name,"status":"passed","details":details})
    idx=make_index(df[["asin","user_id","rating","text","timestamp"]],manifest["dataset_version"])
    if idx.review_id.tolist()!=df.review_id.tolist() or idx.source_id.tolist()!=df.source_id.tolist() or idx.text_hash.tolist()!=df.text_hash.tolist():
        raise ValueError("Stable identifier reconstruction mismatch.")
    passed("stable_review_ids",f"{len(df)} unique deterministic row IDs")
    columns=["review_id","product_split","split","supervised_eligible"]
    validate_splits(df.drop(columns=columns[1:]),df[columns])
    passed("split_leakage","ASIN, normalized text and source IDs isolated in eligible record pool")
    meta=validate_task_manifest(root)
    pairs=pd.read_csv(root/"data/annotation/pair_metadata.csv",keep_default_na=False)
    validate_pairs(pairs,df)
    texts=pd.read_csv(root/"data/annotation/annotation_texts.csv",keep_default_na=False).set_index("pair_id")
    lookup=df.set_index("review_id")
    for r in pairs.itertuples():
        text=texts.loc[r.pair_id]
        if text.text_a!=lookup.loc[r.review_id_a,"text"] or text.text_b!=lookup.loc[r.review_id_b,"text"]:
            raise ValueError("Annotation pair body alignment mismatch.")
    passed("pair_integrity",f"{len(pairs)} unique within-product/split pairs; no self or reversed pairs")
    vectors,index,emeta=load_embedding_cache(root)
    for start in range(0,len(vectors),10000):
        block=np.asarray(vectors[start:start+10000])
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block,axis=1),1.,atol=1e-4):
            raise ValueError("Embedding normalization/nonfinite check failed.")
    passed("semantic_cache","Version, text/config/model revision, array SHA, IDs, finite normalized vectors checked")
    for split in ["train","validation"]:read_labels(root,split,require_complete=False)
    passed("label_structure","Train/validation IDs and allowed labels checked; test values remain sealed")
    try:
        require_training_labels(root)
        passed("training_label_gate","Minimum train/validation class counts met")
    except GateError as e:checks.append({"check":"training_label_gate","status":"pending","details":str(e)})
    handoff_manifest=root/"results/handoff/manifest.json"
    if handoff_manifest.exists():
        hm=read_json(handoff_manifest)
        if hm["dataset_version"] != manifest["dataset_version"]:raise ValueError("Handoff dataset mismatch.")
        for filename,digest in hm["file_sha256"].items():
            if sha_file(handoff_manifest.parent/filename)!=digest:raise ValueError("Handoff checksum mismatch.")
        output=pd.read_csv(handoff_manifest.parent/"pair_scores.csv",keep_default_na=False)
        if output.duplicated(["pair_id","method"]).any():raise ValueError("Repeated handoff pair/method row.")
        if not output.decision.isin(["duplicate","distinct","review"]).all():raise ValueError("Unknown handoff decision.")
        lookup=df.set_index("review_id").asin
        if not (output.review_id_a.map(lookup).eq(output.asin)&output.review_id_b.map(lookup).eq(output.asin)).all():
            raise ValueError("Handoff pair ID/product back-reference failed.")
        scores=pd.to_numeric(output.score,errors="coerce")
        present=output.score.ne("")
        if not np.isfinite(scores[present]).all() or not scores[present].between(-1.0001,1.0001).all():
            raise ValueError("Invalid handoff score.")
        passed("handoff_integrity",f"{hm['pairs']} pairs, five-method preview; IDs and finite available scores verified")
    out={"dataset_version":manifest["dataset_version"],"checks":checks,"study":status(root),
         "label_source":"AI_assisted_review","human_gold_standard":False}
    write_json(root/"results/tables/acceptance_report.json",out)
    print(__import__("json").dumps(out,ensure_ascii=False,indent=2))
    return out
