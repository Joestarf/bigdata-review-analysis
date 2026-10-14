"""Standalone multilingual review-only predictor with original-text features."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .common import ROOT, sha_file, write_csv, sha_text
from .multilingual_features import FEATURE_NAMES, analyze_text, conflict_evidence, pair_language, score_methods, allowed_features
from .multilingual_embedding import CONFIG, MultilingualEncoder

class MultilingualPredictor:
    def __init__(self, model_dir=None):
        self.directory=Path(model_dir or ROOT/"models/multilingual_v2").resolve()
        path=self.directory/"manifest.json"
        if not path.exists():
            raise ValueError(f"Missing multilingual manifest: {path}. Complete training first.")
        self.manifest=json.loads(path.read_text())
        if self.manifest.get("encoder")!=CONFIG or self.manifest.get("feature_names")!=FEATURE_NAMES+["semantic_cosine"]:
            raise ValueError("Model feature or encoder configuration mismatch")
        for relative,expected in self.manifest["files"].items():
            target=(self.directory/relative).resolve()
            if target.parent!=self.directory or not target.is_file() or sha_file(target)!=expected:
                raise ValueError(f"Model checksum mismatch: {relative}")
        code_root=Path(__file__).resolve().parents[1]
        for relative,expected in self.manifest["code_hashes"].items():
            target=(code_root/relative).resolve()
            if not target.is_relative_to(code_root) or not target.is_file() or sha_file(target)!=expected:
                raise ValueError(f"Versioned inference code changed: {relative}; use matching code or train a new revision")
        self.lexical=joblib.load(self.directory/"lexical.joblib")
        self.models={m:joblib.load(self.directory/f"{m}.joblib") for m in ("C","E")}
        self.encoder=None

    def predict_pairs(self,pairs):
        rows=list(pairs)
        if not rows or len(rows)>1000:
            raise ValueError("Provide 1..1000 pairs per call")
        for row in rows:
            for key in ("text_a","text_b"):
                value=row.get(key)
                if not isinstance(value,str) or not value.strip() or len(value)>10000:
                    raise ValueError(f"{key} must be nonempty text up to 10000 characters")
        ids=[r.get("pair_id") or sha_text(json.dumps([r["text_a"],r["text_b"]],ensure_ascii=False)) for r in rows]
        if len(set(ids))!=len(ids):raise ValueError("pair_id must be unique within a call")
        if self.encoder is None:self.encoder=MultilingualEncoder()
        texts=list(dict.fromkeys([r["text_a"] for r in rows]+[r["text_b"] for r in rows]))
        vectors,receipt=self.encoder.encode(texts)
        lookup={t:(v,n) for t,v,n in zip(texts,vectors,receipt["window_counts"])}
        a=[r["text_a"] for r in rows];b=[r["text_b"] for r in rows]
        sim=[float(lookup[aa][0]@lookup[bb][0]) for aa,bb in zip(a,b)]
        x=self.lexical.transform_pairs(a,b,sim)
        allowed=allowed_features(x);scores=score_methods(x,self.models)
        output=[]
        for i,row in enumerate(rows):
            metadata=[analyze_text(t) for t in (a[i],b[i])]
            conflict=conflict_evidence(a[i],b[i])
            flags=[]
            if any(t["short_evidence"] for t in metadata):flags.append("limited_text_evidence")
            if any(t["generic_phrase"] for t in metadata):flags.append("generic_phrase")
            if conflict["candidate_conflict"]:flags.append("specific_detail_difference")
            if any(lookup[t][1]>1 for t in (a[i],b[i])):flags.append("long_text_window_aggregation")
            for method in ("A","B","C","D","E"):
                threshold=self.manifest["configs"][method]["threshold"]
                score=float(scores[method][i])
                output.append({
                    "pair_id":ids[i],"asin":row.get("asin",""),
                    "review_id_a":row.get("review_id_a",""),"review_id_b":row.get("review_id_b",""),
                    "method":method,"score":score,"threshold":threshold,
                    "raw_signal":score>=threshold,
                    "candidate_signal":bool(score>=threshold and allowed[i]),
                    "decision":"review","reason":";".join(flags+["multilingual_validation_not_authorized_for_automatic_weighting"]),
                    "language":pair_language(a[i],b[i]),
                    "conflict_evidence":json.dumps(conflict,ensure_ascii=False),
                    "exact_match":bool(x[i,FEATURE_NAMES.index("exact_match")]),
                    "model_version":self.manifest["schema"]})
        return pd.DataFrame(output)

    def score_product(self,asin,reviews):
        reviews=list(reviews)
        if not isinstance(asin,str) or not asin.strip():raise ValueError("asin is required")
        if len(reviews)>1000:raise ValueError("At most 1000 reviews per product")
        ids=[r.get("review_id") for r in reviews]
        if any(not isinstance(r,str) or not r.strip() for r in ids) or len(ids)!=len(set(ids)):
            raise ValueError("Each review needs a unique nonempty review_id")
        if any(r.get("asin",asin)!=asin for r in reviews):
            raise ValueError("Reviews belong to multiple products")
        for r in reviews:
            if not isinstance(r.get("text"),str) or not r["text"].strip() or len(r["text"])>10000:
                raise ValueError("Each review needs valid text")
        rows=[]
        for a,b in itertools.combinations(reviews,2):
            rows.append({"pair_id":sha_text("|".join(sorted([a["review_id"],b["review_id"]]))),
                         "asin":asin,"review_id_a":a["review_id"],"review_id_b":b["review_id"],
                         "text_a":a["text"],"text_b":b["text"]})
        batches=[self.predict_pairs(rows[i:i+1000]) for i in range(0,len(rows),1000)]
        return pd.concat(batches,ignore_index=True) if batches else pd.DataFrame(
            columns=["pair_id","asin","review_id_a","review_id_b","method","score","decision","reason"])

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir",type=Path,default=ROOT/"models/multilingual_v2")
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    frame=pd.read_csv(args.input,keep_default_na=False)
    if not {"text_a","text_b"}<=set(frame):raise ValueError("Input requires text_a and text_b")
    predictor=MultilingualPredictor(args.model_dir)
    batches=[predictor.predict_pairs(frame.iloc[i:i+1000].to_dict("records")) for i in range(0,len(frame),1000)]
    write_csv(args.output,pd.concat(batches,ignore_index=True) if batches else pd.DataFrame())
