"""Sealed Chinese test and constructed multilingual diagnostics for v2."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from .common import ROOT, sha_file, write_csv, write_json, normalize_text as old_normalize, word_count
from .training import validate_lock, bounded_metrics
from .portable import PortablePredictor
from .multilingual_training import annotate_pairs
from .multilingual_inference import MultilingualPredictor
from .multilingual_features import analyze_text, normalize_text, pair_language
from .chinese_controls import CHINESE_CONTROLS as CONTROLS, ENGLISH_CONTROLS as CONTROLS_EN

def summarize(y, predicted):
    y=np.asarray(y,float);predicted=np.asarray(predicted,bool)
    if np.isfinite(y).any():return bounded_metrics(y,predicted)
    total=int(predicted.sum())
    return {"n":0,"precision":0.,"recall":0.,"f1":0.,"tn":0,"fp":0,"fn":0,"tp":0,
            "predicted_positive":0,"precision_defined":False,"recall_defined":False,
            "full_rows":len(y),"automatic_candidate_positive":total,"unresolved_positive":total,
            "precision_bound_defined":bool(total),"precision_lower":0.,"precision_upper":1. if total else 0.}

def bootstrap(table, predictions, iterations=1000):
    """Resample products, not individual correlated pairs."""
    group=table.groupby("asin").indices
    products=sorted(group);rng=np.random.default_rng(42)
    results=[]
    y=table.target.to_numpy(float);p=np.asarray(predictions,bool)
    for _ in range(iterations):
        selected=rng.choice(products,size=len(products),replace=True)
        idx=np.concatenate([group[x] for x in selected])
        yy,pp=y[idx],p[idx];known=np.isfinite(yy)
        tp=int(((yy==1)&pp).sum());fp=int(((yy==0)&pp).sum());fn=int(((yy==1)&~pp).sum())
        predicted=tp+fp;positive=tp+fn;total=int(pp.sum())
        results.append([tp/predicted if predicted else np.nan,
                        tp/positive if positive else np.nan,
                        2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                        tp/total if total else np.nan])
    arr=np.asarray(results)
    out={"bootstrap_products":len(products),"bootstrap_iterations":iterations,"bootstrap_seed":42}
    for i,name in enumerate(("precision","recall","f1","precision_lower")):
        valid=arr[:,i][np.isfinite(arr[:,i])]
        out[name+"_bootstrap_valid"]=len(valid)
        out[name+"_ci_low"]=float(np.quantile(valid,.025)) if len(valid) else None
        out[name+"_ci_high"]=float(np.quantile(valid,.975)) if len(valid) else None
    return out

def evaluate(root=ROOT):
    validate_lock(root)
    directory=root/"data/multilingual_v2"
    out=root/"results/multilingual_v2";out.mkdir(parents=True,exist_ok=True)
    receipt_path=out/"test_receipt.json"
    if receipt_path.exists():raise ValueError("Final Chinese test already evaluated; keep its locked receipt")
    predictor=MultilingualPredictor()
    lock=predictor.manifest
    # Only after v2 model is locked, validate completion and then parse the sealed test.
    seal=json.loads((directory/"test_seal.json").read_text())
    if sha_file(directory/"test_seal.json")!=lock["test_seal_sha256"] or sha_file(directory/"labels_test.csv")!=seal["labels_sha256"]:
        raise ValueError("Test seal changed after model lock")
    manifest=json.loads((directory/"candidate_manifest.json").read_text())
    table=annotate_pairs(root,"test",manifest).reset_index(drop=True)
    result=predictor.predict_pairs(table.to_dict("records"))
    write_csv(out/"chinese_test_pair_scores.csv",result)
    old=PortablePredictor(root)
    old_result=old.predict_pairs(list(zip(table.text_a,table.text_b)))
    old_result["pair_id"]=np.repeat(table.pair_id.to_numpy(),5)
    old_scores=old_result.pivot(index="pair_id",columns="method",values="score").reindex(table.pair_id)
    new_scores=result.pivot(index="pair_id",columns="method",values="score").reindex(table.pair_id)
    new_allowed=result.pivot(index="pair_id",columns="method",values="candidate_signal").reindex(table.pair_id)
    old_allowed=np.array([word_count(a)>=5 and word_count(b)>=5 for a,b in zip(table.text_a,table.text_b)])
    rows=[]
    sensitivity=[]
    # Duplicate source records are removed; different users can still yield identical text pairs.
    # Report a text-pair-deduplicated sensitivity, without modifying the frozen test or model.
    key=table.apply(lambda r:json.dumps(sorted([normalize_text(r.text_a),normalize_text(r.text_b)]),ensure_ascii=False),axis=1)
    unique=~key.duplicated()
    for version,scores in (("v1",old_scores),("v2",new_scores)):
        for method in ("A","B","C","D","E"):
            threshold=old.manifest["methods"][method]["review_threshold"] if version=="v1" else lock["configs"][method]["threshold"]
            raw=scores[method].to_numpy()>=threshold
            candidate=raw&old_allowed if version=="v1" else new_allowed[method].to_numpy(bool)
            for scope,prediction in (("raw_threshold",raw),("eligible_candidate",candidate)):
                metrics=bounded_metrics(table.target,prediction)
                rows.append({"version":version,"method":method,"scope":scope,
                             "threshold":threshold,**metrics,**bootstrap(table,prediction)})
                reduced=table[unique].reset_index(drop=True)
                sensitivity.append({"version":version,"method":method,"scope":scope,
                    "unique_text_pairs":len(reduced),**bounded_metrics(reduced.target,prediction[unique])})
    metrics=pd.DataFrame(rows)
    write_csv(out/"chinese_test_metrics.csv",metrics)
    write_csv(out/"chinese_test_unique_text_pair_metrics.csv",pd.DataFrame(sensitivity))
    write_csv(out/"chinese_test_labels_with_metadata.csv",table)
    write_json(receipt_path,{
        "model_manifest_sha256":sha_file(root/"models/multilingual_v2/manifest.json"),
        "labels_sha256":sha_file(directory/"labels_test.csv"),"tasks_sha256":sha_file(directory/"tasks_test.csv"),
        "rows":len(table),"products":int(table.asin.nunique()),"labels":table.label.value_counts().to_dict(),
        "language_counts":table.language.value_counts().to_dict(),
        "unique_text_pairs":int(unique.sum()),"first_read_after_lock":True,
        "label_provenance":"Automatically assisted, not an independent human gold standard",
        "scope":"Enriched candidate Chinese auxiliary-domain test; not natural prevalence or phone-accessory accuracy",
        "original_test_used":False,"thresholds_adjusted_after_test":False,
        "original_lock_sha256":sha_file(root/"models/model_lock.json"),
        "original_handoff_sha256":sha_file(root/"deliverables/team_handoff.zip")})
    print(metrics[metrics.scope.eq("eligible_candidate")][["version","method","precision","recall","f1","tp","fp","fn","tn"]].to_string(index=False))
    return metrics

def diagnostic(root=ROOT):
    predictor=MultilingualPredictor()
    rows=[]
    for language,controls in (("zh",CONTROLS),("en",CONTROLS_EN)):
        for record in controls:
            rows.append({**record,"pair_id":language+":"+record["pair_id"],"language_group":language})
    for zh,en in zip(CONTROLS,CONTROLS_EN):
        if zh["pair_id"]!=en["pair_id"]:raise ValueError("Unaligned bilingual controls")
        rows.append({**zh,"pair_id":"mixed:"+zh["pair_id"],"language_group":"mixed",
                     "text_a":zh["text_a"]+" "+en["text_a"],"text_b":zh["text_b"]+" "+en["text_b"],
                     "origin":"constructed_bilingual_concatenation"})
    table=pd.DataFrame(rows)
    results=predictor.predict_pairs(table.to_dict("records"))
    out=root/"results/multilingual_v2"
    write_csv(out/"diagnostic_pair_scores.csv",results)
    summaries=[];diagnostics=[]
    for language,group in table.groupby("language_group",sort=True):
        texts=list(dict.fromkeys(group.text_a.tolist()+group.text_b.tolist()))
        vectors=predictor.lexical.vectorizer.transform([normalize_text(t) for t in texts])
        tokenizer=predictor.encoder.model.tokenizer
        fractions=[];unknowns=0;token_count=0
        for text in texts:
            ids=tokenizer(text,truncation=False,add_special_tokens=False,verbose=False)["input_ids"]
            count=sum(t==tokenizer.unk_token_id for t in ids)
            fractions.append(count/max(1,len(ids)));unknowns+=count;token_count+=len(ids)
        diagnostics.append({"language":language,"unique_inputs":len(texts),
            "zero_lexical_inputs":sum(vectors[i].nnz==0 for i in range(len(texts))),
            "short_inputs":sum(analyze_text(t)["short_evidence"] for t in texts),
            "generic_inputs":sum(analyze_text(t)["generic_phrase"] for t in texts),
            "mean_unknown_fraction":float(np.mean(fractions)),
            "pooled_unknown_fraction":unknowns/max(1,token_count)})
        for category,subset in group.groupby("group"):
            y=subset.label.map({"near":1,"distinct":0}).to_numpy(float)
            for method in ("A","B","C","D","E"):
                scores=results[results.method.eq(method)].set_index("pair_id").reindex(subset.pair_id)
                for scope in ("raw_signal","candidate_signal"):
                    summaries.append({"language":language,"group":category,"method":method,"scope":scope,
                                      **summarize(y,scores[scope].to_numpy(bool))})
    write_csv(out/"diagnostic_summary.csv",pd.DataFrame(summaries))
    write_csv(out/"language_diagnostics.csv",pd.DataFrame(diagnostics))
    write_json(out/"diagnostic_receipt.json",{
        "pairs":len(table),"per_language":table.groupby("language_group").size().to_dict(),
        "model_sha256":sha_file(root/"models/multilingual_v2/manifest.json"),
        "controls_source_sha256":sha_file(root/"src/chinese_controls.py"),
        "used_for_training_or_thresholds":False,"natural_language_gold":False,
        "mixed_origin":"Constructed concatenation of aligned Chinese and English controls; not natural mixed-language evaluation",
        "actual_decision":"review"})
    print(pd.DataFrame(diagnostics).to_string(index=False))


def reviewed_evaluation(root=ROOT):
    """Keep initial sealed results; audit labels after lock, never retune the model."""
    out=root/"results/multilingual_v2"
    table=pd.read_csv(out/"chinese_test_postlock_reviewed_labels.csv",keep_default_na=False)
    table["target"]=pd.to_numeric(table.target,errors="coerce")
    original=json.loads((out/"test_receipt.json").read_text())
    if sha_file(root/"models/multilingual_v2/manifest.json")!=original["model_manifest_sha256"]:
        raise ValueError("Model changed after first sealed evaluation")
    predictor=MultilingualPredictor()
    old=PortablePredictor(root)
    old_result=old.predict_pairs(list(zip(table.text_a,table.text_b)))
    old_result["pair_id"]=np.repeat(table.pair_id.to_numpy(),5)
    new_result=pd.read_csv(out/"chinese_test_pair_scores.csv",keep_default_na=False)
    old_scores=old_result.pivot(index="pair_id",columns="method",values="score").reindex(table.pair_id)
    new_scores=new_result.pivot(index="pair_id",columns="method",values="score").reindex(table.pair_id)
    new_candidates=new_result.pivot(index="pair_id",columns="method",values="candidate_signal").reindex(table.pair_id)
    old_allowed=np.array([word_count(a)>=5 and word_count(b)>=5 for a,b in zip(table.text_a,table.text_b)])
    results=[]
    for version,scores in (("v1",old_scores),("v2",new_scores)):
        for method in "ABCDE":
            threshold=old.manifest["methods"][method]["review_threshold"] if version=="v1" else predictor.manifest["configs"][method]["threshold"]
            raw=scores[method].to_numpy()>=threshold
            candidate=raw&old_allowed if version=="v1" else new_candidates[method].to_numpy(bool)
            for scope,prediction in (("raw_threshold",raw),("eligible_candidate",candidate)):
                results.append({"version":version,"method":method,"scope":scope,
                    "threshold":threshold,**summarize(table.target,prediction),**bootstrap(table,prediction)})
    write_csv(out/"chinese_test_reviewed_metrics.csv",pd.DataFrame(results))
    write_json(out/"test_postlock_review_receipt.json",{
        "model_manifest_sha256":original["model_manifest_sha256"],
        "initial_test_receipt_sha256":sha_file(out/"test_receipt.json"),
        "reviewed_labels_sha256":sha_file(out/"chinese_test_postlock_reviewed_labels.csv"),
        "labels":table.label.value_counts().to_dict(),
        "content_review_sha256":sha_file(root/"data/multilingual_v2/test_postlock_content_review.csv"),
        "labels_reviewed_after_lock_and_initial_evaluation":True,
        "model_retrained":False,"thresholds_changed":False,
        "scope":"Post hoc reference-label quality audit, exploratory comparison, not first sealed gold test",
        "label_provenance":"Automatically assisted judgments, not independent human gold"})
    print(pd.DataFrame(results).query("scope == 'eligible_candidate'")[["version","method","n","precision","recall","f1","tp","fp","fn","tn","precision_lower"]].to_string(index=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["test","diagnostic","reviewed"])
    args=parser.parse_args()
    if args.action=="test":evaluate()
    elif args.action=="reviewed":reviewed_evaluation()
    else:diagnostic()
