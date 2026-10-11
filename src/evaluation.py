"""One final evaluation after model/threshold lock; labels remain separate from training."""
import numpy as np
import pandas as pd
import joblib
from .common import ROOT,SEED,load_study,read_json,write_json,write_csv,sha_file,word_count
from .annotations import read_labels,GateError
from .training import validate_lock,metrics,bounded_metrics,score_methods,METHODS,automatic_allowed
from .features import fit_lexical,indexed_pair_features,text_pair_features
from .embedding import load_embedding_cache,load_encoder,encode_texts
from .synthetic import transform_text,TRANSFORMS

def validate_evaluation(root=ROOT):
    validate_lock(root)
    path=root/"results/tables/final_evaluation.json"
    if not path.exists():raise GateError("Final evaluation is missing.")
    seal=read_json(path)
    if seal["lock_sha256"]!=sha_file(root/"models/model_lock.json") or seal["test_labels_sha256"]!=sha_file(root/"data/annotation/labels_test.csv"):
        raise GateError("Final evaluation inputs changed; create an explicit new experiment version.")
    return seal

def product_bootstrap(frame,iterations=1000):
    groups=[g for _,g in frame.groupby("asin",sort=True)]
    if len(groups)<2 or frame.target.nunique()<2:
        return {"status":"insufficient_products_or_classes","products":len(groups),"valid_resamples":0}
    # Cluster-level confusion counts preserve the exact product resampling scheme.
    counts=np.array([[int(((g.target==0)&~g.predicted).sum()),
                      int(((g.target==0)&g.predicted).sum()),
                      int(((g.target==1)&~g.predicted).sum()),
                      int(((g.target==1)&g.predicted).sum())] for g in groups],dtype=np.int64)
    rng=np.random.default_rng(SEED);samples=[]
    for _ in range(iterations):
        tn,fp,fn,tp=counts[rng.integers(0,len(groups),len(groups))].sum(axis=0)
        if tp+fn==0 or tn+fp==0:continue
        samples.append([tp/(tp+fp) if tp+fp else np.nan,tp/(tp+fn),2*tp/(2*tp+fp+fn)])
    if len(samples)<100:return {"status":"insufficient_valid_resamples","products":len(groups),"valid_resamples":len(samples)}
    arr=np.asarray(samples)
    result={"status":"ok","products":len(groups),"valid_resamples":len(samples),"iterations":iterations,"seed":SEED}
    for i,k in enumerate(["precision","recall","f1"]):
        values=arr[np.isfinite(arr[:,i]),i];valid=len(values)
        result[f"{k}_valid_resamples"]=valid
        result[f"{k}_status"]="ok" if valid>=100 else "undefined_or_insufficient_valid_resamples"
        result[f"{k}_low"]=float(np.percentile(values,2.5)) if valid>=100 else None
        result[f"{k}_high"]=float(np.percentile(values,97.5)) if valid>=100 else None
    return result

def evaluate(root=ROOT):
    if (root/"results/tables/final_evaluation.json").exists():
        validate_evaluation(root)
        print("Sealed final evaluation verified; not rerunning or tuning on test.")
        return
    lock=validate_lock(root) # Must occur before opening any test label.
    test=read_labels(root,"test")
    if test.target.eq(1).sum()<20 or test.target.eq(0).sum()<20:
        raise GateError("Test requires >=20 confirmed positives and >=20 negatives; exploratory labels are insufficient.")
    df,manifest=load_study(root)
    vectorizer=fit_lexical(root)
    embeddings,eindex,emeta=load_embedding_cache(root)
    x=indexed_pair_features(test,df,vectorizer,embeddings)
    models={m:joblib.load(root/f"models/{m}.joblib") for m in ["C","E"]}
    scores=score_methods(x,models)
    allowed=automatic_allowed(test,df)
    lookup=df.set_index("review_id")
    trunc=eindex.set_index("review_id").truncated
    test["short"]=[not v for v in allowed]
    test["truncated"]=[bool(trunc[a] or trunc[b]) for a,b in zip(test.review_id_a,test.review_id_b)]
    definite=test.target.notna().to_numpy()
    comparisons,breakdowns,predictions,intervals=[],[],[],[]
    for m in METHODS:
        config=lock["methods"][m]
        # Review-only fallback is evaluated as a candidate, never automatic-ready.
        threshold=config["review_threshold"]
        predicted=(scores[m]>=threshold)&allowed
        benchmark_review=(~allowed) | (not config["qualified"])
        review=benchmark_review | (lock.get("population_automatic_ready") is not True)
        result=bounded_metrics(test.target.to_numpy(dtype=float),predicted)
        comparisons.append({"method":m,"split":"test","threshold":threshold,"qualified_validation":config["qualified"],
                            "evaluation_mode":"benchmark_qualified_candidate" if config["qualified"] else "exploratory_candidate",
                            "population_automatic_ready":bool(lock.get("population_automatic_ready",False)),
                            **result,"uncertain":int((~definite).sum()),"review":int(review.sum()),
                            "annotation_source":"AI_assisted_review"})
        original_ids=set(pd.read_csv(root/"data/annotation/original600/pair_metadata.csv").pair_id) if (root/"data/annotation/original600/pair_metadata.csv").exists() else set(test.pair_id)
        intermediate=root/"data/annotation/original1200/pair_metadata.csv"
        middle_ids=set(pd.read_csv(intermediate).pair_id) if intermediate.exists() else set(test.pair_id)
        for cohort,cmask in [("original150",test.pair_id.isin(original_ids)),
                            ("extension300",test.pair_id.isin(middle_ids)&~test.pair_id.isin(original_ids)),
                            ("scale550",~test.pair_id.isin(middle_ids))]:
            if cmask.any():
                breakdowns.append({"method":m,"group":cohort,**bounded_metrics(test.loc[cmask,"target"].to_numpy(dtype=float),predicted[cmask]),
                                   "contains_both_classes":test.loc[cmask,"target"].nunique()==2})
        detail=test.loc[definite,["asin","target"]].copy()
        detail["predicted"]=predicted[definite]
        intervals.append({"method":m,**product_bootstrap(detail)})
        groups={"exact":test.label.eq("exact"),"near":test.label.eq("near"),"distinct":test.label.eq("distinct"),
                "short":test.short,"long_truncated":test.truncated,"not_truncated":~test.truncated}
        for name,mask in groups.items():
            mask=mask.to_numpy()&definite
            if mask.any():
                breakdowns.append({"method":m,"group":name,**metrics(test.loc[mask,"target"],predicted[mask]),
                                   "contains_both_classes":test.loc[mask,"target"].nunique()==2,
                                   "review":int(review[mask].sum())})
        for i,r in enumerate(test.itertuples()):
            decision="review" if review[i] else ("duplicate" if predicted[i] else "distinct")
            benchmark_decision="review" if benchmark_review[i] else ("duplicate" if predicted[i] else "distinct")
            predictions.append({"pair_id":r.pair_id,"asin":r.asin,"review_id_a":r.review_id_a,"review_id_b":r.review_id_b,
                                "method":m,"score":float(scores[m][i]),"candidate_positive":bool(predicted[i]),
                                "decision":decision,"benchmark_decision":benchmark_decision,"label":r.label,"reason":r.reason,"short":bool(r.short),"truncated":bool(r.truncated)})
    output=root/"results/tables"
    write_csv(output/"test_comparison.csv",pd.DataFrame(comparisons))
    write_csv(output/"test_breakdowns.csv",pd.DataFrame(breakdowns))
    write_csv(output/"test_product_bootstrap.csv",pd.DataFrame(intervals))
    pred=pd.DataFrame(predictions)
    write_csv(output/"test_pair_predictions.csv",pred)
    # Cases from selected method, or E if none qualified; fallback clearly labeled.
    method=lock["selected_method"] or "E"
    view=pred[pred.method.eq(method)].set_index("pair_id")
    pos=test.label.isin(["exact","near"])
    case_sets={"false_positive":test.label.eq("distinct")&test.pair_id.map(view.candidate_positive),
               "false_negative":pos&~test.pair_id.map(view.candidate_positive),
               "uncertain":test.label.eq("uncertain")}
    cases=[]
    for kind,mask in case_sets.items():
        for r in test[mask].head(10).itertuples():
            a,b=lookup.loc[r.review_id_a],lookup.loc[r.review_id_b]
            evidence=[]
            for m in METHODS:
                row=pred[(pred.pair_id==r.pair_id)&(pred.method==m)].iloc[0]
                evidence.append(f"{m}={row.score:.4f}:{row.decision}")
            cause="通用短语或信息不足" if r.short or r.label=="uncertain" else ("编码器截断长文本" if r.truncated else "共同商品词、具体细节或结论差异需按正文复核")
            cases.append({"kind":kind,"pair_id":r.pair_id,"asin":r.asin,"text_a":a.text,"text_b":b.text,
                          "label":r.label,"label_reason":r.reason,"methods":"; ".join(evidence),"possible_cause":cause,
                          "downstream_risk":"误判可能降权正常评论；待复核结果不能直接降权"})
    write_csv(output/"error_cases.csv",pd.DataFrame(cases))
    # Held-out-source synthetic positives, separate from real test metrics.
    rng=np.random.default_rng(SEED)
    sources=df[(df.split=="test") & df.supervised_eligible & df.text.map(word_count).ge(10)].head(200)
    synthetic=[]
    for r in sources.itertuples():
        for kind in ["exact_copy"]+TRANSFORMS:
            b=r.text if kind=="exact_copy" else transform_text(r.text,kind,rng)
            if b is not None:synthetic.append({"source_review_id":r.review_id,"asin":r.asin,"split":"test","transform":kind,"text_a":r.text,"text_b":b,"label_source":"synthetic"})
    syn=pd.DataFrame(synthetic)
    encoder=load_encoder(emeta["model_revision"])
    texts=list(dict.fromkeys(syn.text_a.tolist()+syn.text_b.tolist()))
    vectors,_,_=encode_texts(encoder,texts,64)
    vlookup=dict(zip(texts,vectors))
    sims=[np.dot(vlookup[a],vlookup[b]) for a,b in zip(syn.text_a,syn.text_b)]
    xs=text_pair_features(vectorizer,syn.text_a,syn.text_b,sims)
    ss=score_methods(xs,models)
    synrows=[]
    for m in METHODS:
        for kind,g in syn.groupby("transform"):
            detect=ss[m][g.index]>=lock["methods"][m]["review_threshold"]
            synrows.append({"method":m,"transform":kind,"n":len(g),"detected":int(detect.sum()),
                            "detection_rate":float(detect.mean()),"note":"positive-only synthetic robustness; precision/F1 undefined"})
    write_csv(output/"synthetic_test_sources.csv",syn)
    write_csv(output/"synthetic_test_detection.csv",pd.DataFrame(synrows))
    write_json(output/"final_evaluation.json",{"dataset_version":manifest["dataset_version"],"lock_sha256":sha_file(root/"models/model_lock.json"),
             "test_labels_sha256":sha_file(root/"data/annotation/labels_test.csv"),"test_rows":len(test),
             "effective_binary_rows":int(definite.sum()),"uncertain_rows":int((~definite).sum()),
             "annotation_source":"AI_assisted_review; no independent human gold standard",
             "natural_prevalence_estimate":False,"bootstrap_iterations":1000,
             "selected_method":lock["selected_method"],"automatic_ready_validation":lock["automatic_ready_validation"]})
    print("Final test evaluation sealed; no further tuning on this test set.")
