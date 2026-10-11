"""Fit only after manual-label and synthetic-review acceptance; never read test labels."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from .common import ROOT, SEED, load_study, write_csv, write_json, read_json, sha_file, word_count, packages
from .annotations import GateError, require_training_labels, validate_task_manifest, require_frozen_guidelines, require_test_seal
from .synthetic import approved_synthetic
from .features import fit_lexical, indexed_pair_features, text_pair_features
from .embedding import load_embedding_cache, load_encoder, encode_texts

METHODS = ["A", "B", "C", "D", "E"]
COST_ORDER = {"A":0, "B":1, "C":2, "D":3, "E":4}

def metrics(y, predicted):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0,1]).ravel()
    return {"n":len(y), "precision":float(precision_score(y,predicted,zero_division=0)),
            "recall":float(recall_score(y,predicted,zero_division=0)),
            "f1":float(f1_score(y,predicted,zero_division=0)),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"predicted_positive":int(predicted.sum()),
            "precision_defined":bool(predicted.sum()>0),"recall_defined":bool(y.sum()>0)}

def bounded_metrics(y, predicted):
    """Uncertain labels remain unresolved; they lower the precision bound."""
    y = np.asarray(y,dtype=float); predicted = np.asarray(predicted,dtype=bool)
    known = np.isfinite(y)
    result = metrics(y[known],predicted[known])
    total = int(predicted.sum()); unresolved = int((predicted & ~known).sum())
    result.update({"full_rows":len(y),"automatic_candidate_positive":total,
                   "unresolved_positive":unresolved,
                   "precision_lower":result["tp"]/total if total else 0.,
                   "precision_upper":(result["tp"]+unresolved)/total if total else 0.})
    return result

def choose_threshold(y, scores, allowed, fixed=None):
    y, scores, allowed = np.asarray(y,dtype=float), np.asarray(scores), np.asarray(allowed,dtype=bool)
    thresholds=[fixed] if fixed is not None else np.unique(scores[allowed]).tolist()
    choices=[]
    for threshold in thresholds:
        result=bounded_metrics(y,(scores>=threshold)&allowed)
        if result["precision_lower"]>=.95 and result["automatic_candidate_positive"]>=30:
            choices.append({"threshold":float(threshold),**result})
    if not choices:return None
    return max(choices,key=lambda x:(x["recall"],x["precision_lower"],x["threshold"]))

def automatic_allowed(pairs,df):
    lookup=df.set_index("review_id").text
    return np.array([word_count(lookup[a])>=5 and word_count(lookup[b])>=5
                     for a,b in zip(pairs.review_id_a,pairs.review_id_b)])

def protected_paths(root):
    paths=["data/processed/reviews.csv","data/processed/review_index.csv","data/processed/split_assignments.csv",
           "data/annotation/pair_metadata.csv","data/annotation/annotation_texts.csv",
           "data/annotation/labels_train.csv","data/annotation/labels_validation.csv",
           "data/annotation/guideline_freeze.json","data/annotation/test_seal.json","report/annotation_guide.md",
           "data/annotation/synthetic_pairs.csv","data/annotation/synthetic_review.csv",
           "models/lexical.joblib","models/lexical_manifest.json",
           "models/embedding_manifest.json","models/embedding_index.csv",
           "models/C.joblib","models/E.joblib",
           "src/common.py","src/features.py","src/embedding.py","src/inference.py","src/training.py","src/evaluation.py","src/annotations.py","src/synthetic.py"]
    paths += [p for p in ["data/annotation/original600/pair_metadata.csv","data/annotation/original1200/pair_metadata.csv"]
              if (root/p).exists()]
    return {p:sha_file(root/p) for p in paths}

def validate_lock(root=ROOT):
    path=root/"models/model_lock.json"
    if not path.exists():raise GateError("No locked model. Finish annotation, approve transformations, then train.")
    lock=read_json(path)
    _,manifest=load_study(root)
    if lock["dataset_version"]!=manifest["dataset_version"]:raise GateError("Locked model dataset mismatch.")
    if lock["packages"] != packages():raise GateError("Locked dependency versions changed.")
    for p,digest in lock["protected_files"].items():
        if not (root/p).exists() or sha_file(root/p)!=digest:raise GateError(f"Locked artifact changed: {p}")
    return lock

def score_methods(x, models):
    return {"A":x[:,4], "B":x[:,0], "C":models["C"].predict_proba(x[:,:5])[:,1],
            "D":x[:,5], "E":models["E"].predict_proba(x)[:,1]}

def train(root:Path=ROOT):
    if (root/"models/model_lock.json").exists():
        validate_lock(root)
        print("Existing locked model verified; no refitting or test-label access.")
        return
    train_labels,val_labels=require_training_labels(root)
    test_seal=require_test_seal(root)
    syn=approved_synthetic(root)
    df,manifest=load_study(root)
    vectorizer=fit_lexical(root)
    embeddings,_,emeta=load_embedding_cache(root)
    train_def=train_labels[train_labels.target.notna()].copy()
    xtrain=indexed_pair_features(train_def,df,vectorizer,embeddings)
    xval=indexed_pair_features(val_labels,df,vectorizer,embeddings)
    encoder=load_encoder(emeta["model_revision"])
    unique=list(dict.fromkeys(syn.text_a.tolist()+syn.text_b.tolist()))
    vectors,_,_=encode_texts(encoder,unique,64)
    lookup=dict(zip(unique,vectors))
    sims=np.array([np.dot(lookup[a],lookup[b]) for a,b in zip(syn.text_a,syn.text_b)])
    xsyn=text_pair_features(vectorizer,syn.text_a,syn.text_b,sims)
    x=np.concatenate([xtrain,xsyn])
    y=np.concatenate([train_def.target.to_numpy(dtype=int),np.ones(len(syn),dtype=int)])
    # Synthetic weight is 25% of manual weight, hence 20% of total training weight.
    weights=np.concatenate([np.ones(len(train_def)),np.full(len(syn),.25*len(train_def)/len(syn))])
    allowed=automatic_allowed(val_labels,df)
    yval=val_labels.target.to_numpy(dtype=float)
    models,configs,rows={},{},[]
    for method,cols in [("C",5),("E",6)]:
        candidates=[]
        for c in [.1,1.,10.]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=SEED))
            model.fit(x[:,:cols],y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
            scores=model.predict_proba(xval[:,:cols])[:,1]
            selection=choose_threshold(yval,scores,allowed)
            # F1 at 0.5 is only a fallback for choosing a review-only candidate.
            fallback=bounded_metrics(yval,(scores>=.5)&allowed)
            candidates.append((selection, fallback, c,model))
        accepted=[r for r in candidates if r[0] is not None]
        chosen=max(accepted,key=lambda r:(r[0]["recall"],r[0]["precision_lower"],-r[2])) if accepted else max(candidates,key=lambda r:(r[1]["f1"],-r[2]))
        selection,_,c,model=chosen
        models[method]=model
        configs[method]={"C":c,"qualified":selection is not None,"threshold":selection["threshold"] if selection else None,
                         "review_threshold":selection["threshold"] if selection else .5}
        joblib.dump(model,root/f"models/{method}.joblib")
    scores=score_methods(xval,models)
    for method in ["A","B","D"]:
        selection=choose_threshold(yval,scores[method],allowed,fixed=1. if method=="A" else None)
        configs[method]={"qualified":selection is not None,"threshold":selection["threshold"] if selection else None,
                        "review_threshold":selection["threshold"] if selection else (1. if method=="A" else .8)}
    qualified=[m for m in METHODS if configs[m]["qualified"]]
    for m in METHODS:
        threshold=configs[m]["review_threshold"]
        result=bounded_metrics(yval,(scores[m]>=threshold)&allowed)
        rows.append({"method":m,**configs[m],**result,"review_rows":int((~allowed).sum()),
                     "uncertain_rows":int(val_labels.target.isna().sum()),"split":"validation"})
    original_path=root/"data/annotation/original600/pair_metadata.csv"
    if original_path.exists():
        original_ids=set(pd.read_csv(original_path).pair_id)
        intermediate=root/"data/annotation/original1200/pair_metadata.csv"
        middle_ids=set(pd.read_csv(intermediate).pair_id) if intermediate.exists() else set(val_labels.pair_id)
        for cohort,mask in [("original150",val_labels.pair_id.isin(original_ids)),
                            ("extension300",val_labels.pair_id.isin(middle_ids)&~val_labels.pair_id.isin(original_ids)),
                            ("scale550",~val_labels.pair_id.isin(middle_ids))]:
            for m in METHODS:
                if mask.any():
                    rows.append({"method":m,"split":"validation","cohort":cohort,**configs[m],
                                 **bounded_metrics(yval[mask],(scores[m][mask]>=configs[m]["review_threshold"])&allowed[mask])})
    best=max(qualified,key=lambda m:(next(r["recall"] for r in rows if r["method"]==m),-COST_ORDER[m])) if qualified else None
    lock={"dataset_version":manifest["dataset_version"],"methods":configs,"selected_method":best,
          "automatic_ready_validation":best is not None,"population_automatic_ready":False,
          "validation_scope":"Fixed enriched AI-reviewed benchmark; not population precision",
          "test_annotation_seal":test_seal,"seed":SEED,"selection_precision":.95,
          "selection_min_predictions":30,"training_manual_rows":len(train_def),"synthetic_rows":len(syn),
          "synthetic_total_weight":float(weights[len(train_def):].sum()),"manual_total_weight":len(train_def),
          "test_labels_accessed":False,"protected_files":protected_paths(root),"packages":packages(),
          "annotation_source":"AI_assisted_review; no independent human gold standard",
          "embedding_model_revision":emeta["model_revision"],
          "score_meanings":{"A":"normalized exact-match flag","B":"character TF-IDF cosine",
                           "C":"logistic classifier score, not calibrated duplicate probability",
                           "D":"normalized encoder cosine","E":"logistic classifier score, not calibrated duplicate probability"}}
    write_csv(root/"results/tables/validation_comparison.csv",pd.DataFrame(rows))
    write_json(root/"models/model_lock.json",lock)
    print(f"Models locked. Selected method={best}; automatic validation target met={best is not None}. Test labels remain unopened.")
