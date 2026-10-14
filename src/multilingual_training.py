"""Train an independent multilingual revision; never consumes the original test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .common import ROOT, load_study, sha_file, write_csv, write_json, packages
from .annotations import read_labels
from .training import validate_lock, bounded_metrics, choose_threshold
from .multilingual_features import MultilingualLexical, FEATURE_NAMES, analyze_text, normalize_text, pair_language, score_methods, allowed_features
from .multilingual_embedding import MultilingualEncoder, CONFIG, digest

METHODS = ("A","B","C","D","E")
MODEL_DIR = "models/multilingual_v2"
OUT_DIR = "results/multilingual_v2"
CODE_FILES = ["src/multilingual_features.py","src/multilingual_embedding.py",
              "src/multilingual_training.py","src/multilingual_inference.py"]

def annotate_pairs(root, split, candidate_manifest):
    directory=root/"data/multilingual_v2"
    labels=pd.read_csv(directory/f"labels_{split}.csv",keep_default_na=False)
    tasks=pd.read_csv(directory/f"tasks_{split}.csv",keep_default_na=False)
    if not labels.pair_id.is_unique or set(labels.pair_id)!=set(tasks.pair_id):
        raise ValueError(f"{split} label IDs do not match fixed blind tasks")
    if not labels.label.isin(["exact","near","distinct","uncertain"]).all() or labels.reason.str.strip().eq("").any():
        raise ValueError(f"Incomplete or invalid {split} labels")
    if sha_file(directory/f"tasks_{split}.csv")!=candidate_manifest["task_sha256"][split]:
        raise ValueError(f"Changed {split} task content")
    meta=pd.read_csv(directory/"pair_metadata.csv",keep_default_na=False)
    meta=meta[meta.split.eq(split)]
    result=meta.merge(tasks,on="pair_id",validate="one_to_one").merge(labels,on="pair_id",validate="one_to_one")
    exact=result.label.eq("exact")
    if any(normalize_text(a)!=normalize_text(b) for a,b in zip(result.loc[exact,"text_a"],result.loc[exact,"text_b"])):
        raise ValueError("Exact label on unequal normalized text")
    result["target"]=result.label.map({"exact":1,"near":1,"distinct":0,"uncertain":np.nan})
    return result

def verify_pairs(table):
    if not table.pair_id.is_unique or (table.review_id_a==table.review_id_b).any():
        raise ValueError("Repeated pair ID or self-pair")
    keys=[tuple(sorted([a,b])) for a,b in zip(table.review_id_a,table.review_id_b)]
    if len(set(keys))!=len(keys): raise ValueError("Reverse or repeated review pair")
    products=table.groupby("asin").split.nunique()
    if products.gt(1).any(): raise ValueError("Product appears in multiple splits")
    texts={}
    for row in table.itertuples():
        for body in (row.text_a,row.text_b):
            texts.setdefault(normalize_text(body),set()).add(row.split)
    common={t for t,splits in texts.items() if len(splits)>1}
    if common: raise ValueError(f"{len(common)} normalized texts leak across labeled sets")

def prepare_training(root=ROOT):
    validate_lock(root)
    df,_=load_study(root)
    bodies=pd.read_csv(root/"data/annotation/annotation_texts.csv",keep_default_na=False)
    rows=[]
    for split in ("train","validation"):
        table=read_labels(root,split).merge(bodies,on="pair_id",validate="one_to_one")
        table["language"]=[pair_language(a,b) for a,b in zip(table.text_a,table.text_b)]
        table["origin"]="amazon_2023_automatically_assisted"
        rows.append(table)
    directory=root/"data/multilingual_v2"
    manifest=json.loads((directory/"candidate_manifest.json").read_text())
    if sha_file(directory/"pair_metadata.csv")!=manifest["metadata_sha256"]:
        raise ValueError("Chinese candidate metadata mismatch")
    for split in ("train","validation"):
        table=annotate_pairs(root,split,manifest)
        table["origin"]="chinese_auxiliary_automatically_assisted"
        rows.append(table)
    columns=["pair_id","asin","review_id_a","review_id_b","split","text_a","text_b",
             "label","reason","target","language","origin"]
    combined=pd.concat([r[columns] for r in rows],ignore_index=True)
    # New NFKC normalization can introduce additional common templates: quarantine them.
    spread={}
    for row in combined.itertuples():
        for text in (row.text_a,row.text_b):
            spread.setdefault(normalize_text(text),set()).add(row.split)
    public={t for t,parts in spread.items() if len(parts)>1}
    quarantine=combined.text_a.map(normalize_text).isin(public)|combined.text_b.map(normalize_text).isin(public)
    write_csv(directory/"normalization_quarantine.csv",combined[quarantine])
    combined=combined[~quarantine].copy()
    verify_pairs(combined)
    train=combined[combined.split.eq("train")]
    val=combined[combined.split.eq("validation")]
    for name,table,p,n in (("train",train,20,50),("validation",val,20,20)):
        if table.target.eq(1).sum()<p or table.target.eq(0).sum()<n:
            raise ValueError(f"{name} does not meet confirmed-positive/negative gate")
    source=pd.read_csv(directory/"reviews.csv",keep_default_na=False)
    source_manifest=json.loads((directory/"source_manifest.json").read_text())
    if sha_file(directory/"reviews.csv")!=source_manifest["sample_sha256"] or sha_file(directory/"reviews.csv")!=manifest["sample_sha256"]:
        raise ValueError("Chinese source sample checksum mismatch")
    lookup=pd.concat([df[["review_id","asin","text","product_split"]],source[["review_id","asin","text","product_split"]]],ignore_index=True).set_index("review_id")
    for row in combined.itertuples():
        for rid,text in ((row.review_id_a,row.text_a),(row.review_id_b,row.text_b)):
            if rid not in lookup.index or lookup.loc[rid,"asin"]!=row.asin or lookup.loc[rid,"product_split"]!=row.split or lookup.loc[rid,"text"]!=text:
                raise ValueError("Pair source/product/split mismatch")
    pool=df[df.split.eq("train") & df.supervised_eligible].text.tolist()
    pool += source[source.product_split.eq("train") & source.supervised_eligible].text.tolist()
    excluded=set(public)
    # Exclude new-normalization overlap against *all* validation-pool records, not just labels.
    excluded.update(df[df.product_split.ne("train")].text.map(normalize_text))
    excluded.update(source[source.product_split.ne("train")].text.map(normalize_text))
    pool=[text for text in pool if normalize_text(text) not in excluded]
    if not pool: raise ValueError("No leakage-free training vocabulary corpus")
    return train,val,list(dict.fromkeys(pool)),int(quarantine.sum())

def language_metrics(table,scores,threshold,allowed):
    rows=[]
    for language,positions in table.groupby("language").indices.items():
        index=np.asarray(positions); y=table.target.to_numpy(float)[index]
        row=bounded_metrics(y,(scores[index]>=threshold)&allowed[index])
        rows.append({"language":language,**row,
                     "min_positive_negative_met":bool((y==1).sum()>=20 and (y==0).sum()>=20)})
    return rows

def train(root=ROOT):
    out=root/MODEL_DIR;out.mkdir(parents=True,exist_ok=True)
    lock_path=out/"manifest.json"
    if lock_path.exists():
        raise ValueError("Multilingual model already locked; use a new revision to retrain")
    seal_path=root/"data/multilingual_v2/test_seal.json"
    if not seal_path.exists():
        raise ValueError("Complete and seal independent Chinese test labels before model locking")
    seal=json.loads(seal_path.read_text())
    directory=root/"data/multilingual_v2"
    if not seal.get("sealed") or seal.get("rows")!=150 or seal.get("labels_sha256")!=sha_file(directory/"labels_test.csv") or seal.get("tasks_sha256")!=sha_file(directory/"tasks_test.csv"):
        raise ValueError("Chinese test completion seal mismatch; no test labels parsed")
    original_sha=sha_file(root/"models/model_lock.json")
    handoff_sha=sha_file(root/"deliverables/team_handoff.zip")
    train_table,val_table,pool,quarantine=prepare_training(root)
    lexical=MultilingualLexical().fit(pool)
    texts=list(dict.fromkeys(train_table.text_a.tolist()+train_table.text_b.tolist()+
                             val_table.text_a.tolist()+val_table.text_b.tolist()))
    encoder=MultilingualEncoder()
    vectors,receipt=encoder.encode(texts,out/"cache")
    lookup={t:v for t,v in zip(texts,vectors)}
    def features(table):
        similarities=[float(lookup[a]@lookup[b]) for a,b in zip(table.text_a,table.text_b)]
        return lexical.transform_pairs(table.text_a.tolist(),table.text_b.tolist(),similarities)
    xtrain=features(train_table);xval=features(val_table)
    known=train_table.target.notna().to_numpy()
    y=train_table.target.to_numpy(float)[known].astype(int)
    # Balance source corpora by aggregate weight; heuristic language tags are report strata only.
    group=train_table.origin.to_numpy()[known]
    counts={g:int((group==g).sum()) for g in set(group)}
    weights=np.array([len(y)/(len(counts)*counts[g]) for g in group])
    yval=val_table.target.to_numpy(float)
    allowed=allowed_features(xval)
    models={};configs={};hyper=[]
    for method,width in (("C",len(FEATURE_NAMES)),("E",len(FEATURE_NAMES)+1)):
        candidates=[]
        for c in (.1,1.,10.):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=42))
            model.fit(xtrain[known,:width],y,standardscaler__sample_weight=weights,
                      logisticregression__sample_weight=weights)
            scores=model.predict_proba(xval[:,:width])[:,1]
            selection=choose_threshold(yval,scores,allowed)
            fallback=bounded_metrics(yval,(scores>=.5)&allowed)
            hyper.append({"method":method,"C":c,"qualified_overall":selection is not None,
                          "review_threshold":selection["threshold"] if selection else .5,**fallback})
            candidates.append((selection,fallback,c,model))
        accepted=[candidate for candidate in candidates if candidate[0] is not None]
        chosen=max(accepted,key=lambda r:(r[0]["recall"],r[0]["precision_lower"],-r[2])) if accepted else max(
            candidates,key=lambda r:(r[1]["f1"],-r[2]))
        selection,_,c,model=chosen
        models[method]=model
        configs[method]={"C":c,"threshold":selection["threshold"] if selection else .5,
                         "qualified_overall":selection is not None}
    scores=score_methods(xval,models)
    for method in ("A","B","D"):
        selection=choose_threshold(yval,scores[method],allowed,fixed=1. if method=="A" else None)
        configs[method]={"threshold":selection["threshold"] if selection else (1. if method=="A" else .8),
                         "qualified_overall":selection is not None}
    result=[]
    for method in METHODS:
        threshold=configs[method]["threshold"]
        grouped=language_metrics(val_table,scores[method],threshold,allowed)
        # Each supported application group needs validation evidence; missing mixed data closes gate.
        per={r["language"]:r for r in grouped}
        gate=all(g in per and per[g]["min_positive_negative_met"] and
                 per[g]["precision_lower"]>=.95 and per[g]["automatic_candidate_positive"]>=30
                 for g in ("en","zh","mixed"))
        configs[method]["qualified_multilingual"]=bool(gate and configs[method]["qualified_overall"])
        result.append({"method":method,"language":"all",**bounded_metrics(yval,(scores[method]>=threshold)&allowed)})
        result.extend({"method":method,**r} for r in grouped)
    joblib.dump(lexical,out/"lexical.joblib")
    for method,model in models.items():joblib.dump(model,out/f"{method}.joblib")
    output=root/OUT_DIR
    write_csv(output/"validation_metrics.csv",pd.DataFrame(result))
    write_csv(output/"hyperparameters.csv",pd.DataFrame(hyper))
    write_csv(root/"data/multilingual_v2/training_input.csv",train_table)
    write_csv(root/"data/multilingual_v2/validation_input.csv",val_table)
    files={p.name:sha_file(p) for p in [out/"lexical.joblib",out/"C.joblib",out/"E.joblib"]}
    write_json(lock_path,{
        "schema":"multilingual-v2.1","encoder":CONFIG,"feature_names":FEATURE_NAMES+["semantic_cosine"],
        "files":files,"code_hashes":{p:sha_file(root/p) for p in CODE_FILES},
        "configs":configs,"actual_decision_policy":"review_only",
        "qualification_requires":["en","zh","mixed"],"label_provenance":"Automatically assisted; not independent human gold",
        "input_hashes":{str(p.relative_to(root)):sha_file(p) for p in [
            root/"data/multilingual_v2/training_input.csv",root/"data/multilingual_v2/validation_input.csv",
            root/"data/multilingual_v2/source_manifest.json",root/"data/multilingual_v2/candidate_manifest.json"]},
        "original_lock_sha256":original_sha,"original_handoff_sha256":handoff_sha,
        "training_confirmed":len(y),"training_languages":train_table.loc[known,"language"].value_counts().to_dict(),
        "training_weight_groups":counts,"language_tags":"Han/Latin script heuristic, not linguistic language detection",
        "vocabulary_corpus_count":len(pool),"vocabulary_sha256":digest(pool),
        "quarantined_pair_count":quarantine,"embedding_receipt":receipt,"packages":packages(),
        "test_seal_sha256":sha_file(seal_path),
        "original_test_used":False,"multilingual_test_used":False})
    validate_lock(root)
    if original_sha!=sha_file(root/"models/model_lock.json") or handoff_sha!=sha_file(root/"deliverables/team_handoff.zip"):
        raise ValueError("Original artifacts changed")
    print(json.dumps({"trained_confirmed_pairs":len(y),"languages":counts,
                      "decision":"review_only","manifest":str(lock_path)},ensure_ascii=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["train"])
    parser.parse_args();train()
