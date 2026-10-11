"""One fixed supplemental batch, before model locking; never inspect labels for mining."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .common import ROOT,SEED,load_study,pair_id,word_count,write_csv,write_json,read_json,sha_file
from .features import fit_lexical
from .embedding import load_embedding_cache
from .pairs import validate_pairs
from .annotations import require_frozen_guidelines,GateError

def extend_pairs(root:Path=ROOT,count=300):
    if (root/"models/model_lock.json").exists():raise GateError("Cannot expand candidate definitions after model lock.")
    out=root/"data/annotation"
    if (out/"extension_manifest.json").exists():
        print("Fixed supplementary batch already exists; no repeated label-driven mining.")
        return
    require_frozen_guidelines(root)
    df,manifest=load_study(root)
    embeddings,_,_=load_embedding_cache(root);vec=fit_lexical(root)
    original=pd.read_csv(out/"pair_metadata.csv",keep_default_na=False)
    existing=set(original.pair_id)
    pools={split:{"lexical":[],"semantic":[]} for split in ["validation","test"]}
    for asin,g in df[df.supervised_eligible & df.split.isin(pools)].groupby("asin",sort=True):
        if len(g)<2:continue
        x=vec.transform(g.text);lex=(x@x.T).toarray()
        v=embeddings[g.index];sem=np.clip(v@v.T,-1,1)
        a,b=np.triu_indices(len(g),1)
        words=np.array([word_count(t) for t in g.text])
        candidates=[]
        ids=g.review_id.tolist();split=g.split.iloc[0]
        for ia,ib in zip(a,b):
            if min(words[ia],words[ib])<8:continue
            pid=pair_id(ids[ia],ids[ib])
            if pid in existing:continue
            candidates.append({"pair_id":pid,"asin":asin,"review_id_a":ids[ia],"review_id_b":ids[ib],
                              "split":split,"lexical_score":float(lex[ia,ib]),"semantic_score":float(sem[ia,ib]),
                              "short_pair":False})
        pools[split]["lexical"].extend(sorted(candidates,key=lambda r:(-r["lexical_score"],r["pair_id"]))[:8])
        low=[r for r in candidates if r["lexical_score"]<.6]
        pools[split]["semantic"].extend(sorted(low,key=lambda r:(-r["semantic_score"],r["pair_id"]))[:8])
    rng=np.random.default_rng(SEED+1)
    additions=[];selection=[]
    for split in pools:
        used=set();per_product={};rows=[]
        for kind,quota in [("lexical",int(count*2/3)),("semantic",count-int(count*2/3))]:
            score="lexical_score" if kind=="lexical" else "semantic_score"
            pool=sorted(pools[split][kind],key=lambda r:(-r[score],r["pair_id"]))
            chosen=0
            for r in pool:
                if r["pair_id"] in used or per_product.get(r["asin"],0)>=5:continue
                rows.append(dict(r,sampling_stratum="extension_"+kind))
                used.add(r["pair_id"]);per_product[r["asin"]]=per_product.get(r["asin"],0)+1;chosen+=1
                if chosen==quota:break
            if chosen!=quota:raise GateError(f"Insufficient fixed extension pool: {split}/{kind} {chosen}/{quota}")
            selection.append({"split":split,"stratum":kind,"quota":quota,"selected":chosen})
        additions.extend(rows[int(i)] for i in rng.permutation(len(rows)))
    new=pd.DataFrame(additions);validate_pairs(new,df)
    backup=out/"original600";backup.mkdir(exist_ok=False)
    for name in ["pair_metadata.csv","annotation_texts.csv","pair_manifest.json","guideline_freeze.json",
                 "labels_train.csv","labels_validation.csv","labels_test.csv"]:
        shutil.copy2(out/name,backup/name)
    combined=pd.concat([original,new],ignore_index=True);validate_pairs(combined,df)
    lookup=df.set_index("review_id").text
    public=pd.DataFrame({"pair_id":combined.pair_id,"text_a":combined.review_id_a.map(lookup),"text_b":combined.review_id_b.map(lookup)})
    for split in pools:
        extra=pd.DataFrame({"pair_id":new.loc[new.split.eq(split),"pair_id"],"label":"","reason":"","annotation_version":""})
        if split == "test":
            # The independent test manager merges this sheet with the old labels.
            write_csv(out/"labels_extension_test.csv",extra)
        else:
            old=pd.read_csv(out/f"labels_{split}.csv",keep_default_na=False,dtype=str)
            write_csv(out/f"labels_{split}.csv",pd.concat([old,extra],ignore_index=True))
        write_csv(out/f"blind_extension_{split}.csv",public[public.pair_id.isin(extra.pair_id)])
    write_csv(out/"pair_metadata.csv",combined);write_csv(out/"annotation_texts.csv",public)
    meta=read_json(backup/"pair_manifest.json")
    prior=meta["pair_table_sha256"]
    meta.update({"pair_table_sha256":sha_file(out/"pair_metadata.csv"),"text_table_sha256":sha_file(out/"annotation_texts.csv"),
                 "original_quotas":meta["quotas"],"quotas":combined.split.value_counts().to_dict(),"candidate_version":2,
                 "extension_policy":"Fixed 300 per held-out split; 2/3 lexical, 1/3 semantic; >=8 words both; cap 5 pairs/product; no label access"})
    write_json(out/"pair_manifest.json",meta)
    freeze=read_json(backup/"guideline_freeze.json")
    freeze["previous_pair_table_sha256"]=freeze["pair_table_sha256"];freeze["pair_table_sha256"]=meta["pair_table_sha256"]
    freeze["extension_authorization"]="User requested continuation; guideline unchanged; original600 preserved."
    write_json(out/"guideline_freeze.json",freeze)
    write_json(out/"extension_manifest.json",{"dataset_version":manifest["dataset_version"],"original_pair_sha256":prior,
               "new_pair_sha256":meta["pair_table_sha256"],"counts":selection,"seed":SEED+1,"labels_used_in_selection":False,
               "rounds":1,"test_label_values_parsed_by_mining":False,"backup":"original600","note":"Enriched benchmark extension, not prevalence estimation; no repeat mining to force desired results."})
    print(combined.split.value_counts().to_dict())
