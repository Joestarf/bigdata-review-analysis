"""Generate a fixed larger blind benchmark before any model fit or test opening."""
from pathlib import Path
from collections import defaultdict
import shutil
import numpy as np
import pandas as pd
from .common import ROOT,SEED,load_study,pair_id,word_count,write_csv,write_json,read_json,sha_file
from .features import fit_lexical
from .embedding import load_embedding_cache
from .pairs import validate_pairs
from .annotations import GateError

TARGET={"train":4000,"validation":1000,"test":1000}

def prepare_scale(root:Path=ROOT):
    if (root/"models/model_lock.json").exists():
        raise GateError("Do not enlarge a benchmark after model lock/test evaluation.")
    out=root/"data/annotation/scale6000"
    df,manifest=load_study(root)
    if (out/"manifest.json").exists():
        existing_manifest=read_json(out/"manifest.json")
        if existing_manifest["dataset_version"]!=manifest["dataset_version"]:
            raise GateError("Existing scale candidates belong to a different dataset.")
        for name,key in [("additional_metadata.csv","additional_metadata_sha256"),("additional_texts.csv","additional_texts_sha256")]:
            if sha_file(out/name)!=existing_manifest[key]:raise GateError("Existing scale candidate checksum mismatch.")
        print("Fixed 6000 benchmark candidates verified; no repeated mining.");return
    existing=pd.read_csv(root/"data/annotation/pair_metadata.csv",keep_default_na=False)
    used=set(existing.pair_id)
    vectorizer=fit_lexical(root);embeddings,_,_=load_embedding_cache(root)
    pools={s:defaultdict(list) for s in TARGET}
    for asin,g in df[df.supervised_eligible & df.split.isin(TARGET)].groupby("asin",sort=True):
        if len(g)<2:continue
        x=vectorizer.transform(g.text);lex=(x@x.T).toarray()
        v=embeddings[g.index];sem=np.clip(v@v.T,-1,1)
        i,j=np.triu_indices(len(g),1);ids=g.review_id.tolist()
        words=np.array([word_count(t) for t in g.text]);split=g.split.iloc[0]
        candidates=[]
        for a,b in zip(i,j):
            pid=pair_id(ids[a],ids[b])
            if pid in used:continue
            candidates.append({"pair_id":pid,"asin":asin,"review_id_a":ids[a],"review_id_b":ids[b],
                               "split":split,"lexical_score":float(lex[a,b]),"semantic_score":float(sem[a,b]),
                               "short_pair":bool(min(words[a],words[b])<5)})
        # Keep bounded top pools, plus a deterministic local random draw for coverage.
        long=[r for r in candidates if not r["short_pair"]]
        pools[split]["lexical"].extend(sorted(long,key=lambda r:(-r["lexical_score"],r["pair_id"]))[:40])
        low=[r for r in long if r["lexical_score"]<.6]
        pools[split]["semantic"].extend(sorted(low,key=lambda r:(-r["semantic_score"],r["pair_id"]))[:40])
        pools[split]["short"].extend(sorted((r for r in candidates if r["short_pair"]),key=lambda r:(-r["lexical_score"],r["pair_id"]))[:20])
        rng=np.random.default_rng(int(asin.encode().hex()[:12],16)+SEED+2)
        pools[split]["random"].extend(candidates[int(k)] for k in rng.permutation(len(candidates))[:40])
    rng=np.random.default_rng(SEED+2);rows=[];quotas=[]
    for split,target in TARGET.items():
        needed=target-int(existing.split.eq(split).sum())
        per_product=defaultdict(int);selected=set()
        allocated={"lexical":int(needed*.4),"semantic":int(needed*.25),"random":int(needed*.2)}
        allocated["short"]=needed-sum(allocated.values())
        selected_rows=[]
        for kind,quota in allocated.items():
            candidates=pools[split][kind]
            if kind=="random":candidates=[candidates[int(k)] for k in rng.permutation(len(candidates))]
            else:candidates=sorted(candidates,key=lambda r:(-r["semantic_score"] if kind=="semantic" else -r["lexical_score"],r["pair_id"]))
            n=0
            for row in candidates:
                if row["pair_id"] in selected or per_product[row["asin"]]>=20:continue
                selected.add(row["pair_id"]);per_product[row["asin"]]+=1
                selected_rows.append(dict(row,sampling_stratum="scale6000_"+kind));n+=1
                if n==quota:break
            if n!=quota:raise GateError(f"Insufficient predeclared candidate stratum {split}/{kind}: {n}/{quota}")
            quotas.append({"split":split,"stratum":kind,"selected":n})
        rows.extend(selected_rows[int(k)] for k in rng.permutation(len(selected_rows)))
    added=pd.DataFrame(rows);validate_pairs(added,df)
    combined=pd.concat([existing,added],ignore_index=True);validate_pairs(combined,df)
    lookup=df.set_index("review_id").text
    public=pd.DataFrame({"pair_id":added.pair_id,"text_a":added.review_id_a.map(lookup),"text_b":added.review_id_b.map(lookup)})
    write_csv(out/"additional_metadata.csv",added)
    write_csv(out/"additional_texts.csv",public)
    for split in TARGET:
        ids=added.loc[added.split.eq(split),"pair_id"]
        blind=public[public.pair_id.isin(ids)].reset_index(drop=True)
        write_csv(out/f"blind_{split}.csv",blind)
        # 200-pair sheets let each reviewer read/save bounded batches without losing progress.
        for k in range(0,len(blind),200):
            write_csv(out/f"batches/{split}_{k//200+1:02d}.csv",blind.iloc[k:k+200])
        write_csv(out/f"labels_{split}.csv",pd.DataFrame({"pair_id":ids,"label":"","reason":"","annotation_version":""}))
    write_json(out/"manifest.json",{"dataset_version":manifest["dataset_version"],"seed":SEED+2,
        "target":TARGET,"base_pair_sha256":sha_file(root/"data/annotation/pair_metadata.csv"),
        "additional_metadata_sha256":sha_file(out/"additional_metadata.csv"),
        "additional_texts_sha256":sha_file(out/"additional_texts.csv"),"quotas":quotas,
        "labels_used_for_selection":False,"model_fit_before_generation":False,
        "policy":"One fixed batch; 40% lexical,25% semantic-lowlex,20% random,15% short; cap20/product; source split inherited; all retained",
        "annotation_source":"AI_assisted_review; no human gold standard",
        "note":"Generated separately while 1200-pair annotation is running. Independent test manager merges final labels; no modeling code reads test values."})
    print({"target":TARGET,"additional":added.split.value_counts().to_dict(),"path":str(out)})

def activate_scale(root:Path=ROOT):
    """Merge candidate definitions once; never parse test labels."""
    from .annotations import require_test_seal, read_labels
    if (root/"models/model_lock.json").exists():raise GateError("Model already locked.")
    out=root/"data/annotation";scale=out/"scale6000";sm=read_json(scale/"manifest.json")
    if sm.get("activated_pair_sha256"):
        if sha_file(out/"pair_metadata.csv")!=sm["activated_pair_sha256"]:
            raise GateError("Activated scale metadata changed.")
        print("6000 candidate definitions already activated.");return
    # Finish the intermediate batch, then archive its immutable materials.
    read_labels(root,"train");read_labels(root,"validation");require_test_seal(root)
    df,manifest=load_study(root)
    if sm["dataset_version"]!=manifest["dataset_version"] or sha_file(out/"pair_metadata.csv")!=sm["base_pair_sha256"]:
        raise GateError("Scale base dataset/candidates changed.")
    for name,key in [("additional_metadata.csv","additional_metadata_sha256"),("additional_texts.csv","additional_texts_sha256")]:
        if sha_file(scale/name)!=sm[key]:raise GateError("Scale candidate checksum mismatch.")
    existing=pd.read_csv(out/"pair_metadata.csv",keep_default_na=False)
    additional=pd.read_csv(scale/"additional_metadata.csv",keep_default_na=False)
    combined=pd.concat([existing,additional],ignore_index=True);validate_pairs(combined,df)
    if combined.split.value_counts().to_dict()!=TARGET:raise GateError("Final fixed quota mismatch.")
    backup=out/"original1200";backup.mkdir(exist_ok=False)
    for name in ["pair_metadata.csv","annotation_texts.csv","pair_manifest.json","guideline_freeze.json",
                 "labels_train.csv","labels_validation.csv","labels_test.csv","test_seal.json"]:
        shutil.copy2(out/name,backup/name)
    texts=pd.concat([pd.read_csv(out/"annotation_texts.csv",keep_default_na=False),
                     pd.read_csv(scale/"additional_texts.csv",keep_default_na=False)],ignore_index=True)
    if not texts.pair_id.is_unique or set(texts.pair_id)!=set(combined.pair_id):raise GateError("Scale body ID mismatch.")
    for split in ["train","validation"]:
        old=pd.read_csv(out/f"labels_{split}.csv",keep_default_na=False,dtype=str)
        new=pd.read_csv(scale/f"labels_{split}.csv",keep_default_na=False,dtype=str)
        write_csv(out/f"labels_{split}.csv",pd.concat([old,new],ignore_index=True))
    write_csv(out/"pair_metadata.csv",combined);write_csv(out/"annotation_texts.csv",texts)
    meta=read_json(out/"pair_manifest.json")
    meta.update({"candidate_version":3,"quotas":TARGET,"pair_table_sha256":sha_file(out/"pair_metadata.csv"),
                 "text_table_sha256":sha_file(out/"annotation_texts.csv"),"scale_policy":sm["policy"]})
    write_json(out/"pair_manifest.json",meta)
    freeze=read_json(out/"guideline_freeze.json")
    freeze["previous_pair_table_sha256"]=freeze["pair_table_sha256"]
    freeze["pair_table_sha256"]=meta["pair_table_sha256"]
    freeze["scale_policy"]="Target6000 fixed before fitting; label definitions unchanged."
    write_json(out/"guideline_freeze.json",freeze)
    sm["activated_pair_sha256"]=meta["pair_table_sha256"];write_json(scale/"manifest.json",sm)
    # Existing test receipt no longer matches new tasks until independent manager merges.
    print("6000 candidate metadata activated. Test manager must merge/seal1000; train/validation supplemental labels may need synchronization.")

def sync_scale_labels(root:Path=ROOT):
    """Overlay independent train/validation judgments only; test manager owns test."""
    out=root/"data/annotation";scale=out/"scale6000";sm=read_json(scale/"manifest.json")
    if not sm.get("activated_pair_sha256"):raise GateError("Activate candidate metadata first.")
    if (root/"models/model_lock.json").exists():raise GateError("Cannot change locked labels.")
    pending={}
    for split in ["train","validation"]:
        old=pd.read_csv(out/f"labels_{split}.csv",keep_default_na=False,dtype=str).set_index("pair_id")
        paths=[scale/f"labels_{split}.csv"]
        if split=="train":
            paths+=sorted(scale.glob("labels_train_*.csv"))
        seen={}
        for path in paths:
            new=pd.read_csv(path,keep_default_na=False,dtype=str).set_index("pair_id")
            if set(new.columns)!={"label","reason","annotation_version"}:
                raise GateError(f"Supplemental label schema invalid: {path.name}")
            if not new.index.is_unique or not new.index.isin(old.index).all():
                raise GateError(f"Supplemental label IDs invalid: {path.name}")
            filled=new.label.ne("")
            if (new.loc[filled,"reason"].str.strip().eq("").any()
                or new.loc[filled,"annotation_version"].ne(sm["dataset_version"]).any()
                or not new.loc[filled,"label"].isin(["exact","near","distinct","uncertain"]).all()):
                raise GateError(f"Supplemental label/reason/version invalid: {path.name}")
            for pid,row in new.loc[filled].iterrows():
                values=tuple(row[col] for col in ["label","reason","annotation_version"])
                if pid in seen and seen[pid]!=values:
                    raise GateError(f"Conflicting completed train shards for pair {pid}")
                seen[pid]=values
            old.loc[new.index[filled],new.columns]=new.loc[filled]
        pending[split]=old.reset_index()
    # Validate both splits before any write. Test values are never parsed here.
    for split,frame in pending.items():
        write_csv(out/f"labels_{split}.csv",frame)
    print("Train/validation supplemental labels synchronized; test labels not parsed.")

if __name__=="__main__":prepare_scale()
