"""Generate a fixed larger blind benchmark before any model fit or test opening."""
from pathlib import Path
from collections import defaultdict
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
    if (out/"manifest.json").exists():
        print("Fixed 6000 benchmark candidates already generated; no repeated mining.");return
    df,manifest=load_study(root)
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

if __name__=="__main__":prepare_scale()
