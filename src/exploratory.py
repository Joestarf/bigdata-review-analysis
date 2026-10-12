"""Describe train/validation labels and uncalibrated A/B/D; never touch test labels."""
import numpy as np
import pandas as pd
from .common import ROOT,load_study,write_csv,write_json
from .annotations import read_labels,require_frozen_guidelines
from .features import fit_lexical,indexed_pair_features
from .embedding import load_embedding_cache
from .training import metrics,automatic_allowed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def explore_labels(root=ROOT):
    require_frozen_guidelines(root)
    df,manifest=load_study(root)
    vectorizer=fit_lexical(root)
    embeddings,_,_=load_embedding_cache(root)
    counts,comparisons,score_rows=[],[],[]
    for split in ["train","validation"]:
        labeled=read_labels(root,split)
        counts.extend({"split":split,"label":label,"n":int(n),"source":"AI_assisted_review"}
                      for label,n in labeled.label.value_counts().items())
        x=indexed_pair_features(labeled,df,vectorizer,embeddings)
        allowed=automatic_allowed(labeled,df)
        definite=labeled.target.notna().to_numpy()
        for m,scores,threshold in [("A",x[:,4],1.),("B",x[:,0],.8),("D",x[:,5],.8)]:
            p=(scores>=threshold)&allowed
            comparisons.append({"split":split,"method":m,"threshold":threshold,"threshold_selected":False,
                                **metrics(labeled.loc[definite,"target"],p[definite]),
                                "uncertain":int((~definite).sum()),"review_short":int((~allowed).sum()),
                                "mode":"exploratory fixed thresholds, not validated automatic classifier"})
            score_rows.extend({"pair_id":r.pair_id,"split":split,"method":m,"score":float(scores[i]),"label":r.label,
                               "candidate_positive":bool(p[i]),"decision":"review","label_reason":r.reason}
                              for i,r in enumerate(labeled.itertuples()))
    out=root/"results/tables"
    val=read_labels(root,"validation")
    view=pd.DataFrame(score_rows)
    decision=view[(view.split=="validation") & (view.method=="D")].set_index("pair_id").candidate_positive
    lookup=df.set_index("review_id")
    masks={"false_positive":val.label.eq("distinct") & val.pair_id.map(decision),
           "false_negative":val.label.isin(["exact","near"]) & ~val.pair_id.map(decision),
           "uncertain":val.label.eq("uncertain")}
    cases=[]
    for kind,mask in masks.items():
        for row in val[mask].head(10).itertuples():
            a,b=lookup.loc[row.review_id_a],lookup.loc[row.review_id_b]
            cases.append({"kind":kind,"pair_id":row.pair_id,"text_a":a.text,"text_b":b.text,"label":row.label,
              "label_reason":row.reason,"scores":"; ".join(f"{s.method}={s.score:.4f}" for s in view[view.pair_id==row.pair_id].itertuples()),
              "scope":"exploratory validation D threshold 0.8; not final test",
              "possible_cause":"common product topic or insufficient specifics; see label reason",
              "downstream_risk":"false positives could downweight normal reviews; outputs remain review-only"})
    write_csv(out/"exploratory_error_cases.csv",pd.DataFrame(cases))
    labels=pd.DataFrame(counts).pivot(index="split",columns="label",values="n").fillna(0).reindex(["train","validation"])
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    bottom=np.zeros(len(labels))
    for label,color in [("near","#187a68"),("distinct","#4878ad"),("uncertain","#c8a555")]:
        values=labels[label].to_numpy()
        ax.bar(labels.index,values,bottom=bottom,label=label,color=color)
        for i,v in enumerate(values): ax.text(i,bottom[i]+v/2,str(int(v)),ha="center",va="center",color="white")
        bottom+=values
    ax.set(ylabel="Review pairs",title="Reference labels after blind review")
    ax.legend(loc="upper right")
    (root/"results/figures").mkdir(parents=True,exist_ok=True)
    fig.savefig(root/"results/figures/annotation_label_distribution.png",dpi=160);plt.close(fig)
    write_csv(out/"annotation_label_counts.csv",pd.DataFrame(counts))
    write_csv(out/"exploratory_baseline_comparison.csv",pd.DataFrame(comparisons))
    write_csv(out/"exploratory_pair_scores.csv",pd.DataFrame(score_rows))
    write_json(out/"exploratory_manifest.json",{"dataset_version":manifest["dataset_version"],"test_labels_opened":False,
               "models_trained":False,"automatic_ready":False,"label_source":"AI_assisted_review"})
    print(pd.DataFrame(counts).to_string(index=False))
