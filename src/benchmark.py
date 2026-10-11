"""Cached A/B/D matrix throughput; excludes labeling, download, and classifier training."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT,load_study,write_csv,write_json
from .features import fit_lexical
from .embedding import load_embedding_cache
from .performance import Monitor

def benchmark(root=ROOT):
    df,manifest=load_study(root)
    vec=fit_lexical(root)
    embeddings,_,emeta=load_embedding_cache(root)
    groups=list(df.groupby("asin",sort=True))
    predictor=None
    if (root/"models/model_lock.json").exists():
        from .inference import Predictor
        predictor=Predictor(root)
    rows=[]
    for n in [100,500,1000]:
        selected=groups[:n]
        checksum=0.; pairs=0; reviews=0
        with Monitor() as monitor:
            for _,g in selected:
                if predictor is not None:
                    scored=predictor.score_product(g)
                    checksum+=float(scored.score.sum())
                    pairs+=len(scored)//5
                else:
                    ids=g.index.to_numpy()
                    x=vec.transform(g.text)
                    lexical=(x@x.T).toarray()
                    semantic=embeddings[ids]@embeddings[ids].T
                    a,b=np.triu_indices(len(g),1)
                    checksum+=float(lexical[a,b].sum()+semantic[a,b].sum())
                    pairs+=len(a)
                reviews+=len(g)
        rows.append({"products":len(selected),"reviews":reviews,"pairs":pairs,**monitor.result,
                     "score_checksum":checksum,"scope":("all five methods, features and decisions; no CSV write" if predictor else "cached TF-IDF and semantic all-pair matrices; no CSV output or C/E classification")})
    write_csv(root/"results/tables/cached_matrix_benchmark.csv",pd.DataFrame(rows))
    write_json(root/"results/tables/encoding_performance.json",emeta["performance"] if "performance" in emeta else emeta)
    figures=root/"results/figures";figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axes[0].plot([r["products"] for r in rows],[r["seconds"] for r in rows],"o-")
    axes[0].set(xlabel="Products",ylabel="Elapsed seconds",title="Five-method inference" if predictor else "Cached pair matrices")
    axes[1].plot([r["products"] for r in rows],[r["peak_rss_mb"] for r in rows],"o-")
    axes[1].set(xlabel="Products",ylabel="Peak process RSS (MB)",title="Absolute peak memory")
    axes[1].set_ylim(0,max(r["peak_rss_mb"] for r in rows)*1.1)
    fig.savefig(figures/"cached_pair_performance.png",dpi=160);plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    return rows
