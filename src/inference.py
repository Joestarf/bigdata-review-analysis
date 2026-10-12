"""Pair interfaces, review-safe previews, and within-product streamed inference."""
from pathlib import Path
import gzip
import csv
import io
import numpy as np
import pandas as pd
import joblib
from .common import ROOT, load_study, pair_id, normalize_text, word_count, write_csv, write_json, read_json, sha_text, sha_file
from .features import fit_lexical, structural_features, text_pair_features, prepare_structure, prepared_structural_features
from .embedding import load_embedding_cache, load_encoder, encode_texts
from .training import METHODS, validate_lock
from .annotations import GateError

class Predictor:
    def __init__(self,root:Path=ROOT):
        self.root=root
        self.df,self.manifest=load_study(root)
        self.vectorizer=fit_lexical(root)
        self.embeddings,self.embedding_index,self.embedding_meta=load_embedding_cache(root)
        self.positions={r:i for i,r in enumerate(self.df.review_id)}
        self.lock=validate_lock(root) if (root/"models/model_lock.json").exists() else None
        self.models={m:joblib.load(root/f"models/{m}.joblib") for m in ["C","E"]} if self.lock else {}
        self.encoder=None

    def _decision(self,m,score,a,b,short=None,exact=None):
        short = word_count(a)<5 or word_count(b)<5 if short is None else short
        if short:
            return "review","short_text; exact_signal="+str(normalize_text(a)==normalize_text(b) if exact is None else exact).lower()
        if not self.lock:
            return "review","annotation_gate_not_met_or_model_not_locked; exact_signal="+str(normalize_text(a)==normalize_text(b) if exact is None else exact).lower()
        config=self.lock["methods"][m]
        if self.lock.get("population_automatic_ready") is not True:
            return "review","enriched_benchmark_only; independent_population_review_required"
        if not config["qualified"]:
            return "review","validation_precision_target_not_met"
        return ("duplicate" if score>=config["threshold"] else "distinct"),"locked_validation_threshold"

    def _format(self,a,b,ids,asin,x,truncated=None):
        columns=["pair_id","asin","review_id_a","review_id_b","method","score","decision","reason"]
        if len(x)==0:return pd.DataFrame(columns=columns)
        scores={"A":x[:,4],"B":x[:,0],"D":x[:,5]}
        if self.lock:
            scores["C"]=self.models["C"].predict_proba(x[:,:5])[:,1]
            scores["E"]=self.models["E"].predict_proba(x)[:,1]
        truncated = [False]*len(a) if truncated is None else truncated
        rows=[]
        short_lookup={t:word_count(t)<5 for t in set(a)|set(b)}
        for i,(ta,tb) in enumerate(zip(a,b)):
            ia,ib=ids[i]
            pid=pair_id(ia,ib)
            short=short_lookup[ta] or short_lookup[tb]
            for m in METHODS:
                score=float(scores[m][i]) if m in scores else None
                if score is not None and not np.isfinite(score):raise ValueError("Nonfinite model score.")
                decision,reason=self._decision(m,score,ta,tb,short=short,exact=bool(x[i,4]))
                if truncated[i]:reason += "; encoder_input_truncated"
                rows.append({"pair_id":pid,"asin":asin,"review_id_a":ia,"review_id_b":ib,
                             "method":m,"score":score,"decision":decision,"reason":reason})
        return pd.DataFrame(rows,columns=["pair_id","asin","review_id_a","review_id_b","method","score","decision","reason"])

    def predict_pairs(self,pairs):
        """Accept iterable of (text_a,text_b); return five rows per pair."""
        pairs=list(pairs)
        if not pairs:return self._format([],[],[],"",np.empty((0,6)))
        if any(len(p)!=2 or not all(isinstance(t,str) and t.strip() for t in p) for p in pairs):
            raise ValueError("Each pair must contain two nonempty strings.")
        a,b=zip(*pairs)
        if self.encoder is None:self.encoder=load_encoder(self.embedding_meta["model_revision"])
        unique=list(dict.fromkeys(list(a)+list(b)))
        vectors,_,_=encode_texts(self.encoder,unique,64)
        lookup=dict(zip(unique,vectors))
        sims=[np.dot(lookup[ta],lookup[tb]) for ta,tb in pairs]
        x=text_pair_features(self.vectorizer,a,b,sims)
        ids=[(f"input_{sha_text(ta)}",f"input_{sha_text(tb)}") for ta,tb in pairs]
        lengths=self.encoder.tokenizer(unique,padding=False,truncation=False,return_length=True,verbose=False)["length"]
        truncated_map=dict(zip(unique,[n>256 for n in lengths]))
        return self._format(a,b,ids,"",x,[truncated_map[ta] or truncated_map[tb] for ta,tb in pairs])

    def score_product(self,reviews):
        """Input dataframe with asin,review_id,text; only one product, unique IDs."""
        if not {"asin","review_id","text"}.issubset(reviews.columns):raise ValueError("Product input needs asin, review_id, text.")
        if reviews.asin.nunique()!=1 or not reviews.review_id.is_unique:raise ValueError("Use one product and unique review IDs.")
        if len(reviews)<2:return self._format([],[],[],str(reviews.asin.iloc[0]) if len(reviews) else "",np.empty((0,6)))
        if not reviews.text.map(lambda t:isinstance(t,str) and bool(t.strip())).all():raise ValueError("Nonempty text required.")
        ids=reviews.review_id.tolist()
        for row in reviews.itertuples():
            if row.review_id in self.positions:
                old = self.df.iloc[self.positions[row.review_id]]
                if old.text != row.text or old.asin != row.asin:
                    raise ValueError("Review ID/body/product mismatch.")
        if all(r in self.positions for r in ids):
            pos=np.array([self.positions[r] for r in ids])
            original=self.df.iloc[pos]
            if original.text.tolist()!=reviews.text.tolist() or original.asin.tolist()!=reviews.asin.tolist():
                raise ValueError("Review ID/body/product mismatch.")
            vec=self.embeddings[pos]
            truncated=self.embedding_index.iloc[pos].truncated.to_numpy(dtype=bool)
        else:
            if self.encoder is None:self.encoder=load_encoder(self.embedding_meta["model_revision"])
            vec,_,_=encode_texts(self.encoder,reviews.text.tolist(),64)
            lengths=self.encoder.tokenizer(reviews.text.tolist(),padding=False,truncation=False,return_length=True,verbose=False)["length"]
            truncated=np.asarray(lengths)>256
        lexical=self.vectorizer.transform(reviews.text)
        lex=(lexical@lexical.T).toarray()
        sem=vec@vec.T
        i,j=np.triu_indices(len(reviews),1)
        texts=reviews.text.tolist()
        prepared=[prepare_structure(text) for text in texts]
        x=np.array([[float(lex[ia,ib])]+prepared_structural_features(prepared[ia],prepared[ib])+[float(sem[ia,ib])]
                    for ia,ib in zip(i,j)],dtype=np.float32)
        return self._format([texts[v] for v in i],[texts[v] for v in j],[(ids[ia],ids[ib]) for ia,ib in zip(i,j)],str(reviews.asin.iloc[0]),x,truncated[i]|truncated[j])

def predict_pairs(pairs,root=ROOT):
    return Predictor(root).predict_pairs(pairs)

def score_product(reviews,root=ROOT):
    return Predictor(root).score_product(reviews)

def handoff(root=ROOT,count=10):
    predictor=Predictor(root)
    products=sorted(predictor.df.asin.unique())[:count]
    reviews=predictor.df[predictor.df.asin.isin(products)]
    results=pd.concat([predictor.score_product(g) for _,g in reviews.groupby("asin",sort=True)],ignore_index=True)
    out=root/"results/handoff"
    write_csv(out/"reviews.csv",reviews[["asin","review_id","user_id","rating","text","timestamp"]])
    write_csv(out/"pair_scores.csv",results)
    write_json(out/"manifest.json",{"dataset_version":predictor.manifest["dataset_version"],"products":products,
               "review_rows":len(reviews),"pairs":len(results)//5,"method_rows":len(results),
               "status":"locked_model" if predictor.lock else "preview_only_no_locked_model",
               "automatic_decisions":bool(predictor.lock and predictor.lock.get("population_automatic_ready",False)),"transitive_groups_created":False,
               "file_sha256":{p:sha_file(out/p) for p in ["reviews.csv","pair_scores.csv"]},
               "model_lock_sha256":sha_file(root/"models/model_lock.json") if predictor.lock else None,
               "score_meanings":predictor.lock["score_meanings"] if predictor.lock else {"C":"untrained","E":"untrained"},
               "note":"Benchmark scores; review decisions require independent review before downstream weighting."})
    return {"products":len(products),"reviews":len(reviews),"pairs":len(results)//5,"path":str(out)}

def validate_inference(root=ROOT):
    """Verify a completed local inference receipt before claiming completion."""
    from .evaluation import validate_evaluation
    validate_lock(root);validate_evaluation(root)
    df,manifest=load_study(root)
    path=root/"results/predictions/pair_scores.csv.gz"
    receipt=root/"results/predictions/manifest.json"
    if not path.exists() or not receipt.exists():raise GateError("Full inference output or receipt is missing.")
    fm=read_json(receipt)
    expected=sum(len(g)*(len(g)-1)//2 for _,g in df.groupby("asin"))
    if (fm.get("dataset_version")!=manifest["dataset_version"]
        or fm.get("pairs")!=expected or fm.get("method_rows")!=expected*len(METHODS)
        or fm.get("products")!=int(df.asin.nunique()) or fm.get("methods")!=METHODS
        or fm.get("output_sha256")!=sha_file(path)
        or fm.get("model_lock_sha256")!=sha_file(root/"models/model_lock.json")
        or fm.get("final_evaluation_sha256")!=sha_file(root/"results/tables/final_evaluation.json")):
        raise GateError("Full inference provenance/count/checksum mismatch.")
    if fm.get("decision_counts")!={"review":expected*len(METHODS)} or fm.get("population_automatic_ready") is not False:
        raise GateError("Unvalidated population automatic decisions present.")
    return fm

def infer_all(root=ROOT):
    predictor=Predictor(root)
    if not predictor.lock:raise GateError("Full inference requires a locked model and final test evaluation.")
    final=root/"results/tables/final_evaluation.json"
    if not final.exists():raise GateError("Run final evaluation before full inference.")
    from .evaluation import validate_evaluation
    validate_evaluation(root)
    path=root/"results/predictions/pair_scores.csv.gz"
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".partial")
    count=0;method_rows=0;decision_counts={};candidate_counts={m:0 for m in METHODS}
    long_ids=set(predictor.df.loc[predictor.df.text.map(word_count).ge(5),"review_id"])
    from .performance import Monitor
    with Monitor() as monitor:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0,filename="") as compressed:
                with io.TextIOWrapper(compressed,encoding="utf-8",newline="") as handle:
                    writer=None
                    for k,(_,g) in enumerate(predictor.df.groupby("asin",sort=True),1):
                        result=predictor.score_product(g)
                        expected=len(g)*(len(g)-1)//2
                        if len(result)!=expected*len(METHODS):
                            raise ValueError("Product inference pair/method count mismatch.")
                        if writer is None:
                            writer=csv.writer(handle);writer.writerow(result.columns)
                        writer.writerows(result.fillna("").itertuples(index=False,name=None))
                        count+=expected;method_rows+=len(result)
                        eligible=result.review_id_a.isin(long_ids)&result.review_id_b.isin(long_ids)
                        for m in METHODS:
                            threshold=predictor.lock["methods"][m]["review_threshold"]
                            candidate_counts[m]+=int((result.method.eq(m)&eligible&result.score.ge(threshold)).sum())
                        for key,value in result.decision.value_counts().items():
                            decision_counts[key]=decision_counts.get(key,0)+int(value)
                        if k%100==0:print(f"Inference products={k}, pairs={count}",flush=True)
    expected_total=sum(len(g)*(len(g)-1)//2 for _,g in predictor.df.groupby("asin"))
    if count!=expected_total:raise ValueError("Full inference count mismatch.")
    temporary.replace(path)
    write_json(path.parent/"manifest.json",{"dataset_version":predictor.manifest["dataset_version"],"pairs":count,
               "method_rows":method_rows,"products":int(predictor.df.asin.nunique()),
               "methods":METHODS,"full_within_product_pairs":True,"no_transitive_clustering":True,
               "output_sha256":sha_file(path),"model_lock_sha256":sha_file(root/"models/model_lock.json"),
               "final_evaluation_sha256":sha_file(final),"performance":monitor.result,
               "decision_counts":decision_counts,"candidate_positive_by_method":candidate_counts,
               "candidate_count_note":"Threshold signals, not confirmed redundant pairs or natural prevalence",
               "score_meanings":predictor.lock["score_meanings"],
               "population_automatic_ready":False,"compression":"gzip with deterministic mtime=0"})
    return {"pairs":count,"method_rows":method_rows,"output":str(path)}
