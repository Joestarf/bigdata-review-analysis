"""Build allowlisted public model/rating package and a local full research archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
from importlib.metadata import version
import pandas as pd
from .common import ROOT, read_json, sha_file, write_json
from .training import validate_lock
from .evaluation import validate_evaluation

PUBLIC_DATA = ["review_weights.csv","product_ratings.csv","product_final_ratings.csv","candidate_pair_signals.csv",
               "weighting_policy.json","rating_adjustment_manifest.json","test_comparison.csv"]
PUBLIC_DOCS = ["portable_model.md","rating_adjustment.md","experiment_results.md"]
DEPENDENCIES = ["numpy","pandas","joblib","scikit-learn","torch","sentence-transformers","psutil","matplotlib"]
FORBIDDEN_COLUMNS = {"user_id","text","text_a","text_b","timestamp","label_reason","reason"}

def blob_sha(data):
    return hashlib.sha256(data).hexdigest()

def zip_write(path,blobs):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".partial")
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(blobs.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644 << 16
            archive.writestr(info,data)
    temporary.replace(path)

def verify_public(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError("Corrupt ZIP member.")
        names=archive.namelist()
        manifest=json.loads(archive.read("bundle_manifest.json"))
        if len(names)!=len(set(names)) or set(names)!=(set(manifest["file_sha256"])|{"bundle_manifest.json"}):
            raise ValueError("Unexpected or duplicate archive member.")
        for name,digest in manifest["file_sha256"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:raise ValueError("Unsafe archive path.")
            data=archive.read(name)
            if blob_sha(data)!=digest:raise ValueError(f"Bundle checksum mismatch: {name}")
            allowed=(name.startswith("src/") and name.endswith(".py")) or name in {
                "models/lexical.joblib","models/C.joblib","models/E.joblib","models/portable_manifest.json",
                "requirements-inference.txt","README.md","data/SOURCE.md"} or name in {
                "data/"+n for n in PUBLIC_DATA} or name in {"report/"+n for n in PUBLIC_DOCS}
            if not allowed:raise ValueError(f"Public member outside allowlist: {name}")
            if name.startswith("data/") and name.endswith(".csv"):
                if FORBIDDEN_COLUMNS & set(pd.read_csv(io.BytesIO(data),nrows=0).columns):
                    raise ValueError(f"Raw/user data columns in public archive: {name}")
        reviews=pd.read_csv(io.BytesIO(archive.read("data/review_weights.csv")),keep_default_na=False)
        products=pd.read_csv(io.BytesIO(archive.read("data/product_ratings.csv")),keep_default_na=False)
        if len(reviews)!=90750 or reviews.asin.nunique()!=1000 or not reviews.review_id.is_unique:
            raise ValueError("Public review-weight count/ID mismatch.")
        if len(products)!=6000 or products.duplicated(["asin","scenario"]).any():
            raise ValueError("Public product/scenario count mismatch.")
        return {"members":len(names),"review_weight_rows":len(reviews),"product_scenario_rows":len(products),
                "bytes":Path(path).stat().st_size,"sha256":sha_file(Path(path))}

def build(root=ROOT):
    lock=validate_lock(root);validate_evaluation(root)
    rating=read_json(root/"results/tables/rating_adjustment_manifest.json")
    if rating["dataset_version"]!=lock["dataset_version"] or rating["model_lock_sha256"]!=sha_file(root/"models/model_lock.json"):
        raise ValueError("Rating/model provenance mismatch.")
    if rating["source_sha256"]!=sha_file(root/"src/rating_adjustment.py"):
        raise ValueError("Weighting implementation changed; regenerate before packaging.")
    for name,digest in rating["file_sha256"].items():
        if sha_file(root/"results/tables"/name)!=digest:raise ValueError(f"Rating artifact changed: {name}")
    dependencies={name:version(name) for name in DEPENDENCIES}
    requirements="\n".join(f"{name}=={value}" for name,value in dependencies.items())+"\n"
    (root/"requirements-inference.txt").write_text(requirements,encoding="utf-8")
    portable={"schema_version":1,"dataset_version":lock["dataset_version"],
        "model_revision":lock["embedding_model_revision"],"model_lock_sha256":sha_file(root/"models/model_lock.json"),
        "feature_order":["char_cosine","word_jaccard","length_ratio","length_difference","exact_match","semantic_cosine"],
        "methods":lock["methods"],"files_sha256":{f"models/{n}":sha_file(root/"models"/n)
            for n in ["lexical.joblib","C.joblib","E.joblib"]},"packages":dependencies,"automatic_production_ready":False}
    write_json(root/"models/portable_manifest.json",portable)
    blobs={}
    for path in sorted((root/"src").glob("*.py")):blobs[str(path.relative_to(root))]=path.read_bytes()
    for name in ["lexical.joblib","C.joblib","E.joblib","portable_manifest.json"]:
        blobs["models/"+name]=(root/"models"/name).read_bytes()
    for name in PUBLIC_DATA:blobs["data/"+name]=(root/"results/tables"/name).read_bytes()
    for name in PUBLIC_DOCS:blobs["report/"+name]=(root/"report"/name).read_bytes()
    blobs["requirements-inference.txt"]=requirements.encode()
    blobs["data/SOURCE.md"]=b"""# Data provenance
Source: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
Category: Cell_Phones_and_Accessories; fixed derived sample 90,750 reviews / 1,000 ASINs.
Public contents: project review IDs, ratings, experimental weights, product aggregates and numerical candidate scores.
No full review-body columns, original user ID columns or timestamps. Review IDs are pseudonymous lookup keys, not guaranteed irreversible anonymization.
Original download revision was not recorded. Existing SHA fixes this sample; re-downloading creates a new version.
The upstream dataset card checked for this handoff did not state an explicit license. No general redistribution permission for original text is assumed; original data stays in the separate local research archive.
"""
    blobs["README.md"]=b"""# Team model and rating handoff

## Prepared rating data
Unzip to a new directory. data/product_final_ratings.csv contains default experimental weighted star ratings for all 1,000 sampled ASINs. data/product_ratings.csv contains six strategies.
Default: light 0.75, moderate 0.50, same wording 0.25, extra five-field record copy 0; short/semantic-only signals retain weight 1.
Formula: sum(weight * rating) / sum(weight). Original stars are not altered.
Recompute from all 90,750 review-level rows without original Amazon data:
    python -m src.rating_adjustment --weights data/review_weights.csv --output recomputed_product_ratings.csv

## Frozen model use
Install requirements-inference.txt in a compatible environment. Original runtime: Arch WSL / Python 3.14. Exact sklearn/joblib versions preserve artifact compatibility. GPU is optional.
The fixed MiniLM revision downloads on first semantic inference unless already cached.
    from src.portable import PortablePredictor
    p = PortablePredictor(".")
    result = p.predict_pairs([("The battery lasts all day.", "A full charge lasts me a whole day.")])

CSV scoring from input columns asin,review_id,text:
    python -m src.portable --bundle . --input new_reviews.csv --output new_pairs.csv
Outputs five method rows per pair; all decisions stay review. Only load trusted joblib files.
models/portable_manifest.json contains the artifact hashes.

## Scope and delivery
These are experimental information-redundancy policies, not authenticity labels or official product stars.
No method met the validation target; reference labels lack independent human verification.
No transitive near-duplicate grouping. Rankings cover only the sampled 1,000 ASINs.
The reports explain policy, actual performance and sampling bias.
bundle_manifest.json checks every other archive member; the separate .sha256 checks this ZIP.
Obtain the separate LOCAL team_research_private.zip from the project owner for original review text, fixed labels and cached vectors. Keep the sealed test as the original final evaluation, not a tuning set.
"""
    manifest={"bundle_version":"team-handoff-v1","dataset_version":lock["dataset_version"],
        "model_lock_sha256":portable["model_lock_sha256"],"policy_version":rating["policy_version"],
        "contains_full_review_body_columns":False,"contains_original_user_ids":False,"review_rows":90750,"products":1000,
        "scenarios":6,"file_sha256":{name:blob_sha(data) for name,data in sorted(blobs.items())}}
    blobs["bundle_manifest.json"]=json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode()
    public=root/"deliverables/team_handoff.zip";zip_write(public,blobs)
    check=verify_public(public)
    (root/"deliverables/team_handoff.sha256").write_text(f"{check['sha256']}  team_handoff.zip\n",encoding="utf-8")
    private=dict(blobs)
    paths=set(lock["protected_files"]) | {
        "requirements.txt","data/processed/dataset_manifest.json","data/annotation/pair_manifest.json","data/annotation/labels_test.csv",
        "models/review_embeddings.npy","models/embedding_index.csv","models/embedding_manifest.json",
        "models/lexical_manifest.json","models/model_lock.json","results/tables/final_evaluation.json",
        "results/tables/test_pair_predictions.csv","results/tables/test_comparison.csv"}
    for relative in sorted(paths):
        path=root/relative
        if not path.is_file():raise ValueError(f"Missing full research input: {relative}")
        private[relative]=path.read_bytes()
    private["PRIVATE_README.md"]=b"""# Full local research handoff
Includes original five-field review bodies and user IDs, stable indices/splits, fixed pair inputs/labels, frozen classifiers/TF-IDF and cached semantic vectors.
Shared public README and portable inference still apply. Original-core Predictor can reuse fixed data/cache only when the original dependency/platform lock matches. requirements.txt preserves the original full environment; on other machines use PortablePredictor.
Do not retune or rerun sealed test to change this experiment's conclusions.
This archive stays local and is excluded from the public repository. The 570 MiB all-pair output is separately in results/predictions/pair_scores.csv.gz and not duplicated here.
Upstream provenance/license uncertainty is documented in data/SOURCE.md; review appropriate source/use terms before further sharing original material.
"""
    private["private_bundle_manifest.json"]=json.dumps({"private_research_archive":True,"dataset_version":lock["dataset_version"],
        "contains_original_review_text":True,"contains_original_user_ids":True,
        "file_sha256":{name:blob_sha(data) for name,data in sorted(private.items())}},ensure_ascii=False,indent=2).encode()
    local=root/"results/handoff/team_research_private.zip";zip_write(local,private)
    with zipfile.ZipFile(local) as archive:
        if archive.testzip() is not None:raise ValueError("Private ZIP CRC failed.")
        pm=json.loads(archive.read("private_bundle_manifest.json"))
        for name,digest in pm["file_sha256"].items():
            if blob_sha(archive.read(name))!=digest:raise ValueError("Private member checksum mismatch.")
    receipt={"public":check,"private":{"bytes":local.stat().st_size,"sha256":sha_file(local),"path":str(local)},
        "model_lock_sha256":portable["model_lock_sha256"],"rating_manifest_sha256":sha_file(root/"results/tables/rating_adjustment_manifest.json")}
    write_json(root/"results/handoff/team_package_receipt.json",receipt)
    (root/"deliverables/README.md").write_text(f"""# 组内交接下载

- [team_handoff.zip](team_handoff.zip)：{check['bytes']/1024/1024:.2f}MiB。含可移植模型、90,750条评论编号/原星级/六策略权重、1,000商品最终星级及候选分数；不含原评论正文字段、原用户标识或时间戳。评论编号可回连固定样本，不能视为不可逆匿名化。
- [校验值](team_handoff.sha256)：ZIP的SHA-256；包内manifest核对逐文件SHA。
- 完整研究包保存在WSL本地results/handoff/team_research_private.zip（{local.stat().st_size/1024/1024:.2f}MiB），含原五字段数据、固定标签及向量缓存，可供组内另行交接；未上传公开仓库。
- 570MiB全量对结果已有本地保存，不重复打包。评分模块只需要包内review_weights.csv。

默认折扣：轻0.75、中0.50、相同措辞0.25、重复记录副本0、短评/仅语义信号1。最终星级为Σw×rating/Σw。属于样本敏感性实验；候选仍需复核，不能判断水军或称为官方商品评分。

模型接口见[portable_model](../report/portable_model.md)，策略及实测结果见[rating_adjustment](../report/rating_adjustment.md)。在新目录解压，按包内README执行；不依赖原工作目录。

发布范围由src/package_handoff.py校验：模型、代码、研究说明及不含正文/用户/时间的衍生评分数据。原data/、models/和其他CSV/JSON仍保持忽略，不强制加入原始文件。
""",encoding="utf-8")
    print(receipt)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--verify",type=Path)
    args=p.parse_args()
    print(verify_public(args.verify)) if args.verify else build(args.root)

if __name__=="__main__":main()
