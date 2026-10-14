"""Allowlisted multilingual inference handoff; no review bodies or vector cache."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from .common import ROOT, sha_file
from .multilingual_inference import MultilingualPredictor

SOURCES=["src/__init__.py","src/common.py","src/multilingual_features.py",
         "src/multilingual_embedding.py","src/multilingual_inference.py","src/multilingual_training.py","src/redundancy_degree.py"]
MODELS=["models/multilingual_v2/"+name for name in ("lexical.joblib","C.joblib","E.joblib","manifest.json")]
EXTRAS=["requirements-inference.txt","report/multilingual_adaptation.md","report/multilingual_degree_policy.md",
        "results/figures/multilingual_input_coverage.png","results/figures/multilingual_reviewed_comparison.png","results/figures/uncertain_degree_assessment.png"]
README="""# 多语言重复内容识别：研究交接包

本包可独立读取中英文及混杂正文，输出A—E分数、暂定是否重复、程度和复核标记。所有决策都是review，
尚不支持自动降权；余弦或逻辑回归分数不是校准后的真实重复概率。
不含原评论、用户、标注正文、向量缓存或评分更新。原英文评分交接包另行保留。

安装 requirements-inference.txt 后运行：
python -m src.multilingual_inference --input your_pairs.csv --output pair_scores.csv
python -m src.redundancy_degree --input your_pairs.csv --output pair_degrees.csv
输入至少两列 text_a,text_b，可带 pair_id,asin,review_id_a,review_id_b。
首次使用按固定提交下载多语言预训练编码器；此下载不在本压缩包中。

完整说明见 report/multilingual_adaptation.md。训练需完整研究仓库和本地冻结数据；
本公开包的训练脚本仅随运行版本保存，不包含重新训练所需数据。
只加载可信来源的joblib文件。模型清单校验文件SHA和配套运行代码版本。
"""

def verify(path):
    expected=set(SOURCES+MODELS+EXTRAS+["README.md","bundle_manifest.json"])
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist())!=expected or len(archive.namelist())!=len(expected):
            raise ValueError("Unexpected or duplicate multilingual bundle members")
        manifest=json.loads(archive.read("bundle_manifest.json"))
        for name in sorted(expected-{"bundle_manifest.json"}):
            if hashlib.sha256(archive.read(name)).hexdigest()!=manifest["files_sha256"].get(name):
                raise ValueError(f"Bundle checksum mismatch: {name}")
        model=json.loads(archive.read("models/multilingual_v2/manifest.json"))
        for name,checksum in model["files"].items():
            if manifest["files_sha256"]["models/multilingual_v2/"+name]!=checksum:
                raise ValueError("Model manifest checksum mismatch")
        for name,checksum in model["code_hashes"].items():
            if manifest["files_sha256"].get(name)!=checksum:
                raise ValueError("Runtime code checksum mismatch")
        if model["actual_decision_policy"]!="review_only":
            raise ValueError("Unvalidated automatic weighting policy in multilingual bundle")
        # Exact allowlist excludes all review, label, raw prediction and cache paths.
    return {"files":len(expected),"sha256":sha_file(Path(path)),"bytes":Path(path).stat().st_size}

def build(root=ROOT):
    MultilingualPredictor(root/"models/multilingual_v2")
    payload={name:(root/name).read_bytes() for name in SOURCES+MODELS+EXTRAS}
    payload["README.md"]=README.encode()
    payload["bundle_manifest.json"]=(json.dumps({
        "schema":"multilingual-research-handoff-v2.1",
        "files_sha256":{name:hashlib.sha256(body).hexdigest() for name,body in payload.items()},
        "original_review_data_included":False,"annotation_data_included":False,
        "vector_cache_included":False,"automatic_weighting_supported":False,
        "encoder_weights_included":False},ensure_ascii=False,indent=2)+"\n").encode()
    target=root/"deliverables/multilingual_model.zip";target.parent.mkdir(exist_ok=True)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,body in sorted(payload.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,body)
    from .common import atomic_bytes
    atomic_bytes(target,stream.getvalue())
    print(json.dumps(verify(target),ensure_ascii=False))
    return target

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",type=Path)
    args=parser.parse_args()
    if args.verify:print(json.dumps(verify(args.verify),ensure_ascii=False))
    else:build()
