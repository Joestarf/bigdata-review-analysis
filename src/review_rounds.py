"""Apply three complementary evidence passes to 6600 labeled research candidates."""
from __future__ import annotations
import argparse
import json
import pandas as pd
from .common import ROOT, write_csv, write_json, sha_file
from .annotations import read_labels
from .training import validate_lock
from .redundancy_degree import DegreePredictor, classify_pair, POLICY_VERSION, POLICY_DIGEST, POLICY_CODE_SHA256
from .multilingual_embedding import digest

def run(root=ROOT,reuse_frozen_scores=False):
    validate_lock(root)
    directory=root/"data/multilingual_v2";out=root/"results/multilingual_v2"
    predictor=DegreePredictor()
    old_bodies=pd.read_csv(root/"data/annotation/annotation_texts.csv",keep_default_na=False)
    inputs=[]
    for split in ("train","validation","test"):
        labels=read_labels(root,split).merge(old_bodies,on="pair_id",validate="one_to_one")
        labels["corpus"]="original_english";inputs.append(labels)
        tasks=pd.read_csv(directory/f"tasks_{split}.csv",keep_default_na=False)
        label_path=directory/("labels_test_postlock_reviewed.csv" if split=="test" else f"labels_{split}.csv")
        labels=pd.read_csv(label_path,keep_default_na=False)
        meta=pd.read_csv(directory/"pair_metadata.csv",keep_default_na=False)
        chinese=meta[meta.split.eq(split)].merge(tasks,on="pair_id",validate="one_to_one").merge(labels,on="pair_id",validate="one_to_one")
        chinese["corpus"]="chinese_auxiliary";inputs.append(chinese)
    table=pd.concat(inputs,ignore_index=True)
    if not table.pair_id.is_unique or len(table)!=6600:raise ValueError("Unexpected research candidate IDs/count")
    outputs=[]
    cached=None
    old_receipt_path=out/"degree_receipt.json"
    cache_path=out/"degree_assessments.csv"
    fingerprint=digest(table[["pair_id","text_a","text_b"]].to_dict("records"))
    if reuse_frozen_scores:
        old_receipt=json.loads(old_receipt_path.read_text())
        if old_receipt["model_manifest_sha256"]!=sha_file(root/"models/multilingual_v2/manifest.json") or sha_file(cache_path)!=old_receipt["outputs_sha256"]["degree_assessments.csv"]:
            raise ValueError("Frozen score model/result mismatch")
        recorded_fingerprint=old_receipt.get("score_input_fingerprint")
        if recorded_fingerprint is not None and recorded_fingerprint!=fingerprint:
            raise ValueError("Frozen score input text/order changed")
        # For pre-fingerprint receipts, verify every frozen source task via the trained manifest.
        if recorded_fingerprint is None:
            model=predictor.model.manifest
            candidate_path=directory/"candidate_manifest.json"
            if sha_file(candidate_path)!=model["input_hashes"]["data/multilingual_v2/candidate_manifest.json"]:
                raise ValueError("Frozen Chinese candidate manifest changed")
            candidates=json.loads(candidate_path.read_text())
            if sha_file(directory/"pair_metadata.csv")!=candidates["metadata_sha256"]:
                raise ValueError("Frozen Chinese pair metadata changed")
            for split in ("train","validation","test"):
                if sha_file(directory/f"tasks_{split}.csv")!=candidates["task_sha256"][split]:
                    raise ValueError("Frozen Chinese task texts changed")
        cached=pd.read_csv(cache_path,keep_default_na=False).set_index("pair_id")
        if set(cached.index)!=set(table.pair_id):raise ValueError("Frozen score pair IDs mismatch")
    columns=["pair_id","asin","review_id_a","review_id_b","text_a","text_b"]
    for start in range(0,len(table),1000):
        records=table.iloc[start:start+1000][columns].to_dict("records")
        if cached is None:
            result=predictor.predict_pairs(records)
        else:
            proposals=[]
            for row in records:
                evidence=json.loads(cached.loc[row["pair_id"],"evidence_round_3"])
                proposals.append(classify_pair(row["text_a"],row["text_b"],evidence["method_scores"],evidence["thresholds"],ids=row))
            result=pd.DataFrame(proposals)
        outputs.append(result)
    result=pd.concat(outputs,ignore_index=True)
    context=table[["pair_id","split","corpus","label"]].rename(columns={"label":"original_reference_label"})
    result=result.merge(context,on="pair_id",validate="one_to_one")
    write_csv(out/"degree_assessments.csv",result)
    pending=result[result.original_reference_label.eq("uncertain")]
    write_csv(out/"uncertain_after_three_passes.csv",pending)
    summary=result.groupby(["corpus","original_reference_label","redundancy_degree"],dropna=False).size().reset_index(name="pairs")
    write_csv(out/"degree_summary.csv",summary)
    # Three previously specified calibration cases are kept local; never fit the model on them.
    tasks=pd.read_csv(directory/"tasks_validation.csv",keep_default_na=False)
    calibration=tasks.iloc[[62,57,52]].copy()
    calibration["expected_degree"]=["none","light","high"]
    calibration["reference_origin"]="project_defined_degree_examples"
    write_csv(directory/"degree_calibration.csv",calibration)
    actual=result[result.pair_id.isin(calibration.pair_id)].copy()
    checked=calibration[["pair_id","expected_degree"]].merge(actual,on="pair_id",validate="one_to_one")
    if not checked.expected_degree.eq(checked.redundancy_degree).all():
        raise ValueError("Specified degree calibration cases failed")
    write_csv(out/"degree_calibration_results.csv",checked)
    write_json(out/"degree_receipt.json",{
        "pairs":len(result),"previous_uncertain_pairs":len(pending),
        "score_input_fingerprint":fingerprint,"reused_frozen_method_scores":bool(reuse_frozen_scores),
        "uncertain_degree_counts":pending.redundancy_degree.value_counts().to_dict(),
        "uncertain_low_evidence_pairs":int(pending.evidence_insufficient.sum()),
        "all_degree_counts":result.redundancy_degree.value_counts().to_dict(),
        "policy_version":POLICY_VERSION,"policy_digest":POLICY_DIGEST,"policy_code_sha256":POLICY_CODE_SHA256,
        "model_manifest_sha256":sha_file(root/"models/multilingual_v2/manifest.json"),
        "degree_calibration_sha256":sha_file(directory/"degree_calibration.csv"),
        "calibration_examples":3,"passed_calibration_examples":len(checked),
        "independent_degree_gold_available":False,
        "reference_relation_scope":"Classifier detects main-content duplication; light partial redundancy is an additional degree rule, not its trained target.",
        "old_test_scope":"Original sealed English test only used for exploratory output, never for retraining or threshold selection.",
        "three_passes":"Wording/general-opinion evidence; contextual specific-fact evidence; frozen method evidence. Not three independent annotators.",
        "automatic_rating_changes":False,
        "outputs_sha256":{p.name:sha_file(p) for p in [out/"degree_assessments.csv",out/"uncertain_after_three_passes.csv",out/"degree_summary.csv",out/"degree_calibration_results.csv"]}})
    print(json.dumps({"pairs":len(result),"previous_uncertain":len(pending),
        "degree_counts":pending.redundancy_degree.value_counts().to_dict(),
        "evidence_insufficient":int(pending.evidence_insufficient.sum()),"calibration_passed":len(checked)},ensure_ascii=False))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuse-frozen-scores",action="store_true")
    args=parser.parse_args()
    run(reuse_frozen_scores=args.reuse_frozen_scores)
