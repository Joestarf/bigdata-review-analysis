from __future__ import annotations
from pathlib import Path
import pandas as pd
from .common import ROOT, LABELS, load_study, read_json, write_json, sha_file

class GateError(ValueError):
    pass

def validate_task_manifest(root=ROOT):
    _, manifest = load_study(root)
    meta = read_json(root / "data/annotation/pair_manifest.json")
    if meta["dataset_version"] != manifest["dataset_version"]:
        raise GateError("Annotation tasks do not match the frozen dataset.")
    for path, key in [("pair_metadata.csv", "pair_table_sha256"), ("annotation_texts.csv", "text_table_sha256")]:
        if sha_file(root / "data/annotation" / path) != meta[key]:
            raise GateError(f"Annotation input changed: {path}")
    return meta

def labels_path(root, split):
    if split not in ["train", "validation", "test"]:
        raise ValueError("Unknown split.")
    return root / "data/annotation" / f"labels_{split}.csv"

def read_labels(root, split, require_complete=True):
    # Only the final evaluator may parse sealed test labels after model lock.
    if split == "test":
        from .training import validate_lock
        validate_lock(root)
        require_test_seal(root)
    meta = validate_task_manifest(root)
    labels = pd.read_csv(labels_path(root, split), keep_default_na=False, dtype=str)
    tasks = pd.read_csv(root / "data/annotation/pair_metadata.csv", keep_default_na=False)
    expected = tasks[tasks.split == split]
    if not labels.pair_id.is_unique or set(labels.pair_id) != set(expected.pair_id):
        raise GateError(f"{split} label IDs do not match tasks.")
    invalid = set(labels.label) - set(LABELS) - {""}
    if invalid:
        raise GateError(f"Unknown labels: {invalid}")
    if require_complete and (labels.label.eq("").any() or labels.reason.str.strip().eq("").any()):
        done = int((labels.label.ne("") & labels.reason.str.strip().ne("")).sum())
        raise GateError(f"{split}: {done}/{len(labels)} AI-assisted judgments complete. Finish local annotation first.")
    if labels.reason.str.len().gt(1000).any():raise GateError("Annotation reason exceeds 1000 characters.")
    labeled = expected.merge(labels, on="pair_id", validate="one_to_one")
    if labeled.label.eq("exact").any():
        from .common import normalize_text
        bodies = pd.read_csv(root / "data/annotation/annotation_texts.csv", keep_default_na=False).set_index("pair_id")
        for pid in labeled.loc[labeled.label.eq("exact"), "pair_id"]:
            row = bodies.loc[pid]
            if normalize_text(row.text_a) != normalize_text(row.text_b):
                raise GateError(f"Exact label has unequal normalized bodies: {pid}")
    labeled["target"] = labeled.label.map(LABELS)
    if require_complete and labeled.annotation_version.ne(meta["dataset_version"]).any():
        raise GateError(f"{split} annotation version mismatch.")
    return labeled

def require_frozen_guidelines(root=ROOT):
    path = root / "data/annotation/guideline_freeze.json"
    if not path.exists():
        raise GateError("Complete the 60-pair pilot and freeze the annotation guidelines in the local interface.")
    freeze = read_json(path)
    meta = validate_task_manifest(root)
    if freeze["dataset_version"] != meta["dataset_version"] or freeze["guide_sha256"] != sha_file(root / "report/annotation_guide.md") or freeze["pair_table_sha256"] != meta["pair_table_sha256"]:
        raise GateError("Annotation guideline or data version changed after freeze. Reconcile pilot judgments before refreezing.")
    return freeze

def require_training_labels(root=ROOT):
    require_frozen_guidelines(root)
    scale=root/"data/annotation/scale6000/manifest.json"
    if scale.exists() and not read_json(scale).get("activated_pair_sha256"):
        raise GateError("Finish and activate the fixed6000 benchmark before training; do not lock the intermediate1200 batch.")
    train = read_labels(root, "train")
    val = read_labels(root, "validation")
    for name, labels, pos, neg in [("train", train, 20, 50), ("validation", val, 20, 20)]:
        p = int(labels.target.eq(1).sum())
        n = int(labels.target.eq(0).sum())
        if p < pos or n < neg:
            raise GateError(f"{name} insufficient confirmed labels: positive={p}/{pos}, negative={n}/{neg}. No guessed labels will be added.")
    return train, val

def require_test_seal(root=ROOT):
    """Check an independent completion receipt and hashes; do not parse test labels."""
    meta = validate_task_manifest(root)
    path = root / "data/annotation/test_seal.json"
    if not path.exists(): raise GateError("Independent test annotation has not been sealed.")
    seal = read_json(path)
    expected = int(meta["quotas"]["test"])
    if (seal.get("dataset_version") != meta["dataset_version"]
        or seal.get("pair_metadata_sha256") != meta["pair_table_sha256"]
        or seal.get("rows") != expected or seal.get("completed") != expected
        or seal.get("sealed") is not True
        or seal.get("labels_sha256") != sha_file(labels_path(root, "test"))):
        raise GateError("Test completion seal or label checksum mismatch.")
    return seal

def status(root=ROOT):
    out = {"stage": "awaiting_preparation", "human_required": False, "additional_label_review_required": True}
    path = root / "data/annotation/pair_manifest.json"
    if not path.exists():
        return out
    meta = validate_task_manifest(root)
    out["stage"] = "awaiting_annotation_acceptance"
    counts = {}
    for split in ["train", "validation"]:
        labels = pd.read_csv(labels_path(root, split), keep_default_na=False, dtype=str)
        counts[split] = {"completed": int((labels.label.ne("") & labels.reason.str.strip().ne("")).sum()), "total": len(labels)}
    try:
        seal = require_test_seal(root)
        counts["test"] = {"completed": seal["completed"], "total": seal["rows"], "sealed": True}
    except GateError:
        counts["test"] = {"completed": None, "total": meta["quotas"]["test"], "sealed": False}
        previous=root/"data/annotation/original1200/test_seal.json"
        if previous.exists():
            receipt=read_json(previous)
            if receipt.get("labels_sha256")==sha_file(previous.parent/"labels_test.csv"):
                counts["test"]["previous_sealed_rows"]=receipt["completed"]
    out.update({"annotation": counts, "pilot": 60, "guidelines_frozen": (root / "data/annotation/guideline_freeze.json").exists(),
                "models_locked": (root / "models/model_lock.json").exists(),
                "final_evaluated": (root / "results/tables/final_evaluation.json").exists()})
    if all(c["completed"] == c["total"] for c in counts.values()):
        out["human_required"] = False
        out["additional_label_review_required"] = True
        try:
            require_training_labels(root)
            out["stage"] = "awaiting_synthetic_review"
            out["human_required"] = False
        except GateError as e:
            out["stage"] = "insufficient_confirmed_labels"
            out["gate_reason"] = str(e)
    if out["models_locked"]:
        out["stage"] = "awaiting_final_evaluation"
    if out["final_evaluated"]:
        out["stage"] = "final_evaluation_sealed"
    scale=root/"data/annotation/scale6000/manifest.json"
    if scale.exists():
        planned=read_json(scale)
        progress={}
        for split in ["train","validation"]:
            sheet=pd.read_csv(scale.parent/f"labels_{split}.csv",keep_default_na=False,dtype=str)
            progress[split]={"additional_completed":int((sheet.label.ne("")&sheet.reason.str.strip().ne("")).sum()),
                             "additional_total":len(sheet)}
        # Independent test manager owns supplemental completion receipt.
        progress["test"]={"status":"independent_blind_review; values_not_read"}
        out["scale6000"]={"target":planned["target"],"activated":bool(planned.get("activated_pair_sha256")),"progress":progress}
        if not planned.get("activated_pair_sha256"):out["stage"]="awaiting_fixed_scale6000_annotation"
    out["label_source"] = "AI_assisted_review"
    out["human_gold_standard"] = False
    write_json(root / "results/tables/study_status.json", out)
    return out
