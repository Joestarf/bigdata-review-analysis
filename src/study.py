"""Run from repository root: .venv/bin/python -m src.study --help."""
import argparse
import json
from pathlib import Path
from .common import ROOT
from .annotations import GateError

def main():
    parser=argparse.ArgumentParser(description="Review redundancy study, with sealed labels and explicit acceptance gates")
    parser.add_argument("--root",type=Path,default=ROOT)
    subs=parser.add_subparsers(dest="command",required=True)
    for name in ["audit","embed","pairs","prepare","synthesize","train","evaluate","handoff","infer","benchmark","validate","status","explore-labels","extend","scale","activate-scale","sync-scale"]:
        subs.add_parser(name)
    serve=subs.add_parser("serve");serve.add_argument("--port",type=int,default=8765)
    args=parser.parse_args();root=args.root.resolve()
    try:
        match args.command:
            case "audit":
                from .audit import run_audit
                run_audit(root)
            case "embed":
                from .embedding import run_embeddings
                run_embeddings(root)
            case "pairs":
                from .pairs import prepare_pairs
                prepare_pairs(root)
            case "prepare":
                from .audit import run_audit
                from .embedding import run_embeddings
                from .pairs import prepare_pairs
                run_audit(root);run_embeddings(root);prepare_pairs(root)
            case "serve":
                from .annotation_server import serve
                serve(root,args.port)
            case "scale":
                from .scale import prepare_scale
                prepare_scale(root)
            case "activate-scale":
                from .scale import activate_scale
                activate_scale(root)
            case "sync-scale":
                from .scale import sync_scale_labels
                sync_scale_labels(root)
            case "extend":
                from .extension import extend_pairs
                extend_pairs(root)
            case "synthesize":
                from .synthetic import prepare_synthetic
                prepare_synthetic(root)
            case "train":
                from .training import train
                train(root)
            case "evaluate":
                from .evaluation import evaluate
                evaluate(root)
            case "handoff":
                from .inference import handoff
                print(json.dumps(handoff(root),ensure_ascii=False,indent=2))
            case "infer":
                from .inference import infer_all
                print(json.dumps(infer_all(root),ensure_ascii=False,indent=2))
            case "benchmark":
                from .benchmark import benchmark
                benchmark(root)
            case "validate":
                from .validation import validate
                validate(root)
            case "status":
                from .annotations import status
                print(json.dumps(status(root),ensure_ascii=False,indent=2))
            case "explore-labels":
                from .exploratory import explore_labels
                explore_labels(root)
    except (GateError,ValueError,FileNotFoundError) as e:
        parser.exit(2,f"Stage cannot proceed: {e}\n")

if __name__=="__main__":main()
