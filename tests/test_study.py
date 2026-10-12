"""Safety and leakage invariants; fixtures never alter the real sample or labels."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.common import COLUMNS,clean_frame,clean_text,sha_text,pair_id,write_csv,write_json,sha_file
from src.audit import make_index,split_records,validate_splits
from src.pairs import validate_pairs
from src.embedding import cache_key,load_embedding_cache
from src.features import structural_features,text_pair_features
from src.training import choose_threshold,train
from src.annotations import GateError,read_labels
from src.synthetic import transform_text,PROTECTED
from src.annotation_server import AnnotationStore

class StudyInvariantTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([["A","u1",5,"It does not support iPhone 14; battery 3000mAh.",1234567890000],
                             ["A","u2",4,"Completely different camera performance.",1234567890001]],columns=COLUMNS)
    def test_cleaner_preserves_negation_numbers_models_and_comparison(self):
        text="<p>Does NOT fit iPhone 14 &amp; 3000mAh; x < y.</p>"
        self.assertEqual(clean_text(text),"Does NOT fit iPhone 14 & 3000mAh; x < y.")
    def test_fractional_invalid_values_not_truncated(self):
        rows=[]
        for rating,stamp in [(2.5,100),(5,100.5),(float("inf"),100),(5,-1),(0,100),(5,253402300800000)]:
            rows.append(["A","u",rating,"Valid information here.",stamp])
        clean,excluded,summary=clean_frame(pd.DataFrame(rows,columns=COLUMNS))
        self.assertEqual(len(clean),0);self.assertEqual(summary["removed_rows"],6)
        self.assertEqual(len(excluded),6)
        self.assertEqual(summary["reason_counts_overlapping"]["invalid_rating"],3)
        self.assertEqual(summary["reason_counts_overlapping"]["invalid_timestamp"],3)
    def test_overlapping_exclusion_counts_and_long_range(self):
        raw=pd.DataFrame([["","",1,"",100],["A","u",5,"x"*10001,100]],columns=COLUMNS)
        clean,excluded,report=clean_frame(raw)
        self.assertEqual(len(excluded),2)
        self.assertEqual(excluded.iloc[0].reasons,"missing_asin|missing_user_id|empty_text")
        self.assertEqual(report["long_text_range"],{"min":10001,"max":10001})
    def test_placeholder_text_rejected_without_losing_valid_information(self):
        raw=self.frame()
        placeholders=pd.DataFrame([["A","u",5,t,100] for t in ["N/A","None","NA"]],columns=COLUMNS)
        clean,excluded,report=clean_frame(pd.concat([raw,placeholders],ignore_index=True))
        self.assertEqual(len(clean),2)
        self.assertEqual(report["reason_counts_overlapping"]["placeholder_text"],3)
        self.assertEqual(len(excluded),3)

    def test_distinct_row_ids_shared_source_hash_deterministic(self):
        raw=pd.concat([self.frame(),self.frame().iloc[:1]],ignore_index=True)
        idx=make_index(raw,"version")
        self.assertTrue(idx.review_id.is_unique)
        self.assertEqual(idx.source_id.iloc[0],idx.source_id.iloc[2])
        pd.testing.assert_frame_equal(idx,make_index(raw,"version"))
        self.assertNotEqual(idx.review_id.iloc[0],make_index(raw,"new_version").review_id.iloc[0])
    def test_template_exclusion_and_source_deduplication(self):
        raw=pd.DataFrame([[str(i),f"u{i}",5,"same public template",i] for i in range(20)],columns=COLUMNS)
        rich=pd.concat([make_index(raw,"v"),raw],axis=1)
        a=split_records(rich)
        self.assertTrue(a.split.eq("template_pool").all())
        self.assertFalse(a.supervised_eligible.any())
        validate_splits(rich,a)
    def pair_fixture(self):
        raw=self.frame()
        rich=pd.concat([make_index(raw,"v"),raw],axis=1)
        rich["split"]="train";rich["supervised_eligible"]=True
        a,b=rich.review_id.tolist()
        pairs=pd.DataFrame([{"pair_id":pair_id(a,b),"review_id_a":a,"review_id_b":b,"asin":"A","split":"train"}])
        return rich,pairs
    def test_pair_symmetry_self_and_cross_product_rejected(self):
        df,pairs=self.pair_fixture();validate_pairs(pairs,df)
        reverse=pairs.copy();reverse[["review_id_a","review_id_b"]]=reverse[["review_id_b","review_id_a"]].to_numpy()
        with self.assertRaises(ValueError):validate_pairs(pd.concat([pairs,reverse]),df)
        bad=pairs.copy();bad["review_id_b"]=bad.review_id_a
        with self.assertRaises(ValueError):validate_pairs(bad,df)
        df.loc[1,"asin"]="B"
        with self.assertRaises(ValueError):validate_pairs(pairs,df)
    def test_cross_split_and_same_source_rejected(self):
        df,pairs=self.pair_fixture();df.loc[1,"split"]="validation"
        with self.assertRaises(ValueError):validate_pairs(pairs,df)
        df.loc[1,"split"]="train";df.loc[1,"source_id"]=df.loc[0,"source_id"]
        with self.assertRaises(ValueError):validate_pairs(pairs,df)
    def test_threshold_requires_30_and_short_review_excluded(self):
        self.assertIsNone(choose_threshold(np.ones(29),np.ones(29),np.ones(29,dtype=bool)))
        self.assertIsNone(choose_threshold(np.ones(35),np.ones(35),np.zeros(35,dtype=bool)))
        chosen=choose_threshold(np.r_[np.ones(30),np.zeros(10)],np.r_[np.ones(30),np.zeros(10)],np.ones(40,dtype=bool))
        self.assertEqual(chosen["predicted_positive"],30)
        self.assertEqual(chosen["precision"],1.)
    def test_training_gate_stops_before_test_read_or_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.training.require_training_labels",side_effect=GateError("missing positive labels")) as gate, \
                 patch("src.training.approved_synthetic") as synthetic,patch("src.training.fit_lexical") as fitting:
                with self.assertRaises(GateError):train(Path(directory))
                gate.assert_called_once();synthetic.assert_not_called();fitting.assert_not_called()
    def test_embedding_key_model_data_text_config_changes(self):
        base=cache_key("v","text","revision")
        for params in [("v2","text","revision"),("v","other","revision"),("v","text","revision2")]:
            self.assertNotEqual(base,cache_key(*params))
        with patch("src.embedding.MAX_TOKENS",128):self.assertNotEqual(base,cache_key("v","text","revision"))
    def test_corrupt_embedding_array_and_index_rejected(self):
        df, _=self.pair_fixture()
        version="v";revision="r"
        import json
        thash=sha_text(json.dumps(df.text.tolist(),ensure_ascii=False,separators=(",",":")))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"models").mkdir()
            path=root/"models/review_embeddings.npy"
            np.save(path,np.zeros((2,384),dtype=np.float32))
            write_csv(root/"models/embedding_index.csv",pd.DataFrame({"review_id":df.review_id}))
            meta={"model_revision":revision,"cache_key":cache_key(version,thash,revision),"array_sha256":sha_file(path),"index_sha256":sha_file(root/"models/embedding_index.csv")}
            write_json(root/"models/embedding_manifest.json",meta)
            with patch("src.embedding.load_study",return_value=(df,{"dataset_version":version})):
                load_embedding_cache(root)
                np.save(path,np.ones((2,384),dtype=np.float32))
                with self.assertRaisesRegex(ValueError,"corrupted"):load_embedding_cache(root)
    def test_synthetic_typo_preserves_protected_numbers_and_models(self):
        text="I cannot recommend this iPhone 14 iphone galaxy A1234 3000mAh because it never worked properly."
        rng=np.random.default_rng(42)
        for _ in range(20):
            changed=transform_text(text,"typo",rng)
            self.assertIsNotNone(changed)
            for token in ["cannot","never","iPhone","iphone","galaxy","14","A1234","3000mAh"]:self.assertIn(token,changed)
    def test_annotation_interface_exposes_only_blind_fields(self):
        store=AnnotationStore.__new__(AnnotationStore)
        import threading
        store.lock=threading.RLock()
        rows=pd.DataFrame([{"pair_id":"p","text_a":"A","text_b":"B","label":"","reason":"",
                            "asin":"secret","rating":5,"score":.99}])
        store.stage_rows=lambda _:rows
        task=store.task("pilot",0)
        self.assertEqual(set(task),{"index","total","completed","pair_id","text_a","text_b","label","reason"})
    def test_feature_inputs_exclude_identity_rating_and_timestamp(self):
        from src.features import LEXICAL_FEATURES,SEMANTIC_FEATURES
        self.assertEqual(len(LEXICAL_FEATURES),5)
        self.assertEqual(len(SEMANTIC_FEATURES),6)
        self.assertFalse(set(SEMANTIC_FEATURES)&{"rating","asin","user_id","timestamp","label","source_id"})
        self.assertEqual(structural_features("battery lasts","battery does not last")[-1],0.)

    def test_real_classifier_fit_and_deterministic_validation_lock_fixture(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from src.training import score_methods
        from src.common import read_json
        # Actual numeric features and sklearn fitting, synthetic fixture only; never real study labels.
        texts=["Specific model A123 battery lasted ten hours with screen active.",
               "Specific model A123 battery lasted ten hours with screen active.",
               "Camera failed completely on Tuesday after charging overnight.",
               "Packaging arrived with broken cable and unusable port."]
        df=pd.DataFrame({"review_id":["a","b","c","d"],"text":texts})
        positive={"review_id_a":"a","review_id_b":"b","target":1}
        negative={"review_id_a":"c","review_id_b":"d","target":0}
        manual=pd.DataFrame([positive]*20+[negative]*50)
        validation=pd.DataFrame([positive]*30+[negative]*30)
        vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5)).fit(texts)
        embeddings=np.eye(4,384,dtype=np.float32);embeddings[1]=embeddings[0]
        syn=pd.DataFrame({"text_a":[texts[0]]*40,"text_b":[texts[0]]*40})
        fixture_vectors=np.ones((1,384),dtype=np.float32)/np.sqrt(384)
        locks=[]
        with patch("src.training.require_training_labels",return_value=(manual,validation)), \
             patch("src.training.require_test_seal",return_value={"sealed":True}), \
             patch("src.training.approved_synthetic",return_value=syn), \
             patch("src.training.load_study",return_value=(df,{"dataset_version":"fixture"})), \
             patch("src.training.fit_lexical",return_value=vectorizer), \
             patch("src.training.load_embedding_cache",return_value=(embeddings,None,{"model_revision":"fixture"})), \
             patch("src.training.load_encoder",return_value=None), \
             patch("src.training.encode_texts",return_value=(fixture_vectors,64,"cpu")), \
             patch("src.training.protected_paths",return_value={}):
            for _ in range(2):
                with tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);(root/"models").mkdir()
                    train(root)
                    lock=read_json(root/"models/model_lock.json")
                    self.assertFalse(lock["test_labels_accessed"])
                    self.assertTrue(lock["automatic_ready_validation"])
                    self.assertAlmostEqual(lock["synthetic_total_weight"],17.5)
                    self.assertEqual(lock["manual_total_weight"],70)
                    locks.append(lock)
        self.assertEqual(locks[0],locks[1])

    def test_product_api_preview_alignment_and_no_cross_product(self):
        from src.inference import Predictor
        from sklearn.feature_extraction.text import TfidfVectorizer
        p=Predictor.__new__(Predictor)
        p.df=self.frame().assign(review_id=["r1","r2"])
        p.positions={"r1":0,"r2":1}
        p.vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5)).fit(p.df.text)
        p.embeddings=np.eye(2,384,dtype=np.float32)
        p.embedding_index=pd.DataFrame({"truncated":[False,False]})
        p.lock=None;p.models={}
        result=p.score_product(p.df[["asin","review_id","text"]])
        self.assertEqual(len(result),5)
        self.assertTrue(result.decision.eq("review").all())
        self.assertTrue(result[result.method.isin(["C","E"])].score.isna().all())
        wrong=p.df.copy();wrong.loc[0,"text"]="Changed text"
        with self.assertRaisesRegex(ValueError,"mismatch"):p.score_product(wrong)
        wrong=p.df.copy();wrong.loc[0,"asin"]="B"
        with self.assertRaises(ValueError):p.score_product(wrong)
        self.assertEqual(len(p.predict_pairs([])),0)

    def test_uncertain_predictions_reduce_precision_lower_bound(self):
        y=np.r_[np.ones(30),np.full(10,np.nan),np.zeros(20)]
        score=np.r_[np.ones(40),np.zeros(20)]
        self.assertIsNone(choose_threshold(y,score,np.ones(60,dtype=bool)))
        from src.training import bounded_metrics
        m=bounded_metrics(y,score>=.5)
        self.assertEqual(m["unresolved_positive"],10)
        self.assertEqual(m["precision"],1.)
        self.assertEqual(m["precision_lower"],.75)
        self.assertEqual(m["precision_upper"],1.)

    def test_test_stage_unavailable_before_final_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            store=AnnotationStore.__new__(AnnotationStore);store.root=Path(directory)
            with patch("src.annotation_server.require_frozen_guidelines"):
                with self.assertRaises(GateError):store.check_stage("test")

    def test_status_uses_seal_not_test_label_reader(self):
        from src.annotations import status
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"data/annotation").mkdir(parents=True)
            (root/"data/annotation/pair_manifest.json").touch()
            seen=[]
            def reader(path,**kwargs):
                seen.append(str(path))
                self.assertNotIn("labels_test.csv",str(path))
                return pd.DataFrame({"label":["near"],"reason":["specific content"]})
            with patch("src.annotations.validate_task_manifest",return_value={"quotas":{"test":450}}), \
                 patch("src.annotations.require_test_seal",return_value={"rows":450,"completed":450}), \
                 patch("src.annotations.pd.read_csv",side_effect=reader), \
                 patch("src.annotations.require_training_labels",side_effect=GateError("fixture")):
                value=status(root)
            self.assertEqual(value["annotation"]["test"]["completed"],450)
            self.assertEqual(len(seen),2)

    def test_known_ids_validated_in_mixed_product_input(self):
        from src.inference import Predictor
        p=Predictor.__new__(Predictor)
        p.df=self.frame().assign(review_id=["r1","r2"])
        p.positions={"r1":0,"r2":1}
        bad=p.df.copy();bad.loc[0,"text"]="Forged body";bad.loc[1,"review_id"]="new"
        with self.assertRaisesRegex(ValueError,"mismatch"):p.score_product(bad)

    def test_text_pair_id_is_direction_independent(self):
        from src.inference import Predictor
        from sklearn.feature_extraction.text import TfidfVectorizer
        class Tokenizer:
            def __call__(self,texts,**kwargs):return {"length":[10]*len(texts)}
        class Encoder:
            tokenizer=Tokenizer()
        p=Predictor.__new__(Predictor);p.encoder=Encoder()
        p.vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5)).fit(["battery life is very good today","camera does not work any more"])
        p.lock=None
        texts=("battery life is very good today","camera does not work any more")
        vectors=np.eye(2,384,dtype=np.float32)
        with patch("src.inference.encode_texts",return_value=(vectors,64,"cpu")):
            out=p.predict_pairs([texts,texts[::-1]])
        self.assertEqual(out.pair_id.iloc[0],out.pair_id.iloc[5])

    def test_unactivated_scale_stops_intermediate_training(self):
        from src.annotations import require_training_labels
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_json(root/"data/annotation/scale6000/manifest.json",{"target":{"train":4000}})
            with patch("src.annotations.require_frozen_guidelines"),patch("src.annotations.read_labels") as reader:
                with self.assertRaisesRegex(GateError,"fixed6000"):require_training_labels(root)
                reader.assert_not_called()

    def test_test_label_reader_requires_model_lock_first(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.annotations.pd.read_csv") as reader:
                with self.assertRaises(GateError):read_labels(Path(directory),"test")
                reader.assert_not_called()

    def test_benchmark_qualified_model_remains_review_for_population(self):
        from src.inference import Predictor
        p=Predictor.__new__(Predictor)
        p.lock={"population_automatic_ready":False,"methods":{"A":{"qualified":True,"threshold":1.}}}
        decision,reason=p._decision("A",1.,"Battery lasts for ten hours.","Battery lasts for ten hours.")
        self.assertEqual(decision,"review")
        self.assertIn("benchmark",reason)

    def test_historical_cohorts_are_protected_when_present(self):
        from src.training import protected_paths
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            names=["data/annotation/original600/pair_metadata.csv","data/annotation/original1200/pair_metadata.csv"]
            for name in names:
                (root/name).parent.mkdir(parents=True,exist_ok=True);(root/name).touch()
            with patch("src.training.sha_file",return_value="fixture"):
                protected=protected_paths(root)
            for name in names:self.assertIn(name,protected)


    def scale_sync_fixture(self,root):
        annotation=root/"data/annotation";scale=annotation/"scale6000"
        write_json(scale/"manifest.json",{"activated_pair_sha256":"fixed","dataset_version":"v"})
        columns=["pair_id","label","reason","annotation_version"]
        write_csv(annotation/"labels_train.csv",pd.DataFrame([["p1","","",""],["p2","","",""]],columns=columns))
        write_csv(annotation/"labels_validation.csv",pd.DataFrame([["v1","","",""]],columns=columns))
        write_csv(scale/"labels_train.csv",pd.DataFrame([["p1","distinct","different facts","v"],["p2","","",""]],columns=columns))
        write_csv(scale/"labels_train_secondary.csv",pd.DataFrame([["p2","uncertain","generic text","v"]],columns=columns))
        write_csv(scale/"labels_validation.csv",pd.DataFrame([["v1","near","same details","v"]],columns=columns))
        return annotation,scale

    def test_scale_sync_merges_independent_shards_without_test_access(self):
        from src.scale import sync_scale_labels
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);annotation,scale=self.scale_sync_fixture(root)
            real_reader=pd.read_csv
            with patch("src.scale.pd.read_csv",wraps=real_reader) as reader:
                sync_scale_labels(root)
                self.assertFalse(any("test" in str(call.args[0]) for call in reader.call_args_list))
            result=real_reader(annotation/"labels_train.csv",keep_default_na=False)
            self.assertEqual(result.label.tolist(),["distinct","uncertain"])

    def test_scale_sync_conflict_rejected_before_writing(self):
        from src.scale import sync_scale_labels
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);annotation,scale=self.scale_sync_fixture(root)
            before=sha_file(annotation/"labels_train.csv")
            write_csv(scale/"labels_train_primary.csv",pd.DataFrame(
                [["p1","near","same details","v"]],columns=["pair_id","label","reason","annotation_version"]))
            with self.assertRaisesRegex(GateError,"Conflicting"):
                sync_scale_labels(root)
            self.assertEqual(before,sha_file(annotation/"labels_train.csv"))

    def test_scale_sync_rejects_stale_review_version(self):
        from src.scale import sync_scale_labels
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);annotation,scale=self.scale_sync_fixture(root)
            frame=pd.read_csv(scale/"labels_train_secondary.csv",keep_default_na=False)
            frame["annotation_version"]="stale"
            write_csv(scale/"labels_train_secondary.csv",frame)
            with self.assertRaisesRegex(GateError,"version invalid"):
                sync_scale_labels(root)


    def test_prepared_features_identical_to_original_definition(self):
        from src.features import word_set,prepare_structure,prepared_structural_features
        texts=["battery does not last 10 hours; A1234.","Battery lasts ten hours.","", "很好", "not GREAT!!!"]
        for a in texts:
            for b in texts:
                wa,wb=word_set(a),word_set(b);la,lb=len(a),len(b)
                expected=[len(wa&wb)/len(wa|wb) if wa|wb else 0.,
                          min(la,lb)/max(la,lb) if max(la,lb) else 1.,
                          abs(la-lb),float(a.casefold()==b.casefold())]
                self.assertEqual(prepared_structural_features(prepare_structure(a),prepare_structure(b)),expected)


    def test_reporting_refuses_unlocked_test_before_table_read(self):
        from src.reporting import build_report
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.reporting.pd.read_csv") as reader:
                with self.assertRaises(GateError):build_report(Path(directory))
                reader.assert_not_called()

    def test_full_inference_writes_deterministic_gzip_and_provenance(self):
        from src.inference import infer_all
        from types import SimpleNamespace
        import gzip
        from src.common import read_json
        data=pd.DataFrame({"asin":["A","A"],"review_id":["r1","r2"],
                           "text":["A battery lasts ten hours daily.","This phone does not charge at all."]})
        output=pd.DataFrame([{"pair_id":"p","asin":"A","review_id_a":"r1","review_id_b":"r2",
                              "method":m,"score":.6,"decision":"review","reason":"fixture"} for m in ["A","B","C","D","E"]])
        lock={"methods":{m:{"review_threshold":.5} for m in ["A","B","C","D","E"]},"score_meanings":{}}
        predictor=SimpleNamespace(lock=lock,df=data,manifest={"dataset_version":"fixture"},score_product=lambda _:output)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_json(root/"models/model_lock.json",{"fixture":True})
            write_json(root/"results/tables/final_evaluation.json",{"fixture":True})
            with patch("src.inference.Predictor",return_value=predictor),patch("src.evaluation.validate_evaluation"):
                infer_all(root)
            path=root/"results/predictions/pair_scores.csv.gz"
            self.assertEqual(path.read_bytes()[4:8],bytes(4))
            with gzip.open(path,"rt") as file:self.assertEqual(len(pd.read_csv(file)),5)
            receipt=read_json(path.parent/"manifest.json")
            self.assertEqual(receipt["pairs"],1);self.assertEqual(receipt["method_rows"],5)
            self.assertEqual(receipt["output_sha256"],sha_file(path))
            self.assertEqual(receipt["candidate_positive_by_method"],{m:1 for m in ["A","B","C","D","E"]})
            self.assertEqual(receipt["decision_counts"],{"review":5})


    def test_bootstrap_precision_undefined_with_no_predictions(self):
        from src.evaluation import product_bootstrap
        frame=pd.DataFrame({"asin":["A","A","B","B"],"target":[0,1,0,1],"predicted":[False]*4})
        result=product_bootstrap(frame,200)
        self.assertEqual(result["status"],"ok")
        self.assertIsNone(result["precision_low"]);self.assertEqual(result["precision_valid_resamples"],0)
        self.assertEqual(result["recall_low"],0.);self.assertEqual(result["f1_high"],0.)
        self.assertEqual(result,product_bootstrap(frame,200))

    def test_bounded_precision_is_undefined_without_candidate_predictions(self):
        from src.training import bounded_metrics
        result=bounded_metrics([1,0,np.nan],[False,False,False])
        self.assertFalse(result["precision_bound_defined"])


    def test_inference_receipt_rejects_modified_output(self):
        from src.inference import validate_inference
        data=pd.DataFrame({"asin":["A","A"]})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"results/predictions/pair_scores.csv.gz"
            path.parent.mkdir(parents=True);path.write_bytes(b"fixture")
            write_json(root/"models/model_lock.json",{"fixture":True})
            write_json(root/"results/tables/final_evaluation.json",{"fixture":True})
            receipt={"dataset_version":"v","pairs":1,"method_rows":5,"products":1,
                     "methods":["A","B","C","D","E"],"output_sha256":sha_file(path),
                     "model_lock_sha256":sha_file(root/"models/model_lock.json"),
                     "final_evaluation_sha256":sha_file(root/"results/tables/final_evaluation.json"),
                     "decision_counts":{"review":5},"population_automatic_ready":False}
            write_json(path.parent/"manifest.json",receipt)
            with patch("src.inference.validate_lock"),patch("src.evaluation.validate_evaluation"),\
                 patch("src.inference.load_study",return_value=(data,{"dataset_version":"v"})):
                self.assertEqual(validate_inference(root)["pairs"],1)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(GateError,"checksum"):validate_inference(root)


    def test_synthetic_variants_cannot_collide_with_held_out_text(self):
        from src.synthetic import generate_synthetic
        text="This comfortable replacement is working perfectly with my phone every single day"
        train_frame=pd.DataFrame({"review_id":[f"r{i}" for i in range(1000)],
                                  "asin":["A"]*1000,"split":["train"]*1000,
                                  "supervised_eligible":[True]*1000,"text":[text]*1000})
        held_out=pd.DataFrame({"review_id":["v"],"asin":["B"],"split":["validation"],
                               "supervised_eligible":[True],"text":[text+"!"]})
        with self.assertRaisesRegex(GateError,"punctuation"):
            generate_synthetic(pd.concat([train_frame,held_out],ignore_index=True))

if __name__=="__main__":unittest.main()
