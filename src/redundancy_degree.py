"""Three complementary evidence passes for provisional information redundancy."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from .multilingual_features import normalize_text, analyze_text, conflict_evidence

POLICY_VERSION="degree-policy-v1.2"
POLICY_RULES={
    "normal_generic_opinions":"Similar quality, packaging and delivery praise is not duplicate content.",
    "light":"A specific fact is shared while substantial new information remains.",
    "medium":"Main content substantially overlaps with limited additional information; heuristic.",
    "high":"Main specific facts are equivalent or sufficiently informative wording is almost identical.",
    "decision":"Provisional relationship and degree; do not automatically discount ratings.",
    "score_semantics":"Frozen similarity/classifier scores are evidence, not duplicate probability or severity.",
}
POLICY_DIGEST=hashlib.sha256(json.dumps(POLICY_RULES,sort_keys=True).encode()).hexdigest()
POLICY_CODE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

MASS=r"(\d+(?:\.\d+)?)\s*(kg|千克|公斤|g|克)(?![a-z0-9_])"
GENERIC=re.compile(r"质量|品质|包装|物流|快递|配送|价格|满意|不错|很好|好评|喜欢|推荐|划算|品质|描述|值得|quality|packaging|delivery|shipping|satisfied|good|great|excellent|recommend|price",re.I)
DETAIL=re.compile(r"\d|赠|送了|切片|油性|气味|味道|口感|保湿|不刺激|不油腻|补水|肤质|浓香|吸收|异味|配料|小时|分钟|尺寸|型号|实际|标称|充电|续航|电池|屏幕|接口|安装|松动|破损|裂|霉|碎|gift|free|slice|oily|greasy|scent|taste|moist|battery|screen|charging|broken|crack|model|received only",re.I)
GIFT_OBJECTS={
    "red_dates":r"红枣|jujubes?|red dates",
    "tweezers":r"镊子|tweezers",
    "spoon":r"小勺|勺子|spoon",
    "cup":r"杯子|cup",
    "cable":r"充电线|数据线|(?:usb[- ]?c |charging )?cable",
    "ginseng_slices":r"西洋参片|ginseng slices",
}
ATTRIBUTE_VALUES={
    "texture_oily":r"油性(?:很|非常)?足|油性很好|oily texture",
    "slice_service":r"(?:代|帮忙|帮我|商家.{0,5})切片|(?:seller|shop).{0,25}slic",
    "not_greasy":r"不油腻|不黏腻|不粘腻|non[- ]?greasy|not greasy|not sticky",
    "rose_scent":r"玫瑰(?:花)?香|rose scent",
    "no_irritation":r"温和不刺激|无刺激|不刺激|does not irritate|non[- ]?irritating",
    "overnight_moisture":r"(?:一晚|整晚|晚上).{0,12}(?:不干|保湿)|overnight moisture",
}
def _mass(value,unit):
    return round(float(value)*(1000 if unit in ("kg","千克","公斤") else 1),6)

def specific_facts(text):
    """Return typed facts only. A topic name such as 'shipping' is never a fact."""
    text=normalize_text(text);facts={};spans=[]
    combined=re.search(MASS+r"[^。.!?！？;；]{0,14}(?:送到|收到|实收|只有|actual(?:ly)? received)[^\d]{0,8}"+MASS,text,re.I)
    if combined:
        declared=_mass(combined.group(1),combined.group(2))
        received=_mass(combined.group(3),combined.group(4))
        facts["declared_mass_g"]=declared;facts["received_mass_g"]=received
        spans.append(combined.span())
    else:
        patterns={
            "declared_mass_g":r"(?:标称|标注|上面写(?:着|了)?|写着|写了|claimed|advertised|listed)[^\d。.!?！？;；]{0,12}"+MASS,
            "received_mass_g":r"(?:实收|实到|收到|送到|到手|actual(?:ly)? received)[^\d。.!?！？;；]{0,12}"+MASS,
        }
        for role,pattern in patterns.items():
            match=re.search(pattern,text,re.I)
            if match:
                facts[role]=_mass(match.group(1),match.group(2));spans.append(match.span())
    clauses=re.split(r"[。.!?！？;；,，\n]",text)
    for clause in clauses:
        # Bind explicit negation to a named property; do not flip from any negative word.
        battery_claim=re.search(r"电池.{0,10}耐用|battery.{0,30}durable",clause,re.I)
        if battery_claim and not re.search(r"不是不|不能说不|没有不|not undurable",battery_claim.group(),re.I):
            facts["battery_durable"]=not bool(re.search(r"不(?:太|够|那么)?耐用|not.{0,8}durable|isn.t.{0,8}durable",battery_claim.group(),re.I))
        # ASCII boundaries also match identifiers immediately adjacent to Han characters.
        model_claims={}
        for match in re.finditer(r"(?<![a-z0-9_])(iphone|ipad|galaxy|pixel|thinkpad|macbook|xiaomi|redmi|oneplus|oppo|vivo)\s*[- ]?([a-z]?\d+[a-z]?)(?![a-z0-9_])",clause,re.I):
            prefix=clause[max(0,match.start()-24):match.start()]
            cues=list(re.finditer(r"不(?:支持|兼容|适用|适配|适合)|不能(?:适配|兼容|使用)?|does not support|cannot support|not compatible(?: with)?|not for|适配|适用|支持|兼容|fits?|compatible(?: with)?|supports?|for",prefix,re.I))
            last=cues[-1].group().casefold() if cues else ""
            negative=bool(last and re.match(r"不|不能|does not|cannot|not ",last))
            role="incompatible" if negative else "supported" if last else "mentioned"
            key="model_"+role+"_"+match.group(1)
            model_claims.setdefault(key,set()).add(match.group(2))
        for key,values in model_claims.items():
            facts[key]=sorted(set(facts.get(key,[]))|values)
        if re.search(r"赠|送了|赠品|附赠|free|gift",clause,re.I):
            for name,pattern in GIFT_OBJECTS.items():
                if re.search(pattern,clause,re.I):facts["gift_"+name]=True
        for name,pattern in ATTRIBUTE_VALUES.items():
            if re.search(pattern,clause,re.I):facts[name]=True
    return facts

def _generic(text):
    # This does not claim language-wide coverage. Unrecognized details remain reviewable.
    n=normalize_text(text)
    return bool(GENERIC.search(n)) and not bool(DETAIL.search(n))

def _coverage(a,b):
    def grams(text):
        compact=re.sub(r"\s+","",normalize_text(text))
        return {compact[i:i+3] for i in range(max(0,len(compact)-2))}
    aa,bb=grams(a),grams(b)
    return len(aa&bb)/max(1,len(aa)),len(aa&bb)/max(1,len(bb))

def classify_pair(a,b,method_scores=None,thresholds=None,*,ids=None):
    if any(not isinstance(t,str) or not t.strip() or len(t)>10000 for t in (a,b)):
        raise ValueError("Provide nonempty text up to 10000 characters")
    aa,bb=analyze_text(a),analyze_text(b)
    exact=aa["normalized"]==bb["normalized"]
    fa,fb=specific_facts(a),specific_facts(b)
    shared={key:fa[key] for key in sorted(fa.keys()&fb.keys())
            if fa[key]==fb[key] and not key.startswith("model_")}
    # A shared product/model name is context, not a shared experience or event.
    differences={key:[fa[key],fb[key]] for key in sorted(fa.keys()&fb.keys()) if fa[key]!=fb[key]}
    families={key.removeprefix("model_supported_") for key in fa.keys()|fb.keys() if key.startswith("model_supported_")}
    for family in sorted(families):
        key_supported="model_supported_"+family;key_incompatible="model_incompatible_"+family
        a_yes_b_no=set(fa.get(key_supported,[]))&set(fb.get(key_incompatible,[]))
        a_no_b_yes=set(fa.get(key_incompatible,[]))&set(fb.get(key_supported,[]))
        for model in sorted(a_yes_b_no|a_no_b_yes):
            differences["model_compatibility_"+family+"_"+model]=["supported","incompatible"] if model in a_yes_b_no else ["incompatible","supported"]
    conflict=conflict_evidence(a,b)
    generic=_generic(a) and _generic(b)
    cover_a,cover_b=_coverage(a,b)
    scores={str(k):float(v) for k,v in (method_scores or {}).items()}
    threshold_map=thresholds or {}
    active=[k for k,v in scores.items() if threshold_map.get(k) is not None and v>=threshold_map[k]]
    short=aa["short_evidence"] or bb["short_evidence"]
    quantity_pair={"declared_mass_g","received_mass_g"}<=shared.keys()
    reason="no_confirmed_shared_specific_information";degree="none";confidence="low"
    if differences or conflict["candidate_conflict"]:
        reason="same_attribute_detail_difference";confidence="medium"
    elif generic and not (exact and min(len(aa["normalized"]),len(bb["normalized"]))>=70):
        reason="normal_similar_generic_opinions";confidence="medium"
    elif quantity_pair and fa==fb and len(aa["normalized"])<120 and len(bb["normalized"])<120:
        degree="high";confidence="high";reason="equivalent_declared_and_received_quantity_facts"
    elif exact and not short and (shared or min(len(aa["normalized"]),len(bb["normalized"]))>=40):
        degree="high";confidence="high";reason="informative_exact_wording"
    elif shared:
        meaningful_keys={key for key in fa.keys()|fb.keys() if not key.startswith("model_")}
        coverage=len(shared)/max(1,len(meaningful_keys))
        if coverage>=.8 and min(cover_a,cover_b)>=.65 and len(shared)>=2:
            degree="high";confidence="medium";reason="main_specific_facts_and_wording_overlap"
        elif coverage>=.6 and min(cover_a,cover_b)>=.4 and len(shared)>=2:
            degree="medium";reason="mostly_shared_specific_facts_with_local_additions"
        else:
            degree="light";confidence="medium";reason="shared_specific_fact_with_new_or_unparsed_information"
    elif min(cover_a,cover_b)>=.8 and not short and scores.get("B",0)>=.8 and scores.get("D",0)>=.8:
        degree="high";confidence="medium";reason="strong_whole_text_wording_and_semantic_agreement"
    elif min(cover_a,cover_b)>=.55 and not short and ("C" in active or "E" in active) and scores.get("D",0)>=.8:
        degree="medium";reason="partial_wording_and_frozen_model_agreement"
    # Missing information yields a provisional negative estimate, never a confirmed negative.
    unresolved=confidence=="low" or bool(differences) or conflict["candidate_conflict"]
    return {
        **{k:(ids or {}).get(k,"") for k in ("pair_id","asin","review_id_a","review_id_b")},
        "estimated_is_duplicate":degree!="none","redundancy_degree":degree,
        "judgment_status":"provisional","confidence_band":confidence,
        "needs_review":True,"evidence_insufficient":bool(unresolved),"decision":"review",
        "evidence_round_1":json.dumps({"normalized_equal":exact,"normal_generic_opinions":generic,
                                     "short_evidence":short,"wording_coverage":[cover_a,cover_b]},ensure_ascii=False),
        "evidence_round_2":json.dumps({"facts_a":fa,"facts_b":fb,"shared_facts":shared,
                                     "fact_differences":differences,"detail_conflict_cues":conflict},ensure_ascii=False),
        "evidence_round_3":json.dumps({"method_scores":scores,"thresholds":threshold_map,
                                     "active_methods":active,"score_meaning":"uncalibrated evidence only"},ensure_ascii=False),
        "reason":reason,"model_version":"multilingual-v2.1","policy_version":POLICY_VERSION,
        "policy_digest":POLICY_DIGEST,"policy_code_sha256":POLICY_CODE_SHA256,
    }

class DegreePredictor:
    def __init__(self,model_dir=None):
        from .multilingual_inference import MultilingualPredictor
        self.model=MultilingualPredictor(model_dir)
    def predict_pairs(self,pairs):
        import pandas as pd
        pairs=list(pairs)
        if not pairs:return pd.DataFrame()
        scored=self.model.predict_pairs(pairs)
        table=scored.pivot(index="pair_id",columns="method",values="score")
        ordered=scored.pair_id.drop_duplicates().tolist()
        thresholds={m:self.model.manifest["configs"][m]["threshold"] for m in "ABCDE"}
        output=[]
        for row,pid in zip(pairs,ordered):
            output.append(classify_pair(row["text_a"],row["text_b"],table.loc[pid].to_dict(),
                          thresholds,ids={**row,"pair_id":pid}))
        return pd.DataFrame(output)

    def score_product(self,asin,reviews):
        import pandas as pd
        if isinstance(reviews,pd.DataFrame):reviews=reviews.to_dict("records")
        reviews=list(reviews)
        scores=self.model.score_product(asin,reviews)
        if scores.empty:
            return pd.DataFrame(columns=["pair_id","asin","review_id_a","review_id_b",
                                         "estimated_is_duplicate","redundancy_degree","decision"])
        lookup={row["review_id"]:row["text"] for row in reviews}
        matrix=scores.pivot(index="pair_id",columns="method",values="score")
        thresholds={m:self.model.manifest["configs"][m]["threshold"] for m in "ABCDE"}
        output=[]
        for row in scores.drop_duplicates("pair_id").to_dict("records"):
            output.append(classify_pair(lookup[row["review_id_a"]],lookup[row["review_id_b"]],
                          matrix.loc[row["pair_id"]].to_dict(),thresholds,ids=row))
        return pd.DataFrame(output)

if __name__=="__main__":
    import argparse
    import pandas as pd
    from .common import write_csv
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir",type=Path)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    frame=pd.read_csv(args.input,keep_default_na=False)
    if not {"text_a","text_b"}<=set(frame):
        raise ValueError("Input requires text_a and text_b columns")
    predictor=DegreePredictor(args.model_dir)
    parts=[predictor.predict_pairs(frame.iloc[i:i+1000].to_dict("records")) for i in range(0,len(frame),1000)]
    write_csv(args.output,pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(
        columns=["pair_id","estimated_is_duplicate","redundancy_degree","judgment_status","needs_review","decision"]))
