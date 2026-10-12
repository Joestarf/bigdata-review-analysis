"""Experimental review weights and sample product ratings; frozen models stay unchanged."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, sha_file, read_json, write_json, write_csv, word_count
from .inference import Predictor, validate_inference

SCENARIOS = ["baseline", "record_only", "wording_group", "graded_conservative", "graded_default", "graded_strong"]
DEFAULT_POLICY = {
    "policy_version": "rating-weight-v1",
    "interpretation": "experimental information-redundancy weights, not validated authenticity or production scores",
    "short_min_words": 5,
    "pair_rule": "maximum direct evidence per review; no transitive closure",
    "record_copy_weight": 0.0,
    "semantic_only_weight": 1.0,
    "short_or_undetermined_weight": 1.0,
    "graded_conservative": {"light": .90, "moderate": .75, "same_wording": .50},
    "graded_default": {"light": .75, "moderate": .50, "same_wording": .25},
    "graded_strong": {"light": .50, "moderate": .25, "same_wording": .10},
}
LEVELS = {0: "no_lexical_signal", 1: "light_candidate", 2: "moderate_candidate",
          3: "same_wording", 4: "extra_record_copy"}

def validate_policy(policy):
    if policy.get("short_min_words") != 5:
        raise ValueError("This version keeps the frozen five-word review boundary.")
    for key in ["record_copy_weight", "semantic_only_weight", "short_or_undetermined_weight"]:
        if policy.get(key) != DEFAULT_POLICY[key]:
            raise ValueError(f"Reserved evidence rule cannot change: {key}")
    for name in SCENARIOS[3:]:
        values = policy.get(name, {})
        w = [values.get(k) for k in ["same_wording", "moderate", "light"]]
        if any(not isinstance(x, (int, float)) or not np.isfinite(x) for x in w):
            raise ValueError(f"Invalid policy weights: {name}")
        if not (0 < w[0] <= w[1] <= w[2] <= 1):
            raise ValueError("Weights must satisfy 0 < same_wording <= moderate <= light <= 1.")

def aggregate_ratings(weights):
    """Offline recomputation needs only ASIN, rating and six weight columns."""
    required = {"asin", "review_id", "rating"} | {f"weight_{s}" for s in SCENARIOS}
    if not required.issubset(weights.columns):
        raise ValueError(f"Missing rating inputs: {sorted(required - set(weights.columns))}")
    if not weights.review_id.is_unique or weights.asin.isna().any():
        raise ValueError("Use unique review IDs and nonempty ASINs.")
    if not weights.asin.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("ASINs must be nonempty strings.")
    rating = pd.to_numeric(weights.rating, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(rating).all() or not np.isin(rating, [1,2,3,4,5]).all():
        raise ValueError("Ratings must be integer stars from 1 to 5.")
    frames = []
    for scenario in SCENARIOS:
        w = pd.to_numeric(weights[f"weight_{scenario}"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(w).all() or not ((w >= 0) & (w <= 1)).all():
            raise ValueError("Weights must be finite and in [0,1].")
        work = pd.DataFrame({"asin": weights.asin, "weighted_stars": rating*w,
                             "weight": w, "weight_sq": w*w, "kept": w > 0})
        g = work.groupby("asin", sort=True).agg(review_count=("weight","size"),
            positive_weight_reviews=("kept","sum"), effective_weight=("weight","sum"),
            weighted_star_sum=("weighted_stars","sum"), squared_weight_sum=("weight_sq","sum"))
        if g.effective_weight.le(0).any():
            raise ValueError("A product has zero total weight; final rating is undefined.")
        g["final_rating"] = g.weighted_star_sum/g.effective_weight
        g["kish_effective_n"] = g.effective_weight**2/g.squared_weight_sum
        g["scenario"] = scenario
        # Ties use six decimal places, never CSV display rounding.
        g["sample_rank"] = g.final_rating.round(6).rank(method="min", ascending=False).astype(int)
        frames.append(g.reset_index())
    products = pd.concat(frames, ignore_index=True)
    base = products[products.scenario.eq("baseline")].set_index("asin")
    products["baseline_rating"] = products.asin.map(base.final_rating)
    products["baseline_rank"] = products.asin.map(base.sample_rank)
    products["rating_delta"] = products.final_rating - products.baseline_rating
    products["rank_change"] = products.baseline_rank - products.sample_rank
    if not products.final_rating.between(1-1e-12,5+1e-12).all():
        raise ValueError("Weighted rating outside [1,5].")
    return products.drop(columns=["squared_weight_sum"])

def classify_reviews(df, predictor):
    n = len(df)
    copy = df.source_id.duplicated().to_numpy()
    keep = ~copy
    long = df.text.map(word_count).ge(5).to_numpy()
    level = np.zeros(n, dtype=np.int8)
    level[copy] = 4
    sizes = np.ones(n, dtype=int)
    eligible = df[keep & long]
    count = eligible.groupby(["asin", "text_hash"]).review_id.transform("size").to_numpy()
    sizes[eligible.index] = count
    level[keep & long & (sizes > 1)] = 3
    semantic = np.zeros(n, dtype=bool)
    partner = np.full(n, "", dtype=object)
    evidence_pair = np.full(n, "", dtype=object)
    positions = dict(zip(df.review_id, range(n)))
    candidates = []
    thresholds = {m: predictor.lock["methods"][m]["review_threshold"] for m in ["B","C","D","E"]}
    for k, (asin, group) in enumerate(df.groupby("asin", sort=True), 1):
        # Exact source copies do not create additional near-duplicate evidence.
        group = group[keep[group.index] & long[group.index]]
        if len(group) >= 2:
            scores = predictor.score_product(group)
            wide = scores.pivot(index=["pair_id","asin","review_id_a","review_id_b"], columns="method", values="score").reset_index()
            mask = wide.A.eq(1)
            for m in thresholds:
                mask |= wide[m].ge(thresholds[m])
            wide = wide[mask].sort_values("pair_id")
            for r in wide.itertuples(index=False):
                i,j = positions[r.review_id_a], positions[r.review_id_b]
                lex = r.B >= thresholds["B"]
                cls = r.C >= thresholds["C"] or r.E >= thresholds["E"]
                strength = 3 if r.A == 1 else (2 if lex and r.E >= thresholds["E"] else (1 if lex or cls else 0))
                sem = r.D >= thresholds["D"]
                semantic[i] |= sem; semantic[j] |= sem
                # Only direct matching peers contribute. Equal strength keeps the smallest pair ID.
                for a,b in [(i,j),(j,i)]:
                    if strength > level[a] or (strength == level[a] and strength > 0 and
                       (not evidence_pair[a] or r.pair_id < evidence_pair[a])):
                        level[a] = strength
                        partner[a] = df.review_id.iloc[b]
                        evidence_pair[a] = r.pair_id
                candidates.append({"pair_id":r.pair_id, "asin":asin, "review_id_a":r.review_id_a,
                    "review_id_b":r.review_id_b, **{m:float(getattr(r,m)) for m in ["A","B","C","D","E"]},
                    "direct_level": LEVELS[strength], "semantic_signal":bool(sem),
                    "decision":"review", "status":"experimental_signal_not_confirmed_redundancy"})
        if k % 100 == 0:
            print(f"Weighting evidence products={k}, candidate pairs={len(candidates)}", flush=True)
    out = df[["asin","review_id","rating"]].copy()
    out["degree"] = [LEVELS[v] for v in level]
    out["short_text"] = ~long
    out["semantic_signal"] = semantic
    out["unique_wording_group_size"] = sizes
    out["direct_peer_review_id"] = partner
    out["direct_pair_id"] = evidence_pair
    out["evidence_status"] = np.select([level==4, level==3, (level==1)|(level==2)],
        ["record_duplicate", "observed_same_wording", "unconfirmed_model_candidate"], default="undetermined_or_no_lexical_signal")
    out["requires_review"] = level != 4
    return out, pd.DataFrame(candidates, columns=["pair_id","asin","review_id_a","review_id_b","A","B","C","D","E",
                                                "direct_level","semantic_signal","decision","status"])

def apply_weights(reviews, policy):
    validate_policy(policy)
    out = reviews.copy()
    copy = out.degree.eq("extra_record_copy")
    same = out.degree.eq("same_wording")
    out["weight_baseline"] = 1.
    out["weight_record_only"] = (~copy).astype(float)
    out["weight_wording_group"] = np.where(copy, 0., np.where(same, 1./out.unique_wording_group_size, 1.))
    for scenario in SCENARIOS[3:]:
        cfg = policy[scenario]
        out[f"weight_{scenario}"] = out.degree.map({"extra_record_copy":0., "same_wording":cfg["same_wording"],
            "moderate_candidate":cfg["moderate"], "light_candidate":cfg["light"], "no_lexical_signal":1.})
    if out[[f"weight_{s}" for s in SCENARIOS]].isna().any().any():
        raise ValueError("Unknown degree; no weight assigned.")
    return out

def make_report(root, products, weights, receipt):
    def table(frame):
        cols = frame.columns.tolist()
        return "| "+" | ".join(cols)+" |\n| "+" | ".join(["---"]*len(cols))+" |\n"+"\n".join(
            "| "+" | ".join(str(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))
    summary=[]
    for name,g in products.groupby("scenario",sort=False):
        summary.append({"策略":name, "星级变化商品":int(g.rating_delta.abs().gt(1e-9).sum()),
            "平均绝对变化":round(float(g.rating_delta.abs().mean()),6),
            "最大绝对变化":round(float(g.rating_delta.abs().max()),6),
            "样本名次变化商品":int(g.rank_change.ne(0).sum())})
    top=products[products.scenario.eq("graded_default")].copy()
    top=top.loc[top.rating_delta.abs().sort_values(ascending=False).head(10).index,
                ["asin","review_count","effective_weight","baseline_rating","final_rating","rating_delta"]].round(6)
    degrees=weights.degree.value_counts().rename_axis("degree").reset_index(name="reviews")
    report=f"""# 评论折扣与商品样本星级

## 计算方法

最终星级 = Σ(评论权重 × 原星级) / Σ评论权重。折扣降低评论的计入权重，原星级不改写；重复好评与重复差评使用相同规则。结果在1—5之间。数据版本：{receipt['dataset_version']}。这是当前90,750条、1,000 ASIN样本的评分影响实验，不是Amazon全量或官方星级。

|程度或证据|默认权重|含义|
|---|---|---|
|没有词面重复信号|1|保持计入|
|轻度候选|0.75|B、C或E超过已锁定的探索阈值，尚未确认|
|中度候选|0.50|B与E同时超过已锁定阈值，尚未确认|
|相同措辞|0.25|同商品规范化正文相同且两条均至少5词|
|五字段完全一致的额外记录|0|每组保留固定行顺序中的首条，其他副本不重复计入|
|不足5词或仅语义相近|1|证据不足，保持权重；记录副本规则优先|

A为正文精确匹配；B字符TF-IDF阈值{receipt['thresholds']['B']}，C/E逻辑回归阈值{receipt['thresholds']['C']}/{receipt['thresholds']['E']}，D语义阈值{receipt['thresholds']['D']}。阈值、模型和封存测试均未修改。这里的轻/中度是实验政策对候选信号的分级，不是经验证的真实重复程度。相同措辞也不能证明评论者复制或虚假。

每条评论取它与其他评论的最强直接证据；同级以最小pair_id选可追溯的直接对。A像B、B像C不会将A与C合并。成对证据对两端使用相同规则，不按星级、用户或时间选择一端；每条评论独立取自身最强直接证据，因此同一对的两条评论最终权重可能不同。短评保持权重，不把通用短赞语直接当已确认冗余。

## 其他策略与参数

- baseline：全部评论权重1，保留原样本平均星级。
- record_only：只排除额外记录副本。
- wording_group：先排除记录副本，再让同商品至少5词的同正文组每条权重1/组内唯一记录数，使整组总权重为1；短评保持1。
- graded_conservative：轻/中/相同措辞为0.90/0.75/0.50。
- graded_default：轻/中/相同措辞为0.75/0.50/0.25。
- graded_strong：轻/中/相同措辞为0.50/0.25/0.10。

每种策略独立计算分母。effective_weight为权重之和，Kish有效样本量为(Σw)^2/Σw²，不是独立信息条数。名次仅在这1,000个ASIN内按星级保留6位小数排序，同分采用最小名次；rank_change为原名次减新名次，正值表示上升。用于敏感性分析，不能代表类别总体排名。很多商品原评分相同或很接近；其他商品的微小变化也会改变未改评分商品的名次，因此名次变化不能单独证明实际质量次序改变。

## 实际结果

{table(pd.DataFrame(summary))}

评论分级（以额外记录副本优先，互斥）：

{table(degrees)}

默认策略星级变化最大的10个商品：

{table(top)}

## 交接文件与复现

- results/tables/review_weights.csv：全部90,750条稳定评论编号（可回连本地正文）、ASIN、原星级、证据级别、直接对及六种权重；不含评论正文、原用户标识和时间戳。
- results/tables/product_ratings.csv：1,000商品×六种策略，原始及最终星级、有效权重和样本名次。
- results/tables/product_final_ratings.csv：默认折扣的1,000商品结果。
- results/tables/candidate_pair_signals.csv：保留候选的五方法分数、直接级别和review标记。
- results/tables/weighting_policy.json及rating_adjustment_manifest.json：参数、输入版本及文件校验值。
- results/figures/rating_discount_effect.png：默认策略星级变化。

运行 .venv/bin/python -m src.rating_adjustment 生成固定版本实验。使用 --policy 指定同结构政策文件可改变折扣但不改模型；使用 --weights review_weights.csv --output product_ratings.csv 可仅从衍生数据重新计算六策略。下游报告默认显示graded_default，同时保留baseline与wording_group对照。

## 使用范围

当前五方法没有达到验证精确率目标；标注未经独立人类验证。本次折扣由实验政策指定，没有从测试集拟合，也没有证明其能改善真实商品质量排序。所有模型候选的原decision仍为review。主张自动使用前需要独立复核和新的自然抽样验证；不能把本次加权星级称为纠正后的真值。有限流式样本、每商品最多100条、缺少parent_asin等限制继续适用。
"""
    (root/"report/rating_adjustment.md").write_text(report,encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d=products[products.scenario.eq("graded_default")]
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axs[0].scatter(d.baseline_rating,d.final_rating,s=10,alpha=.4)
    axs[0].plot([1,5],[1,5],"--",color="grey")
    axs[0].set(xlabel="Original sample rating",ylabel="Weighted sample rating",
               title="Default experimental discount",xlim=(1,5),ylim=(1,5))
    axs[1].hist(d.rating_delta,bins=40)
    axs[1].set(xlabel="Weighted minus original stars",ylabel="Products",title="Rating change (1,000 ASINs)")
    fig.savefig(root/"results/figures/rating_discount_effect.png",dpi=160)
    plt.close(fig)

def run(root=ROOT, policy_path=None):
    validate_inference(root)
    predictor=Predictor(root)
    # load_study uses a fixed RangeIndex; preserve the original order for source copies.
    df=predictor.df.reset_index(drop=True)
    policy=read_json(policy_path) if policy_path else dict(DEFAULT_POLICY)
    reviews,pairs=classify_reviews(df,predictor)
    weights=apply_weights(reviews,policy)
    products=aggregate_ratings(weights)
    out=root/"results/tables"
    write_csv(out/"review_weights.csv",weights)
    write_csv(out/"candidate_pair_signals.csv",pairs)
    write_csv(out/"product_ratings.csv",products)
    write_csv(out/"product_final_ratings.csv",products[products.scenario.eq("graded_default")])
    write_json(out/"weighting_policy.json",policy)
    receipt={"dataset_version":predictor.manifest["dataset_version"],"policy_version":policy["policy_version"],
        "model_lock_sha256":sha_file(root/"models/model_lock.json"),"final_evaluation_sha256":sha_file(out/"final_evaluation.json"),
        "thresholds":{m:predictor.lock["methods"][m]["review_threshold"] for m in ["A","B","C","D","E"]},
        "products":int(weights.asin.nunique()),"reviews":len(weights),"candidate_pairs":len(pairs),
        "degree_counts":weights.degree.value_counts().to_dict(),"automatic_production_ready":False,
        "no_transitive_groups":True,"source_sha256":sha_file(root/"src/rating_adjustment.py"),
        "file_sha256":{name:sha_file(out/name) for name in ["review_weights.csv","candidate_pair_signals.csv",
            "product_ratings.csv","product_final_ratings.csv","weighting_policy.json"]}}
    write_json(out/"rating_adjustment_manifest.json",receipt)
    make_report(root,products,weights,receipt)
    print({"products":receipt["products"],"reviews":len(weights),"candidate_pairs":len(pairs),"degree_counts":receipt["degree_counts"]})
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--policy",type=Path)
    p.add_argument("--weights",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.weights:
        if not args.output:p.error("--weights requires --output")
        write_csv(args.output,aggregate_ratings(pd.read_csv(args.weights,keep_default_na=False)))
    else:
        run(args.root,args.policy)

if __name__=="__main__":main()
