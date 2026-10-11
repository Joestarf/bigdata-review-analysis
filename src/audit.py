from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, COLUMNS, SEED, sha_file, sha_text, normalize_text, word_count, read_reviews, clean_frame, write_csv, write_json, read_json, packages

def make_index(df, version):
    ids = []
    for i, row in enumerate(df.itertuples(index=False, name=None)):
        asin, user, rating, text, stamp = row
        canonical = json.dumps([str(asin), str(user), float(rating), str(text), int(stamp)], ensure_ascii=False, separators=(",", ":"))
        ids.append({"review_id": sha_text(f"{version}:row:{i}"), "source_id": sha_text(canonical),
                    "text_hash": sha_text(normalize_text(text)), "dataset_version": version})
    return pd.DataFrame(ids)

def split_records(df):
    products = np.array(sorted(df.asin.unique()))
    np.random.default_rng(SEED).shuffle(products)
    n = len(products)
    train_n, val_n = int(n * .7), int(n * .15)
    mapping = {str(p): "train" if i < train_n else "validation" if i < train_n + val_n else "test"
               for i, p in enumerate(products)}
    out = df[["review_id"]].copy()
    out["product_split"] = df.asin.map(mapping)
    cross = pd.DataFrame({"text_hash": df.text_hash, "split": out.product_split}).groupby("text_hash").split.nunique()
    shared = set(cross[cross > 1].index)
    out["split"] = out.product_split
    out.loc[df.text_hash.isin(shared), "split"] = "template_pool"
    # Stable first copy for supervised learning; duplicates remain in the frozen sample and index.
    out["supervised_eligible"] = ~df.source_id.duplicated() & ~df.text_hash.isin(shared)
    return out

def run_audit(root: Path = ROOT):
    processed = root / "data/processed/reviews.csv"
    rawpath = root / "data/interim/reviews_sample.csv"
    df = read_reviews(processed)
    cleaned_current, invalid_current, _ = clean_frame(df)
    if len(cleaned_current) != len(df):
        raise ValueError("Frozen processed sample contains invalid records; inspect before rebuilding.")
    version = sha_file(processed)
    idx = make_index(df, version)
    rich = pd.concat([idx, df], axis=1)
    sampling = read_json(root / "data/interim/sampling_metadata.json")
    newer, excluded, cleaning = clean_frame(read_reviews(rawpath))
    cleaning["frozen_processed_rows"] = len(df)
    cleaning["improved_cleaning_matches_frozen_data"] = newer.equals(df)
    if len(newer) == len(df):
        cleaning["text_differences_against_frozen_data"] = int(newer.text.ne(df.text).sum())
        changed = newer.text.ne(df.text)
        deltas = pd.DataFrame({"review_id": idx.loc[changed,"review_id"], "frozen_text": df.loc[changed,"text"], "improved_text": newer.loc[changed,"text"]})
        write_csv(root / "results/tables/cleaning_text_differences.csv", deltas)
    cleaning["note"] = "Audit does not overwrite the current five-field sample."
    write_csv(root / "results/tables/excluded_rows.csv", excluded)
    write_json(root / "results/tables/cleaning_audit.json", cleaning)

    bytext = rich.groupby("text_hash").agg(rows=("review_id", "size"), products=("asin", "nunique"),
                    users=("user_id", "nunique"), ratings=("rating", "nunique"), text=("text", "first"))
    within = rich.groupby(["asin", "text_hash"]).agg(rows=("review_id", "size"), users=("user_id", "nunique"),
                    ratings=("rating", "nunique"), text=("text", "first")).reset_index()
    repeated = within[within.rows > 1].copy()
    repeated["extra_rows"] = repeated.rows - 1
    repeated["star_conflict"] = repeated.ratings > 1
    product = rich.groupby("asin").size().rename("reviews").to_frame()
    pstats = repeated.groupby("asin").agg(repeated_groups=("text_hash", "size"), extra_rows=("extra_rows", "sum"),
                                          cross_user_groups=("users", lambda x: int((x > 1).sum())),
                                          conflicting_rating_groups=("star_conflict", "sum"))
    product = product.join(pstats).fillna(0)
    for c in product.columns:
        product[c] = product[c].astype(int)
    product["extra_row_share"] = product.extra_rows / product.reviews
    write_csv(root / "results/tables/product_redundancy.csv", product.reset_index())
    write_csv(root / "results/tables/within_product_repeat_groups.csv", repeated.drop(columns="text"))
    length = rich.text.map(word_count)
    band = pd.cut(length, bins=[-1, 4, 19, float("inf")], labels=["<5 words", "5-19 words", ">=20 words"])
    bandframe = rich.assign(length_band=band, record_extra=rich.source_id.duplicated(),
                           global_text_extra=rich.text_hash.duplicated(),
                           within_text_extra=rich.duplicated(["asin", "text_hash"]))
    bstats = bandframe.groupby("length_band", observed=False).agg(rows=("review_id", "size"),
                        record_extra_rows=("record_extra", "sum"), global_text_extra_rows=("global_text_extra", "sum"),
                        within_product_extra_rows=("within_text_extra", "sum")).reset_index()
    write_csv(root / "results/tables/redundancy_by_length.csv", bstats)
    cross = bytext[(bytext.rows > 1) & (bytext.products > 1)]
    examples = pd.concat([bytext[bytext.rows > 1].sort_values("rows", ascending=False).head(10),
                          bytext[(bytext.rows > 1) & bytext.text.map(word_count).ge(20)].sort_values("rows", ascending=False).head(10)]).drop_duplicates()
    write_csv(root / "results/tables/repeat_examples.csv", examples.reset_index()[["text_hash", "rows", "products", "ratings", "text"]])
    stats = {"rows": len(df), "products": int(df.asin.nunique()), "users": int(df.user_id.nunique()),
             "record_extra_rows": int(rich.source_id.duplicated().sum()),
             "global_text_extra_rows": int(rich.text_hash.duplicated().sum()),
             "within_product_groups": len(repeated), "within_product_extra_rows": int(repeated.extra_rows.sum()),
             "within_product_cross_user_groups": int((repeated.users > 1).sum()),
             "cross_product_groups": len(cross), "cross_product_group_rows": int(cross.rows.sum()),
             "overlap_note": "These counts overlap and must not be added.", "short_rows": int((length < 5).sum())}
    manifest = {"artifact_version": 1, "dataset_version": version,
                "files": {"processed": {"path": str(processed.relative_to(root)), "sha256": version, "rows": len(df), "fields": COLUMNS},
                          "interim": {"path": str(rawpath.relative_to(root)), "sha256": sha_file(rawpath), "rows": len(read_reviews(rawpath)), "fields": COLUMNS}},
                "source_revision": "unrecorded", "sampling": sampling, "packages": packages(),
                "processing": {"existing_sample_frozen": True, "new_cleaner_recipe": "HTML known tags, entities, whitespace, controls; strict integer rating and milliseconds"},
                "limitations": ["Bounded stream with local shuffling, not a full-category probability sample.",
                                "Top observed ASINs, cap 100 per product. Counts are sample counts, not full product totals.",
                                "parent_asin unavailable; parent variants may cross splits."]}
    assignments = split_records(rich)
    splitstats = assignments.groupby("split").agg(rows=("review_id", "size"), supervised_rows=("supervised_eligible", "sum")).reset_index()
    write_csv(root / "data/processed/review_index.csv", idx)
    write_csv(root / "data/processed/split_assignments.csv", assignments)
    write_csv(root / "results/tables/split_summary.csv", splitstats)
    manifest["index_hashes"] = {p: sha_file(root / p) for p in ["data/processed/review_index.csv", "data/processed/split_assignments.csv"]}
    write_json(root / "data/processed/dataset_manifest.json", manifest)
    write_json(root / "results/tables/redundancy_audit.json", stats)
    validate_splits(rich, assignments)
    report = f"""# 数据版本与质量审计

固定数据版本：{version}

保留 {len(df):,} 条评论、{df.asin.nunique():,} 个 ASIN。原始与处理后文件均已记录 SHA-256；原下载仓库提交版本未记录，无法追溯补填。

五字段重复额外记录 {stats['record_extra_rows']} 条；同商品规范化文本重复 {stats['within_product_groups']} 组、额外 {stats['within_product_extra_rows']} 条；全样本文本重复额外 {stats['global_text_extra_rows']} 条。统计相互重叠，不能相加。全样本重复比例不能当作商品内部的冗余比例。

{splitstats.to_string(index=False)}

按 ASIN 划分 70/15/15；跨集合相同文本全部进入公共模板池；全字段重复副本不进入监督学习。原样本仍完整保留。由于没有 parent_asin，无法保证父商品款式完全隔离。

当前抽样优先选择有限扫描中评论较多的商品且每商品最多100条；缓冲区打乱不是对完整类别均匀随机抽样。结果只能支持本样本的实验。

改进清洗与原样本一致：{cleaning['improved_cleaning_matches_frozen_data']}。本次原抽样审计排除 {cleaning['removed_rows']} 条占位正文（如 N/A、None）；旧通用HTML正则误删比较符号或尖括号中的正文，改进规则在 {cleaning.get('text_differences_against_frozen_data',0)} 条中保留了这些信息。差异逐条存于 cleaning_text_differences.csv。当前实验固定原 CSV 与其已标注正文，未覆盖；如切换改进正文，应建立新版本并重算相关缓存及标签。排除原因与长度见 cleaning_audit.json。
"""
    (root / "report").mkdir(exist_ok=True)
    (root / "report/data_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"audit": stats, "splits": splitstats.to_dict("records"), "dataset_version": version}, indent=2))
    return stats

def validate_splits(rich, assignments):
    a = rich.merge(assignments, on="review_id", validate="one_to_one")
    eligible = a[a.supervised_eligible]
    for field in ["asin", "text_hash", "source_id"]:
        if eligible.groupby(field).split.nunique().max() > 1:
            raise ValueError(f"Leakage across splits: {field}")
    if not rich.review_id.is_unique or len(assignments) != len(rich):
        raise ValueError("Index uniqueness/coverage failed.")
