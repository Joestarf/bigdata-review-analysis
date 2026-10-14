"""Frozen A-E Chinese diagnostics; source snippets and authored controls, no retraining."""
import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, normalize_text, word_count, sha_file, write_csv, write_json
from .features import word_set, text_pair_features
from .portable import PortablePredictor
from .training import validate_lock

SOURCE_COMMIT="7f7175fd717c5dfdd74bfad9e0e46d5390bdec51"
SOURCE_REPO="https://github.com/livingbody/Commodity_Review_Sentiment_Predictions"
SOURCE_URL=f"https://raw.githubusercontent.com/livingbody/Commodity_Review_Sentiment_Predictions/{SOURCE_COMMIT}/javaroom.ipynb"
SOURCE_NOTEBOOK_SHA256="b66037eff872a4ce38653c40389941c1b81aa7c10116eccda737a4a26115e2b4"
HAN=re.compile(r"[\u3400-\u9fff]")
TEST_COLUMNS=["数据ID","用户ID","商品ID","评论时间戳","评论标题","评论内容"]

def source_examples(root):
    directory=root/"data/chinese_checks"
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"source_javaroom.ipynb"
    if not path.exists():
        # Read JSON as data only; never execute the external notebook or its shell commands.
        with urllib.request.urlopen(SOURCE_URL,timeout=30) as response:
            body=response.read(1024*1024+1)
        if len(body)>1024*1024:raise ValueError("Source notebook exceeds bounded download limit.")
        path.write_bytes(body)
    if sha_file(path)!=SOURCE_NOTEBOOK_SHA256:raise ValueError("Pinned source notebook checksum mismatch.")
    nb=json.loads(path.read_text())
    records=[]
    for index,cell in enumerate(nb["cells"]):
        command="".join(cell.get("source",[])).strip()
        if command not in ["!head 训练集.csv","!head 测试集.csv","!tail 测试集.csv"]:continue
        raw="".join("".join(output.get("text",[])) for output in cell.get("outputs",[]))
        reader=csv.DictReader(io.StringIO(raw),fieldnames=TEST_COLUMNS if command.startswith("!tail") else None)
        for row in reader:
            if not row.get("数据ID") or not row.get("评论内容"):raise ValueError("Malformed source CSV output.")
            if None in row:raise ValueError("Unexpected source CSV fields; do not use the upstream comma-index parser.")
            records.append({"sample_id":row["数据ID"],"product_id":row["商品ID"],"text":row["评论内容"],
                            "source_split":"train" if row["数据ID"].startswith("TRAIN") else "test",
                            "source_cell":index,"origin":"published_notebook_csv_output"})
    examples=pd.DataFrame(records).drop_duplicates("sample_id")
    if len(examples)==0:raise ValueError("No actual notebook CSV examples.")
    write_csv(directory/"source_examples.csv",examples)
    return examples,{"repository":SOURCE_REPO,"commit":SOURCE_COMMIT,"url":SOURCE_URL,
                    "notebook_sha256":sha_file(path),"source_rows":len(examples),
                    "source_products":int(examples.product_id.nunique()),
                    "available_original_within_product_pairs":int(sum(n*(n-1)//2 for n in examples.groupby("product_id").size())),
                    "full_competition_csv_available":False}

def make_pairs(examples,controls):
    expected={"paraphrase":10,"negation":10,"numeric_model":10,"same_sentiment_different_information":10,"generic_uncertain":6}
    if controls.group.value_counts().to_dict()!=expected or not controls.pair_id.is_unique:
        raise ValueError("Authored diagnostic quota/IDs changed; review benchmark definition.")
    if not controls.origin.eq("authored_diagnostic_control").all():
        raise ValueError("Control provenance mismatch.")
    rows=controls.to_dict("records")
    for row in examples.itertuples():
        transforms={"exact_copy":row.text,"outer_whitespace":"  "+row.text+"  ",
            "punctuation":row.text+"。",
            "cjk_spacing":HAN.sub(lambda m:m.group(0)+" ",row.text).strip()}
        for kind,variant in transforms.items():
            rows.append({"pair_id":f"source_{row.sample_id}_{kind}","topic":row.product_id,
                "group":kind,"label":"known_positive_transform","text_a":row.text,"text_b":variant,
                "reason":"已知同源文本复制或信息不变的形式变换；不是真实两位用户重复标签",
                "origin":"controlled_variant_of_notebook_example"})
    pairs=pd.DataFrame(rows)
    if not pairs.pair_id.is_unique:raise ValueError("Repeated diagnostic pair ID.")
    pairs["expected_binary"]=pairs.label.map({"near":1,"distinct":0,"known_positive_transform":1})
    pairs["pair_family"]=np.where(pairs.origin.eq("authored_diagnostic_control"),"authored_control","source_transform")
    return pairs

def metrics(pairs,signals):
    rows=[]
    for group,g in pairs.groupby("group",sort=False):
        for method in ["A","B","C","D","E"]:
            score=signals.loc[g.index,method].to_numpy()
            prediction=signals.loc[g.index,method+"_signal"].to_numpy(dtype=bool)
            labels=g.expected_binary.to_numpy(dtype=float)
            eligible=np.isfinite(labels)
            y=labels[eligible].astype(int);p=prediction[eligible]
            tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum())
            fn=int(((y==1)&~p).sum());tn=int(((y==0)&~p).sum())
            rows.append({"group":group,"method":method,"pairs":len(g),"known_positives":int((y==1).sum()),
                "known_negatives":int((y==0).sum()),"uncertain":int((~eligible).sum()),"signal_positive":int(prediction.sum()),
                "tp":tp,"fp":fp,"fn":fn,"tn":tn,"mean_score":float(score.mean()),"min_score":float(score.min()),
                "max_score":float(score.max()),"positive_transform_detection":tp/(tp+fn) if tp+fn else np.nan,
                "negative_control_false_signal":fp/(fp+tn) if fp+tn else np.nan,
                "actual_decision":"review","scope":"controlled diagnostics, not natural Chinese accuracy"})
    return pd.DataFrame(rows)

def report(root,source,pairs,summary,diagnostics,receipt):
    def table(frame):
        return "| "+" | ".join(frame.columns)+" |\n| "+" | ".join(["---"]*len(frame.columns))+" |\n"+"\n".join(
            "| "+" | ".join(str(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))
    scopes=summary[["group","method","pairs","known_positives","known_negatives","signal_positive","tp","fp","fn","tn"]]
    pure=diagnostics[diagnostics.han_chars.gt(0)&diagnostics.ascii_word_count.eq(0)]
    report=f"""# 固定重复识别模型的中文诊断

## 结论

当前实现不宜直接用于中文近似重复识别。规范化精确匹配仍可识别{source['source_rows']}对全部原样复制，但词面特征覆盖和短评规则存在明显中文适配问题，语义阈值也混淆关键否定及数字/型号差异。中英文配对对照同时表明，否定和数值辨别并非只有中文才困难；不能把所有误信号归因于语言，也不能仅替换编码器就宣称解决。

这是有来源样例和构造控制对的功能诊断，不是自然中文准确率；实际识别决策全部保持review。

## 范围与来源

测试现有A—E重复识别方法在中文输入上的特征、阈值信号和运行行为。不是复现另一项情感/星级预测任务，不重新训练，不改动原模型、阈值、英语数据、封存测试或交接包。

来源：[livingbody/Commodity_Review_Sentiment_Predictions]({SOURCE_REPO})，固定提交{SOURCE_COMMIT}。该仓库仅提交README、notebook和requirements，完整竞赛CSV不在仓库。只从公开notebook的CSV输出解析{source['source_rows']}条实际中文评论，使用标准CSV解析，不执行外部notebook，也不复用其按逗号位置截断正文的读取方法。完整竞赛数据本次未取得。

这{source['source_rows']}条评论对应{source['source_products']}商品，可用原始同商品评论对为{source['available_original_within_product_pairs']}，因此没有建立自然中文重复率或真实同商品效果评价。其星级标签不能作为重复标签。README里的720万条是上游数据介绍，不是本次实际取得的数量。

## 诊断组成

- 来源评论构造同源完整复制、外侧空格、追加标点、汉字间空格四类，共{int(pairs.pair_family.eq('source_transform').sum())}对。标签依据是明确的信息不变变换，单独报告检出，不用作真实用户重复评价。
- 独立编写46对控制样例：10对具体信息一致的改写、10对关键否定差异、10对数字/型号差异、10对情感同向但细节不同、6对通用短评不确定。标签是构造情形的设计判断，未经独立人类复核，不是自然标注金标准。
- 总计{len(pairs)}对；模型输入只有正文，不输入星级、来源类别或控制标签。没有用这些标签调阈值。
- 既有探索阈值：A=1、B=0.8、C/E=0.5、D=0.8。表格的正信号只表示超过阈值；实际decision始终review，不能称自动确认重复。

## 实际特征与输入处理

共{len(diagnostics)}个不同正文输入：
- 字符TF-IDF零向量：{int(diagnostics.lexical_zero.sum())}（{diagnostics.lexical_zero.mean()*100:.2f}%）。
- 中文中按现有空格规则被标为不足5词：{int(diagnostics.legacy_short.sum())}（{diagnostics.legacy_short.mean()*100:.2f}%）。
- 有汉字且没有英文字母/数字词的输入：{len(pure)}；其原英文正则词集合全部为空。
- 逐输入平均的编码器未知token比例：{diagnostics.unknown_fraction.mean()*100:.2f}%；超过256 word pieces而截断：{int(diagnostics.truncated.sum())}条。

原TF-IDF固定在英语Amazon训练语料上拟合；即使分析器按字符工作，也不能假定固定词表覆盖中文。word_set仅匹配a-z/0-9，不处理汉字；word_count以空格切分，中文不插空格的长句也可能被标为短评。中文句子是否加空格因此会影响旧短评门槛，这是实现限制，不是中文内容真的信息不足。

固定MiniLM模型卡标记language=en：[官方说明](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。中文可被分词/输出向量并不证明向量理解中文细节；未知token、共同字符和句型都需结合控制对看。以下误信号是此诊断集观察，不据其直接推断成因或自然总体精确率。

## 各组阈值信号

{table(scopes)}

tp/fn仅指明确的同源变换或构造正例；fp/tn仅指构造负例。不确定组不算二分类正确率。每类10或28对很小、且并非自然随机抽样，不能合并成中文真实准确率，也不能直接比较上游情感分类准确率。

## 可复现文件

运行 .venv/bin/python -m src.chinese_probe。
本地data/chinese_checks/保存来源notebook、实际样例、控制对、诊断评论对及固定来源清单。results/tables/chinese_probe_*.csv保存逐对五分数/信号、特征诊断与分组统计；results/tables/chinese_probe_receipt.json记录来源SHA、控制SHA、原模型锁SHA和输出SHA。results/figures/chinese_probe_signals.png展示分组检出/误信号率。原公开交接包SHA仍为{receipt['original_handoff_sha256']}。

## 后续适配

中文应用应单独设计汉字长度/分词、中文词面表示和保留否定/数字/型号的特征，并考虑经中文或多语言语料训练的语义编码器。随后用同商品真实中文评论对重新划分训练/验证/测试，建立与情感标签独立的重复标签，重新锁定阈值。不能只把旧词集合改成中文后继续复用原C/E分类器，因为输入特征分布已经改变；也不能让插入空格成为通过旧短评门槛的正式方案。

本次没有训练新中文模型或产生中文评分/排名结论。中文数据来源、诊断构造、阈值信号和正式识别效果分开报告。
"""
    sample=pairs[pairs.pair_id.eq("ctrl_011")].iloc[0]
    vectorizer=__import__("joblib").load(root/"models/lexical.joblib")
    projected=vectorizer.transform([sample.text_a,sample.text_b])
    feature_names=vectorizer.get_feature_names_out()
    retained=[feature_names[projected[i].indices].tolist() for i in range(2)]
    report+="\n\n## 构造案例的特征追踪\n\n"
    report+=f"ctrl_011：{sample.text_a} / {sample.text_b}。这对核心结论相反的输入在现有TF-IDF里仅保留{retained[0]}和{retained[1]}，两个稀疏向量相同；因此B=1不能解释为中文信息相同。这是可直接观察的投影缺失，不是根据相似度猜测其成因。案例为构造控制，不是两位真实用户的评论。\n"
    (root/"report/chinese_probe.md").write_text(report,encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups=[("exact_copy","Exact copy","positive_transform_detection"),("punctuation","Punctuation","positive_transform_detection"),
        ("cjk_spacing","CJK spacing","positive_transform_detection"),("paraphrase","Paraphrase","positive_transform_detection"),
        ("negation","Negation","negative_control_false_signal"),("numeric_model","Number/model","negative_control_false_signal"),
        ("same_sentiment_different_information","Same sentiment","negative_control_false_signal")]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for ax,subset,title in [(axes[0],groups[:4],"Positive control signal rate"),(axes[1],groups[4:],"Negative control false signal rate")]:
        x=np.arange(len(subset));width=.15
        for k,m in enumerate(["A","B","C","D","E"]):
            values=[summary.loc[summary.group.eq(g)&summary.method.eq(m),metric].iloc[0] for g,_,metric in subset]
            ax.bar(x+(k-2)*width,values,width,label=m)
        ax.set(xticks=x,xticklabels=[n for _,n,_ in subset],ylim=(0,1.05),ylabel="Fraction of control pairs",title=title)
        ax.tick_params(axis="x",rotation=20)
    axes[0].legend(ncol=5,loc="upper center")
    fig.savefig(root/"results/figures/chinese_probe_signals.png",dpi=160);plt.close(fig)

def run(root=ROOT):
    lock=validate_lock(root)
    baseline={p:sha_file(root/p) for p in ["models/model_lock.json","data/processed/reviews.csv",
        "data/annotation/labels_test.csv","deliverables/team_handoff.zip"]}
    examples,source=source_examples(root)
    control_path=root/"data/chinese_checks/controls.csv"
    from .chinese_controls import CHINESE_CONTROLS, ENGLISH_CONTROLS
    english_path=root/"data/chinese_checks/controls_en.csv"
    if not control_path.exists():write_csv(control_path,pd.DataFrame(CHINESE_CONTROLS))
    if not english_path.exists():write_csv(english_path,pd.DataFrame(ENGLISH_CONTROLS))
    controls=pd.read_csv(control_path,keep_default_na=False)
    if not controls.equals(pd.DataFrame(CHINESE_CONTROLS)):
        raise ValueError("Chinese controls changed relative to frozen authored fixtures.")
    if not pd.read_csv(english_path,keep_default_na=False).equals(pd.DataFrame(ENGLISH_CONTROLS)):
        raise ValueError("Matched English controls changed; create a separate diagnostic version.")
    pairs=make_pairs(examples,controls)
    predictor=PortablePredictor(root)
    scores=predictor.predict_pairs(list(zip(pairs.text_a,pairs.text_b)))
    if len(scores)!=len(pairs)*5 or not scores.decision.eq("review").all():
        raise ValueError("Frozen inference shape/review handling mismatch.")
    # _score emits pair-major/method-major rows, map diagnostics back by fixed input order.
    scores["diagnostic_pair_id"]=np.repeat(pairs.pair_id.to_numpy(),5)
    signal=scores.pivot(index="diagnostic_pair_id",columns="method",values="score").reindex(pairs.pair_id).reset_index(drop=True)
    for method in ["A","B","C","D","E"]:
        signal[method+"_signal"]=signal[method].ge(lock["methods"][method]["review_threshold"])
    pair_results=pd.concat([pairs,signal],axis=1)
    pair_results["legacy_short_pair"]=pairs.text_a.map(word_count).lt(5)|pairs.text_b.map(word_count).lt(5)
    pair_results["operational_decision"]="review"
    texts=list(dict.fromkeys(pairs.text_a.tolist()+pairs.text_b.tolist()))
    matrix=predictor.vectorizer.transform(texts)
    tokenizer=predictor.encoder.tokenizer
    diagnostics=[]
    for i,text in enumerate(texts):
        token_ids=tokenizer(text,truncation=False,add_special_tokens=False,verbose=False)["input_ids"]
        unknown=sum(t==tokenizer.unk_token_id for t in token_ids)
        diagnostics.append({"text_hash":hashlib.sha256(text.encode()).hexdigest(),"text":text,"chars":len(text),
            "han_chars":len(HAN.findall(text)),"legacy_words":word_count(text),"legacy_short":word_count(text)<5,
            "ascii_word_count":len(word_set(text)),"lexical_nonzero":int(matrix[i].nnz),"lexical_zero":matrix[i].nnz==0,
            "token_count":len(token_ids),"unknown_tokens":unknown,"unknown_fraction":unknown/len(token_ids) if token_ids else 0.,
            "truncated":len(token_ids)+2>256})
    diagnostics=pd.DataFrame(diagnostics)
    summary=metrics(pairs,signal)
    out=root/"results/tables"
    write_csv(out/"chinese_probe_pairs.csv",pair_results)
    write_csv(out/"chinese_probe_feature_diagnostics.csv",diagnostics)
    write_csv(out/"chinese_probe_summary.csv",summary)
    write_json(root/"data/chinese_checks/source_manifest.json",source)
    write_csv(root/"data/chinese_checks/diagnostic_pairs.csv",pairs)
    bilingual,language_diagnostics=matched_language_probe(root,controls,predictor,english_path)
    receipt={"matched_english_pairs":46,"execution_device":str(predictor.encoder.device),"english_control_sha256":sha_file(english_path),"controlled_fixture_sha256":sha_file(root/"src/chinese_controls.py"),"source":source,"control_sha256":sha_file(control_path),"pairs":len(pairs),
        "original_model_lock_sha256":baseline["models/model_lock.json"],
        "original_handoff_sha256":baseline["deliverables/team_handoff.zip"],"no_training_or_threshold_tuning":True,
        "source_sha256":sha_file(root/"src/chinese_probe.py"),"natural_chinese_accuracy_estimated":False,
        "full_chinese_dataset_available":False,"file_sha256":{n:sha_file(out/n) for n in
            ["chinese_probe_pairs.csv","chinese_probe_feature_diagnostics.csv","chinese_probe_summary.csv", "chinese_probe_bilingual_pairs.csv","chinese_probe_bilingual_summary.csv","chinese_probe_language_diagnostics.csv"]}}
    report(root,source,pairs,summary,diagnostics,receipt)
    append_language_report(root,bilingual,language_diagnostics)
    for p,digest in baseline.items():
        if sha_file(root/p)!=digest:raise ValueError(f"Frozen artifact changed during diagnostic: {p}")
    write_json(out/"chinese_probe_receipt.json",receipt)
    print(json.dumps({"source":source,"control_pairs":len(controls),"all_pairs":len(pairs),
        "unique_inputs":len(texts),"zero_vector_inputs":int(diagnostics.lexical_zero.sum()),
        "short_flag_inputs":int(diagnostics.legacy_short.sum()),"full_dataset_used":False},ensure_ascii=False,indent=2))
    print(summary.loc[summary.group.isin(["paraphrase","negation","numeric_model"]) & summary.method.isin(["C","D","E"]), ["group","method","tp","fp","fn","tn"]].to_string(index=False))
    return receipt

def matched_language_probe(root,controls,predictor,english_path):
    english=pd.read_csv(english_path,keep_default_na=False)
    if not controls[["pair_id","topic","group","label"]].equals(english[["pair_id","topic","group","label"]]):
        raise ValueError("Bilingual cases/labels must align one-to-one.")
    summaries=[];outputs=[];languages=[]
    for name,frame in [("zh",controls),("en",english)]:
        output=predictor.predict_pairs(list(zip(frame.text_a,frame.text_b)))
        output["case_id"]=np.repeat(frame.pair_id.to_numpy(),5)
        scores=output.pivot(index="case_id",columns="method",values="score").reindex(frame.pair_id).reset_index(drop=True)
        for m in ["A","B","C","D","E"]:
            scores[m+"_signal"]=scores[m].ge(predictor.manifest["methods"][m]["review_threshold"])
        evaluation=frame.copy()
        evaluation["expected_binary"]=frame.label.map({"near":1,"distinct":0})
        values=metrics(evaluation,scores);values["language"]=name;summaries.append(values)
        rows=pd.concat([evaluation,scores],axis=1);rows["language"]=name;outputs.append(rows)
        texts=list(dict.fromkeys(frame.text_a.tolist()+frame.text_b.tolist()))
        lexical=predictor.vectorizer.transform(texts)
        ratios=[];token_count=0;unknown_count=0
        for t in texts:
            tokens=predictor.encoder.tokenizer(t,truncation=False,add_special_tokens=False,verbose=False)["input_ids"]
            unknown=sum(v==predictor.encoder.tokenizer.unk_token_id for v in tokens)
            ratios.append(unknown/len(tokens) if tokens else 0.)
            token_count+=len(tokens);unknown_count+=unknown
        languages.append({"language":name,"control_pairs":len(frame),"unique_inputs":len(texts),
            "zero_lexical_inputs":int(sum(lexical[i].nnz==0 for i in range(len(texts)))),
            "legacy_short_inputs":sum(word_count(t)<5 for t in texts),
            "empty_ascii_word_sets":sum(not word_set(t) for t in texts),
            "mean_input_unknown_fraction":float(np.mean(ratios)),
            "pooled_token_unknown_fraction":unknown_count/token_count})
    bilingual=pd.concat(summaries,ignore_index=True);diagnostics=pd.DataFrame(languages)
    out=root/"results/tables"
    write_csv(out/"chinese_probe_bilingual_pairs.csv",pd.concat(outputs,ignore_index=True))
    write_csv(out/"chinese_probe_bilingual_summary.csv",bilingual)
    write_csv(out/"chinese_probe_language_diagnostics.csv",diagnostics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    x=np.arange(3);w=.35
    for offset,language in [(-.5,"zh"),(.5,"en")]:
        row=diagnostics[diagnostics.language.eq(language)].iloc[0]
        values=[row.zero_lexical_inputs/row.unique_inputs,row.legacy_short_inputs/row.unique_inputs,
                row.mean_input_unknown_fraction]
        bars=ax.bar(x+offset*w,np.array(values)*100,w,label=language)
        ax.bar_label(bars,fmt="%.1f%%")
    ax.set(xticks=x,xticklabels=["Zero TF-IDF vector","Legacy short flag","Mean unknown tokens"],
           ylim=(0,105),ylabel="Percent",title="Matched 46-case language diagnostics")
    ax.legend()
    fig.savefig(root/"results/figures/chinese_probe_language_coverage.png",dpi=160);plt.close(fig)
    return bilingual,diagnostics

def append_language_report(root,bilingual,diagnostics):
    def table(frame):
        return "| "+" | ".join(frame.columns)+" |\n| "+" | ".join(["---"]*len(frame.columns))+" |\n"+"\n".join(
            "| "+" | ".join(str(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))
    counts=bilingual[["language","group","method","pairs","tp","fp","fn","tn"]]
    path=root/"report/chinese_probe.md"
    with path.open("a",encoding="utf-8") as f:
        f.write("\n\n## 同案例的中英文对照\n\n")
        f.write("对全部46个构造案例补充忠实英文翻译，保持标签、否定、数字、型号和信息差异不变；未挑选有利案例或调整阈值。中文结果查看后才增加此对照，因此属于探索性诊断，不是事先设计的正式跨语言效果检验。两种语言各46对，共92个控制对运行。\n\n")
        f.write("A精确匹配，B字符TF-IDF，C词面逻辑回归，D固定MiniLM语义相似度，E词面加语义逻辑回归。逐输入平均未知token比例与合并所有token计算的比例分别报告。\n\n")
        f.write(table(diagnostics.round(6)))
        f.write("\n\n"+table(counts))
        f.write("\n\n分组误信号需与语言特征覆盖一起解读；英语控制也可能出现否定或数字误信号，不能把全部错误归因于中文。此对照没有替代同商品真实中文标注。构造中英文案例随src/chinese_controls.py保存，运行脚本可重建本地CSV，公开来源原文及结果明细仍按忽略规则留本地。\n")

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT)
    run(p.parse_args().root)
if __name__=="__main__":main()
