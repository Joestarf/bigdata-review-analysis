"""Build aggregate figures and a reproducible report after the final evaluation."""
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT,read_json,write_csv,write_json,sha_file
from .training import METHODS,validate_lock
from .evaluation import validate_evaluation
from .inference import validate_inference

def markdown_table(frame):
    columns=frame.columns.tolist()
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for values in frame.itertuples(index=False,name=None):
        lines.append("| "+" | ".join(str(v) for v in values)+" |")
    return "\n".join(lines)

def percent(value):
    return f"{float(value)*100:.2f}%"

def build_report(root:Path=ROOT):
    lock=validate_lock(root);seal=validate_evaluation(root)
    tables=root/"results/tables";figures=root/"results/figures";figures.mkdir(parents=True,exist_ok=True)
    test=pd.read_csv(tables/"test_comparison.csv")
    val=pd.read_csv(tables/"validation_comparison.csv")
    main_val=val[val.cohort.isna()] if "cohort" in val else val
    ci=pd.read_csv(tables/"test_product_bootstrap.csv").set_index("method")
    audit=read_json(tables/"redundancy_audit.json")
    emeta=read_json(root/"models/embedding_manifest.json")
    inference=validate_inference(root)
    bench=pd.read_csv(tables/"cached_matrix_benchmark.csv")
    labels=[]
    for split in ["train","validation","test"]:
        sheet=pd.read_csv(root/f"data/annotation/labels_{split}.csv",keep_default_na=False)
        for label in ["exact","near","distinct","uncertain"]:
            labels.append({"split":split,"label":label,"n":int(sheet.label.eq(label).sum())})
    counts=pd.DataFrame(labels);write_csv(tables/"final_annotation_counts.csv",counts)
    rows=[]
    for r in test.itertuples():
        interval=ci.loc[r.method]
        precision=percent(r.precision) if bool(r.precision_defined) else "undefined (no positive prediction)"
        recall=percent(r.recall) if bool(r.recall_defined) else "undefined (no confirmed positive)"
        rows.append({"method":r.method,"threshold":f"{r.threshold:.6f}","n":r.n,"TP":r.tp,"FP":r.fp,
                     "FN":r.fn,"TN":r.tn,"precision":precision,"recall":recall,"F1":percent(r.f1),
                     "precision bound":f"{percent(r.precision_lower)} to {percent(r.precision_upper)}" if r.precision_bound_defined else "undefined",
                     "unresolved positive":r.unresolved_positive})
    display=pd.DataFrame(rows)
    x=np.arange(len(test));width=.25
    fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
    for offset,column in [(-1,"precision"),(0,"recall"),(1,"f1")]:
        values=test[column].where(test.precision_defined) if column=="precision" else test[column]
        ax.bar(x+offset*width,values,width,label=column.title())
        if column=="precision":
            for position,defined in zip(x,test.precision_defined):
                if not defined:ax.text(position-width,.02,"N/A",ha="center",fontsize=8)
    ax.set(xticks=x,xticklabels=test.method,ylim=(0,1.05),ylabel="Score on definite AI-reviewed labels",
           title="Held-out ASIN benchmark (uncertain reported separately)")
    ax.legend();fig.savefig(figures/"test_model_comparison.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(14,3.2),layout="constrained")
    vmax=max(test[["tn","fp","fn","tp"]].to_numpy().ravel())
    for ax,r in zip(axes,test.itertuples()):
        matrix=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int)
        ax.imshow(matrix,cmap="Blues",vmin=0,vmax=vmax)
        for (i,j),n in np.ndenumerate(matrix):ax.text(j,i,str(n),ha="center",va="center")
        ax.set(title=r.method,xticks=[0,1],xticklabels=["Distinct","Duplicate"],
               yticks=[0,1],yticklabels=["Distinct","Duplicate"],xlabel="Candidate prediction")
    axes[0].set_ylabel("AI reference label")
    fig.savefig(figures/"test_confusion_matrices.png",dpi=160);plt.close(fig)
    pivot=counts.pivot(index="split",columns="label",values="n").reindex(["train","validation","test"])
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    pivot[["exact","near","distinct","uncertain"]].plot.bar(stacked=True,ax=ax)
    ax.set(xlabel="",ylabel="Pairs",title="Fixed 6,000-pair AI annotation");ax.tick_params(axis="x",rotation=0)
    fig.savefig(figures/"final_label_composition.png",dpi=160);plt.close(fig)
    synthetic=pd.read_csv(tables/"synthetic_test_detection.csv")
    cases=pd.read_csv(tables/"error_cases.csv",keep_default_na=False)
    selected=lock["selected_method"] or "none"
    scope="验证选型目标没有方法达标；五方法保留为探索性候选。" if selected=="none" else f"在固定富集验证集上选定{selected}；不能据此宣称自然场景精确率达到95%。"
    e=test.set_index("method").loc["E"];c=test.set_index("method").loc["C"]
    delta=f"E相对C的测试precision差{(e.precision-c.precision)*100:+.2f}个百分点、recall差{(e.recall-c.recall)*100:+.2f}个百分点、F1差{(e.f1-c.f1)*100:+.2f}个百分点。这是观察差异，不是显著性结论，也不能据测试差异重选模型。"
    interval_rows=[]
    for m in METHODS:
        r=ci.loc[m]
        interval_rows.append({"method":m,"products":int(r.get("products",0)),"status":r.status,
             "precision 95% CI":f"{percent(r.precision_low)} to {percent(r.precision_high)}" if r.status=="ok" and r.get("precision_status")=="ok" else "undefined/unavailable",
             "recall 95% CI":f"{percent(r.recall_low)} to {percent(r.recall_high)}" if r.status=="ok" else "unavailable",
             "F1 95% CI":f"{percent(r.f1_low)} to {percent(r.f1_high)}" if r.status=="ok" else "unavailable"})
    benchmark_display=bench[["products","reviews","pairs","seconds","peak_rss_mb","gpu_peak_mb"]].copy()
    benchmark_display[["seconds","peak_rss_mb","gpu_peak_mb"]]=benchmark_display[["seconds","peak_rss_mb","gpu_peak_mb"]].round(3)
    try:git_tip=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    except subprocess.SubprocessError:git_tip="unavailable"
    narrative=f"""# 组员一真实实验与交接报告

## 结论及范围

已完成固定数据审计、6,000对真实评论的AI辅助标注、2,000个合成训练正例与100个变换审阅、A—E五方法训练/对照、参数锁定后一次最终测试，以及全部同ASIN对推断。{scope}

标签未经独立人类金标准复核；候选按相似度、随机和短文配额富集。评价只说明这一固定AI标注集上的表现，不能估计自然重复率或判断水军。全量输出全部review，不直接赋予降权权重，不生成传递重复组。

## 数据与版本

- 固定样本{audit['rows']:,}行、{audit['products']:,}个ASIN、{audit['users']:,}个用户。
- SHA-256：{lock['dataset_version']}。
- 五字段额外副本{audit['record_extra_rows']}行；同商品文本重复{audit['within_product_groups']}组、额外{audit['within_product_extra_rows']}行；跨商品文本重复{audit['cross_product_groups']}组。
- 全样本措辞重复额外{audit['global_text_extra_rows']}行，与同商品口径不同且有重叠，不能相加。
- 有界扫描100万、局部缓冲打乱、按观测频次挑1,000商品、每商品最多100条；不是类别总体均匀随机抽样。未保留parent_asin，原下载revision未记录。
- 改进清洗与旧版本12条差异留表；原五字段CSV未改写。异常排除及全部字段见data_audit.md。

## 标签与训练

{markdown_table(counts)}

同商品分组划分700/150/150个ASIN；跨集合相同规范化正文进入公共模板池。保留原600与1200批次，最终批次固定在模型拟合之前，不依据测试成绩补标。训练只用确定AI标签{lock['training_manual_rows']}对；uncertain不强行转为负例。

合成训练{lock['synthetic_rows']}对，总权重{lock['synthetic_total_weight']:.2f}，真实确定训练权重{lock['manual_total_weight']}；合成占合并权重20%。复制、空格、大小写、标点和受限拼写变动各审20例，来源均为训练评论。

TF-IDF仅拟合训练记录，char_wb 3—5 gram/min_df2/max100,000。C/E数值特征标准化后L2逻辑回归，只在验证集比较C=0.1/1/10；实际配置见models/model_lock.json。模型只用正文，不输入商品、用户、评分、时间或标签来源。

## 验证选型及测试

资格要求完整验证批次的保守precision下界≥95%，且至少30个允许正预测；不确定正预测纳入分母。验证结果见validation_comparison.csv，包括原150、增补300、新增550三个批次。选型结果：{selected}。不达标配置按已锁定的探索阈值评价，不能解释为自动识别部署。

最终测试共{seal['test_rows']}对；确定二分类{seal['effective_binary_rows']}对、不确定{seal['uncertain_rows']}对。阈值及模型在读取测试标签前锁定，之后没有据测试调参。

{markdown_table(display)}

无正预测时precision及其区间无定义，图中标为N/A；CSV数值0需结合precision_defined=False解读，不能解释为产生了错误正预测。精确匹配A只覆盖相同措辞，监督池移除公共模板和记录副本后，自然完全匹配正例可能很少。

### 商品bootstrap，1,000次、seed42

{markdown_table(pd.DataFrame(interval_rows))}

重采样单位为ASIN，有效重采样不足时标为unavailable。区间反映该富集AI基准的不确定性，未证明自然场景95%精确率。

### 语义增量

{delta}

MiniLM固定revision {emeta['model_revision']}，384维、float32归一化、最长256 word pieces；{emeta['truncated_rows']}条评论被截断。长文本分组、exact/near/distinct/short及各批次结果见test_breakdowns.csv。编码器未微调。共同商品词、相似情绪与否定差异仍需看正文判断。

## 错误与合成检查

error_cases.csv提供{len(cases)}个错误/不确定案例及正文、参考理由、五方法分数和下游风险；各类最多10个、不足全部展示。具体原因复核记录见error_cases_review.csv（若存在）。全量待复核不会自动误降权。

合成测试只使用测试集合原评论，单独报告已知复制/扰动检出率，precision/F1因只有正例而无定义。详见synthetic_test_detection.csv，共{len(synthetic)}个方法/变换组合；不替代真实评价。

## 性能与全量结果

模型首次加载/下载{emeta['model_loading_or_download_seconds']:.2f}秒，独立正文GPU编码{emeta['encoding']['seconds']:.2f}秒，torch峰值分配显存{emeta['encoding']['gpu_peak_mb']:.2f}MiB。缓存推断使用CPU词面/分类及现有向量，不重复下载或编码。

{markdown_table(benchmark_display)}

表为完整五方法的特征、评分及决策，不含CSV写出。RSS为单进程绝对峰值，显存为本过程torch分配值，不代表整机占用。

全部{inference['products']}个商品、{inference['pairs']:,}对、{inference['method_rows']:,}方法行已写至results/predictions/pair_scores.csv.gz。含压缩写出的耗时{inference['performance']['seconds']:.2f}秒、峰值RSS {inference['performance']['peak_rss_mb']:.2f}MiB。输出SHA及模型锁/评价SHA见manifest.json。阈值候选数量只是模型信号，不能作为已确认重复数或自然重复率。

固定10商品交接位于results/handoff/，正文及review_id可回连；predict_pairs与score_product具体调用见runbook.md及handoff_contract.md。原五字段接口保持不变。每对五行包含pair_id、asin、两个review_id、method、score、decision、reason。余弦和未校准分类器分数不能称真实重复概率。

## 验收、独立审核与Git

独立GPT-6.1 Sol三轮提出九项实质问题，已经修正并记录third_party_review.md。工程测试及实际全量验收见implementation_status.md与acceptance_report.json；夹具成绩不当真实模型成绩。

Git历史误上传已获用户明确同意后清理，旧历史备份在仓库外，本地数据完整保留。当前只提交代码、说明、安全图和无输出notebook模板；data/、模型、CSV/JSON结果、带输出notebook保持本地。报告生成时Git代码提交：{git_tip}。

## 文件与后续建议

完整路径/SHA清单位于artifact_inventory.csv。主要结果：test_comparison.csv、test_breakdowns.csv、test_product_bootstrap.csv、test_pair_predictions.csv、synthetic_test_detection.csv、error_cases.csv；主要图：test_model_comparison.png、test_confusion_matrices.png、final_label_composition.png、cached_pair_performance.png。

组员二接入时先保留review标记，另行讨论降权政策。若要宣称自然场景可自动降权，后续需要独立人类双人标注和自然抽样的验证集，再固定新的实验版本；当前结论足以交付方法比较与复核接口。
"""
    (root/"report/experiment_results.md").write_text(narrative,encoding="utf-8")
    inventory=[]
    for directory in ["data","models","results","report","notebooks"]:
        for path in sorted((root/directory).rglob("*")):
            if path.is_file() and path.suffix in [".csv",".json",".joblib",".npy",".gz",".md",".png",".ipynb"] and path.name!="artifact_inventory.csv":
                inventory.append({"path":str(path.resolve()),"relative_path":str(path.relative_to(root)),
                                  "bytes":path.stat().st_size,"sha256":sha_file(path)})
    write_csv(tables/"artifact_inventory.csv",pd.DataFrame(inventory))
    write_json(tables/"report_receipt.json",{"model_lock_sha256":sha_file(root/"models/model_lock.json"),
               "final_evaluation_sha256":sha_file(tables/"final_evaluation.json"),
               "report_sha256":sha_file(root/"report/experiment_results.md"),"git_tip_at_generation":git_tip})
    print(f"Report and figures created; {len(inventory)} artifacts inventoried.")

