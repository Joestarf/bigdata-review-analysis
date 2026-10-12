# Duplicate Review Detection and Ranking Impact Analysis

## Course
Big Data Introduction

## Topic
Detecting redundant product reviews and analyzing their influence on product rating and ranking.

## Dataset
Amazon Reviews 2023

## Methods

- Data preprocessing
- Text similarity analysis
- Duplicate review detection
- Review weighting
- Rating and ranking simulation

## Structure

- src: source code
- notebooks: experiments
- results: outputs
- report: documents

## Data exploration pipeline

The initial study sample uses McAuley-Lab/Amazon-Reviews-2023, category Cell_Phones_and_Accessories. The category source is about 20.8 million ratings, 11.6 million reviewers, and 1.3 million products according to the dataset card. The code reads a bounded, shuffled stream from the Hugging Face repository; it does not save the full category file.

From the project root in Arch WSL, run:

~~~bash
.venv/bin/python src/download_amazon.py --max-scan 1000000 --target-products 1000 --per-product-cap 100
.venv/bin/python src/preprocess.py
.venv/bin/python src/explore_data.py
~~~

The current run scanned 1,000,000 streamed records and produced 90,750 processed reviews from 1,000 products and 71,133 users. The downloaded sample selects products with the most observations in the bounded stream and caps each product at 100 reviews. This gives a practical development dataset, but it is not a representative probability sample and its review counts should not be interpreted as full product totals when drawing ranking conclusions.

preprocess.py normalizes whitespace and HTML entities, removes malformed rows and text longer than 10,000 characters, and preserves duplicate reviews for analysis. The output schema is asin,user_id,rating,text,timestamp; timestamp remains the source Unix timestamp in milliseconds.

The exact-text repeat check case-folds text and collapses whitespace. It measures repeated wording, not reviewer intent, authenticity, or coordination.

Generated artifacts:

- data/processed/reviews.csv
- results/tables/data_summary.json
- results/tables/source_fields.csv
- results/tables/rating_distribution.csv
- results/tables/product_review_counts.csv
- results/figures/star_distribution.png
- results/figures/review_length_distribution.png
- results/figures/top_product_review_counts.png

Dataset source: [Amazon Reviews 2023 on Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).


## 组员一：重复识别实验

当前固定样本90,750条/1,000ASIN。评论对基准固定为6,000对（训练4000、验证1000、测试1000；保留原600与1200批次），采用按商品划分和正文隔离。标注来源与适用范围见标注指南；全量交接保持待复核。

实测结果见 [实验与交接报告](report/experiment_results.md)，完成状态见 [实现状态](report/implementation_status.md)，复现及接口见 [运行说明](report/runbook.md)，标注定义见 [标注指南](report/annotation_guide.md)。本地审计notebook已执行；仓库提供[无输出复现模板](notebooks/data_quality_audit_template.ipynb)。

~~~bash
cd /home/joestar/code/bigdata-review-analysis
.venv/bin/python -m src.study prepare
.venv/bin/python -m src.study explore-labels
.venv/bin/python -m src.study validate
.venv/bin/python -m src.study serve --port 8765
~~~

本地复核页面：http://localhost:8765 。稳定旁表位于data/processed/；标注位于data/annotation/；语义缓存位于models/；10商品交接预览位于results/handoff/。五字段reviews.csv保持原版本，改进清洗与旧版的12条正文差异已经单独记录。

6,000对参考标签及五方法训练、最终测试已经完成；没有方法达到预设验证选型目标。当前词面与词面加语义逻辑回归的测试结果一致，全部候选用于复核。详细指标、样本数和适用范围见实验报告。

模型代码仅使用正文特征，测试标签与训练隔离。冻结预训练MiniLM参数，不训练星级、情感或评论者身份模型，识别接口保持待复核，不生成传递重复组。独立的折扣模块提供可调整的评分影响实验，不能作为自动识别达标或官方评分的证据。


## 分级折扣与组内交接

默认评论权重：轻度候选0.75、中度候选0.50、相同措辞0.25、五字段额外记录副本0；短评/仅语义信号保持1。最终星级为Σ(权重×原星级)/Σ权重。另提供较轻、较重、仅记录去重及每措辞组计一次的对照。

实测默认策略改变304个商品的样本星级，平均绝对变化约0.00263星、最大约0.04737星。详细定义与结果见[评分折扣实验](report/rating_adjustment.md)。

组员可下载[交接包](deliverables/team_handoff.zip)，查看[包说明](deliverables/README.md)及[可移植模型接口](report/portable_model.md)。公开包包含模型、90,750条编号/星级/权重数据和1,000商品六策略结果；原正文字段、原用户标识及完整研究包保留本地。
