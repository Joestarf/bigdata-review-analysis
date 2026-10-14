# 组内交接说明：评论冗余识别与评分影响分析

这份说明是组员接手本项目的入口。研究对象是**评论文字的信息冗余**，相似措辞不代表虚假评论，也不判断评论者是否为“水军”。

**直接做评分、排名分析：下载评分与数据包。需要识别新的中文、英文或混杂评论：再下载多语言模型包。需要原始正文或重新训练：联系数据负责人索取本地研究包。**

## 1. 下载与分工

| 文件 | 下载入口 | 包含内容 | 适合的工作 |
| --- | --- | --- | --- |
| 评分与数据包 team_handoff.zip，约5.46 MiB | [直接下载](https://raw.githubusercontent.com/Joestarf/bigdata-review-analysis/main/deliverables/team_handoff.zip) · [GitHub文件页](../deliverables/team_handoff.zip) | 原英文识别模型、90,750条评论的编号/星级/六策略权重、1,000商品的评分和样本排名 | 评分与排名影响分析、统计图表、报告 |
| 多语言模型包 multilingual_model.zip，约1 MB | [直接下载](https://raw.githubusercontent.com/Joestarf/bigdata-review-analysis/main/deliverables/multilingual_model.zip) · [GitHub文件页](../deliverables/multilingual_model.zip) | 五方法推断、暂定重复程度、模型文件及运行说明 | 新评论识别、中文/英文/混杂输入、案例分析 |
| 项目源码与报告 | [GitHub仓库](https://github.com/Joestarf/bigdata-review-analysis) | 工程代码、方法说明、聚合结果和图表 | 理解方法、撰写报告、后续开发 |

两个压缩包分别解压到不同目录，避免覆盖同名src和models目录。只查看评分CSV或画图不需要下载语义编码器，也不需要GPU。

## 2. 评分分析组员先看哪些数据

解压team_handoff.zip后，优先查看以下文件。这里的路径是**压缩包内部路径**：

| 文件 | 行数或范围 | 用法 |
| --- | --- | --- |
| data/product_final_ratings.csv | 1,000个商品 | 默认折扣策略的商品评分、原评分、变化和样本名次 |
| data/product_ratings.csv | 1,000商品 × 6策略，共6,000行 | 比较不降权、只去记录副本、同措辞组计一次及三种折扣强度 |
| data/review_weights.csv | 90,750条评论 | 每条评论的原星级、编号、直接证据和六种权重 |
| data/candidate_pair_signals.csv | 原评分实验保留的候选对 | 查看五种方法的信号与复核标记；不是所有评论对 |
| data/weighting_policy.json | 策略配置 | 核对折扣参数 |
| data/test_comparison.csv | 原英文模型评价 | 查看模型效果与未达标情况 |
| data/SOURCE.md、bundle_manifest.json | 来源与校验说明 | 确认版本和文件一致性 |

商品结果表中的主要字段：

| 字段 | 含义 |
| --- | --- |
| asin | 商品标识；连接商品表使用此键 |
| scenario | 策略名；比较策略时用asin与scenario共同定位一行 |
| review_count | 当前样本内评论数，不是商品全量评论数 |
| baseline_rating / final_rating | 原样本平均星级 / 加权星级 |
| rating_delta | final_rating减baseline_rating；正值表示提高 |
| baseline_rank / sample_rank | 在当前1,000商品中的原名次 / 新名次 |
| rank_change | baseline_rank减sample_rank；正值表示上升 |
| effective_weight | 评论权重之和，不是独立用户数或独立信息数 |

六种scenario分别是baseline、record_only、wording_group、graded_conservative、graded_default、graded_strong。默认展示graded_default，同时保留baseline与record_only等对照。

**评论编号review_id用于关联固定样本。公开包没有评论正文、原用户ID或时间戳；需要正文案例时索取本地研究包。不要用CSV当前行号代替review_id。**

## 3. 折扣与最终评分如何解释

评分计算为：

**加权商品星级 = Σ(评论权重 × 原星级) / Σ评论权重。**

折扣改变评论的计入权重，原始1—5星不改写。重复好评与重复差评使用相同规则。

| 原英文评分实验的证据级别 | 默认权重 |
| --- | ---: |
| 没有词面信号、短评或只有语义相近 | 1.00 |
| 轻度候选 | 0.75 |
| 中度候选 | 0.50 |
| 同商品相同规范化措辞且满足长度条件 | 0.25 |
| 五字段完全相同记录的额外副本 | 0.00，每组保留首条 |

这些是**声明的实验政策**，不是从真实商品质量拟合出来的最优折扣。

原评分实验的“轻/中度候选”和新多语言接口的light/medium/high不是同一套定义。新程度接口尚未生成对应商品权重；不能直接把high替换成旧包的same_wording，也不能把新模型分数直接当权重。

已有实测结果：

| 策略 | 星级变化商品数 | 平均绝对星级变化，按全部1,000商品计算 | 最大绝对变化 | 样本名次变化商品数 |
| --- | ---: | ---: | ---: | ---: |
| 仅记录副本去重 record_only | 208 | 0.002120 | 0.047374 | 705 |
| 默认折扣 graded_default | 304 | 0.002631 | 0.047374 | 751 |
| 较强折扣 graded_strong | 304 | 0.003113 | 0.047374 | 810 |

相比record_only，默认文本候选折扣额外改变124个商品，平均绝对增量约0.000546星。整体星级影响较小，变化不能全部归因于近似重复模型。

名次只在本样本1,000个ASIN之间比较，按星级保留6位小数、同分采用最小名次。大量评分接近或并列，会使很小的评分变化造成较多名次变化。**不能据此声称降权后更接近真实商品质量排序，也不能称为Amazon官方评分或排名。**

## 4. 模型调用组员怎么开始

### 4.1 安装环境

原运行环境是Arch WSL、Python 3.14。推荐使用同版本Python和包内锁定依赖；GPU可选。

进入所选压缩包的解压根目录，确认这里能看到src、models和requirements-inference.txt。创建虚拟环境：

~~~bash
python -m venv .venv
~~~

Windows PowerShell安装依赖：

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
~~~

Linux/WSL安装依赖：

~~~bash
.venv/bin/python -m pip install -r requirements-inference.txt
~~~

下面示例中的python指该虚拟环境的解释器；可以替换为上面的完整解释器路径。

首次进行模型推断需要联网下载固定版本的预训练编码器，编码器权重不在ZIP内。原英文包使用all-MiniLM-L6-v2，多语言包使用paraphrase-multilingual-MiniLM-L12-v2。只读取现成评分CSV不需要这一步。

### 4.2 多语言评论对：CSV调用

在multilingual_model.zip的解压根目录准备UTF-8的pairs.csv，至少包含text_a、text_b；建议同时保留pair_id、asin、review_id_a、review_id_b：

~~~csv
pair_id,asin,review_id_a,review_id_b,text_a,text_b
example-1,EXAMPLE,r1,r2,上面写了500克我收到的只有350克左右,重量不够差距太大了500克送到只有350克
~~~

~~~bash
python -m src.multilingual_inference --input pairs.csv --output pair_scores.csv
python -m src.redundancy_degree --input pairs.csv --output pair_degrees.csv
~~~

pair_scores.csv每对输出A—E五行分数；pair_degrees.csv每对输出一行暂定关系与程度。正文必须非空且不超过10,000字符；同一次调用pair_id须唯一。CSV入口按最多1,000对分批处理。

### 4.3 多语言单商品：Python调用

在多语言包解压根目录运行：

~~~python
from src.redundancy_degree import DegreePredictor

predictor = DegreePredictor()
reviews = [
    {"review_id": "r1", "text": "上面写了500克，我收到的只有350克左右。"},
    {"review_id": "r2", "text": "重量不够差距太大了，500克送到只有350克。"},
]
result = predictor.score_product("EXAMPLE", reviews)
result.to_csv("product_degrees.csv", index=False, encoding="utf-8-sig")
~~~

输入须为同一商品，review_id为非空且唯一的字符串。一次单商品调用最多1,000条评论；评论对数量按n×(n−1)/2增长，建议先用少量评论接入。分组调用不同商品，保留asin、review_id_a和review_id_b。

### 4.4 原英文模型与评分重算

在team_handoff.zip的解压根目录，用包含asin、review_id、text的UTF-8评论CSV调用原英文模型：

~~~bash
python -m src.portable --bundle . --input new_reviews.csv --output new_pairs.csv
~~~

仅从公开包中的权重重新计算六策略评分，不需要原始评论正文：

~~~bash
python -m src.rating_adjustment --weights data/review_weights.csv --output recomputed_product_ratings.csv
~~~

原英文模型用于原英文实验；新的中文或混杂正文优先使用多语言包。两种模型的调用参数不同，以本节示例为准。

## 5. 输出怎么接入下游

五种方法分别是A精确匹配、B字符TF-IDF、C词面逻辑回归、D语义相似度、E词面加语义逻辑回归。相似度和分类器分数不是校准后的真实重复概率，也不能横向当作同一量纲。

多语言程度表主要字段：

| 字段 | 如何使用 |
| --- | --- |
| pair_id、asin、review_id_a、review_id_b | 回连输入评论，保留直接评论对关系 |
| estimated_is_duplicate | 暂定关系；light/medium/high为true，none为false |
| redundancy_degree | none、light、medium、high |
| confidence_band | 证据强度类别，不是概率 |
| evidence_insufficient | 是否仍缺乏足够证据 |
| decision、needs_review | 当前统一为review、true，保留复核要求 |
| reason、evidence_round_1/2/3 | 判断理由及措辞、具体事实、冻结模型的证据 |
| model_version、policy_version | 记录运行模型与程度规则版本 |

程度口径：

- 正常相似的质量、包装、物流好评：不重复。
- 共同一个具体事实，但另一条有新的主要信息：轻度。
- 主要内容明显重叠但仍有少量新信息：中度，当前为启发式。
- 主要具体事实基本相同，或有足够信息的正文几乎相同：高度。

none且evidence_insufficient=true表示**尚无确认重复的正证据**，不是已经证明不重复。不要仅凭estimated_is_duplicate=false把证据不足样本当可靠负例。A像B、B像C也不能自动推出A像C或把三者合并成重复组。

## 6. 结果与报告可以怎么写

已完成原英语90,750条/1,000商品的处理、6,000对参考标注任务，以及1,000商品内部4,146,325对的五方法推断。新增中文辅助样本11,663条/254商品及600对任务；新多语言模型用2,453对确定参考标签训练。语义编码器参数冻结，重新训练的是词面表示与C/E逻辑回归。

中文后验质量复核参考集有150对，其中114对确定标签、26对重复。以下为原阈值信号，不是附加保守筛选后的自动决策：

| 方法 | 真正例 | 误判为重复 | 精确率 | 召回率 |
| --- | ---: | ---: | ---: | ---: |
| 旧词面逻辑回归 | 4 | 1 | 80.0% | 15.4% |
| 新词面逻辑回归 | 14 | 2 | 87.5% | 53.8% |
| 新词面加语义逻辑回归 | 14 | 3 | 82.4% | 53.8% |
| 新纯语义方法 | 24 | 38 | 38.7% | 92.3% |

中文输入覆盖有所改善，当前对照中词面逻辑回归更合适；加入语义特征没有超过词面方法，纯语义高召回伴随较多误判。附加既定保守筛选后，新C/E只留下5个确定正例；筛选词表偏手机配件，对部分中文辅助领域过于保守。

6,600对候选已给出暂定程度：高度37、中度37、轻度10、无确认重复证据6,516。原不确定2,667对中1,922对仍低证据。三项程度定义案例全部通过，但没有独立程度评价集。

报告必须同时说明：

- 标签是辅助判断，尚未经独立人工金标准验证；中文表是后验标签质量复核比较，原封存结果另行保留，模型与阈值没有随复核调整。
- 当前没有方法达到预设95%精确率及足够预测数量等选型条件，不能宣称已实现可靠自动降权。
- 候选刻意富集相似内容，候选集重复比例不是自然评论重复率。
- 中文辅助领域与手机配件不同；构造的混杂控制不能替代自然混杂评论评价。
- 商品样本不是全类别均匀随机样本，每商品最多100条，缺少parent_asin；不代表全量评分或父商品层面独立。
- 加权结果是样本敏感性实验，不是商品质量真值；不判断用户真假。

详细依据：[英文实验结果](experiment_results.md) · [评分实验](rating_adjustment.md) · [多语言结果](multilingual_adaptation.md) · [程度口径](multilingual_degree_policy.md)。

## 7. 原始数据、标注与训练复现

公开包不含原始正文、用户ID、时间戳、标签正文、向量缓存或完整评论对输出。需要这些材料时，联系数据负责人，私下获取：

| 本地研究包 | 内容与用途 |
| --- | --- |
| results/handoff/team_research_private.zip，约160 MiB | 原英语五字段数据、索引、标签、模型及缓存；原英语实验复现 |
| results/handoff/multilingual_research_private.zip，约7.4 MiB | 新中文样本、候选、标签草稿及复核记录、模型和结果；多语言补充复现 |

两个研究包在项目负责人的Arch WSL工程中：/home/joestar/code/bigdata-review-analysis/。没有公开下载链接。需要完整多语言训练复现时，取得**两个研究包及项目源码**；中文补充包依赖原英语研究材料。不要覆盖原始样本、原标签或封存测试，也不要用现有测试集反复调阈值再当独立测试。

多语言公开ZIP中的训练脚本随版本保存，但该ZIP缺少完整训练输入，不能仅靠公开ZIP重新训练。完整流程见[运行说明](runbook.md)和[多语言报告](multilingual_adaptation.md)。

## 8. 常见接入问题

| 问题 | 处理 |
| --- | --- |
| 找不到src模块或模型清单 | 进入相应压缩包的解压根目录；检查没有多套一层目录 |
| 模型文件SHA或运行代码版本不匹配 | 使用同一个完整ZIP内的代码与模型；不要混合两个包或其他版本 |
| 首次编码器下载失败 | 检查到Hugging Face的网络连接，或事先缓存清单指定的确切模型版本 |
| joblib/sklearn加载不兼容 | 优先使用原Python 3.14及requirements-inference.txt锁定版本；保留实际错误说明 |
| 没有GPU | 使用CPU；计算速度会不同，不影响只读取CSV分析 |
| CSV字段缺失、空正文、重复编号 | 对照第4节修正输入；保留同商品范围和有效编号 |
| 想展示评论案例但公开CSV没有正文 | 索取本地研究包，按review_id回连；不要猜测正文 |
| 分数高但显示review | 当前是设计行为；分数高不等于已验证的自动重复标签 |

只从可信项目来源取得并加载joblib模型。包内manifest校验配套文件；原评分包另附[ZIP校验文件](../deliverables/team_handoff.sha256)。

## 9. 本次公开包校验值

以下校验值对应本说明整理时的交接包。后续更换模型包时应同步更新说明。

~~~text
team_handoff.zip
fef2dcb66c24517c8d28ee2607c9cd7dc211af1a17a8c68e178d4dc24c03463c

multilingual_model.zip
c9fa82c977311a196c6a28d704faa849d7a89b3e78fa53b8e170b62c765f67bb
~~~

## 10. 给组员的接手顺序

1. 先读本说明，按自己的任务下载一个或两个公开ZIP，并分别解压。
2. 评分分析先打开product_final_ratings.csv和product_ratings.csv，保留baseline及record_only对照。
3. 模型接入按第4节用示例输入跑通，再替换为自己的同商品评论。
4. 写报告时把第6节的结果范围和限制一起说明。
5. 需要正文案例或重训，再向数据负责人索取本地研究包。

本次模型与结果对应源码提交74f6a89；本说明的新增不改变已有数据、模型、阈值或评分结果。
