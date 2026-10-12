# 组员一真实实验与交接报告

## 结论及范围

已完成固定数据审计、6,000对真实评论的自动辅助标注、2,000个合成训练正例与100个变换审阅、A—E五方法训练/对照、参数锁定后一次最终测试，以及全部同ASIN对推断。验证选型目标没有方法达标；五方法保留为探索性候选。

标签未经独立人类金标准复核；候选按相似度、随机和短文配额富集。评价只说明这一固定参考标签集上的表现，不能估计自然重复率或判断水军。全量输出全部review，不直接赋予降权权重，不生成传递重复组。

## 数据与版本

- 固定样本90,750行、1,000个ASIN、71,133个用户。
- SHA-256：8f1037a6987436a7c19b56766c42969526374145b8d02dde02caa5b1b750f0ec。
- 五字段额外副本231行；同商品文本重复594组、额外632行；跨商品文本重复1075组。
- 全样本措辞重复额外6855行，与同商品口径不同且有重叠，不能相加。
- 有界扫描100万、局部缓冲打乱、按观测频次挑1,000商品、每商品最多100条；不是类别总体均匀随机抽样。未保留parent_asin，原下载revision未记录。
- 改进清洗与旧版本12条差异留表；原五字段CSV未改写。异常排除及全部字段见data_audit.md。

## 方法定义

|方法|实现|分数含义|
|---|---|---|
|A|规范化精确匹配|相同正文标记0/1|
|B|字符TF-IDF阈值|字符3—5 gram余弦相似度|
|C|词面特征逻辑回归|未校准分类器输出|
|D|固定MiniLM语义向量阈值|语义向量余弦相似度|
|E|词面加语义特征逻辑回归|未校准分类器输出|

## 标签与训练

| split | label | n |
| --- | --- | --- |
| train | exact | 1 |
| train | near | 184 |
| train | distinct | 2073 |
| train | uncertain | 1742 |
| validation | exact | 0 |
| validation | near | 83 |
| validation | distinct | 467 |
| validation | uncertain | 450 |
| test | exact | 0 |
| test | near | 44 |
| test | distinct | 661 |
| test | uncertain | 295 |

同商品分组划分700/150/150个ASIN；跨集合相同规范化正文进入公共模板池。保留原600与1200批次，最终批次固定在模型拟合之前，不依据测试成绩补标。训练只用确定参考标签2258对；uncertain不强行转为负例。

合成训练2000对，总权重564.50，真实确定训练权重2258；合成占合并权重20%。复制、空格、大小写、标点和受限拼写变动各审20例，来源均为训练评论。

TF-IDF仅拟合训练记录，char_wb 3—5 gram/min_df2/max100,000。C/E数值特征标准化后L2逻辑回归，只在验证集比较C=0.1/1/10；实际配置见models/model_lock.json。模型只用正文，不输入商品、用户、评分、时间或标签来源。

## 验证选型及测试

资格要求完整验证批次的保守precision下界≥95%，且至少30个允许正预测；不确定正预测纳入分母。验证结果见validation_comparison.csv，包括原150、增补300、新增550三个批次。选型结果：none。不达标配置按已锁定的探索阈值评价，不能解释为自动识别部署。

最终测试共1000对；确定二分类705对、不确定295对。阈值及模型在读取测试标签前锁定，之后没有据测试调参。

| method | threshold | n | TP | FP | FN | TN | precision | recall | F1 | precision bound | unresolved positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1.000000 | 705 | 0 | 0 | 44 | 661 | undefined (no positive prediction) | 0.00% | 0.00% | undefined | 0 |
| B | 0.800000 | 705 | 0 | 1 | 44 | 660 | 0.00% | 0.00% | 0.00% | 0.00% to 0.00% | 0 |
| C | 0.500000 | 705 | 2 | 1 | 42 | 660 | 66.67% | 4.55% | 8.51% | 11.76% to 94.12% | 14 |
| D | 0.800000 | 705 | 15 | 143 | 29 | 518 | 9.49% | 34.09% | 14.85% | 7.21% to 31.25% | 50 |
| E | 0.500000 | 705 | 2 | 1 | 42 | 660 | 66.67% | 4.55% | 8.51% | 11.76% to 94.12% | 14 |

C/E各只有3个确定标签上的正预测（2真阳性、1假阳性），另有14个不确定正候选；66.67%不包含这些不确定项，不能视为稳定高精确率。

无正预测时precision及其区间无定义，图中标为N/A；CSV数值0需结合precision_defined=False解读，不能解释为产生了错误正预测。精确匹配A只覆盖相同措辞，监督池移除公共模板和记录副本后，自然完全匹配正例可能很少。

### 商品bootstrap，1,000次、seed42

| method | products | status | precision 95% CI | recall 95% CI | F1 95% CI |
| --- | --- | --- | --- | --- | --- |
| A | 141 | ok | undefined/unavailable | 0.00% to 0.00% | 0.00% to 0.00% |
| B | 141 | ok | 0.00% to 0.00% | 0.00% to 0.00% | 0.00% to 0.00% |
| C | 141 | ok | 0.00% to 100.00% | 0.00% to 12.20% | 0.00% to 21.29% |
| D | 141 | ok | 4.82% to 14.86% | 20.75% to 45.66% | 8.00% to 22.03% |
| E | 141 | ok | 0.00% to 100.00% | 0.00% to 11.76% | 0.00% to 20.41% |

重采样单位为ASIN，有效重采样不足时标为unavailable。区间反映该富集参考标签基准的不确定性，未证明自然场景95%精确率。

### 语义增量

E相对C的测试precision差+0.00个百分点、recall差+0.00个百分点、F1差+0.00个百分点。这是观察差异，不是显著性结论，也不能据测试差异重选模型。语义向量缓存约133MiB，额外编码耗时见性能节；当前配置没有观察到这一成本带来的增量收益。

MiniLM固定revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41，384维、float32归一化、最长256 word pieces；1324条评论被截断。长文本分组、exact/near/distinct/short及各批次结果见test_breakdowns.csv。编码器未微调。共同商品词、相似情绪与否定差异仍需看正文判断。

## 错误与合成检查

error_cases.csv提供21个错误/不确定案例及正文、参考理由、五方法分数和下游风险；各类最多10个、不足全部展示。案例按E方法已锁定的探索阈值整理，并附五种方法分数；不是全体方法错误的全集。具体原因复核记录见error_cases_review.csv及error_analysis.md。全量待复核不会自动误降权。

合成测试只使用测试集合原评论，单独报告已知复制/扰动检出率，precision/F1因只有正例而无定义。详见synthetic_test_detection.csv，共25个方法/变换组合；C/E对本批复制及扰动全部检出，但真实近似重复召回仍低，说明这些合成变换的难度与真实改写不同，不替代真实评价。

## 性能与全量结果

模型首次加载/下载35.74秒，独立正文GPU编码45.91秒，torch峰值分配显存387.99MiB。缓存推断使用CPU词面/分类及现有向量，不重复下载或编码。

| products | reviews | pairs | seconds | peak_rss_mb | gpu_peak_mb |
| --- | --- | --- | --- | --- | --- |
| 100 | 9258 | 430688 | 12.252 | 747.211 | 0 |
| 500 | 45585 | 2092467 | 61.535 | 759.277 | 0 |
| 1000 | 90750 | 4146325 | 103.268 | 765.59 | 0 |

表为完整五方法的特征、评分及决策，不含CSV写出。RSS为单进程绝对峰值，显存为本过程torch分配值，不代表整机占用。

全部1000个商品、4,146,325对、20,731,625方法行已写至results/predictions/pair_scores.csv.gz。含压缩写出的耗时914.90秒、峰值RSS 658.24MiB。输出SHA及模型锁/评价SHA见manifest.json。阈值候选数量只是模型信号，不能作为已确认重复数或自然重复率。

固定10商品交接位于results/handoff/，正文及review_id可回连；predict_pairs与score_product具体调用见runbook.md及handoff_contract.md。原五字段接口保持不变。每对五行包含pair_id、asin、两个review_id、method、score、decision、reason。余弦和未校准分类器分数不能称真实重复概率。

## 工程验收与文件管理

工程问题与修正见third_party_review.md；实际验收见acceptance_report.json。夹具成绩不当作真实模型成绩。生成报告时的代码版本：73aa9d90b75ef286c9e923b7a7acc8916af92d2a。

原始数据、标签、模型、CSV/JSON结果和带输出notebook保留本地；仓库提供代码、研究说明、图及无输出notebook模板。

## 文件与后续建议

完整路径/SHA清单位于artifact_inventory.csv。主要结果：test_comparison.csv、test_breakdowns.csv、test_product_bootstrap.csv、test_pair_predictions.csv、synthetic_test_detection.csv、error_cases.csv；主要图：test_model_comparison.png、test_confusion_matrices.png、final_label_composition.png、cached_pair_performance.png。

组员二接入时先保留review标记，另行讨论降权政策。若要宣称自然场景可自动降权，后续需要独立人类双人标注和自然抽样的验证集，再固定新的实验版本；当前结论足以交付方法比较与复核接口。
