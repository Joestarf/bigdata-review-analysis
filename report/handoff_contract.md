# 评论冗余分析交接契约

## 1. 目标与范围

本契约定义原始评论、稳定标识、成对预测和单个 ASIN 评分的交接约定，供分析流程与后续调用方对接。研究问题是同一商品评论之间的信息冗余，不判断水军，也不以重复性推断评论真实性。

数据约 90,750 行、1,000 个 ASIN。现有 `reviews.csv` 保持五个字段：`asin`、`user_id`、`rating`、`text`、`timestamp`。原始重复评论必须保留，不得去重覆盖。另提供旁表 `review_index`，字段为 `review_id`、`source_id`、`text_hash`、`dataset_version`。`review_id` 在同一数据版本中唯一且稳定；`source_id` 用于追溯来源记录，`text_hash` 标识正文内容，`dataset_version` 标记数据版本。旁表不能取代原始文件。

当前比较范围为同一 ASIN 内的评论对，总计 4,146,325 对。计算样本是有限的流式样本，热门 ASIN 优先，且每个商品最多 100 条评论；它不是全量评论对，也不是均匀随机样本。因此不能把该样本的评论计数解释为全体评论总体的排名或比例。

## 2. 接口约定

### 输入与输出

| 接口/文件 | 输入 | 输出或保证 |
|---|---|---|
| `reviews.csv` | 原始评论数据 | 保持五字段及原始重复记录；不在此契约内改变字段或清洗语义。 |
| `review_index` | 与数据版本对应的来源记录 | `review_id`、`source_id`、`text_hash`、`dataset_version`；`review_id` 唯一稳定。 |
| `predict_pairs` | 两段评论正文 | 返回该方法的分数/判断；输入是正文对，不要求调用方提供星级、用户或情感信息。 |
| `score_product` | 一个 ASIN 的评论及其 ID | 对该商品范围内的评论生成成对比较/汇总所需的评分结果；输入 ID 用于关联稳定评论记录。 |
| 预测结果表 | 一行一对预测 | 必须包含 `pair_id`、`asin`、`review_id_a`、`review_id_b`、`method`、`score`、`decision`、`reason`。 |

预测表列定义：

| 列 | 定义 |
|---|---|
| `pair_id` | 评论对的稳定标识。 |
| `asin` | 该评论对所属商品标识；成对输入应属于同一 ASIN。 |
| `review_id_a`、`review_id_b` | 两条评论在 `review_index` 中的稳定 ID。 |
| `method` | 产生本行输出的方法标识。 |
| `score` | 方法输出的分数；除非另有校准证据，不得称为真实概率。 |
| `decision` | `duplicate`、`distinct` 或 `review`。`review` 表示需人工复核/证据不足。 |
| `reason` | 简明说明输出依据或触发复核的原因；不得伪装成人工标签。 |

对短而泛化的评论，默认 `decision=review`。此默认用于降低证据不足时的自动判断，不代表已完成标注。四类参考标签 `exact`、`near`、`distinct`、`uncertain` 与三类预测决策不是一一等价：预测决策是系统输出，本次AI辅助标签是评估参考（未经人类金标准复核），应分字段、分表保存。

## 3. 方法标识

当前计划比较的方法如下；这些名称表示方法设计，不表示已训练或已验证：

| 方法 | 说明 |
|---|---|
| A | 精确匹配基线。 |
| B | 字符级 TF-IDF 基线。 |
| C | 逻辑回归基线。 |
| D | 预训练 MiniLM 语义方法。 |
| E | 结合词面与语义特征的逻辑回归方法。 |

不训练星级预测模型或情感模型。不得把星级、情感或用户信息用作重复性标签依据。AI辅助真实对标签与合成标签分开管理；随机评论对不能直接作为负例。

## 4. 接入示例

以下仅说明数据如何交给接口，不假设或规定任何命令行名称、参数格式、文件路径或调用方式。具体调用形式由实现方提供。

```text
单对预测：
predict_pairs(text_a="Battery lasts about ten hours per charge.",
              text_b="I get roughly 10 hours from one charge.")

单商品评分：
score_product(asin=<一个ASIN>,
              reviews=[{review_id: <稳定ID>, text: <正文>}, ...])
```

结果示意仅展示字段形状，不代表真实运行结果：

```text
pair_id | asin | review_id_a | review_id_b | method | score | decision | reason
<id>    | <ASIN> | <id-a> | <id-b> | <method> | <方法分数> | <duplicate/distinct/review> | <说明>
```

调用方应使用与输入正文一致的数据版本及 `review_id` 映射；不应仅依赖正文哈希合并记录，因为相同正文可能来自不同的原始评论，原始重复记录须保留。

## 5. 解释与使用限制

- `score` 是方法输出分数，不自动等于概率、置信度或人工一致率；方法之间的分数未必可直接比较。
- `decision` 表示当前方法给出的操作性结果。`review` 应保留为可复核结果，不要强行并入 `duplicate` 或 `distinct`。
- 不做基于预测边的传递闭包分组。若 A 与 B 判为重复、B 与 C 判为重复，不据此自动推断 A 与 C 重复或将三者归为同一组。
- 结果范围受同 ASIN、热门商品优先、每商品 cap 100 的有限流式采样限制；不代表全量或均匀随机覆盖。不要据此用评论计数给全部商品做总体排名。
- 当前固定真实对标注集为6,000对（train4,000、validation1,000、test1,000），保留原600与1200阶段，先以 train 60 对进行 pilot。测试标签在阈值锁定后才用于最终评价。不得泄漏测试标签用于选择方法或阈值。
- 本契约不报告标注完成情况、模型训练状态、阈值、指标或实测结果；任何后续结果应附对应数据版本、方法版本和评估范围。

## 6. 实现对照

实际调用为 predict_pairs([(text_a,text_b), ...]) 和 score_product(DataFrame[asin,review_id,text])，每对5行；正文接口使用虚拟文本编号，商品接口使用真实review_id。当前状态与输入失败处理以 runbook.md 和 implementation_status.md 为准。模型锁定前的预览C/E分数为空；锁定后的完整交接提供五方法分数。由于只有富集AI基准证据，完整交接也保持decision=review，不作为自动降权证据。


## 7. 分级折扣实验与交接包

识别接口的decision=review继续保留。独立的src.rating_adjustment按已声明政策模拟评分变化，不修改模型锁或封存测试，不把候选改标为已确认重复。

公开team_handoff.zip提供可移植识别模型、全部review_weights及六策略product_ratings。默认product_final_ratings用于当前样本的下游对照。折扣规则、证据层级、加权均值和样本名次见rating_adjustment.md；模型从src.portable加载，不需要原始数据或完整向量缓存。

完整版team_research_private.zip含原五字段数据、固定标签和缓存，保存在本地results/handoff/。评论编号是可回连键，公开包去除正文字段、原用户ID和时间戳，但不宣称不可逆匿名化。发布清单与校验方式见deliverables/README.md。
