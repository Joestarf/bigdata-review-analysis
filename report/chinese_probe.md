# 固定重复识别模型的中文诊断

## 结论

当前实现不宜直接用于中文近似重复识别。规范化精确匹配仍可识别28对全部原样复制，但词面特征覆盖和短评规则存在明显中文适配问题，语义阈值也混淆关键否定及数字/型号差异。中英文配对对照同时表明，否定和数值辨别并非只有中文才困难；不能把所有误信号归因于语言，也不能仅替换编码器就宣称解决。

这是有来源样例和构造控制对的功能诊断，不是自然中文准确率；实际识别决策全部保持review。

## 范围与来源

测试现有A—E重复识别方法在中文输入上的特征、阈值信号和运行行为。不是复现另一项情感/星级预测任务，不重新训练，不改动原模型、阈值、英语数据、封存测试或交接包。

来源：[livingbody/Commodity_Review_Sentiment_Predictions](https://github.com/livingbody/Commodity_Review_Sentiment_Predictions)，固定提交7f7175fd717c5dfdd74bfad9e0e46d5390bdec51。该仓库仅提交README、notebook和requirements，完整竞赛CSV不在仓库。只从公开notebook的CSV输出解析28条实际中文评论，使用标准CSV解析，不执行外部notebook，也不复用其按逗号位置截断正文的读取方法。完整竞赛数据本次未取得。

这28条评论对应28商品，可用原始同商品评论对为0，因此没有建立自然中文重复率或真实同商品效果评价。其星级标签不能作为重复标签。README里的720万条是上游数据介绍，不是本次实际取得的数量。

## 诊断组成

- 来源评论构造同源完整复制、外侧空格、追加标点、汉字间空格四类，共112对。标签依据是明确的信息不变变换，单独报告检出，不用作真实用户重复评价。
- 独立编写46对控制样例：10对具体信息一致的改写、10对关键否定差异、10对数字/型号差异、10对情感同向但细节不同、6对通用短评不确定。标签是构造情形的设计判断，未经独立人类复核，不是自然标注金标准。
- 总计158对；模型输入只有正文，不输入星级、来源类别或控制标签。没有用这些标签调阈值。
- 既有探索阈值：A=1、B=0.8、C/E=0.5、D=0.8。表格的正信号只表示超过阈值；实际decision始终review，不能称自动确认重复。

## 实际特征与输入处理

共204个不同正文输入：
- 字符TF-IDF零向量：149（73.04%）。
- 中文中按现有空格规则被标为不足5词：170（83.33%）。
- 有汉字且没有英文字母/数字词的输入：155；其原英文正则词集合全部为空。
- 逐输入平均的编码器未知token比例：67.10%；超过256 word pieces而截断：0条。

原TF-IDF固定在英语Amazon训练语料上拟合；即使分析器按字符工作，也不能假定固定词表覆盖中文。word_set仅匹配a-z/0-9，不处理汉字；word_count以空格切分，中文不插空格的长句也可能被标为短评。中文句子是否加空格因此会影响旧短评门槛，这是实现限制，不是中文内容真的信息不足。

固定MiniLM模型卡标记language=en：[官方说明](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。中文可被分词/输出向量并不证明向量理解中文细节；未知token、共同字符和句型都需结合控制对看。以下误信号是此诊断集观察，不据其直接推断成因或自然总体精确率。

## 各组阈值信号

| group | method | pairs | known_positives | known_negatives | signal_positive | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paraphrase | A | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 0 |
| paraphrase | B | 10 | 10 | 0 | 3 | 3 | 0 | 7 | 0 |
| paraphrase | C | 10 | 10 | 0 | 3 | 3 | 0 | 7 | 0 |
| paraphrase | D | 10 | 10 | 0 | 5 | 5 | 0 | 5 | 0 |
| paraphrase | E | 10 | 10 | 0 | 3 | 3 | 0 | 7 | 0 |
| negation | A | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 10 |
| negation | B | 10 | 0 | 10 | 2 | 0 | 2 | 0 | 8 |
| negation | C | 10 | 0 | 10 | 2 | 0 | 2 | 0 | 8 |
| negation | D | 10 | 0 | 10 | 6 | 0 | 6 | 0 | 4 |
| negation | E | 10 | 0 | 10 | 2 | 0 | 2 | 0 | 8 |
| numeric_model | A | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 10 |
| numeric_model | B | 10 | 0 | 10 | 1 | 0 | 1 | 0 | 9 |
| numeric_model | C | 10 | 0 | 10 | 2 | 0 | 2 | 0 | 8 |
| numeric_model | D | 10 | 0 | 10 | 7 | 0 | 7 | 0 | 3 |
| numeric_model | E | 10 | 0 | 10 | 2 | 0 | 2 | 0 | 8 |
| same_sentiment_different_information | A | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 10 |
| same_sentiment_different_information | B | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 10 |
| same_sentiment_different_information | C | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 10 |
| same_sentiment_different_information | D | 10 | 0 | 10 | 2 | 0 | 2 | 0 | 8 |
| same_sentiment_different_information | E | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 10 |
| generic_uncertain | A | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| generic_uncertain | B | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| generic_uncertain | C | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| generic_uncertain | D | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| generic_uncertain | E | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| exact_copy | A | 28 | 28 | 0 | 28 | 28 | 0 | 0 | 0 |
| exact_copy | B | 28 | 28 | 0 | 5 | 5 | 0 | 23 | 0 |
| exact_copy | C | 28 | 28 | 0 | 6 | 6 | 0 | 22 | 0 |
| exact_copy | D | 28 | 28 | 0 | 28 | 28 | 0 | 0 | 0 |
| exact_copy | E | 28 | 28 | 0 | 6 | 6 | 0 | 22 | 0 |
| outer_whitespace | A | 28 | 28 | 0 | 28 | 28 | 0 | 0 | 0 |
| outer_whitespace | B | 28 | 28 | 0 | 5 | 5 | 0 | 23 | 0 |
| outer_whitespace | C | 28 | 28 | 0 | 6 | 6 | 0 | 22 | 0 |
| outer_whitespace | D | 28 | 28 | 0 | 28 | 28 | 0 | 0 | 0 |
| outer_whitespace | E | 28 | 28 | 0 | 6 | 6 | 0 | 22 | 0 |
| punctuation | A | 28 | 28 | 0 | 0 | 0 | 0 | 28 | 0 |
| punctuation | B | 28 | 28 | 0 | 3 | 3 | 0 | 25 | 0 |
| punctuation | C | 28 | 28 | 0 | 4 | 4 | 0 | 24 | 0 |
| punctuation | D | 28 | 28 | 0 | 28 | 28 | 0 | 0 | 0 |
| punctuation | E | 28 | 28 | 0 | 4 | 4 | 0 | 24 | 0 |
| cjk_spacing | A | 28 | 28 | 0 | 0 | 0 | 0 | 28 | 0 |
| cjk_spacing | B | 28 | 28 | 0 | 1 | 1 | 0 | 27 | 0 |
| cjk_spacing | C | 28 | 28 | 0 | 4 | 4 | 0 | 24 | 0 |
| cjk_spacing | D | 28 | 28 | 0 | 28 | 28 | 0 | 0 | 0 |
| cjk_spacing | E | 28 | 28 | 0 | 4 | 4 | 0 | 24 | 0 |

tp/fn仅指明确的同源变换或构造正例；fp/tn仅指构造负例。不确定组不算二分类正确率。每类10或28对很小、且并非自然随机抽样，不能合并成中文真实准确率，也不能直接比较上游情感分类准确率。

## 可复现文件

运行 .venv/bin/python -m src.chinese_probe。
本地data/chinese_checks/保存来源notebook、实际样例、控制对、诊断评论对及固定来源清单。results/tables/chinese_probe_*.csv保存逐对五分数/信号、特征诊断与分组统计；results/tables/chinese_probe_receipt.json记录来源SHA、控制SHA、原模型锁SHA和输出SHA。results/figures/chinese_probe_signals.png展示分组检出/误信号率。原公开交接包SHA仍为fef2dcb66c24517c8d28ee2607c9cd7dc211af1a17a8c68e178d4dc24c03463c。

## 后续适配

中文应用应单独设计汉字长度/分词、中文词面表示和保留否定/数字/型号的特征，并考虑经中文或多语言语料训练的语义编码器。随后用同商品真实中文评论对重新划分训练/验证/测试，建立与情感标签独立的重复标签，重新锁定阈值。不能只把旧词集合改成中文后继续复用原C/E分类器，因为输入特征分布已经改变；也不能让插入空格成为通过旧短评门槛的正式方案。

本次没有训练新中文模型或产生中文评分/排名结论。中文数据来源、诊断构造、阈值信号和正式识别效果分开报告。


## 构造案例的特征追踪

ctrl_011：这只充电器支持 PD 快充，给手机充电很快。 / 这只充电器不支持 PD 快充，手机只能慢慢充。。这对核心结论相反的输入在现有TF-IDF里仅保留[' pd', ' pd ', 'pd ']和[' pd', ' pd ', 'pd ']，两个稀疏向量相同；因此B=1不能解释为中文信息相同。这是可直接观察的投影缺失，不是根据相似度猜测其成因。案例为构造控制，不是两位真实用户的评论。


## 同案例的中英文对照

对全部46个构造案例补充忠实英文翻译，保持标签、否定、数字、型号和信息差异不变；未挑选有利案例或调整阈值。中文结果查看后才增加此对照，因此属于探索性诊断，不是事先设计的正式跨语言效果检验。两种语言各46对，共92个控制对运行。

A精确匹配，B字符TF-IDF，C词面逻辑回归，D固定MiniLM语义相似度，E词面加语义逻辑回归。逐输入平均未知token比例与合并所有token计算的比例分别报告。

| language | control_pairs | unique_inputs | zero_lexical_inputs | legacy_short_inputs | empty_ascii_word_sets | mean_input_unknown_fraction | pooled_token_unknown_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zh | 46 | 92 | 60 | 86 | 59 | 0.671815 | 0.672566 |
| en | 46 | 92 | 0 | 7 | 0 | 0.0 | 0.0 |

| language | group | method | pairs | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zh | paraphrase | A | 10 | 0 | 0 | 10 | 0 |
| zh | paraphrase | B | 10 | 3 | 0 | 7 | 0 |
| zh | paraphrase | C | 10 | 3 | 0 | 7 | 0 |
| zh | paraphrase | D | 10 | 5 | 0 | 5 | 0 |
| zh | paraphrase | E | 10 | 3 | 0 | 7 | 0 |
| zh | negation | A | 10 | 0 | 0 | 0 | 10 |
| zh | negation | B | 10 | 0 | 2 | 0 | 8 |
| zh | negation | C | 10 | 0 | 2 | 0 | 8 |
| zh | negation | D | 10 | 0 | 6 | 0 | 4 |
| zh | negation | E | 10 | 0 | 2 | 0 | 8 |
| zh | numeric_model | A | 10 | 0 | 0 | 0 | 10 |
| zh | numeric_model | B | 10 | 0 | 1 | 0 | 9 |
| zh | numeric_model | C | 10 | 0 | 2 | 0 | 8 |
| zh | numeric_model | D | 10 | 0 | 7 | 0 | 3 |
| zh | numeric_model | E | 10 | 0 | 2 | 0 | 8 |
| zh | same_sentiment_different_information | A | 10 | 0 | 0 | 0 | 10 |
| zh | same_sentiment_different_information | B | 10 | 0 | 0 | 0 | 10 |
| zh | same_sentiment_different_information | C | 10 | 0 | 0 | 0 | 10 |
| zh | same_sentiment_different_information | D | 10 | 0 | 2 | 0 | 8 |
| zh | same_sentiment_different_information | E | 10 | 0 | 0 | 0 | 10 |
| zh | generic_uncertain | A | 6 | 0 | 0 | 0 | 0 |
| zh | generic_uncertain | B | 6 | 0 | 0 | 0 | 0 |
| zh | generic_uncertain | C | 6 | 0 | 0 | 0 | 0 |
| zh | generic_uncertain | D | 6 | 0 | 0 | 0 | 0 |
| zh | generic_uncertain | E | 6 | 0 | 0 | 0 | 0 |
| en | paraphrase | A | 10 | 0 | 0 | 10 | 0 |
| en | paraphrase | B | 10 | 0 | 0 | 10 | 0 |
| en | paraphrase | C | 10 | 1 | 0 | 9 | 0 |
| en | paraphrase | D | 10 | 6 | 0 | 4 | 0 |
| en | paraphrase | E | 10 | 2 | 0 | 8 | 0 |
| en | negation | A | 10 | 0 | 0 | 0 | 10 |
| en | negation | B | 10 | 0 | 3 | 0 | 7 |
| en | negation | C | 10 | 0 | 5 | 0 | 5 |
| en | negation | D | 10 | 0 | 8 | 0 | 2 |
| en | negation | E | 10 | 0 | 6 | 0 | 4 |
| en | numeric_model | A | 10 | 0 | 0 | 0 | 10 |
| en | numeric_model | B | 10 | 0 | 3 | 0 | 7 |
| en | numeric_model | C | 10 | 0 | 3 | 0 | 7 |
| en | numeric_model | D | 10 | 0 | 6 | 0 | 4 |
| en | numeric_model | E | 10 | 0 | 3 | 0 | 7 |
| en | same_sentiment_different_information | A | 10 | 0 | 0 | 0 | 10 |
| en | same_sentiment_different_information | B | 10 | 0 | 0 | 0 | 10 |
| en | same_sentiment_different_information | C | 10 | 0 | 0 | 0 | 10 |
| en | same_sentiment_different_information | D | 10 | 0 | 0 | 0 | 10 |
| en | same_sentiment_different_information | E | 10 | 0 | 0 | 0 | 10 |
| en | generic_uncertain | A | 6 | 0 | 0 | 0 | 0 |
| en | generic_uncertain | B | 6 | 0 | 0 | 0 | 0 |
| en | generic_uncertain | C | 6 | 0 | 0 | 0 | 0 |
| en | generic_uncertain | D | 6 | 0 | 0 | 0 | 0 |
| en | generic_uncertain | E | 6 | 0 | 0 | 0 | 0 |

分组误信号需与语言特征覆盖一起解读；英语控制也可能出现否定或数字误信号，不能把全部错误归因于中文。此对照没有替代同商品真实中文标注。构造中英文案例随src/chinese_controls.py保存，运行脚本可重建本地CSV，公开来源原文及结果明细仍按忽略规则留本地。
