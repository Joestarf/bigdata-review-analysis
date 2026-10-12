# 运行与交接说明

## 环境与状态

Windows宿主，archlinux WSL，项目 /home/joestar/code/bigdata-review-analysis，现有.venv/Python3.14及RTX3060 Laptop 6GB。固定Amazon样本90750评论、1000ASIN。已完成6000对判读（4000/1000/1000）、合成样本检查、五方法训练和锁定后的最终测试；参考标签未经独立人类验证。没有方法满足验证选型目标，全量交接保留待复核。实测结果见experiment_results.md；历史批次保存在data/annotation/original600与original1200。

## 运行顺序

下列命令是从固定输入重建实验的阶段顺序。已有模型锁和最终评价时，直接查看交付文件或运行validate/report；不要再次拟合、调阈值或重跑测试来改进已封存结果。性能基准单独运行，避免其他计算任务争用资源。

~~~powershell
wsl -d archlinux
~~~

~~~bash
cd /home/joestar/code/bigdata-review-analysis
.venv/bin/python -m src.study prepare
.venv/bin/python -m src.study extend
.venv/bin/python -m src.study scale
# 完成原1200阶段，再激活6000定义，独立测试管理者合并并封存1000对。
.venv/bin/python -m src.study activate-scale
# 逐批完成scale6000/batches中的盲审并保存独立标签；训练分工可用labels_train_*.csv，重叠已完成判断冲突会拒绝。
.venv/bin/python -m src.study sync-scale
.venv/bin/python -m src.study status
.venv/bin/python -m src.study synthesize
# 逐对审阅synthetic_review.csv的100个变换例；审阅须记录来源，全部通过后继续。
.venv/bin/python -m src.study train
.venv/bin/python -m src.study evaluate
.venv/bin/python -m src.study benchmark
.venv/bin/python -m src.study infer
.venv/bin/python -m src.study handoff
.venv/bin/python -m src.study validate
.venv/bin/python -m src.study report
.venv/bin/python -m unittest discover -s tests -v
~~~

prepare复核文件SHA、稳定编号、划分和缓存；不会覆盖已有标签。原CSV保留五字段，dataset_version为文件SHA；review_id由版本和固定行号生成。原下载源revision未记录；未来下载会固定revision。重做默认preprocess会因12条新版清洗差异拒绝覆盖旧版本；请--output另存并重建版本及相关缓存。

扩样仅在模型锁定前进行，固定候选一次生成，不按测试成绩补标。原600备份保留。新的6000候选先单独生成，避免覆盖正在进行的1200阶段判断；激活后由独立测试管理者合并测试标签，建模端只查看完成清单与文件SHA，不打开测试值。

## 标注、门槛与封存

本地页面 study serve --port 8765 只显示两段正文和理由。60对pilot固定规范。测试由独立管理任务在盲表中判断；建模界面直到最终评价封存后才开放测试只读查看，始终不允许网页修改。test_seal.json记录版本、行数、完成数与标签文件SHA，不记录分布。

训练至少20确定正例/50负例，验证及测试各至少20正例/20负例。2000合成训练正例（1000完整复制、4种扰动各250）来源仅限训练记录，变换每种先逐对看20例。合成总权重为确定真实训练权重的25%，即合并权重20%。不把随机对直接当负例。

资格按完整验证批次：确定TP/全部允许正预测的保守precision下界≥95%，至少30个允许正预测，优先召回、同分优先低成本。uncertain另报，保留上下界。无方法达标则所有配置仅用于探索复核；即使富集AI基准达标，全量仍review，因为没有自然场景独立精确率证据。模型锁定后才启封一次最终测试，不能据测试五方法比较再挑模型。

## Python接口

~~~python
from src.inference import Predictor
from src.common import load_study
p = Predictor()
result = p.predict_pairs([("Battery lasts ten hours after a full charge.",
                           "Battery does not last ten hours after a full charge.")])
df, _ = load_study()
g = df[df.asin == sorted(df.asin.unique())[0]]
product_scores = p.score_product(g[["asin", "review_id", "text"]])
~~~

每对输出5行：pair_id、asin、review_id_a、review_id_b、method、score、decision、reason。A规范化匹配0/1，B字符TFIDF余弦，C词面逻辑回归输出，D编码器余弦，E词面+语义逻辑回归输出；分类器分数未校准，不能称真实重复概率。正文接口虚拟ID不能回连原CSV；交换两段正文保持同一pair_id。商品接口要求单ASIN、唯一ID、非空正文，每个已知ID的正文和ASIN必须核对，即使同时包含新ID。

少于5词默认review并保留匹配信号。当前C/E已训练，五种方法均输出完整分数。逐商品计算全部对，不创建90k全局稠密矩阵；无降权权重、无自动传递聚类。错误消息指出缺标签、版本或校验失败，不能复用过期缓存/锁。

## Git归档

按阶段提交和推送代码、说明与安全图。保留用户原忽略设置；原始大样本、133MiB向量缓存及全量对输出留在WSL，不上传GitHub大文件。标签、CSV/JSON结果和带输出的notebook留本地，不通过强制加入绕过忽略规则。提交前运行scripts/check_git_scope.py。最终报告列生成时的代码版本、验收结果及留在本地的文件。

全量结果gzip使用固定mtime=0；manifest记录4,146,325评论对、五方法行数、输出SHA、模型锁SHA、最终评价SHA和实测耗时/内存。validate对总量、校验值和待复核决策范围核对。

report命令必须在最终评价、全量推断及性能基准完成后运行，生成汇总报告、三张对照图和本地逐文件SHA清单；不重新拟合或重跑测试。
