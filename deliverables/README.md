# 组内交接下载

- [team_handoff.zip](team_handoff.zip)：5.46MiB。含可移植模型、90,750条评论编号/原星级/六策略权重、1,000商品最终星级及候选分数；不含原评论正文字段、原用户标识或时间戳。评论编号可回连固定样本，不能视为不可逆匿名化。
- [校验值](team_handoff.sha256)：ZIP的SHA-256；包内manifest核对逐文件SHA。
- 完整研究包保存在WSL本地results/handoff/team_research_private.zip（160.03MiB），含原五字段数据、固定标签及向量缓存，可供组内另行交接；未上传公开仓库。
- 570MiB全量对结果已有本地保存，不重复打包。评分模块只需要包内review_weights.csv。

默认折扣：轻0.75、中0.50、相同措辞0.25、重复记录副本0、短评/仅语义信号1。最终星级为Σw×rating/Σw。属于样本敏感性实验；候选仍需复核，不能判断水军或称为官方商品评分。

模型接口见[portable_model](../report/portable_model.md)，策略及实测结果见[rating_adjustment](../report/rating_adjustment.md)。在新目录解压，按包内README执行；不依赖原工作目录。

发布范围由src/package_handoff.py校验：模型、代码、研究说明及不含正文/用户/时间的衍生评分数据。原data/、models/和其他CSV/JSON仍保持忽略，不强制加入原始文件。


## 中文输入诊断

固定模型已使用28条公开中文样例的112个同源变换、46个构造控制对及其46个英文对照进行诊断。中文词面覆盖及按空格计词存在适配问题，否定与数字辨别在英文中也有误信号。目前不宜把既有模型/阈值直接用于中文近似重复或自动评分降权；原交接包及英语结果没有修改。测试规模与实际分组结果见[中文诊断报告](../report/chinese_probe.md)。
