# 复现实验所需本地输入

GitHub归档仅含代码、说明、安全图和无输出notebook模板；原始数据、缓存、标签、结果JSON/CSV及执行notebook均在WSL本地，未随仓库发布。

精确复现固定样本需保留`data/interim/reviews_sample.csv`、`data/interim/sampling_metadata.json`及`data/processed/`下的`reviews.csv`、`review_index.csv`、`split_assignments.csv`、`dataset_manifest.json`。目标数据SHA-256：`8f1037a6987436a7c19b56766c42969526374145b8d02dde02caa5b1b750f0ec`。6000对还需`data/annotation/`主配对/正文/标签/指南冻结/测试封存，以及`scale6000/`和`original600/`、`original1200/`清单；模型需`models/review_embeddings.npy`、`embedding_index.csv`、`embedding_manifest.json`及模型锁、词面缓存。不得臆造标签。

已有副本按runbook运行：`.venv/bin/python -m src.study prepare`、`.venv/bin/python -m src.study status`、`.venv/bin/python -m src.study evaluate`、`.venv/bin/python -m src.study validate`；其余阶段命令见runbook。原始Hugging Face下载revision未记录，重下不能保证同SHA；应另建数据版本，重建索引、缓存、配对并重新标注，不能声称精确复现。
