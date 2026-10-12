# Luna 标注进度与复核交接

## 范围与状态

- 已逐对阅读 `blind_train.csv` 源行 2880–2999（0-based），每 20 对保存一次；只查看 pair_id 与两段正文。新增结果写入 `labels_train_final200.csv` 行 80–199，包含 label、reason、annotation_version。
- 对新增 near 的根审核意见已应用 6 项标签修正。
- 200 表原有前 80 行属于此前强复审结果。本轮对其中 near/uncertain 子集的临时改动已撤销，并依据 `data/checkpoints/20261012T014903Z/scale6000/labels_train_final200.csv` 精确恢复这 80 行的 label、reason、annotation_version。该 80 行的复核不计入本轮结论。
- 已重新阅读 `labels_train_root_extra.csv` 中 2 个 near 与 12 个 uncertain，其中 6 个 uncertain 改为 near，其余 8 个维持原标签。
- 状态：进度已保存，等待更强复审；尚未最终验收。标签为 AI 辅助逐对判断，不是独立人类金标准。

## 当前标签计数

| 文件 | exact | near | distinct | uncertain | 行数 |
|---|---:|---:|---:|---:|---:|
| `labels_train_final200.csv` | 0 | 65 | 55 | 80 | 200 |
| `labels_train_root_extra.csv` | 0 | 8 | 16 | 6 | 30 |

## 2880–2999 根审核修正

以下 pair 首轮暂标为 near，根审核指出共同点过泛或未达到整体具体信息重复标准，现已修正：

| pair_id | 旧 → 新 | 理由 |
|---|---|---|
| `0f11e84b5ea1dcb38f19962434f2a1da35e313b269c26ea53cdfd9dd830eb082` | near → uncertain | 两条都是不足五词的泛化快充短评，不能仅凭共同主题判为重复。 |
| `f5b156eb21f8abe665abffa2ee6a023644df4a431a7f6b43d4ab73a654183fec` | near → uncertain | 两条都提到易安装，但一条仅此一句，另一条只是泛化赞语，缺少足够具体的共同信息。 |
| `b3a19975cb55e4e5d10b07f91bd86050a1a60665a1bc7eff9a7f708b979a749c` | near → distinct | 虽都提到附送胶贴，第二条另指出产品容易刮伤，新增实质负面体验使整体信息不同。 |
| `e3380e25dc80441bbaf78917d36844b53edfc7a83898507631b6c9abd3a43e74` | near → uncertain | 一条仅表达喜欢充电器，另一条泛称易充电，缺少共同的具体性能细节。 |
| `748f61342be4c3dd9ca8ace3fe4cfe5df5f9a65c1e06bb3fba3e2fb20ccdc46c` | near → distinct | 一条说换笔尖可恢复逐渐变差的书写表现，另一条说备用笔尖避免丢弃手写笔，具体问题和结论不同。 |
| `4fcf1bd3ced66f20bdc9526044ba6b82313d7f10f38df67cbab047c291a56b75` | near → uncertain | 一条说安装清晰且坚固，另一条仅说说明清楚；“清晰”所指不同，缺少足够共同细节。 |

## root_extra 复核修正

| pair_id | 旧 → 新 | 理由 |
|---|---|---|
| `467f5f953c3cc7c59968ad35f718e7992da00b61940f288c5c43e22888a80346` | uncertain → near | 两条都明确评价磁铁吸附力强；第二条具体说可牢固吸附到车载磁吸支架。 |
| `b32ac7d8cb6b6819ab54fa1f02d7aa19d367ec480ddccfbc33e513eb4502be5c` | uncertain → near | 两条都说摩托车手机支架安装或固定手机；第二条明确补充能稳固夹住手机。 |
| `1e390b501d2e2e907d86541fad053ef9370f3fbe54f5c3604a5acfabdbdb563e` | uncertain → near | 两条都明确说 Apple Pencil 充电器能为笔充电；第二条补充每晚使用场景。 |
| `d29c3e3426dff12fa43bb510ea28d20e2ff0e113c749df672891c48e0c29ef1f` | uncertain → near | 两条都具体说车载杯架支架适配杯架并稳固放手机；第二条补充可调节和价格。 |
| `7709401b150506d03ef090449e0f00ec8e8a36d3dbf49a76968e06c73662b1d8` | uncertain → near | 两条都具体评价屏幕保护膜容易安装；一条说无需费力对齐，另一条补充到货快。 |
| `8af3d3db9c434d1814c99cb6df07da79a6401ce34f3152cd6235b91921cb2733` | uncertain → near | 两条都具体称手机壳纤薄；第二条补充手感好。 |

## 原 80 行临时编辑回滚记录

本轮曾将下列原 uncertain 临时改为 near；随后按要求从 checkpoint 精确恢复为原 uncertain、reason 与版本。本列表为操作留痕，临时 near 不属于最终结果：

- `99232d062b50d8d0a3d211f0b38c2ef5f85b96b1f1e7322b80171c5041b9336c`
- `7fb9722483f928c03ef3941129580c12e8fec9201611bf613d19569fd81bcd7d`
- `9fe3342a8143fa2a830cf839afd392091fa489c43c7887d00359a9a8b4fcff0c`
- `77841280f3769ff3eec60a9a6864b1a03f07d303d0ab59d7ae8becd16516ad5b`
- `eb99589388c20673ecf0687d11ac85db8346e701ae44052f8187ce9a55eb50c6`
- `3d520cba3e2a00f2326defd7183b07041a89444880437341f0c6af9b0239b0cd`
- `c592a3092d614401f3629875c57c0739956f1b0baff341239f54407b107a1635`
- `2db88c3be5fee69180b6927eceddd6b664381d863593f915fbc5f8410ef9b8c5`
- `4ad3d8550939f40ebd1819d64984570f17822fa7769017e2517e7b9e100d60ce`
- `9f6f0530fe4c20e4b50b3669469287d22a7376c317b956e63463a83cd339718d`
- `0ed77cedb9ecc78caf1b6401b378afb956c878b970831126638974a554a562b8`
- `c81647288372938ece38ba686de7e6cb46400390dbd9230320c15dd3a9f64748`
- `b374e98261bc5571515eaec40d120a8cfa33d618a3fa1d2929071e3c281f5237`
- `23463298f7ace949ec472f626f9ad9905a41344318a485eb39e0199b2bfa6f12`
- `3d155fa46fdc38b18f16efa1fef967cd83bf3ce2ddc18dbc4938fe28cad40c32`
- `0cd296d0e01ac8d3a3b179dd4ad7f98b34d908e18bd92fab77b5db23f6059b78`
- `0b5e8590c1cd381adc2b5df153129626ee2fd6411479545e58ac23937f9e00ea`
- `b0c08852f0ecb3624884beab6c4980bf554cb3f073cde33bb7f7990db9bf0fbd`
- `127d67d6dc4dee4e49a25f8ab59248c1587e29ca0a12ffeecac659a660bf9dd0`

## SHA-256 与边界

- `blind_train.csv`: `ae5a1f2fe884ada4f2d414641b58ed6365c04088b1dcec0eb7aeb439392af9b8`
- `labels_train_final200.csv`: `2002dc5662d324816af6a3e10e3baf9ad1cb3cbf65f4e274db1a14dda28157a0`
- `labels_train_root_extra.csv`: `0314a6e62562461a64772e8857d52bd98fe37ae9c30bb4fbcee3b5d04fa8d5ca`

标注版本沿用 `8f1037a6987436a7c19b56766c42969526374145b8d02dde02caa5b1b750f0ec`。未查看测试材料、分数、隐藏 metadata、用户或星级；未运行训练或 Git 操作。
