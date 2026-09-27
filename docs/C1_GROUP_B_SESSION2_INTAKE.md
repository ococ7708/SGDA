# C1 组员 B（Session 2）结果接收记录

接收日期：2026-09-27
状态：**运行产物通过机器核验并已合并；复现来源材料尚不完整**

## 1. 组员自述信息

- 负责人：组员 B
- 数据范围：SEED-IV Session 2
- 实验方案：E0–E5
- 随机种子：42、43、44
- 完成情况：270/270，无失败 run
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- 运行设备：`cuda:0`
- 组员声明未修改代码或配置
- 数据路径：`D:\脑机数据\脑机接口\SEED_IV\`
- CLIP 路径：`D:\脑机数据\脑机接口\local_clip_model`
- 组员声明环境文件为 `pytorch_environment.yml`
- 组员声明数据哈希文件为 `seediv_data_hashes.csv`
- Git commit SHA：组员使用解压目录运行，无法提供

以上内容来自组员文字总结；除归档中可交叉验证的字段外，不等同于机器采集的运行 provenance。

## 2. 接收归档

六个分卷完整，7-Zip 22.01 识别为同一套 6-volume 归档。归档 CRC 全量测试结果为 `Everything is Ok`。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `结果.7z.001` | 524288000 | `30EC7ECEF5536733E4F7EF9AA6AFFE0AEA0CE3AEDF496AD357C72C6C203BA45C` |
| `结果.7z.002` | 524288000 | `7566757D1689E0A35BB4953872C3D9D8D4014B2FDC20EEF79392A2907BED16F8` |
| `结果.7z.003` | 524288000 | `86FA82D99E5DDC86DB9F0C3328212738A1A6BD0B4BF358AAF514DA08C2C8181E` |
| `结果.7z.004` | 524288000 | `72FA46786C736033BACF023E9506D36864C281CA899C0F8A85EA41ED2A785EFA` |
| `结果.7z.005` | 524288000 | `4B5D25570E6EABD84529E079D9D71333DA776FEA9BF1D2EF5889A819AF5F2AA5` |
| `结果.7z.006` | 413950362 | `BEEBEA8B51AA1F4AFCB5429823834FAE6E20C754B64DCE22DBA95715B6F9A004` |

归档解压统计：295 个目录、1622 个文件、解压后 3,684,462,655 字节。其中 1620 个文件属于 270 个运行目录，另有 `fold_results.csv` 和 `summary.json`。

## 3. 机器核验结果

- 顶层 effective-config hash：`592351bfb649e601`，与当前正式配置一致。
- 270 个预期 cell 全部存在：6 variants × 3 seeds × 15 targets。
- 每个变体均为 45 个 cell；所有 cell 均严格属于 Session 2。
- 每个 cell 均存在且非空：`COMPLETE`、`run_manifest.json`、`metrics.json`、`epoch_metrics.json`、`last.pt`、`target_best.pt`。
- 所有 `COMPLETE` 内容均为 `PASS`。
- 270 个 `run_manifest.json` 均可解析；variant、seed、session、target 与目录一致。
- 270 个 manifest 的 `effective_config` 与当前 `configs/seediv_c1_full45.json` 解析出的正式配置完全一致，且 `smoke=false`、`original_sgda_modified=false`。
- 270 个 `metrics.json` 与各自 200-epoch 历史的首次最高 accuracy、best epoch 一致；数值均为有限值。
- 归档自带 `fold_results.csv` 含 270 行；`summary.json` 正确记录其余两个 session 共缺 540 cells。
- 2026-09-27 追加严格审计：逐一读取全部 270 个 `target_best.pt`，均可由 PyTorch 正常加载；checkpoint 内 epoch/metrics 与外部 `metrics.json` 一致，并且对应各自 200-epoch 历史中首次达到最高 target accuracy 的 epoch。异常数为 0。
- 对解压暂存目录和正式整合目录的全部 1622 个文件逐个计算 SHA-256：文件集合完全相同，缺失 0、额外 0、内容哈希差异 0。由此确认整合过程未改变归档中的任何文件字节。

## 4. 合并位置与阶段汇总

结果已合并到：

```text
results/seediv_c1/592351bfb649e601/
```

随后使用以下命令重新生成部分汇总：

```powershell
C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe analysis\summarize_c1_results.py --allow-partial
```

当前状态为 `complete=270, missing=540`。Session 2 的 target-best accuracy 均值如下；每项 `n=45`，这里只是接收校验统计，不是完整三 session 的最终结论。

| Variant | Mean accuracy | Sample SD |
|---|---:|---:|
| E0 | 75.9354% | 9.4281% |
| E1 | 76.1791% | 9.6159% |
| E2 | 76.2330% | 9.7181% |
| E3 | 76.3690% | 8.7595% |
| E4 | 76.2670% | 9.3343% |
| E5 | 75.6746% | 9.4430% |

## 5. 未闭合的 provenance 项

1. 归档未包含组员所述的 `pytorch_environment.yml`。
2. 归档未包含组员所述的 `seediv_data_hashes.csv`。
3. 无 Git commit SHA，因此无法仅凭产物独立证明运行代码与某个仓库提交逐字节一致。配置哈希和 manifest 与当前正式配置一致，只能证明配置层面一致。
4. GPU 型号、数据路径、CLIP 路径和“未修改代码”目前是组员声明；现有 runner manifest 只记录逻辑设备 `cuda:0` 和 `original_sgda_modified=false`，未自动采集这些外部事实。

在论文复现或最终归档前，应向组员 B 补收环境文件与数据 hash 文件；Git SHA 缺失应作为既成限制保留，不得补造。后续 Session 1/3 结果只有在同一配置哈希和同样的 cell 级核验通过后，才合并到本目录。三份齐全后必须去掉 `--allow-partial` 运行正式汇总。
