# Results

本目录保存经过审核、适合直接提交到普通 Git 的小型实验结果。原始 rollout、训练日志、checkpoint、模型权重和大体积 JSONL 不进入仓库；它们应保存在实验服务器、Git LFS、GitHub Release 或模型/数据平台中。

## Stage 1

`stage1_final/` 对应 MiniMind-64M 的 Agent-SFT、GRPO、CISPO、DAPO、GSPO 正式比较：

| 文件 | 内容 |
|---|---|
| `training_runs.csv` | 12 个正式 RL run 的预算、耗时、token、checkpoint hash 与权重变化审计 |
| `validation_by_checkpoint.csv` | 每个 checkpoint 合并三个 decode seed 后的 validation 指标 |
| `validation_by_algorithm.csv` | 跨 training seed 的正式算法聚合 |
| `validation_paired_bootstrap.csv` | 四算法相对 Agent-SFT 的题目级区间 |
| `test_by_checkpoint.csv` | Agent-SFT 与三个 DAPO checkpoint 的 official test 指标 |
| `test_by_algorithm.csv` | official test 的算法级聚合 |
| `test_paired_bootstrap.csv` | DAPO 相对 Agent-SFT 的 paired bootstrap |
| `failure_modes.csv` | validation/test 首要失败类型 |
| `stage1_evidence.json` | 数据、代码、协议、预算、winner 与 SHA-256 证据 |
| `generated_artifacts.sha256` | Stage 1 生成结果的校验和 |

## Stage 2 Track 2

`stage2_track2/` 对应 Qwen3-4B 的 Base、Agent-SFT(A)、GRPO(A)、GRPO(B)、Additional-SFT(B) 五臂 official test：

| 文件 | 内容 |
|---|---|
| `official_test_metrics.csv` | 五臂 1,319 题指标与 Wilson 区间 |
| `official_test_statistics.json` | 20,000 次 paired bootstrap 与 exact McNemar 统计 |
| `TRACK2_OFFICIAL_TEST_REPORT.md` | 机器结果的短报告 |
| `final_results_audit.json` | 数据哈希、训练预算、adapter、轨迹覆盖和 canonical manifest 审计 |

完整解释、图表和结论边界见 [`docs/PROJECT_SUMMARY_REPORT.md`](../docs/PROJECT_SUMMARY_REPORT.md)。数据切分哈希见 `dataset/manifests/`。

## Stage 2 Track 1

`stage2_track1/pure_grpo_termination_summary.json` 保存 cold-start Pure GRPO 的人工中断快照和恢复元数据。该 run 曾在 5,363/6,726 组时停止，随后从第 5,350 组持久 checkpoint 恢复；该文件不是完成的四臂 Track 1 结果，最终应由 terminal audit 取代。

完整实验设计见 [`docs/STAGE2_TRACK1_EXPERIMENT_REPORT.md`](../docs/STAGE2_TRACK1_EXPERIMENT_REPORT.md)；Pure 的中断与恢复证据见 [`docs/STAGE2_TRACK1_TERMINATION_REPORT.md`](../docs/STAGE2_TRACK1_TERMINATION_REPORT.md)。

## 完整性

仓库根目录执行以下命令可以核对本目录内容：

```bash
sha256sum -c results/RESULTS_MANIFEST.sha256
```
