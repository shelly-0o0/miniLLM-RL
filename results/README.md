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

`stage2_track1/` 保存 Qwen3-4B Base、Pure GRPO、SFT-only、SFT→GRPO 四臂正式结果：

| 文件 | 内容 |
|---|---|
| `TRACK1_OFFICIAL_TEST_REPORT.md` | 1,319 题四臂 official-test 简报 |
| `official_test_metrics.csv` | 四臂指标与 Wilson 95% 区间 |
| `official_test_statistics.json` | paired bootstrap 与 exact McNemar 统计 |
| `final_results_audit.json` | 数据、训练预算、adapter、validation/test 覆盖和终态审计 |
| `training_dynamics.json` | Pure/warm 训练窗口、累计 token 与拒绝计数 |
| `pure_grpo_termination_summary.json` | Pure 的历史中断、恢复和最终完成元数据 |

Track 1 终态审计为 `PASS`；official-test strict 为 Base 0%、Pure 0%、SFT-only 37.604%、SFT→GRPO 67.475%。Pure answer 达 36.922% 但工具执行为 0；warm GRPO 相对 SFT-only 增加 29.871 pp。

完整实验设计和分析见 [`docs/STAGE2_TRACK1_EXPERIMENT_REPORT.md`](../docs/STAGE2_TRACK1_EXPERIMENT_REPORT.md)；Pure 的中断与恢复证据见 [`docs/STAGE2_TRACK1_TERMINATION_REPORT.md`](../docs/STAGE2_TRACK1_TERMINATION_REPORT.md)。

## Stage 2 SVAMP

`stage2_svamp/` 保存从 GSM8K Additional-SFT(B) adapter warm-start、在 SVAMP 上继续 GRPO 的跨数据集实验：

| 文件 | 内容 |
|---|---|
| `SVAMP_OFFICIAL_TEST_REPORT.md` | 184 题 holdout 指标、配对统计和解释边界 |
| `official_test_metrics.csv` | 两臂 holdout 指标与 Wilson 95% 区间 |
| `official_test_statistics.json` | 20,000 次 paired bootstrap 与 exact McNemar 统计 |
| `final_results_audit.json` | 数据、训练预算、adapter、holdout 覆盖和 manifest 审计 |

SVAMP warm-GRPO 完成 816 groups / 6,528 trajectories，审计 `PASS`；holdout strict 从 67.935% 提升到 76.087%，绝对增量 8.152 pp。

## 完整性

仓库根目录执行以下命令可以核对本目录内容：

```bash
sha256sum -c results/RESULTS_MANIFEST.sha256
```
