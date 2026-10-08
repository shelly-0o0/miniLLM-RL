# Stage 2 SVAMP warm-GRPO 结果

起点为 GSM8K Track 2 的 `Additional-SFT(B)` adapter；在 816 条 SVAMP train prompt 上运行 816 groups × 8 trajectories 的 warm-start GRPO，再在隔离的 184 题 holdout 上与未更新 adapter 作同题比较。

| 模型 | Strict | Answer | Format valid | Tool execution | Evidence | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|
| Additional-SFT(B) zero-shot | 67.935% | 71.739% | 98.913% | 99.457% | 70.109% | 39.91 |
| SVAMP warm GRPO | **76.087%** | **79.891%** | **99.457%** | **99.457%** | **76.630%** | 38.69 |
| 绝对变化 | **+8.152 pp** | **+8.152 pp** | +0.543 pp | 0 pp | **+6.522 pp** | -1.22 |

同题配对统计：

- GRPO-only 成功：26 题；
- zero-shot-only 成功：11 题；
- 共同成功：114 题；共同失败：33 题；
- strict 差值的 20,000 次 paired-bootstrap 95% CI：`[+1.630, +14.674] pp`；
- exact McNemar：`p=0.0200739`。

训练完成 816/816 groups、6,528 trajectories、816 optimizer updates，耗时 11,515.21 秒（3.20 单卡小时），KL rejection 为 0；最大 rollout log-prob MAE 为 0.010271，最大 KL k3 为 0.068794。数据、adapter、训练预算、holdout 覆盖和 manifest 终态审计为 `PASS`。

训练首 100→末 100 组的 strict trajectory accuracy 从 65.000% 升至 77.125%，evidence coverage 从 66.375% 升至 77.625%，平均 action token/group 从 335.52 降至 326.15；全程平均 KL k3 为 0.001817。

区间以 184 道 holdout 题为抽样单位，只描述单 training/decode seed 42 下的逐题差异，不能替代跨训练 seed 重复。
