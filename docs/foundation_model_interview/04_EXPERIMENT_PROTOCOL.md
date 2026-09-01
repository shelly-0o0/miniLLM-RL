# 实验与数据真实性规范

## 1. 什么才叫“真实量化结果”

必须同时满足：真实运行、固定未见测试集、明确 checkpoint、完整配置、可追踪 commit、逐样本证据、至少三训练 seed、均值与方差、没有挑最好一次冒充平均结果。

训练日志的 Reward 上升不是最终结果；上游论文数字也不是个人结果；随机小模型 smoke test 只证明代码路径可运行。

## 2. 实验目录建议

```text
out/
├── run_meta/
│   ├── git_commit.txt
│   ├── pip_freeze.txt
│   ├── nvidia_smi.txt
│   └── agent_split_manifest.json
├── metrics/
│   ├── grpo_s42.jsonl
│   ├── grpo_s43.jsonl
│   ├── grpo_s44.jsonl
│   ├── ...
│   └── algorithm_comparison.csv
├── eval_rlvr/
│   ├── summary.csv
│   ├── algorithm_comparison.csv
│   └── trajectories.jsonl
├── agent_grpo_s42_768.pth
├── agent_cispo_s42_768.pth
├── agent_dapo_s42_768.pth
└── agent_gspo_s42_768.pth
```

每次 run 使用独立 `metrics_path` 和 `save_weight`，禁止多个 seed 追加到同一个 checkpoint 名。

## 3. 两阶段实验

### 3.1 Smoke/正确性阶段

- 数据 16～64 条；
- `max_gen_len=64/128`；
- 每算法 2～5 updates；
- 打开 debug；
- 检查 reward、success、mask、old/current/ref logp 对齐；
- 第一次 policy forward 的 `rollout_logprob_mae` 接近 0、`rollout_ratio_mean` 接近 1；
- ratio 第二个 policy epoch 后不再恒为 1；
- DAPO 只保留部分成功组；
- GSPO sequence ratio 不是 token ratio 的算术平均。

此阶段结果不能写简历。

### 3.2 正式阶段

- 固定训练、验证、测试 manifest；
- 四算法从同一 `agent_sft` 初始化；`agent_sft` 只使用训练集 oracle 工具轨迹，评测问题与标签不参与冷启动；
- 主对照统一 `reward_mode=strict`，shaped curriculum 单列消融；
- 三个训练 seed；
- 固定 update budget 和解码设置；
- 记录额外 rollout budget；
- 每隔固定 update 在验证集评测，最终只在测试集一次性报告；
- 测试轨迹全量保存。

## 4. 一键训练

首次实验先运行 `scripts/split_agent_dataset.py` 固化 train/eval 和 manifest；切分命令见端到端指南第 4 节。之后才执行：

```bash
MM_GPUS=1 \
MM_SEEDS="42 43 44" \
MM_DATA_PATH="../dataset/agent_rl_math_train.jsonl" \
MM_BATCH_SIZE=2 \
MM_GENERATIONS=8 \
MM_POLICY_EPOCHS=4 \
bash scripts/run_rlvr_ablation.sh
```

若工具任务与数学任务分开，再各跑一套 `MM_DATA_PATH`，不要在最后只给混合总分。

## 5. 固定集评测

训练 seed 的权重名全部列出：

```bash
python scripts/eval_agent_rlvr.py \
  --weights \
agent_sft,agent_math_grpo_s42,agent_math_grpo_s43,agent_math_grpo_s44,\
agent_math_cispo_s42,agent_math_cispo_s43,agent_math_cispo_s44,\
agent_math_dapo_s42,agent_math_dapo_s43,agent_math_dapo_s44,\
agent_math_gspo_s42,agent_math_gspo_s43,agent_math_gspo_s44 \
  --reference_weight agent_sft \
  --data_path dataset/agent_rl_math_test.jsonl \
  --seeds 42 \
  --limit 256 \
  --num_generations 1 \
  --output_dir out/eval_rlvr
```

`agent_sft` 是未做 RL 的共同初始化，必须和训练后权重跑同一固定集，才能计算真实改善。这里评测 seed 固定为 42；跨训练 seed 的方差来自 12 个训练后 checkpoint。若要衡量 sampling variance，可再用多个评测 seed，但统计时需做两层方差分解，不能把 3 个 eval seed 当作 3 个独立训练 run。

## 6. 训练指标聚合

```bash
python scripts/summarize_rl_metrics.py 'out/metrics/*.jsonl' \
  --last_n 20 \
  --output out/metrics/algorithm_comparison.csv
```

该表使用每 run 最后 N 条日志的均值，然后再跨 seed 汇总。它描述训练末期动态，不替代固定测试集。

固定集结果单独聚合：

```bash
python scripts/summarize_eval_results.py \
  out/eval_rlvr/summary.csv \
  --output out/eval_rlvr/algorithm_comparison.csv
```

该脚本先平均同一 checkpoint 的解码 seed，再把不同训练 seed checkpoint 当作独立 run 计算均值和 sample std。

## 7. 主要指标定义

```text
Task Accuracy = successful trajectories / all evaluated trajectories
Tool Execution Rate = successfully executed calls / parsed calls
Tool Evidence Coverage = GT items supported by replayed tool results / all GT items
Unfinished Rate = trajectories hitting max turns / all trajectories
Dynamic Acceptance = effective prompt groups / candidate prompt groups
Clip Fraction = clipped valid action tokens or sequences / valid units
Temporal Stability = std(metric over final N logged updates)
Seed Stability = std(final metric across independent training seeds)
```

KL 为 sampled k3 estimator；长度只数 assistant action token，不数 prompt/tool observation。

`max_total_len` 必须覆盖完整的 prompt、assistant action 和 tool observation。代码对超限轨迹直接报错，不允许 rollout 后左截断；否则 old/current log-prob 的条件上下文不一致，ratio、clip fraction 和 KL 都不再可信。

## 8. 统计报告

每算法至少给：

```text
mean = (1/n) Σ x_i
sample std = sqrt(Σ(x_i-mean)²/(n-1))
```

三 seed 很少，标准差估计不稳定，需如实说明。样本级准确率可对固定测试集做 bootstrap CI；训练 seed 仍是更高层随机性，不能只对轨迹 bootstrap 后假装覆盖训练方差。

本项目最终主表（固定集能力 + 训练采样成本）为：

| Algorithm | Math Acc | Tool Success | 两任务 Δ vs SFT | Math/Tool candidate groups | Math/Tool wall time per run |
|---|---:|---:|---:|---:|---:|
| GRPO | 48.4375% | 5.7292% | 0 / 0 | 10.0 / 10.0 | 32.41 / 28.97 s |
| CISPO | 48.4375% | 5.7292% | 0 / 0 | 10.0 / 10.0 | 32.79 / 28.80 s |
| DAPO | 48.4375% | 5.7292% | 0 / 0 | 160.3 / 103.7 | 487.04 / 285.19 s |
| GSPO | 48.4375% | 5.7292% | 0 / 0 | 10.0 / 10.0 | 28.77 / 28.16 s |

响应长度在数学/Tool 固定集分别都是 80.19/92.34 tokens；四算法任务指标的跨训练 seed 标准差均为 0。DAPO 的候选组与 wall time 明显增加，但当前预算没有固定集能力收益。

## 9. Ablation 设计

DAPO 逐项：

1. GRPO；
2. + asymmetric clip；
3. + soft overlong；
4. + token mean；
5. + dynamic sampling。

GSPO：

1. GRPO token ratio；
2. 只换 sequence ratio/clip；
3. Dense vs MoE；
4. clip grid；
5. policy update epochs 1/2/4。

CISPO：

1. GRPO；
2. CISPO sample mean（上游旧实现风格）；
3. CISPO 论文 token mean；
4. upper clip grid；
5. beta 0 vs KL penalty。

每次只改变一个因素。不能把新数据、更长输出、更多 rollout 与新算法同时引入后把全部提升归因于算法。

## 10. 失败实验也要保留

记录失败原因类别：OOM、NaN、ratio/clip 爆炸、reward hacking、verifier error、environment timeout、dynamic buffer 不足、SGLang policy sync 失败。面试时一个有证据的失败定位往往比一条漂亮曲线更能证明工程能力。

## 11. 简历数字写法模板

本项目完成实测后的可用写法：

> 在同一 Agent SFT 初始化、相同 10-group 更新上限和 3 个训练随机种子下，对 GRPO/CISPO/DAPO/GSPO 进行统一对照；DAPO 在组合 Tool-Use 中将保留组零方差率降至 0%，但候选组/action tokens 分别约为普通算法的 10.4/10.9 倍。固定集数学准确率/工具成功率为 48.4375%/5.7292%，四算法相对 baseline 的 delta 均为 0，定位到 64M 模型当前 `10 updates / 1e-7` 条件只完成机制与成本验证、收益不显著。

这类负结果不能改写成“显著提升”。若后续扩大预算取得收益，必须另设 validation 调参并一次性报告未触碰的 test，保留当前配置作为预注册负对照。

## 12. 面试证据包

- 30 秒：简历三条和最终主表。
- 3 分钟：架构图、训练阶段图、四算法差异图、Agent trajectory。
- 10 分钟：loss 公式、old/ref、mask、dynamic buffer、reward hacking 案例。
- 深挖：展示 unit test、某条 false positive 轨迹、SGLang staleness 修复和失败实验。

每个数字能回答四个问题：在哪个数据集？哪个 commit/config？几次独立训练？原始逐样本证据在哪里？
