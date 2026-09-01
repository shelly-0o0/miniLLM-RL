# RL 不增长诊断与后续优化报告

## 1. 本轮要解决的问题

上一轮已经在统一框架中实现并测试 GRPO、CISPO、DAPO、GSPO，但固定集结果显示：数学任务的四算法准确率都为 `48.4375%`，组合 Tool-Use 都为 `5.7292%`，与共同 Agent-SFT 初始化完全一致。这个结果能证明代码路径可运行，却不能证明 RL 改善了模型。

本轮不改变评测口径来“制造提升”，而是依次回答四个问题：

1. RL checkpoint 是否真的保存了参数更新？
2. importance ratio 是否比较了同一数值精度、同一 token 轨迹上的行为策略与当前策略？
3. 严格二值奖励是否给出了足够的组内方差？
4. Tool-Use 模型是否已经达到适合在线 RL 的冷启动成功率？

## 2. 根因定位

### 2.1 FP16 落盘抹除了小学习率更新

旧 RL 使用约 `1e-7` 的学习率，却把所有训练权重强制转换为 FP16 后保存。逐参数比较发现：

- 数学 GRPO 相对 Agent-SFT 的相对 L2 变化只有约 `5.19e-7`，仅 `0.8458%` 参数在 FP16 文件中不同；
- 数学 DAPO 的相对 L2 变化约 `1.02e-6`，仅 `1.029%` 参数不同；
- 数学与 Tool-Use 的 GSPO checkpoint 与初始化逐参数完全相同。

这说明优化器可能产生过微小更新，但 FP16 的量化间隔把大部分更新舍入回原值。对应修复位于 `trainer/train_agent.py::checkpoint_state_dict`：RL 默认以 FP32 保存，BF16/FP16 只作为显式可选的部署精度。

### 2.2 rollout log-prob 与训练 log-prob 不在同一数值条件下

旧实现直接把 KV-cache rollout 保存的 BF16 log-prob 当作行为策略概率，再与重新前向得到的概率计算比率。即使参数尚未更新，缓存路径、精度和轨迹序列化差异也会使 ratio 偏离 1。GSPO 的序列级 clip 很窄，偏差足以让整条序列被裁剪，最终 policy loss 为 0。

修复后，训练侧在完全相同的 packed multi-turn token 轨迹上重新计算行为策略 log-prob；rollout 引擎的值保留为同步审计指标。第一次更新前应满足：

```text
rollout_logprob_mae ≈ 0
rollout_ratio_mean  ≈ 1
```

GSPO smoke test 中 clip fraction 恢复为 0，但五个 prompt group 全部零方差，因此 checkpoint 仍不变化。这把“错误裁剪”和“奖励无方差”两个原因成功分离。

### 2.3 严格二值奖励导致大量零方差组

GRPO 系方法使用同一 prompt 的组相对优势。若一组 `G` 条轨迹全部成功或全部失败，则标准化后的 advantage 全为 0；增加 policy epoch 也不会产生梯度。

旧训练中 GRPO/CISPO 的 zero-variance group rate 约为 90%。DAPO 会丢弃这些组，因此算法没有错误，但需要生成大量候选轨迹才能凑够有效组。Tool challenge 上原 Agent-SFT 的成功率太低，动态采样有效率只有几个百分点。

### 2.4 Tool-Use 冷启动能力不足

原 Agent-SFT 在组合 Tool challenge 训练提示上的严格成功率不足 1%，固定 holdout 也只有约 5%。此时直接在线 RL 的绝大多数采样都是失败轨迹，reward 无法指出“已经正确调用一个工具但漏掉另一个工具”比“完全乱答”更接近目标。

因此，本轮采用经典的 cold-start → online RL 顺序：先只用训练划分中的 oracle 工具轨迹做少量课程 SFT，把策略带入可探索区域，再使用严格、可验证的 DAPO 奖励。课程数据不读取评测答案，也不使用 holdout 问题。

## 3. 代码优化

### 3.1 保留真实 RL 更新

`trainer/train_agent.py` 新增：

- `--checkpoint_dtype {float32,bfloat16,float16}`，默认 `float32`；
- 原子写入临时文件后 `os.replace`，防止中断留下半个 checkpoint；
- 单元测试同时检查默认 FP32 与显式 FP16 行为。

一次有效 DAPO smoke update 的 FP32 权重相对初始化变化为：相对 L2 `9.87e-6`，约 `98.60%` 参数发生非零变化，证明优化器更新不再被落盘精度抹掉。

### 3.2 修复行为策略重要性比率

在 optimizer step 之前，使用当前未更新策略对 exact packed trajectory 重新前向，得到 `old_per_token_logps`；参考模型也在相同 autocast 精度下计算。rollout 保存概率只计算同步误差，不再直接污染 PPO/DAPO/GSPO ratio。

这不是把 old policy 设成 current policy：同一批轨迹的第一次更新前二者相同是 on-policy 的定义；第二个及后续 policy epoch 仍使用冻结的 old log-prob，因此 ratio 会随参数更新而变化。

### 3.3 提升可探索性与奖励可诊断性

新增 `--rollout_temperature`，允许不改代码地校准组内探索。奖励函数同时输出：

- 最终答案是否正确；
- 格式是否合法；
- 工具调用参数是否合法；
- 工具是否成功执行；
- 必需工具覆盖率；
- observation 证据覆盖率；
- 是否在最大轮数/长度处未完成。

`strict` 模式仍只用可验证任务成功作为优化目标；`shaped` 模式加入工具覆盖、证据覆盖和无关工具惩罚，仅用于课程诊断，不能替代正式 task success。

### 3.4 构造无泄漏、类别平衡的 Tool 课程集

`scripts/prepare_agent_sft_data.py::balanced_limit` 从 challenge 训练划分按类别确定性抽样。课程集共 96 条，六类各 16 条：

- exchange_then_math；
- time_and_unit；
- translation_and_math；
- unit_then_math；
- weather_and_exchange；
- weather_time_then_math。

课程问题与 96 条 holdout 的精确交集为 0。训练脚本新增 `--save_resume 0`，短程课程 SFT 不保存体积较大的 Adam 状态。

## 4. Tool-Use 优化实验

### 4.1 公平协议

- 训练硬件：单张 RTX 4090 24GB；
- 基座：64M Dense Agent-SFT；
- 课程 SFT：96 条 challenge-train oracle 轨迹，2 epochs，batch 8，BF16，LR `5e-6`；
- DAPO：从课程 SFT 初始化，严格二值 reward，`G=4`，30 个有效 prompt group，2 policy epochs，LR `1e-7`，温度 `0.9`，FP32 checkpoint；
- 评测：固定 96 条 challenge holdout，解码种子 101/102/103，每个 checkpoint 共 288 条轨迹；
- 三个解码 seed 是同一个训练 checkpoint 的重复生成测量，不冒充三个独立训练 seed。

### 4.2 固定集结果

下表是三个解码 seed 的均值；括号内为解码 seed 间样本标准差。

| Checkpoint | Reward | Task Acc | Format | Tool Exec | Required Coverage | Evidence Coverage | Avg Len | Unfinished | KL vs Agent-SFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原 Agent-SFT | -0.9028 (0.0120) | 4.8611% (0.6014pp) | 98.9583% | 71.1806% | 56.9444% | 38.0208% | 92.81 | 0% | 0 |
| 课程 Agent-SFT | -0.2847 (0.0318) | 35.7639% (1.5912pp) | 99.6528% | 90.9722% | 85.9954% | 47.2222% | 100.97 | 0.3472% | 0.03450 |
| 课程 SFT + DAPO | -0.2778 (0.0120) | 36.1111% (0.6014pp) | 98.9583% | 90.5671% | 85.5324% | 47.9167% | 100.94 | 1.0417% | 0.03515 |

相对原 Agent-SFT：

- 课程 SFT 将准确率提高 `30.9028pp`，工具执行率提高 `19.7917pp`，必需工具覆盖率提高 `29.0509pp`；
- DAPO 最终相对原模型提高 `31.25pp`；
- DAPO 相对课程 SFT 只增加 `0.3472pp` 准确率和 `0.6944pp` 证据覆盖率，同时格式、执行率和 unfinished 略有退化。

因此可以确认 cold-start 解决了主要瓶颈；当前单训练 seed、30 有效组只说明 DAPO 没有把收益完全破坏，并产生很小的正向点估计，不能声称该增量具有统计显著性。

### 4.3 DAPO 训练动态

- 为获得 30 个有效组共采样 280 个候选组，即 1,120 条轨迹；最终动态采样接受率 `10.7143%`；
- 2 policy epochs 产生 60 次 optimizer step；总训练时间约 `891.36s`；
- 有效组平均训练准确率 `59.17%`，组内 accuracy 只可能为 25%/50%/75%；
- clip fraction 均值 `0.00170`，最大 `0.02036`；
- rollout log-prob MAE 均值 `0.00123`，rollout ratio 均值 `0.99990`；
- local k3 KL 的 token P95 均值 `5.11e-4`；
- 平均响应长度 `110.85`，P95 均值 `114.69`，没有出现持续长度爆炸；
- FP32 DAPO checkpoint 相对课程 SFT 的相对 L2 为 `1.165e-5`，`99.449%` 参数有非零变化。

这些数字证明动态采样、非对称 clip、token-level loss 和旧策略比率确实进入了训练；它们不等价于证明 DAPO 比其他算法更高分。

### 4.4 修复后的四算法补充对照

为避免只用 DAPO 代表全部后训练算法，又从完全相同的课程 Agent-SFT、训练 seed 242、`G=4`、30 个 nominal update groups、2 policy epochs 和 LR `1e-7` 补跑 GRPO、CISPO 与 GSPO。四个 FP32 checkpoint 相对共同初始化的权重变化为：

| Algorithm | Relative L2 | Non-zero parameter fraction |
|---|---:|---:|
| GRPO | `8.012e-6` | 97.822% |
| CISPO | `8.459e-6` | 95.940% |
| DAPO | `1.165e-5` | 99.449% |
| GSPO | `7.472e-6` | 98.781% |

这证明修复后四种算法都产生了可保存的真实更新。成本口径并不相同：GRPO/CISPO/GSPO 各消费 30 个候选组（120 条轨迹，约 94–98 秒），DAPO 为得到 30 个非零方差组消费 280 个候选组（1,120 条轨迹，891 秒）。GSPO 在已记录的第 20/30 组出现 sequence ClipFrac `1.0/0.5`，说明论文大模型配置的 `3e-4/4e-4` 序列裁剪区间对本 64M 策略偏紧。

因此这里的“相同预算”严格指相同目标 update groups 和 optimizer steps，不是相同 rollout tokens、FLOPs 或墙钟时间；面试时必须同时报告 DAPO 的额外采样成本。

固定 96 条 Tool holdout、三个解码 seed、以课程 SFT 为 KL reference 的最终结果如下。`Decode std` 是同一 checkpoint 的生成随机性，不是跨训练 seed 稳定性。

| Checkpoint | Reward | Task Acc | Delta vs curriculum | Decode std | RL-only KL | Tool Exec | Avg Len | Unfinished |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 课程 SFT | -0.28472 | 35.7639% | 0pp | 1.5912pp | 0 | 90.9722% | 100.97 | 0.3472% |
| GRPO | -0.28472 | 35.7639% | 0pp | 3.0070pp | `7.40e-6` | 90.9722% | 100.67 | 0.6944% |
| CISPO | -0.29861 | 35.0694% | -0.6944pp | 2.4056pp | `1.71e-5` | 89.8148% | 101.07 | 0.6944% |
| DAPO | -0.27778 | 36.1111% | +0.3472pp | 0.6014pp | `3.24e-5` | 90.5671% | 100.94 | 1.0417% |
| GSPO | -0.28472 | 35.7639% | 0pp | 3.0070pp | `1.25e-5` | 90.6250% | 100.74 | 0.6944% |

四算法因此都完成了实现、真实更新和统一评测，但没有任何一个在当前单训练 seed、30-update 预算下显示可靠的显著收益。DAPO 点估计最高且三个解码 seed 的波动最小，但只比课程 SFT 多 `0.3472pp`，同时 unfinished 更高、候选轨迹成本约为其他算法的 9.33 倍；不能据此宣称 DAPO 优于另外三种方法。GRPO/GSPO 的均值与课程模型相同，CISPO 点估计略低。

## 5. 数学推理后续优化

数学基线已经具有约 46%–48% 的严格成功率，不需要额外课程 SFT。本轮直接从共同 Agent-SFT 初始化运行 DAPO：`G=4`、温度 `0.9`、30 个有效组、2 policy epochs、LR `1e-7`、FP32 保存。

训练动态进一步暴露了零方差成本：

- 30 个有效组来自 492 个候选组，接受率仅 `6.0976%`；
- 共生成 1,968 条轨迹、执行 60 次 optimizer step，耗时约 `1,217.56s`；
- 有效组平均 task accuracy `58.33%`，reward 的时间标准差 `0.2887`；
- clip fraction 均值 `0.00131`，rollout log-prob MAE `0.00158`，ratio 均值 `0.99975`；
- token local k3 KL 的 P95 均值 `5.88e-5`，平均响应长度 83.0；
- FP32 checkpoint 相对 Agent-SFT 的相对 L2 为 `1.008e-5`，`99.419%` 参数发生非零变化。

固定 128 条数学 holdout、三个解码 seed 的结果为：

| Checkpoint | Reward | Task Acc | Format | Avg Len | Unfinished | KL vs Agent-SFT |
|---|---:|---:|---:|---:|---:|---:|
| Agent-SFT | -0.06771 | 46.6146% | 71.6146% | 79.87 | 28.3854% | 0 |
| Agent-SFT + DAPO | -0.06250 | 46.8750% | 71.6146% | 80.35 | 28.3854% | `7.66e-6` |

DAPO 的点估计仅提高 `0.2604pp`，相当于 384 条解码轨迹中约多 1 条成功；分 seed 的 accuracy 变化分别为 `+1.5625pp`、`-2.3438pp`、`+1.5625pp`，方向不一致。因此结论仍是：修复后模型参数确实更新，数学固定集没有显示具有统计意义的收益。继续放大动态采样轮数会增加 94% 无梯度候选的成本，下一步更应改进难度分层、探索温度或过程级可验证信号。

## 6. 最终结论与简历边界

可以据实表述：

> 定位 64M 模型 RLVR 固定集不增长来自 FP16 checkpoint 舍入、行为策略 log-prob 数值不一致及严格二值奖励零方差；将 RL 权重改为 FP32 保存，在 exact multi-turn token 轨迹上重算行为策略概率，并构造无 holdout 泄漏的类别平衡 Tool-Use cold-start 数据。96 条独立 Tool holdout、三个解码种子下，严格任务成功率由 4.86% 提升至 35.76%；再从同一课程 SFT、相同 30-update 目标补跑 GRPO/CISPO/DAPO/GSPO，准确率分别为 35.76%/35.07%/36.11%/35.76%。四算法均有真实参数更新，DAPO 点估计最高但独立增量仅 +0.35pp 且 rollout 成本约 9.33 倍，未宣称统计显著或算法优越性。

不能表述：

- “DAPO 单独把成功率从 4.86% 提升到 36.11%”；
- “三个解码 seed 已证明跨训练 seed 稳定性”；
- “DAPO/GSPO 在本项目上显著优于 GRPO/CISPO”；
- “shaped reward 的上升等价于严格任务成功率上升”。

## 7. 可复现入口与证据

- 核心训练：`trainer/train_agent.py`；
- 四算法目标：`trainer/policy_optimization.py`；
- 课程数据：`scripts/prepare_agent_sft_data.py`；
- 数学优化训练：`scripts/run_math_dapo_optimization.sh`；
- 数学固定集评测：`scripts/run_math_optimization_eval.sh`；
- 固定集汇总：`scripts/summarize_eval_results.py`；
- 训练指标汇总：`scripts/summarize_rl_metrics.py`；
- 论文和官方仓库映射：[SOURCES.md](./SOURCES.md)。

远程复现命令（长任务使用 `screen`）：

```bash
cd /root/minimind
screen -dmS rl_opt_math_dapo bash -lc \
  'cd /root/minimind && bash scripts/run_math_dapo_optimization.sh'
screen -ls
tail -f out/optimization/logs/opt_math_dapo_strict_s342.log

# 训练结束并确认 out/opt_math_dapo_strict_s342_768.pth 存在后执行
screen -dmS rl_opt_math_eval bash -lc \
  'cd /root/minimind && bash scripts/run_math_optimization_eval.sh'
tail -f out/optimization/logs/math_128_3seed_eval.log

screen -dmS rl_opt_tool_ablation bash -lc \
  'cd /root/minimind && bash scripts/run_tool_corrected_ablation.sh'
# 三条补跑完成后，与已有 DAPO 做统一评测
screen -dmS rl_opt_tool_ablation_eval bash -lc \
  'cd /root/minimind && bash scripts/run_tool_corrected_ablation_eval.sh'
```

Tool-Use 本轮核心产物：

```text
out/optimization/eval/tool_full_3seed/summary.csv
out/optimization/eval/tool_full_3seed/checkpoint_comparison.csv
out/optimization/eval/tool_corrected_ablation_3seed/summary.csv
out/optimization/eval/tool_corrected_ablation_3seed/checkpoint_comparison_vs_curriculum.csv
out/optimization/eval/math_128_3seed/summary.csv
out/optimization/eval/math_128_3seed/checkpoint_comparison.csv
out/optimization/metrics/tool_corrected_ablation_training.csv
out/optimization/metrics/opt_tool_dapo_strict_s242.jsonl
out/optimization/logs/opt_tool_dapo_strict_s242.log
/root/autodl-tmp/minimind_optimized_weights/
```
