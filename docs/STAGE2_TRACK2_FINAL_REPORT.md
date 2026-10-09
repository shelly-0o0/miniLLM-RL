# Stage 2 Track 2 最终报告：Qwen3-4B Agent-SFT 与 Agentic GRPO

## 1. 结论摘要

本 Track 在 Qwen3-4B-Base 上完成了一个可审计的五臂 Agentic RL 实验：Base、Agent-SFT(A)、GRPO(A)、GRPO(B) 和 Additional-SFT(B)。GSM8K 官方 test 的 1,319 道题只在训练和超参数决策全部冻结后使用。最终 artifact audit 为 `PASS`。

主要结果不是“GRPO 全面胜出”，而是：

1. Agent-SFT(A) 使模型从 Base 的 0% 严格成功率提升到 0.910%，并学会部分真实工具协议；
2. GRPO(A/B) 分别达到 2.578%/2.654%，相对 Agent-SFT(A) 提升 1.668/1.744 个百分点；
3. GRPO(A) 与 GRPO(B) 的差异仅 0.076 个百分点，题目级配对区间跨 0，不能声称新数据 B 上的 RL 优于已见数据 A；
4. Additional-SFT(B) 达到 42.532%，显著高于 GRPO(B) 的 2.654%。在当前小模型、可靠 oracle 可用且 strict reward 稀疏的条件下，继续监督学习远比直接 strict-GRPO 有效。

这是一项有价值的负结果：RL 确实带来可测的小幅改进，但奖励密度不足以替代高质量 Agent oracle 监督。

## 2. 研究问题与五臂设计

GSM8K 官方训练集先按题目 ID 确定性划为互斥 A/B 池，官方 1,319 题 test 始终隔离。所有 downstream 分支从同一个 Agent-SFT(A) adapter 独立分叉，避免串行继承造成归因混乱。

| 实验臂 | 初始化 | 后续数据与目标 | 回答的问题 |
|---|---|---|---|
| Base | Qwen3-4B-Base | 无 | 未后训练模型能否完成工具协议 |
| Agent-SFT(A) | Base | A 池可靠 oracle，assistant-only SFT | RL 系统冷启动训练的作用 |
| GRPO(A) | Agent-SFT(A) | A 池在线 rollout，strict RLVR | 已见题上继续 RL 的作用 |
| GRPO(B) | Agent-SFT(A) | 与 A 互斥的 B 池在线 rollout，strict RLVR | 新题上在线 RL 的作用 |
| Additional-SFT(B) | Agent-SFT(A) | B 池可靠 oracle，assistant-only SFT | 同一 B 数据上继续 SFT 与 RL 的差异 |

Additional-SFT(B) 与 GRPO(B) 使用相同 B 题 ID 和相同 A 起点，但监督强度并不相同：前者直接看到验证过的工具轨迹，后者只通过自己生成的轨迹和最终 verifier 奖励学习。因此结果说明的是“在当前监督可用性与预算下的训练路线差异”，不能外推成所有场景下 SFT 必然优于 RL。

## 3. 为什么它是真正的 Agentic RL

GRPO 的每条轨迹不是单轮输出一个答案，而是：

```text
question
→ policy 生成 <tool_call> 与 calculate_math 参数
→ 环境解析并执行有界算术表达式
→ observation 回填对话
→ policy 根据执行结果继续生成最终答案
→ verifier 联合检查数值答案、格式、required tool、真实执行和结果证据
→ 组内相对优势与 KL 正则更新 LoRA policy
```

严格任务成功要求轨迹完成、工具调用合法且真实执行、覆盖 required tool、最终答案正确，并且答案得到执行结果支持。只猜对数字、输出伪造 observation 或仅满足格式都不能获得成功奖励。

GRPO 每个问题采样 `G=8` 条轨迹。对组内奖励 `r_i` 标准化得到优势：

```text
A_i = (r_i - mean(r)) / (std(r) + eps)
```

策略项使用 clipped importance ratio；参考模型 KL 使用非负 k3 估计：

```text
ratio = exp(log πθ - log πold)
δ = log πref - log πθ
KL_k3 = exp(δ) - δ - 1
```

rollout 固定为 `temperature=1, top_k=0, top_p=1`，使采样分布与训练重算的完整 softmax 一致。组级 KL 在任何 backward 前检查；非有限值硬失败，有限但超过阈值的重尾组有界拒绝且不更新参数。

## 4. 训练完成证据

| 分支 | 数据行/候选组 | Rollout | Optimizer update | 安全拒绝 | 最大组 KL | 最大 logprob MAE |
|---|---:|---:|---:|---:|---:|---:|
| Agent-SFT(A) | 3,692 | — | 231 SFT steps | — | — | — |
| GRPO(A) | 3,686 | 29,488 | 3,686 | 0 | 0.040555 | 0.061996 |
| GRPO(B) | 3,686 | 29,488 | 3,686 | 0 | 0.041090 | 0.047598 |
| Additional-SFT(B) | 3,686 | — | 231 SFT steps | — | — | — |

四个 adapter 均包含 506 个有限 tensor、35,502,080 个 LoRA 参数。GRPO(A/B) 都从同一个 Agent-SFT(A) 哈希起点训练，候选预算、G、采样和优化配置一致。

## 5. 官方测试结果

所有模型在完全相同的 1,319 题、decode seed 42 和 full-softmax 采样设置下评测。严格成功率区间为按题目计算的 95% Wilson 区间。

| 模型 | 严格成功率 | 答案准确率 | 格式合法率 | 工具执行率 | 证据覆盖率 | 平均输出 token |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.000% [0.000%, 0.290%] | 4.246% | 0.000% | 0.000% | 0.000% | 384.00 |
| Agent-SFT(A) | 0.910% [0.521%, 1.583%] | 11.372% | 8.567% | 38.666% | 5.080% | 52.88 |
| GRPO(A) | 2.578% [1.850%, 3.580%] | 14.936% | 9.022% | 42.835% | 7.809% | 60.74 |
| GRPO(B) | 2.654% [1.914%, 3.668%] | 16.907% | 9.553% | 45.603% | 8.567% | 59.43 |
| Additional-SFT(B) | 42.532% [39.890%, 45.218%] | 48.294% | 97.195% | 99.040% | 43.973% | 80.30 |

Base 的平均输出恰为 384 token，且工具调用为 0，证实其主要问题是不会稳定进入 Agent 协议并收束。GRPO 提升了答案、工具执行和证据覆盖，但严格成功仍低，说明正确数字与“由真实工具结果支持的完整轨迹”之间存在巨大差距。

## 6. 同题配对比较

以下差值均在同一道题上配对。区间为 20,000 次 paired bootstrap；p 值为 exact two-sided McNemar。

| Treatment − Control | 严格成功率差值 | 95% 配对区间 | McNemar p |
|---|---:|---:|---:|
| Agent-SFT(A) − Base | +0.910 pp | [+0.455, +1.440] pp | 0.000488 |
| GRPO(A) − Agent-SFT(A) | +1.668 pp | [+0.682, +2.654] pp | 0.001641 |
| GRPO(B) − Agent-SFT(A) | +1.744 pp | [+0.758, +2.729] pp | 0.000824 |
| GRPO(B) − GRPO(A) | +0.076 pp | [−1.061, +1.213] pp | 1.0 |
| Additional-SFT(B) − Agent-SFT(A) | +41.622 pp | [+38.969, +44.276] pp | 4.83e−160 |
| GRPO(B) − Additional-SFT(B) | −39.879 pp | [−42.608, −37.074] pp | 7.42e−138 |

GRPO(A) 与 GRPO(B) 分别有 34 个只由 treatment 成功的题；相对 Agent-SFT(A)，control-only success 为 12/11，说明改进不是简单包含关系。GRPO(B) 与 GRPO(A) 则是 31 对 30，几乎完全对称。

## 7. 为什么 strict-GRPO 改进有限

1. **奖励稀疏。** 正式训练末 200 组的零方差比例为 A 82%、B 84.5%。这些组的标准化优势为 0，无法提供组内策略排序信号。
2. **成功链条较长。** 模型必须同时学会 schema、工具参数、真实执行、observation 利用、最终答案和证据对齐。任何一环失败都会失去 strict 成功。
3. **起点只使用一半 oracle。** Agent-SFT(A) 在官方 test 上只有 0.910% 严格成功率，初始 on-policy 成功密度偏低。
4. **继续 SFT 的信号密度高。** Additional-SFT(B) 每条样本都有可靠 action oracle；它几乎消除了工具协议问题，使优化主要聚焦于算式与答案。
5. **没有事后修改目标。** 虽然 shaped reward 在 probe 中能产生方差，正式 A/B 仍保持预注册的 strict RLVR，避免看到稀疏性后更换目标并把课程结果冒充正式结果。

因此，正确结论不是“RL 无效”，而是“在这个初始化、模型规模和奖励定义下，strict-GRPO 有小幅有效提升，但远不足以替代可靠 Agent oracle”。

## 8. 审计与结果产物

最终审计验证：

- A/B/test 题目边界与 SHA-256；
- 两个 SFT 和两个 GRPO 的终态记录；
- GRPO 候选组、轨迹、更新与拒绝计数；
- 所有 adapter tensor 有限；
- 五臂配置、adapter 哈希和 1,319 条轨迹覆盖；
- canonical manifest 状态为 `COMPLETE`。

审计结果为：

```text
Stage 2 Track 2 result audit: PASS
```

关键入口：

- 构建全过程：`docs/TRACK2_AB_AGENTIC_RL_BUILD_LOG.md`
- 正式评测：`scripts/evaluate/evaluate_qwen_stage2.py`
- 分片原子合并：`scripts/evaluate/merge_qwen_stage2_shards.py`
- 终态审计：`scripts/audit_qwen_track2_results.py`
- 配对统计：`scripts/analyze_qwen_track2_results.py`
- 机器可读结果：`out/stage2_track2/qwen3_4b/analysis/official_test_statistics.json`

## 9. 结论边界

- 只有一个训练 seed 和一个 decode seed。题目级置信区间不能替代跨训练 seed 方差；“显著”只表示这次训练下的逐题配对差异。
- 没有做相同 wall-clock/FLOPs 的 SFT/RL 预算配平；Additional-SFT 使用 oracle，而 GRPO 使用在线稀疏奖励。
- 结果适用于 Qwen3-4B-Base、当前 LoRA 配置、calculator 环境和 GSM8K，不等于普遍的 SFT/RL 排名。
- Stack1 Pure GRPO 与 SFT→GRPO 的长程四臂实验没有作为本 Track 完成条件，不能用本报告声称 full-scale Base-init-vs-Agent-SFT-init GRPO 结论。

在这些边界内，Track 2 已完整回答预设问题：Agent-SFT 提供必要行为先验；strict Agentic GRPO 可以进一步改善，但信号稀疏；当可靠工具 oracle 可获得时，继续 SFT 是当前最有效的路线。

## 10. 多维补充分析

### 10.1 从最终分数拆解行为漏斗

以下计数都来自同一 1,319 道 official test：

| 模型 | Answer Acc | Tool Execution | Evidence Coverage | Strict Acc | Strict/Answer | Strict/Evidence |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4.246% | 0.000% | 0.000% | 0.000% | 0.0% | — |
| Agent-SFT(A) | 11.372% | 38.666% | 5.080% | 0.910% | 8.0% | 17.9% |
| GRPO(A) | 14.936% | 42.835% | 7.809% | 2.578% | 17.3% | 33.0% |
| GRPO(B) | 16.907% | 45.603% | 8.567% | 2.654% | 15.7% | 31.0% |
| Additional-SFT(B) | 48.294% | 99.040% | 43.973% | 42.532% | 88.1% | 96.7% |

Base 约 4.25% 的正确答案全部缺乏 Agent 行为，说明 answer-only accuracy 会高估部署价值。Agent-SFT 和 GRPO 已经获得 38%–46% 的工具执行率，但大部分执行没有转化为正确证据。Additional-SFT 则几乎消除了“有证据但 strict 失败”的损失，剩余上限主要由数学答案准确率决定。

### 10.2 在线训练是否真的在学习

| 指标 | GRPO(A) 前 200 | GRPO(A) 末 200 | GRPO(B) 前 200 | GRPO(B) 末 200 |
|---|---:|---:|---:|---:|
| Strict trajectory acc | 0.8125% | 2.6875% | 0.9375% | 2.2500% |
| Answer accuracy | 11.00% | 17.94% | 14.44% | 15.94% |
| Tool execution | 35.00% | 42.22% | 38.81% | 44.09% |
| Evidence coverage | 5.13% | 9.19% | 5.75% | 8.25% |
| Zero-variance groups | — | 82.0% | — | 84.5% |

所以 GRPO 不是 loss 近零的完全空转：末段 strict、工具执行和证据覆盖都高于初段。但全程零方差组率仍为 A 84.75%、B 87.76%，绝大多数候选组没有提供相对排序梯度。这解释了为何改善存在，却远小于密集 oracle 监督。

### 10.3 计算成本与监督密度

| 分支 | Wall time | 相对同侧 SFT | 信号形式 | Official-test strict |
|---|---:|---:|---|---:|
| Agent-SFT(A) | 0.461 h | 1.0× | 每个 assistant token 有 oracle | 0.910% |
| GRPO(A) | 15.208 h | 33.0× | 29,488 on-policy trajectories | 2.578% |
| Additional-SFT(B) | 0.461 h | 1.0× | 每个 assistant token 有 oracle | 42.532% |
| GRPO(B) | 15.885 h | 34.5× | 29,488 on-policy trajectories | 2.654% |

这些是观测 wall time，不是严格 FLOPs 配平；序列长度、生成与 teacher forcing 的内核路径也不同。但在当前实现中，Additional-SFT 不仅更准确，也明显更便宜。GRPO 的计算先用于生成 G=8 轨迹，而 B 全程约 87.76% 的组又没有组内奖励方差，形成双重效率损失。

### 10.4 效果量与成功集合

- GRPO(A/B) 相对 0.910% 基线的相对提升为约 183.3%/191.7%，但绝对值仍只有 2.578%/2.654%。相对百分比不能代替绝对部署成功率。
- GRPO(A) vs Agent-SFT 的 discordant pair 是 34:12，且共同成功为 0；GRPO(B) vs Agent-SFT 为 34:11，共同成功仅 1。RL 在改变哪些题会成功，而不是严格保留所有基线成功题。
- GRPO(B) vs GRPO(A) 为 31:30，只有 4 道共同成功，说明两条单-seed policy 的成功集合差异很大，而总分几乎相同。
- Additional-SFT(B) vs GRPO(B) 为 542:16，另有 19 道共同成功；优势不是少量边缘题造成的。

### 10.5 统计强度与边界

六项配对比较做保守 Bonferroni 校正时，阈值约为 `0.0083`。Agent-SFT vs Base、GRPO(A/B) vs Agent-SFT、Additional-SFT 相关主差异仍低于该阈值；GRPO(B) vs GRPO(A) 仍为 `p=1.0`。但这些 p 值和 bootstrap 区间只使用题目作为抽样单位，不能覆盖重新训练 adapter 的波动。

因此 Track 2 的证据强度应分层表述：

1. “这一次训练得到的五个 checkpoint 在同题 test 上有差异”——证据充分；
2. “GRPO(A/B) 在新的 training seed 上仍会稳定提高约 1.7 pp”——尚未验证；
3. “SFT 普遍优于 RL”——实验设计不支持；
4. “在可靠 oracle 已存在、当前 QLoRA 和 strict reward 下，追加 SFT 是更优工程选择”——与本轮数据一致。

### 10.6 数值稳定性与策略漂移

GRPO(A/B) 全程平均 KL k3 约为 0.00442/0.00274，最大审计 KL 约为 0.0406/0.0411，均无安全拒绝和非有限 tensor。rollout logprob MAE 在训练前后没有恶化，说明性能受限主要不是优化爆炸，而是可用 reward signal 太少。单纯进一步收紧 KL 很可能降低探索，放宽 KL 也不会自动创造正确轨迹；优先级应放在更好的 Agent-SFT 冷启动训练、过程反馈或提高有效组比例。
