# Stage 2 Track 1（Stack 1）实验设计报告

> 规范名称：Stage 2 Track 1；早期工作记录中的 “Stack 1” 指同一实验
> 模型：`Qwen/Qwen3-4B-Base` + NF4 QLoRA
> 任务：GSM8K 多轮 calculator Agent
> 训练种子 / decode seed：42 / 42
> 当前状态：`SFT→GRPO` 已完成；`Pure GRPO` 从持久 checkpoint 恢复，四臂 validation/test 正在补齐

## 1. 实验要回答的问题

Track 1 的核心问题不是“哪个 checkpoint 分数最高”，而是 **Agent-SFT 提供的行为先验是否改变 GRPO 的可优化性**。实验把总效果拆成四条路线：

```text
                              ┌─ Base（不训练）
Qwen3-4B-Base ────────────────┤
                              └─ Pure GRPO（cold start）

Qwen3-4B-Base ── Agent-SFT ───┬─ SFT only
                              └─ SFT → GRPO（warm start）
```

四臂分别支持以下受控比较：

| 比较 | 估计的效应 | 主要解释 |
|---|---|---|
| Base → SFT only | 监督行为先验 | SFT 是否教会模型进入工具协议、使用 observation 并正确停止 |
| Base → Pure GRPO | cold-start RL 总效应 | 不依赖行为示范，课程奖励本身能否让 Base 跨过结构化动作边界 |
| SFT only → SFT→GRPO | warm-start 后的 RL 增量 | 已有 Agent 行为后，GRPO 是否继续提高严格任务成功率 |
| Pure GRPO ↔ SFT→GRPO | 初始化效应 | 在相同 GRPO 数据、目标、奖励和预算下，warm start 是否改变训练可达性 |

因此 **SFT only 必须评测**。如果只有 SFT→GRPO 的分数，就无法区分“性能来自 SFT”还是“GRPO 在 SFT 基础上继续带来增益”；也不能把 warm 分支的总提升全部归因给 RL。同理，Base 是两条路线的共同原点，Pure GRPO 是检验 RL 能否从零建立 Agent 协议的必要对照。

这是一项机制导向的四臂消融，而不是严格的 FLOPs 或 wall-clock 配平实验。SFT 使用可靠 oracle action，GRPO 使用 8 倍在线采样和 verifier reward，两者监督密度与计算量不同；可以比较在当前协议下的训练路线效果，但不能据此声称普遍的 SFT-vs-RL 计算效率因果关系。

## 2. 数据边界

所有标签均来自 GSM8K official train；validation/test 标签只供 verifier 使用，不进入参数更新。

| 集合 | 行数 | 用途 | SHA-256 |
|---|---:|---|---|
| `train_rl.jsonl` | 6,726 | 两条 GRPO 分支的共同 prompt 候选池 | `ec9aecf7083386226e17f9eead206b4e045e952ac2c28080f9c9120da74fbf3e` |
| `train_sft.jsonl` | 6,637 | 可靠 Agent oracle 轨迹 | `132a54eb2266ac75ea44c4055ca6ce3661df5b80aa556ff1b602366cdbf772cb` |
| `validation_rl.jsonl` | 747 | 冻结训练后的诊断评测；规范子集取 128 题 | `a7f6c3d91f63269ab31be576d352ccbb19e6280f1fb3818e2eb4d78b8be13700` |
| `test_rl.jsonl` | 1,319 | official test 全量终态评测 | `909ce396576b5cfe2c64889c0e2159632d532b77b8854740e951aa06e41b7ca9` |

`train_sft` 比 `train_rl` 少 89 条，因为这些题没有可安全回放且与答案一致的 calculator annotation；它们可以作为无标签 RL prompt，但不能伪造成监督轨迹。数据 manifest 明确规定：SFT 只读取 train 标签，validation/test 只由 evaluator 和 verifier 读取。

## 3. 共同模型与 Agent 环境

四臂使用同一个 `Qwen/Qwen3-4B-Base` revision `main`。模型以 NF4 4-bit、double quant、bfloat16 compute 加载，并把 `<|im_end|>` 作为 EOS。所有正式 adapter 使用 rank-16 LoRA：

```text
r = 16, alpha = 32, dropout = 0.05, bias = none
target = q/k/v/o_proj + gate/up/down_proj + lm_head
```

Agent 每题最多进行 3 轮交互。模型必须生成带标签的 `calculate_math` JSON action；parser 校验标签、JSON、工具名和参数，安全 calculator 执行有界算术表达式，环境回填 tool observation，模型再继续生成动作或最终答案。

正式 strict success 同时要求：

1. 最终答案正确；
2. 工具标签平衡且结构合法；
3. required tool 实际被调用；
4. 所有调用均合法并真实执行；
5. 执行证据支持 ground truth；
6. 轨迹不是未完成状态。

因此，猜中数字、在标签外输出裸 JSON、伪造 observation、调用错误表达式后再猜答案，均不能算严格成功。

## 4. 两阶段 Agent-SFT 设计

Track 1 的正式 SFT adapter 不是一次普通语言建模微调，而是两阶段行为修复。

### 4.1 第一阶段：完整 Agent-SFT

第一阶段对全部 6,637 条可靠轨迹训练 1 epoch，使用 assistant-only cross entropy。只有 assistant action token 进入 loss；system、user 和 tool observation 只作为条件上下文：

```text
system     工具规则与 schema                   条件 token
user       数学问题                            条件 token
assistant  <tool_call>...</tool_call>           监督 token
tool       <tool_response>...</tool_response>   条件 token
assistant  最终答案                            监督 token
```

主要超参数为：最大长度 1,536，micro batch 1，梯度累积 16，学习率 `1e-4`，cosine scheduler，warmup ratio 0.03，gradient clip 1.0。该阶段让模型学习选择工具、生成参数、读取 observation、继续作答与停止。

### 4.2 第二阶段：控制 token / `lm_head` 修复

早期 probe 发现 `all-linear` LoRA 没有可靠覆盖 tied `lm_head`，导致工具结构控制 token 难以成为 Top-1。正式修复从第一阶段 adapter 继续训练，在 LoRA target 中显式加入 `lm_head`，再次完整遍历 6,637 条轨迹 1 epoch，并对以下结构 token 使用权重 8：

```text
<tool_call>   </tool_call>   <|im_end|>
```

该产物 `sft_lm_head_w8_s42_adapter` 同时承担两个角色：

- `SFT only` 的终态评测对象；
- `SFT→GRPO` 的唯一初始化起点。

这样可避免 warm 分支与 SFT-only 使用不同的监督 checkpoint，从而保证 `SFT only → SFT→GRPO` 只增加 GRPO 处理。

## 5. 两条 GRPO 分支

### 5.1 唯一设计差异：初始化

| 配置 | Pure GRPO | SFT→GRPO |
|---|---|---|
| 初始 policy | Base 上 fresh LoRA | 修复后的 SFT adapter |
| 冻结 reference | 对应起始 policy 的冻结副本 | 对应起始 policy 的冻结副本 |
| 训练 prompts | 相同 6,726 条 | 相同 6,726 条 |
| reward | 相同 shaped curriculum | 相同 shaped curriculum |
| rollout / 优化预算 | 完全相同 | 完全相同 |

除初始化外，正式配置保持一致：每个 prompt 采样 `G=8` 条轨迹，最多 3 轮，每轮最多 384 个 action token，总上下文不超过 2,500；temperature 1.0、top-k 0、top-p 1.0；训练 1 epoch、每批 1 个 prompt group、每组更新 1 epoch，学习率 `1e-6`，KL 系数 `β=0.02`，clip `ε=0.2`，gradient clip 1.0。

每个分支计划消费 6,726 个候选组，即 53,808 条 on-policy 轨迹。两条分支都在单张可见 GPU 内放置 policy 和 reference，以避免跨设备加载差异。

### 5.2 组内优势与 clipped surrogate

对同一 prompt 的 8 条轨迹，先计算组内标准化优势：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-4)
ρ_i,t = exp(log πθ(a_i,t|s_i,t) - log πold(a_i,t|s_i,t))
```

策略损失使用 PPO 风格 clipped surrogate；序列内部先对 action token 求均值，再对序列求均值，避免长输出仅因 token 更多而拥有更大权重。reference KL 使用非负 k3 估计：

```text
δ_i,t = log πref(a_i,t|s_i,t) - log πθ(a_i,t|s_i,t)
KL_k3 = exp(δ_i,t) - δ_i,t - 1
L = -mean(min(ρA, clip(ρ, 1-ε, 1+ε)A)) + β·mean(KL_k3)
```

如果组内奖励完全相同，标准化优势自然为 0，该组不会产生策略排序梯度。报告必须同时呈现成功率、group reward std 或零方差组，而不能只看“训练步数已增加”。

## 6. 训练奖励与最终指标的分工

两条 Track 1 GRPO 分支都使用同一套有界 shaped curriculum，目的是给完全不会工具协议的 Base 提供比 ±1 strict reward 更密集的中间信号。奖励考虑：

- 工具标签与 JSON 格式；
- 工具名、参数合法性和实际执行；
- required tool 覆盖；
- 最终答案 verified fraction；
- tool evidence 对 ground truth 的覆盖；
- 重复调用惩罚和 strict task success bonus。

总 shaped reward 截断到 `[-6, 6]`。标签外裸 JSON 最多贡献有界的 protocol-progress shaping；它不会被执行，也不能得到 strict success。protocol progress 对结构标记、JSON 顶层键、工具名、参数、required tool、执行和 evidence 分阶段计分，并对多个候选取最大值，避免靠重复文本累积奖励。

训练奖励与最终结论严格分离：

- **训练**使用 shaped reward，解决探索和协议可达性；
- **validation/test**始终使用 strict verifier，衡量真实 Agent 成功。

因此 shaped reward 上升只能说明模型更接近目标协议，不能单独解释为 Agent 能力提升。

## 7. On-policy 一致性与安全门

### 7.1 行为策略一致性

rollout 保存 behavior-policy action log-prob ledger，训练时重新计算 current/reference log-prob。为了使 ratio 和 Monte Carlo KL 有定义，采样必须与 ledger 的 full-softmax 分布一致，所以正式配置固定 temperature 1、top-k 0、top-p 1；不能在不修正 ledger 的情况下加入温度缩放或截断采样。

current-policy 重算时关闭 LoRA dropout，但保留 train mode 以支持 gradient checkpointing。每组 rollout log-prob mean absolute error 必须不高于 0.1。

### 7.2 KL 重尾隔离

组级 action-token mean `KL_k3` 阈值为 10。超过阈值的候选组采用 `reject_group`：

- 候选组计入已消费预算；
- 不执行 backward，不把异常梯度写入 policy；
- 保存独立诊断；
- 总拒绝数不得超过 64，连续拒绝不得超过 3；越界立即 hard stop。

该设计把孤立的重尾漂移与持续数值故障区分开。报告不能只呈现平均 KL，必须同时报告被拒绝组数、最大值和最长连续拒绝。

### 7.3 Checkpoint 与精确恢复

每 50 个候选组保存 policy/reference adapter、optimizer、scheduler、候选计数、拒绝计数和 Python/Torch/CUDA RNG 状态。恢复必须从持久 checkpoint 的计数边界继续。

Pure GRPO 曾在候选组 5,363 处人工中断。原始 metrics/log 已按 SHA-256 归档；活动 metrics 回退到持久化的第 5,350 组后恢复，避免 5,351–5,363 的临时记录与重跑结果重复。该操作改变的是日志续接边界，不改变预注册训练配置或总候选预算。

## 8. Validation、official test 与分片合并

四臂使用同一 evaluator、同一多轮工具环境和同一 strict verifier：

| 项目 | Validation | Official test |
|---|---:|---:|
| 数据集 | `validation_rl.jsonl` | `test_rl.jsonl` |
| 题目数 | 规范子集 128 / 全集 747 | 全量 1,319 |
| decode seed | 42 | 42 |
| 最大轮数 | 3 | 3 |
| 每轮最大生成 | 384 | 384 |
| sampling | T=1, top-k=0, top-p=1 | T=1, top-k=0, top-p=1 |

`require_all_runs: true` 禁止静默跳过任何一臂。每臂先生成独立 shard 和 manifest，再由 merge 脚本验证：label、题目覆盖、seed、sampling 参数、limit、数据/config/adapter hash、有限指标以及 `(seed, index)` 唯一性；全部通过后才原子写入 canonical merge。

SFT→GRPO 的 1,319 题 official-test shard 已先完成，strict task accuracy 为 67.4754%。由于 official test 已被查看，它只能作为终态报告结果，不能再用于选择学习率、reward、checkpoint 或 early stopping。后补的 validation 主要承担四臂覆盖、诊断和审计作用，不得反过来改变已经训练完成的模型。

### 8.1 需要报告的指标

最终四臂至少同时报告：

- strict task accuracy；
- answer accuracy；
- format valid rate；
- tool execution success；
- evidence coverage；
- 平均 action token 与未完成率；
- 逐题配对差异与置信区间（如适用）。

只报告 answer accuracy 会把“猜对数字”与“完成 Agent 交互”混为一谈；只报告 tool execution 又无法说明数学答案和证据是否正确。

## 9. Fail-closed 终态审计

最终结果只有通过审计才能进入正式结论。审计条件包括：

1. SFT 数据行数、assistant-only 监督、finite tensor 与 adapter hash 正确；
2. 每条 GRPO 分支都有 terminal record，恰好消费 6,726 groups / 53,808 trajectories；
3. 正常更新组与被拒绝组共同覆盖全部候选组；
4. metrics、adapter tensor、log-prob MAE 和 accepted-group KL 全部有限且在阈值内；
5. KL 拒绝数不超过 64，连续拒绝不超过 3；
6. validation/test 四臂齐全，manifest 为 `COMPLETE`；
7. 题目标签顺序、数据 hash、期望 prompt 数和 adapter hash 一致。

任一条件失败，都不得用已有部分 shard 拼成“完整四臂结果”。

## 10. 假设、判据与解释边界

### 10.1 预期模式

- 若 Base 与 Pure 都几乎没有合法工具动作，而 SFT-only 与 SFT→GRPO 有稳定执行，支持“行为先验决定协议可达性”；
- 若 SFT→GRPO 明显高于 SFT-only，支持“warm-start 后 RL 有独立增量”；
- 若 SFT-only 与 SFT→GRPO 接近，说明 warm 分支高分主要来自 SFT，而不是 GRPO；
- 若 Pure 完成后接近或超过 warm 分支，则反驳“本设置必须依赖 SFT 才能学会 Agent 协议”的强版本；
- 若 shaped reward 上升但 strict/tool execution 不升，只能说明课程奖励改善局部代理指标，不能说明 Agent 能力形成。

### 10.2 不能越过的结论边界

本实验只有一个 training seed，因此题目级 bootstrap 不能替代跨训练 seed 方差。两阶段 SFT 重复遍历同一可靠轨迹，而两条 GRPO 使用 8 倍 on-policy rollout，计算预算并不等量。official test 已用于终态报告，后续不得据此调参。结论只适用于当前 Qwen3-4B-Base、LoRA 容量、GSM8K calculator 环境、shaped reward、`G=8` 和给定预算。

## 11. 当前执行状态

截至本报告更新：

- `SFT only` adapter 已完成；
- `SFT→GRPO` 已完成 6,726/6,726 groups 和 53,808 trajectories，无 KL 拒绝；
- `SFT→GRPO` full official-test shard 已完成；
- `Pure GRPO` 的中断快照已归档，并从 group 5,350 持久 checkpoint 恢复；
- Base、SFT-only、SFT→GRPO validation 以及 Base、SFT-only official-test shard 已进入补测流程；
- Pure 完成后才运行自己的 validation/test，并执行四臂 merge 与终态 audit。

因此当前可以报告设计、单臂事实和中断机制，但完整四臂效果量仍应等待 terminal adapter、全部 shard 与审计结果。Pure 中断和恢复的证据边界见 [Stage 2 Track 1 Pure GRPO 中断与恢复记录](STAGE2_TRACK1_TERMINATION_REPORT.md)。

## 12. 可复现入口

| 文件 | 作用 |
|---|---|
| `dataset/manifests/gsm8k_agent.json` | 数据行数、边界与 SHA-256 |
| `configs/qwen3_4b/lora_sft.yaml` | 第一阶段 Agent-SFT |
| `configs/qwen3_4b/lora_sft_lm_head.yaml` | `lm_head` / 控制 token 修复 |
| `configs/qwen3_4b/pure_grpo.yaml` | cold-start GRPO |
| `configs/qwen3_4b/sft_grpo.yaml` | warm-start GRPO |
| `configs/qwen3_4b/eval_matrix.yaml` | 四臂统一评测协议 |
| `scripts/run_qwen_stage2.sh` | 原始训练与评测入口 |
| `scripts/resume_and_finalize_qwen_track1.sh` | 恢复、补测、合并与审计控制器 |
| `scripts/audit_qwen_stage2_results.py` | fail-closed 结果审计 |

