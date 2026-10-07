# miniLLM-RL：GSM8K Agentic RL 项目总结报告

> 报告日期：2026-10-07
> 项目仓库：`Mini-RL` / `gsm8k-agentic-rl`
> 正式完成范围：Stage 1 MiniMind-64M；Stage 2 Track 2 Qwen3-4B
> 负结果范围：Stage 2 Track 1 Pure GRPO 于 79.74% 预算处人工终止
> 最终回归：72/72 tests PASS
> Stage 2 Track 2 结果审计：PASS

## 1. 执行摘要

本项目完成了从 GSM8K 数据准备、Agent 工具轨迹构造、行为冷启动、在线强化学习、冻结验证、官方测试到结果审计的一整套 Agentic RL 实验闭环。模型并非只输出固定格式的数学答案，而是在 rollout 中生成 `calculate_math` 工具调用，由环境真实执行表达式、回填 observation，再由模型继续生成最终回答；严格 verifier 同时检查答案、调用、执行、必需工具覆盖和证据一致性。

项目形成了两层互补证据：

1. **Stage 1 MiniMind-64M 算法比较**：从同一个 Agent-SFT 起点独立训练 GRPO、CISPO、DAPO、GSPO，各 3 个训练种子。DAPO 在 validation 排名第一，并在未参与选择的 1,319 道 official test 上将严格成功率从 2.1986% 提升到 3.3190%，绝对提升 1.1204 个百分点。
2. **Stage 2 Track 2 Qwen3-4B 路线比较**：完成 Base、Agent-SFT(A)、GRPO(A)、GRPO(B)、Additional-SFT(B) 五臂实验。GRPO 相对 Agent-SFT 取得小幅但可测的增益；使用可靠 B 轨迹继续 SFT 达到 42.5322% 严格成功率，远高于 GRPO(B) 的 2.6535%。这说明当前模型、初始化和稀疏 strict reward 下，高质量 Agent oracle 的监督密度远高于纯在线 RL 信号。

核心结果概览：

| 阶段 | 模型规模 | 正式问题 | 最佳正式结果 | 主要结论 |
|---|---:|---|---:|---|
| Stage 1 | MiniMind 63.91M | 四种 group-relative RL 算法谁更优 | DAPO test strict 3.3190% | DAPO 相对 Agent-SFT +1.1204 pp |
| Stage 2 Track 2 | Qwen3-4B + QLoRA | Agent warm start、已见/未见 RL 数据与追加 SFT 的差异 | Additional-SFT(B) strict 42.5322% | GRPO 有小幅增益，但可靠 oracle 下追加 SFT 明显更强 |
| Stage 2 Track 1 | Qwen3-4B + QLoRA | Base / Pure GRPO / SFT only / SFT→GRPO | 未形成完整四臂结果 | SFT→GRPO 已完成；Pure 在 42,904 条 rollout 后主动终止并保留为负结果 |

![项目技术链路与两阶段实验](assets/project_summary/project_flow.png)

*图 1　项目技术链路。数据边界、真实 calculator 环境和严格 verifier 贯穿两个模型阶段。*

## 2. 研究目标与 Agentic RL 定义

### 2.1 项目要回答的问题

本项目并不把“输出一个带 XML 标签的答案”视为 Agentic RL。正式实验试图回答以下问题：

- 小模型能否通过 Agent-SFT 获得工具调用行为先验；
- 在同一行为冷启动点上，不同 group-relative 策略目标是否产生可复现差异；
- 强化学习得到的收益能否在未参与选模的 official test 上保持；
- 对更大的 Qwen3-4B，当同一批题既可用于在线 RL 又有可靠 oracle 时，GRPO 和继续 SFT 的效果有何差异；
- 稀疏奖励、控制 token、padding、精度、量化、日志与 KL 稳定性等工程因素如何影响可优化性。

### 2.2 本项目中 Agentic RL 的判定标准

一条正式轨迹经历以下状态转换：

```text
GSM8K question + tool schema
  → policy 生成 <tool_call> 和 calculate_math 参数
  → parser 校验标签、JSON、工具名与参数
  → safe_calculate 执行有界算术文法
  → environment 回填 tool observation
  → policy 基于 observation 继续生成最终答案
  → verifier 联合检查答案、调用、执行、覆盖和证据
  → reward 与组内相对优势更新 policy
```

因此它具备 Agentic RL 的三个关键成分：模型产生动作、环境改变状态并返回 observation、奖励由完整交互轨迹决定。只猜对数字、伪造工具结果、输出正确格式但没有真实执行、执行错误表达式后再猜答案，都不能得到严格成功奖励。

### 2.3 SFT 是否真正调用工具

SFT 是离线行为克隆，训练时不重新调用 calculator。数据生成阶段已经把 GSM8K 解答中的 `<<expression=result>>` 标注交给安全 calculator 回放，并只保留表达式、结果和最终答案彼此一致的可靠轨迹：

```text
system     工具规则与 schema                 条件 token
user       数学问题                          条件 token
assistant  <tool_call>...</tool_call>        监督 token
tool       <tool_response>...</tool_response> 条件 token
assistant  最终答案                          监督 token
```

因此 SFT 不只是学习“最终答案格式”，还学习选择工具、生成合法参数、结束工具调用、读取 observation、继续作答和停止；但 tool observation 是环境状态，不是模型动作，不进入 assistant-only loss。真正的在线工具执行发生在 RL rollout 和评测阶段。

## 3. 数据集、边界与可复现性

### 3.1 Stage 1 数据

GSM8K 官方 train 的 7,473 题按固定种子进行题目级拆分；官方 test 始终保持隔离。

| 文件 | 行数 | 用途 | SHA-256 |
|---|---:|---|---|
| `train_rl.jsonl` | 6,726 | 正式 RL prompt | `ec9aecf7083386226e17f9eead206b4e045e952ac2c28080f9c9120da74fbf3e` |
| `train_sft.jsonl` | 6,637 | 可靠 Agent oracle 轨迹 | `132a54eb2266ac75ea44c4055ca6ce3661df5b80aa556ff1b602366cdbf772cb` |
| `validation_rl.jsonl` | 747 | 算法选择，不更新参数 | `a7f6c3d91f63269ab31be576d352ccbb19e6280f1fb3818e2eb4d78b8be13700` |
| `test_rl.jsonl` | 1,319 | 选择结束后的最终测试 | `909ce396576b5cfe2c64889c0e2159632d532b77b8854740e951aa06e41b7ca9` |

`train_sft` 与 `train_rl` 并非题目互斥：SFT 先在训练题的可靠 oracle 上建立行为先验，RL 再在训练 prompt 上在线采样。真正需要互斥的是 train、validation、test。`train_sft` 少 89 条，是因为这些题没有可安全回放且与答案一致的 calculator annotation；它们仍可用于 RL，但不能伪造成监督标签。

assistant-only mask 抽查 128 条的结果为：

| 指标 | 结果 |
|---|---:|
| Assistant supervised tokens | 16,724 |
| Non-pad tokens | 68,921 |
| Assistant token ratio | 24.2655% |
| Mask mismatch | 0 |
| Zero-supervision rows | 0 |

### 3.2 Stage 2 Track 2 数据

官方 7,473 条 train 以 seed 42 一次性划成互斥 A/B，官方 test 仍完全隔离。

| 集合 | 行数 | 说明 |
|---|---:|---|
| A source / RL | 3,736 | Agent-SFT(A) 来源和 A 候选池 |
| B source / RL | 3,737 | GRPO(B) 与 Additional-SFT(B) 来源 |
| A reliable SFT | 3,692 | 可验证 A oracle |
| B reliable SFT | 3,686 | 可验证 B oracle |
| A compare RL | 3,686 | 与 B 等量的正式 GRPO(A) prompt |
| B compare RL | 3,686 | 与 Additional-SFT(B) 完全同题 |
| Official test | 1,319 | 只在全部训练冻结后评测 |

A∩B、A∩test、B∩test 均为空。GRPO(B) 和 Additional-SFT(B) 使用相同 3,686 个题目 ID、相同 Agent-SFT(A) 起点，但监督信号不同：前者只看到自己生成轨迹的 verifier reward，后者直接看到可靠 action oracle。因此该比较衡量当前信号条件下的训练路线差异，而不是普遍证明 SFT 总是优于 RL。

## 4. 算法原理

### 4.1 Assistant-only SFT

设所有 assistant 动作 token 的位置集合为 \(\mathcal A\)，监督目标为：

```text
L_SFT = -(1 / |A|) Σ[t∈A] log πθ(x_t | x_<t)
```

user 和 tool observation 只作为条件上下文。这样既能让模型学习 observation 后的动作，又不会让它把环境输出当成自己要生成的 token。

### 4.2 GRPO 的组内优势

每个 prompt 采样 `G=8` 条轨迹，组内标准化优势为：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + ε)
ratio_i,t = exp(log πθ(a_i,t|s_i,t) - log πold(a_i,t|s_i,t))
```

策略目标采用 PPO 风格的 clipped ratio。参考策略 KL 使用非负 k3 估计：

```text
δ = log πref - log πθ
KL_k3 = exp(δ) - δ - 1
L = -mean(clipped_surrogate) + β · mean(KL_k3)
```

如果一个组内 8 条轨迹全部成功或全部失败，奖励方差为 0，标准化优势也为 0；这正是稀疏 reward 下 loss 接近 0、训练看似运行但几乎没有学习信号的主要原因。

### 4.3 四种 group-relative 目标的差异

| 算法 | 核心设计 | 本项目中的作用 |
|---|---|---|
| GRPO | token-level ratio clipping + 组内相对优势 | 共同基线 |
| CISPO | 对重要性权重进行裁剪，保留不同的目标梯度结构 | 检查 clipping 形式是否改善稳定性 |
| DAPO | 非对称 clipping、dynamic sampling、overlong 等机制 | 聚焦有奖励方差的有效组 |
| GSPO | 先形成长度归一化的 sequence ratio，再做序列级更新 | 检查 token/sequence 归约差异 |

Stage 1 固定相同候选 rollout 预算，因此 DAPO 丢弃零方差组后有效更新数更少；这比强行让 DAPO 多采样直到获得相同有效组更公平，因为后者会额外消费大量 rollout 计算。

### 4.4 严格奖励与课程奖励

Stage 1 和 Stage 2 Track 2 的正式主结果使用严格确定性 RLVR：

```text
r = +1  当且仅当轨迹完成、答案正确、工具调用合法、
        工具真实执行、required tool 被覆盖且执行证据支持答案
r = -1  其他情况
```

Stage 2 Track 1 为解决 Base 完全不会产生协议边界的问题，引入有上界的 `protocol_progress` shaped curriculum，用于识别 JSON schema、工具名、参数合法性、可执行性和结果证据。裸 JSON 仍不会被环境执行，也不会算严格成功。该路线仍在运行，所以 shaped 结果不与 Track 2 的 strict-GRPO 正式结论混合。

## 5. 已完成的工程工作

### 5.1 数据与安全环境

- 下载并规范化 GSM8K official train/test；
- 建立固定切分、题目级泄漏检查、行数和 SHA-256 manifest；
- 实现有界算术 parser，拒绝任意 Python 求值和危险表达式；
- 统一 SFT 数据构造、RL rollout 和 evaluator 使用的 calculator 与答案解析逻辑；
- 构造多轮对话、tool observation、action mask 与严格证据验证；
- 为 reward hack、错误表达式后猜答案、substring 数字匹配、未完成轨迹等情况建立回归测试。

### 5.2 模型与训练入口

- MiniMind Dense 63.91M 的 Full-SFT、Agent-SFT、GRPO/CISPO/DAPO/GSPO 统一训练链；
- Qwen3-4B-Base 的 NF4 QLoRA、PEFT adapter、assistant-only SFT 和在线 Agentic GRPO；
- 支持 fresh adapter、已有 adapter 继续训练、冻结 reference、checkpoint/resume 和运行元数据；
- rollout 使用真实多轮工具执行，tool observation 不计入 policy action ledger；
- 训练、probe、评测、分片合并、配对统计和 fail-closed 审计均有独立脚本与固定配置。

### 5.3 关键故障与修复

| 问题 | 原因 | 修复与证据 |
|---|---|---|
| `requirements.txt` 不存在 | 在 home 而非仓库根目录执行 | 切换到项目目录；依赖本身未损坏 |
| SFT `datasets.CastError` | JSONL 顶层新增 `id`，固定 features 只声明 conversations | 改为推断顶层列；mask mismatch=0、zero supervision=0 |
| 工具 schema 异常导致 crash | 模型可能生成 dict 类型 tool name | 类型门禁；无效调用记失败，不让训练崩溃 |
| DAPO resume 超预算 | resume 段重置 candidate counter | 归档无效 run，从共同 Agent-SFT fresh 重训 |
| Qwen 首轮无合法工具调用 | inference prompt 与 SFT target 的 thinking 前缀模式不一致 | 逐轮 prompt 模式解析和严格 prefix audit |
| 结构 token 学不会 | `all-linear` LoRA 明确不覆盖 tied `lm_head` | 扩展 `lm_head` rank-16 LoRA；结构 Top-1 从 0 升至 98.09% |
| 1-step SFT smoke 不改权重 | warmup/cosine 唯一步 learning rate 为 0 | smoke 改为 2 steps 并检查参数变化 |
| GRPO KL 爆炸 | current forward 单侧启用 LoRA dropout | 保持训练模式但关闭 Dropout；复测组 KL 降至 0.00803 |
| batch 生成 padding 污染 action | 较短序列 PAD 被计入 ledger | 修复 PAD mask，只保留真实 action token |
| shard 或结果不完整也可能被合并 | 旧流程缺少完整性硬门禁 | 原子分片合并、canonical manifest、1,319 题覆盖与哈希审计 |

## 6. Stage 1：MiniMind-64M 正式结果

### 6.1 训练协议与计算量

所有 12 个正式 RL run 都从同一个 `agent_sft_768.pth` 独立加载。共同配置为 6,726 candidate groups、每组 8 条轨迹、2 policy epochs、学习率 `3e-7`、最多 3 轮工具交互、FP32 checkpoint。

| 算法 | 训练 seeds | Candidate groups | Rollout trajectories | Effective groups | Optimizer updates | Generated tokens | 单卡时长合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 3 | 20,178 | 161,424 | 20,178 nominal | 40,356 | 19,940,206 | 26.69 h |
| CISPO | 3 | 20,178 | 161,424 | 20,178 nominal | 40,356 | 19,722,940 | 33.22 h |
| DAPO | 3 | 20,178 | 161,424 | 3,745 | 7,490 | 18,196,702 | 26.65 h |
| GSPO | 3 | 20,178 | 161,424 | 20,178 nominal | 40,356 | 19,940,733 | 30.71 h |
| **合计** | **12 runs** | **80,712** | **645,696** | — | **128,558** | **77,800,581** | **117.26 h** |

所有 run 均满足预算完整、checkpoint tensor 全部有限、91/91 tensor 相对 Agent-SFT 发生改变，并保存独立 SHA-256。

### 6.2 Validation 选模

每个 checkpoint 在相同 747 题上使用 decode seeds 42/43/44。统计顺序是先合并同 checkpoint 的 decode seeds，再跨三个独立 training seeds 计算均值和样本标准差。

| 模型/算法 | Training runs | Strict Task Acc | Answer Acc | Evidence Coverage | Avg Length | KL k3 |
|---|---:|---:|---:|---:|---:|---:|
| Full-SFT | 1 | 0.2231% | 1.2941% | 1.2048% | 149.09 | 0 vs Full-SFT |
| Agent-SFT | 1 | 1.9188% | 2.0973% | 1.9634% | 123.91 | 0.42208 vs Full-SFT |
| GSPO | 3 | 2.2906% ± 0.2690 pp | 2.5584% | 2.4096% | 125.36 | 0.000692 vs Agent-SFT |
| CISPO | 3 | 2.3948% ± 0.2201 pp | 2.7369% | 2.4840% | 123.59 | 0.000993 vs Agent-SFT |
| GRPO | 3 | 2.4691% ± 0.4000 pp | 2.7220% | 2.5138% | 125.75 | 0.000984 vs Agent-SFT |
| **DAPO** | **3** | **2.9154% ± 0.3847 pp** | **3.1980%** | **2.9451%** | **114.10** | **0.014320 vs Agent-SFT** |

![Stage 1 validation 算法比较](assets/project_summary/stage1_validation.png)

*图 2　Stage 1 固定 validation 的严格任务成功率。误差线为跨 training seed 的样本标准差。*

相对 Agent-SFT 的题目级 paired bootstrap：

| 算法 | Strict Δ | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| CISPO | +0.4760 pp | [-0.1934, +1.1304] | 0.9201 |
| GRPO | +0.5503 pp | [-0.1487, +1.2494] | 0.9386 |
| GSPO | +0.3719 pp | [-0.3570, +1.0858] | 0.8385 |
| **DAPO** | **+0.9966 pp** | **[+0.2826, +1.6808]** | **0.9963** |

DAPO 按预先约定的 validation 主指标胜出。official test 在选模完成前没有用于调参、换算法或挑训练种子。

### 6.3 Official test

| 指标 | Agent-SFT | DAPO，3 training seeds | Δ |
|---|---:|---:|---:|
| Strict Task Acc | 2.1986% | **3.3190% ± 0.3047 pp** | **+1.1204 pp** |
| Answer Acc | 2.5272% | **3.6475%** | **+1.1204 pp** |
| Reward | -0.95603 | **-0.93362** | +0.02241 |
| Format Valid | **99.7220%** | 99.6967% | -0.0253 pp |
| Tool Call Valid | **99.9874%** | 99.9836% | -0.0038 pp |
| Tool Execution Success | 99.7914% | **99.8874%** | +0.0960 pp |
| Required Tool Coverage | **100.0000%** | 99.9916% | -0.0084 pp |
| Evidence Coverage | 2.2492% | **3.3274%** | **+1.0783 pp** |
| Avg Response Length | 126.19 | **114.72** | -11.47 tokens |
| KL k3 vs Agent-SFT | 0 | 0.014527 ± 0.001306 | +0.014527 |

![Stage 1 official test](assets/project_summary/stage1_test.png)

*图 3　Stage 1 official test 上 Agent-SFT 与 DAPO 的严格成功、答案准确和证据覆盖。*

DAPO 三个 training seed 的严格成功率分别为 3.2853%、3.6391%、3.0326%，均高于 Agent-SFT 的 2.1986%。题目级 paired bootstrap 给出 `+1.1204 pp`，95% CI `[+0.5391, +1.7185]`，bootstrap 中 Δ>0 的比例为 0.9999。

### 6.4 失败归因

| 模型 | Success | Answer incorrect | Evidence not grounded | Format invalid | Unfinished |
|---|---:|---:|---:|---:|---:|
| Agent-SFT | 2.199% | 97.195% | 0.329% | 0.253% | 0.025% |
| DAPO | 3.319% | 96.058% | 0.312% | 0.253% | 0.051% |

工具调用和执行率已经接近 100%，所以 Stage 1 的主要瓶颈不是 XML/JSON 外形或 calculator runtime，而是从文字题构造正确算式、多步中间结果组合、最终数字与执行证据对齐。DAPO 的收益来自正确答案和证据覆盖的同步上升，不是依靠格式 bonus 刷分。

### 6.5 效果量、稳定性与计算效率

只看 `+1.1204 pp` 容易低估或高估 Stage 1 的意义，因此同时报告绝对变化、相对变化和训练稳定性：

| 维度 | Agent-SFT | DAPO | 变化 | 解释 |
|---|---:|---:|---:|---|
| Strict Task Acc | 2.1986% | 3.3190% | +1.1204 pp / +50.96% relative | 主指标有实质增益，但绝对能力仍低 |
| Answer Acc | 2.5272% | 3.6475% | +1.1204 pp / +44.33% relative | strict 增量几乎完全由答案改善贡献 |
| Evidence Coverage | 2.2492% | 3.3274% | +1.0783 pp / +47.94% relative | 答案与执行证据的一致性同步改善 |
| Tool Execution | 99.7914% | 99.8874% | +0.0960 pp | 工具层已饱和，不是主要收益来源 |
| Avg Response Length | 126.19 | 114.72 | -11.47 / -9.09% | 更短输出伴随更高准确率，没有靠延长轨迹取胜 |

DAPO 的三个 test seed 标准差为 `0.3047 pp`，变异系数约 `9.18%`，最高与最低 seed 相差 `0.6065 pp`。三个 seed 都超过 Agent-SFT，方向一致；但样本仍只有 3 个，所以应表述为“跨三个训练 seed 稳定为正”，而不是已经精确估计了训练随机性分布。

| 算法 | Rollout/h | Generated token/s | Effective group rate | 计算解读 |
|---|---:|---:|---:|---|
| GRPO | 6,047 | 207.49 | 100% nominal | 吞吐最高，validation 增益区间仍跨 0 |
| CISPO | 4,860 | 164.94 | 100% nominal | 本轮最慢，未换来显著更高主指标 |
| DAPO | 6,058 | 189.69 | 18.56% | 大量组被 dynamic sampling 丢弃，但保留组更有学习信息 |
| GSPO | 5,257 | 180.38 | 100% nominal | 序列级归约未优于 DAPO |

DAPO 的 optimizer updates 只有其他方法约 18.6%，总 wall time 却与 GRPO 接近。这说明本项目的大头成本是生成 8 条候选轨迹，而不是 backward；dynamic sampling 能提高更新信息密度，却不能省掉已经完成的 rollout 成本。若未来希望降低计算量，需要在生成前预测低价值组或采用更便宜的探索，而不是只在生成后丢弃。

## 7. Stage 2 Track 2：Qwen3-4B 五臂正式结果

### 7.1 实验设计

| 实验臂 | 初始化 | 后续数据与目标 | 回答的问题 |
|---|---|---|---|
| Base | Qwen3-4B-Base | 无 | 未后训练模型能否完成 Agent 协议 |
| Agent-SFT(A) | Base | A reliable oracle | 行为冷启动的总体作用 |
| GRPO(A) | Agent-SFT(A) | A compare，strict RLVR | 已见 SFT 题上的在线 RL 增量 |
| GRPO(B) | Agent-SFT(A) | B compare，strict RLVR | 新题上的在线 RL 增量 |
| Additional-SFT(B) | Agent-SFT(A) | B reliable oracle | 同一 B 题上继续监督学习的增量 |

### 7.2 训练完成证据

| 分支 | 数据行/候选组 | Rollout | Optimizer updates | Wall time | KL rejection | 最大组 KL |
|---|---:|---:|---:|---:|---:|---:|
| Agent-SFT(A) | 3,692 | — | 231 SFT steps | 0.461 h | — | — |
| GRPO(A) | 3,686 | 29,488 | 3,686 | 15.208 h | 0 | 0.040555 |
| GRPO(B) | 3,686 | 29,488 | 3,686 | 15.885 h | 0 | 0.041090 |
| Additional-SFT(B) | 3,686 | — | 231 SFT steps | 0.461 h | — | — |

四个 adapter 均含 506 个有限 tensor、35,502,080 个 LoRA 参数。GRPO(A/B) 从同一 Agent-SFT(A) adapter 哈希起点独立训练，组大小、采样、候选组和优化配置一致。最终 artifact audit 检查了数据哈希、adapter 哈希、预算计数、非有限 tensor、五臂配置、每臂 1,319 条轨迹覆盖和 canonical manifest，结果为 `PASS`。

### 7.3 Official test 五臂结果

所有模型使用同一 1,319 题、decode seed 42 和 full-softmax 采样。括号内为 strict accuracy 的 95% Wilson 区间。

| 模型 | 严格成功率 | 答案准确率 | 格式合法率 | 工具执行率 | 证据覆盖率 | 平均输出 token |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.000% [0.000%, 0.290%] | 4.246% | 0.000% | 0.000% | 0.000% | 384.00 |
| Agent-SFT(A) | 0.910% [0.521%, 1.583%] | 11.372% | 8.567% | 38.666% | 5.080% | 52.88 |
| GRPO(A) | 2.578% [1.850%, 3.580%] | 14.936% | 9.022% | 42.835% | 7.809% | 60.74 |
| GRPO(B) | 2.654% [1.914%, 3.668%] | 16.907% | 9.553% | 45.603% | 8.567% | 59.43 |
| **Additional-SFT(B)** | **42.532% [39.890%, 45.218%]** | **48.294%** | **97.195%** | **99.040%** | **43.973%** | 80.30 |

![Stage 2 Track 2 五臂指标](assets/project_summary/stage2_metrics.png)

*图 4　Stage 2 Track 2 五臂 official test。Additional-SFT(B) 在协议执行与严格成功上显著领先。*

Base 的平均输出恰为生成上限 384 token，工具执行率为 0，说明它可以偶尔猜中数字，却不能稳定进入 Agent 协议并收束。Agent-SFT 使协议行为变得可达；GRPO 进一步提高答案、工具执行和证据覆盖，但 strict success 仍受长链条失败和奖励稀疏限制。

### 7.4 同题配对比较

| Treatment − Control | Strict Δ | 95% paired bootstrap CI | Exact McNemar p |
|---|---:|---:|---:|
| Agent-SFT(A) − Base | +0.910 pp | [+0.455, +1.440] | 0.000488 |
| GRPO(A) − Agent-SFT(A) | +1.668 pp | [+0.682, +2.654] | 0.001641 |
| GRPO(B) − Agent-SFT(A) | +1.744 pp | [+0.758, +2.729] | 0.000824 |
| GRPO(B) − GRPO(A) | +0.076 pp | [-1.061, +1.213] | 1.0 |
| Additional-SFT(B) − Agent-SFT(A) | +41.622 pp | [+38.969, +44.276] | 4.83e-160 |
| GRPO(B) − Additional-SFT(B) | -39.879 pp | [-42.608, -37.074] | 7.42e-138 |

![Stage 2 Track 2 配对差值](assets/project_summary/stage2_pairwise.png)

*图 5　五臂严格成功率的同题配对差值与 95% bootstrap 区间。跨 0 的 GRPO(B)−GRPO(A) 不能解释为稳定优势。*

### 7.5 行为漏斗与条件转化

把 strict success 拆成“答对、执行工具、证据覆盖、最终严格成功”可以定位每条训练路线的损失发生在哪一层。下表的计数都来自同一 1,319 道 official test；条件转化率是诊断量，不是新的优化指标。

| 模型 | Answer Acc | Tool Execution | Evidence Coverage | Strict Acc | Strict / Answer | Strict / Evidence |
|---|---:|---:|---:|---:|---:|---:|
| Base | 4.246% | 0.000% | 0.000% | 0.000% | 0.0% | — |
| Agent-SFT(A) | 11.372% | 38.666% | 5.080% | 0.910% | 8.0% | 17.9% |
| GRPO(A) | 14.936% | 42.835% | 7.809% | 2.578% | 17.3% | 33.0% |
| GRPO(B) | 16.907% | 45.603% | 8.567% | 2.654% | 15.7% | 31.0% |
| Additional-SFT(B) | 48.294% | 99.040% | 43.973% | 42.532% | 88.1% | 96.7% |

由此可见：

1. Base 约 4.25% 的答案正确全部是非 Agent 路径，answer-only 指标会把它的可用性高估；
2. Agent-SFT(A) 的工具执行率已到 38.67%，但 strict 只有 0.91%，主要损失发生在“正确表达式与证据落地”；
3. GRPO 把 `Strict/Answer` 从 8.0% 提高到约 16%–17%，说明它不仅提高猜中数字的概率，也改善了部分答案到可验证轨迹的转化；
4. Additional-SFT(B) 的 `Strict/Evidence=96.7%`，表明一旦形成正确证据，几乎都能完成最终严格轨迹；剩余主要瓶颈已转到数学答案本身。

### 7.6 训练动态、信号密度与成本

| 诊断 | GRPO(A) | GRPO(B) | 含义 |
|---|---:|---:|---|
| 全程零方差组率 | 84.75% | 87.76% | 只有约 15.25% / 12.24% 的组产生组内排序信号 |
| 末 200 组零方差率 | 82.0% | 84.5% | 后期略有改善，但稀疏性仍很高 |
| 首 200 组 strict trajectory acc | 0.8125% | 0.9375% | 起点成功密度极低 |
| 末 200 组 strict trajectory acc | 2.6875% | 2.2500% | 在线训练信号确实改善，不是完全空转 |
| 首→末工具执行率 | 35.0%→42.22% | 38.81%→44.09% | 协议执行有渐进提升 |
| 首→末 evidence coverage | 5.13%→9.19% | 5.75%→8.25% | 改善幅度仍不足以接近追加 SFT |
| 全程平均 KL k3 | 0.00442 | 0.00274 | 策略漂移总体受控 |

Additional-SFT(B) 用约 `0.461 h` 完成一轮，而 GRPO(B) 用约 `15.885 h`，观测 wall time 相差约 `34.5×`；GRPO(A) 约为 Agent-SFT(A) 的 `33.0×`。这不是严格 FLOPs 配平实验，不能当作普遍速度定律，但至少在本实现和硬件下，Additional-SFT 同时表现得更准确、更稳定且更便宜。原因是 SFT 每条样本都提供 token 级正确动作，而 GRPO 必须先付出 G=8 生成成本，且约 88% 的 B 组没有相对优势信号。

### 7.7 效果量、数据新颖性与统计解释

- GRPO(A) 相对 Agent-SFT(A) 为 `+1.668 pp`，相对提升约 `+183.3%`；GRPO(B) 为 `+1.744 pp`，相对提升约 `+191.7%`。相对数值很大是因为基线只有 0.910%，不能掩盖绝对成功率仍只有约 2.6%。
- GRPO(B) 与 GRPO(A) 的 discordant pair 为 31 对 30，几乎完全对称；`+0.076 pp` 不支持“新题 B 的 RL 泛化更强”或“已见题 A 更容易”中的任何一个方向。
- GRPO(A) 相对 Agent-SFT 的成功集合并非包含关系：34 道 treatment-only、12 道 control-only、0 道共同成功。GRPO 在重新分配成功题，而不是简单保留所有旧能力再增加新题。
- GRPO(B) 与 Agent-SFT 有 1 道共同成功、34 道 treatment-only、11 道 control-only，同样存在遗忘或随机替换。因此总准确率上升不等于逐题单调改进。
- 六项配对比较若作保守 Bonferroni 校正，阈值约为 `0.0083`；Agent-SFT vs Base、两条 GRPO vs Agent-SFT、Additional-SFT 的主要差异仍低于该阈值，B−A 仍完全不显著。该检查只缓解题目级多重比较，不解决单 training seed 限制。

### 7.8 结果解释

1. **Agent-SFT 是必要的行为冷启动。** 它把 Base 的 strict success 从 0 提升到 0.910%，并建立部分真实工具执行能力。
2. **strict-GRPO 并非完全无效。** A/B 两条 GRPO 分支相对 Agent-SFT 分别提高 1.668/1.744 pp，题目级配对区间均高于 0。
3. **没有证据表明 B 上 RL 优于 A 上 RL。** GRPO(B)−GRPO(A) 只有 +0.076 pp，区间跨 0，McNemar p=1.0。
4. **当前主要矛盾是信号密度。** GRPO 正式训练末 200 组的零方差比例约为 A 82%、B 84.5%，大量 prompt group 无法提供组内排序梯度。
5. **可靠 oracle 可用时，继续 SFT 更有效。** Additional-SFT(B) 每条样本都有正确工具动作，而 GRPO(B) 只能从自己的稀疏成功中学习；当前预算下前者形成数量级优势。

这不是“RL 失败”或“SFT 永远优于 RL”的普遍结论。它只说明在 Qwen3-4B-Base、当前 LoRA 配置、single training seed、strict reward 和高质量 oracle 可用的条件下，追加监督比稀疏 on-policy reward 更有效。

## 8. Stage 2 Track 1 终止状态

Track 1 的原计划是四臂 `Base / Pure GRPO / SFT only / SFT→GRPO`，用于直接研究 warm start 是否让课程式 Agent RL 更容易优化。它与 Track 2 的互斥 A/B 五臂实验回答不同问题。Pure 分支于 2026-10-08 01:31（Asia/Shanghai）主动终止，因此 Track 1 不再被描述为进行中，也不形成完整四臂正式结果。

| 分支 | 当前状态 | 证据 | 是否进入正式结论 |
|---|---|---|---|
| SFT only | Adapter 已完成 | 完整 SFT 与控制 token 修复产物存在 | 暂不单独报告四臂效果 |
| SFT→GRPO | **完成** | 6,726/6,726 groups；冻结 test shard strict 67.475% | 可报告单臂结果，不归因全部增益给 GRPO |
| Pure GRPO | **人工终止** | 5,363/6,726 groups，42,904 rollouts，5,352 updates | 只作为 cold-start 负结果 |
| Base | 无训练 | 作为最终统一评测基线 | 尚未形成 Track 1 正式矩阵 |

Pure 共完成计划预算的 `79.735%`，消耗 `194,575 s`（54.05 h）。停止后确认 tmux、训练 PID 和 GPU 3 显存占用均退出；滚动 checkpoint 与原始证据保留，但没有导出终态 adapter。

为了避免以单个 batch 误判，对前 200 组、末 200 组和全程 5,352 个正常更新记录做了聚合：

| Pure GRPO 指标 | 前 200 组 | 末 200 组 | 全程均值 |
|---|---:|---:|---:|
| Shaped reward | -2.354 | +0.250 | -0.674 |
| Answer accuracy | 5.375% | 43.313% | 25.997% |
| Protocol progress | 0.712 | 1.468 | 1.403 |
| Group reward std | 0.790 | 1.139 | 1.013 |
| Strict task accuracy | 0% | 0% | 0% |
| Tool execution rate | 0% | 0% | 0.00234% |
| Action tokens/group | 3,070.1 | 3,072.0 | 3,071.7 |
| KL k3 | 0.0379 | 0.0571 | 0.0416 |

这说明 shaped curriculum 提高了答案命中和“接近协议”的行为，却没有跨过可执行工具边界：末 200 组工具调用仍为 0，轨迹继续消耗 `8×384=3,072` 个 action token 并撞满长度。11 个候选组因组级 KL 超过 10 被安全拒绝；它们均为孤立重尾事件，最长连续拒绝为 1，没有非有限值。终止原因是继续计算的信息收益很低，而不是训练进程崩溃。

作为对照，已完成的 SFT→GRPO 在训练 prompt 上从前 200 组到末 200 组的 strict trajectory accuracy 由 51.0% 升到 77.0%，工具执行率由 99.32% 升到 99.94%，末 200 组 evidence coverage 为 77.88%；冻结 test shard strict accuracy 为 67.475%。这些证据支持 warm start 让可优化行为高密度存在，但因为 Pure 未完成、SFT-only/Base 未作 canonical merge，不能写成严格预算配平的四臂因果结论。

完整终止依据、发布边界和原始日志哈希见 `docs/STAGE2_TRACK1_TERMINATION_REPORT.md`。

## 9. 综合分析

### 9.1 两个阶段共同说明了什么

- **可优化行为先验比算法公式更先决。** MiniMind 通过 Agent-SFT 已能稳定执行协议，group-relative RL 才能比较；Qwen Base 在控制 token 上失配时，即使答案偶尔正确也无法进入 Agent 环境。
- **工具执行率不等于任务能力。** Stage 1 的工具执行接近 100%，strict success 仍只有约 3%，瓶颈已转移到算式规划和证据组合。
- **奖励密度决定 RL 的有效样本率。** DAPO 只保留约 18.56% 有效组；Track 2 strict-GRPO 后期约八成组零方差。训练脚本在跑、loss 有数值，不代表每组都提供学习信号。
- **监督和 RL 的比较必须明确 oracle 条件。** Additional-SFT(B) 的强结果来自密集且可靠的 action oracle，不能与没有 oracle 的现实任务直接类比。
- **评测必须 fail-closed。** 数据哈希、分片数、题目覆盖、adapter hash、预算、finite tensor 和 manifest 任一不满足，结果都不能进入正式统计。

### 9.2 协议能力、数学能力与证据能力必须分开看

本项目的指标不是同一件事的重复测量，而是一个串联成功链：

```text
进入格式 → 生成合法 tool call → 工具执行 → 得到正确中间结果
→ 最终答案正确 → 答案被执行证据支撑 → strict success
```

Stage 1 的协议与工具执行已经饱和，因此继续优化 parser 或格式 reward 的边际价值很低；Stage 2 Agent-SFT/GRPO 的工具执行只有 39%–46%，协议仍是重要瓶颈；Additional-SFT 把协议层提升到约 99%，随后数学答案准确率成为新的上限。不同阶段不能只用同一个“格式问题”解释。

### 9.3 KL、输出长度与策略退化

- Stage 1 DAPO 的 test KL 为 `0.01453±0.00131`，伴随输出缩短 9.09% 和准确率上升，属于受控改变而非长度膨胀。
- Track 2 GRPO(A/B) 的全程平均 KL 分别约 0.00442/0.00274，最大审计 KL 约 0.041，且 0 次安全拒绝；正式 strict-GRPO 数值稳定。
- Track 1 Pure 的常规 KL 均值仍小，但存在 11 个大于阈值的重尾组，最大被拒绝组 KL 达 157,931.45。安全门阻止异常组反向传播，因此不能只看平均 KL，也必须保留尾部诊断。
- Base 与 Pure 大量轨迹撞满 384 token；SFT 后平均 action token 显著下降。收束能力本身是 Agent warm start 的关键收益。

### 9.4 监督效率与 RL 适用条件

Additional-SFT 的优势不能简单归结为“算法更强”，因为它拥有 GRPO 不拥有的 action oracle。但在 oracle 已经存在的本项目条件下，拒绝使用这些标签、改用 8 倍在线采样并没有资源优势。更合理的工程路线是：先用全部可靠 oracle 建立高密度行为分布，再把 RL 用于没有 oracle、需要探索、需要优化不可微目标或需要超越示范的部分。

### 9.5 跨阶段比较的边界

MiniMind-64M DAPO 与 Qwen3-4B GRPO 的 strict accuracy 都处在约 2%–3% 区间，不能据此说模型规模没有作用。两阶段的 base checkpoint、SFT 数据覆盖、prompt 模板、adapter、reward curriculum、decode seeds 和训练预算都不同。真正可比较的是各自阶段内部的受控增量：Stage 1 比较算法目标，Track 2 比较训练路线，Track 1 比较是否 warm start。

### 9.6 可以据实声称的结论

1. 项目实现了真正的多轮 calculator Agent-SFT → Agentic RL → frozen validation/test 闭环；
2. Stage 1 完成四算法、三训练种子、共同候选预算的正式比较，DAPO 获得稳定的小幅增益；
3. Stage 2 Track 2 完成五臂 official test 和配对统计，GRPO 有小幅正增益，Additional-SFT(B) 在当前条件下显著更强；
4. 正式成功不能由格式奖励、猜答案或伪造工具结果获得；
5. 测试、权重、数据边界和结果产物均可审计。

### 9.7 不能声称的结论

1. 不能声称达到 GSM8K SOTA 或模型已具备强数学推理能力；
2. 不能把工具执行率接近 100% 等同于解题正确；
3. Stage 2 Track 2 只有一个 training seed，题目级区间不能替代跨训练 seed 方差；
4. Additional-SFT 与 GRPO 没有做严格 wall-clock/FLOPs 等预算配平；
5. Track 1 Pure GRPO 已人工终止，不能声称存在完成的 Pure-vs-warm-start 四臂结论；
6. official test 已用于最终报告，不应继续用它选择学习率、reward 或 checkpoint。

## 10. 完成度与后续优先级

| 工作项 | 状态 | 说明 |
|---|---|---|
| GSM8K 下载、切分、manifest、泄漏检查 | 完成 | 数据哈希固定 |
| 安全 calculator、多轮环境、严格 verifier | 完成 | 训练和评测共用实现 |
| Stage 1 Agent-SFT | 完成 | 6,637 reliable oracle |
| Stage 1 四算法 × 三 seeds | 完成 | 12 个正式 run |
| Stage 1 validation 选模与 official test | 完成 | DAPO 胜出 |
| Qwen3-4B QLoRA Agent-SFT/GRPO 基础设施 | 完成 | 支持 adapter 与真实工具 rollout |
| Stage 2 Track 2 五臂训练 | 完成 | A/B 边界与预算一致 |
| Stage 2 Track 2 1,319 题评测与最终审计 | 完成 | Audit PASS |
| Stage 2 Track 1 SFT→GRPO | 完成训练与单臂 test shard | strict 67.475%；无完整四臂 merge |
| Stage 2 Track 1 Pure GRPO | 人工终止 | 5,363/6,726 groups；11 次有界 KL 拒绝；无终态 adapter |

若继续推进，优先级应为：

1. 不恢复当前 Pure checkpoint；若重开 cold-start 研究，应预先注册新的协议可达性门槛与早停规则；
2. 若要完成 Track 1 四臂比较，必须重新定义并完整运行 Pure 分支，再统一评测 Base、SFT only、Pure GRPO 和 SFT→GRPO；
3. 若有额外预算，优先补 Stage 2 Track 2 的 training seeds，而不是反复在 official test 上调参；
4. 新的算法改进应在新的 validation 或嵌套验证上进行，重点优化结构化动作可达性、算式规划、证据利用和非零方差组比例。

## 11. 结果与证据索引

### 11.1 人类可读文档

- `docs/STAGE1_GSM8K_FINAL_REPORT.md`：Stage 1 正式报告；
- `docs/STAGE2_TRACK2_FINAL_REPORT.md`：Stage 2 Track 2 正式报告；
- `docs/TRACK2_AB_AGENTIC_RL_BUILD_LOG.md`：Track 2 设计、实现和逐步工作记录；
- `docs/STAGE2_QWEN3_4B_BUILD_LOG.md`：Qwen3-4B Stack/Track 1 搭建记录；
- `docs/STAGE2_TRACK1_TERMINATION_REPORT.md`：Pure GRPO 终止证据与负结果边界；
- `docs/WORKLOG.md`：项目时间线与关键修复记录。

### 11.2 Stage 1 机器可读结果

| 文件 | 内容 |
|---|---|
| `out/stage1_final/training_runs.csv` | 12 个正式 run 的预算、耗时、token 与 checkpoint 审计 |
| `out/stage1_final/validation_by_algorithm.csv` | validation 的正式算法聚合 |
| `out/stage1_final/validation_paired_bootstrap.csv` | 相对 Agent-SFT 的题目级区间 |
| `out/stage1_final/test_by_algorithm.csv` | official test 聚合 |
| `out/stage1_final/test_paired_bootstrap.csv` | DAPO vs Agent-SFT test 配对区间 |
| `out/stage1_final/failure_modes.csv` | 首要失败原因 |
| `out/stage1_final/stage1_evidence.json` | 协议、winner、计数和各类 SHA-256 |

### 11.3 Stage 2 Track 2 机器可读结果

| 文件 | 内容 |
|---|---|
| `out/stage2_track2/qwen3_4b/analysis/official_test_metrics.csv` | 五臂 test 指标与 Wilson 区间 |
| `out/stage2_track2/qwen3_4b/analysis/official_test_statistics.json` | 配对 bootstrap 与 McNemar 统计 |
| `out/run_meta/qwen3_stage2_track2_results_audit.json` | 数据、训练、adapter、轨迹和 manifest 最终审计 |
| `dataset/manifests/gsm8k_track2.json` | A/B/test 行数、边界和内容哈希 |

### 11.4 Stage 2 Track 1 终止证据

| 文件 | 内容 |
|---|---|
| `results/stage2_track1/pure_grpo_termination_summary.json` | 终止计数、前后窗口指标、安全拒绝和原始日志哈希 |

## 12. 最终结论

项目已完成两个可以独立审计的 Agentic RL 正式实验层级，并保留一个可审计的冷启动负结果。Stage 1 证明：在已经掌握工具协议的小模型上，group-relative RL 能带来真实但有限的提升，DAPO 在相同候选预算下表现最好。Stage 2 Track 2 进一步证明：Agent-SFT 提供了必要的行为先验，strict-GRPO 能继续改进，但奖励稀疏使有效学习组不足；当可靠的完整 Agent oracle 可用时，继续监督学习在当前设置下远强于单纯依赖在线严格奖励。Track 1 Pure 则表明，dense shaping 可以改善答案和局部协议分数，却未必能让 base model 跨过结构化工具执行边界。

最重要的项目成果不是单一准确率数字，而是一条可复现、可归因、可拒绝伪成功的工程与实验链：数据边界明确，工具在环境中真实执行，reward 与证据绑定，训练预算可核验，validation 与 test 职责分离，结果经过逐题统计与 fail-closed 审计。这使后续对 reward curriculum、模型规模、工具规划或多 seed 稳定性的研究有了可信基线。
