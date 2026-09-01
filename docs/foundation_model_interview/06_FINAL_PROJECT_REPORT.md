# 轻量级大语言模型全流程训练与 Agent RLVR 系统：最终项目报告

> 本报告是项目总入口。它说明实际完成了什么、为什么这么实现、如何复现、证据在哪里，以及面试时哪些结论可以说、哪些不能说。公式和逐行代码推导分别见文末的专题报告链接。

> 后续更新：初始四算法 Formal 之后又完成了 checkpoint 精度、行为 log-prob 对齐和 Tool 冷启动优化；新增数字与结论边界见 [09_RL_OPTIMIZATION_FOLLOWUP.md](./09_RL_OPTIMIZATION_FOLLOWUP.md)。本文原有负结果继续保留，不做事后删除。

## 1. 项目定位与结论边界

项目以 MiniMind commit `393e387e9ad99f0f04c296e4c5e7353f4444629f` 为工程基线，在单张 RTX 4090 23.52 GiB、PyTorch 2.6.0+cu124、BF16 环境上完成以下工作：

1. 跑通 64M Dense Decoder-only 模型的 Pretrain、SFT、LoRA、DPO、知识蒸馏与 Agent SFT；
2. 对 RMSNorm、RoPE/YaRN、GQA、KV Cache、SwiGLU、PyTorch SDPA Flash kernel 与 Top-1 稀疏 MoE 做代码级和系统级验证；
3. 构建可确定性重放的多轮工具环境与 `Assistant → Tool → Observation → Assistant` 轨迹训练；
4. 在统一训练器内实现并测试 GRPO、CISPO、DAPO、GSPO；
5. 以固定数据划分、共同 SFT 初始化、三训练 seed 和三解码 seed 对比 Reward、准确率、KL、长度、格式、工具执行与稳定性；
6. 保存原始 JSONL、CSV、逐轨迹记录、环境信息和 SHA-256，不用论文数字或训练 loss 冒充本机泛化结果。

需要明确：这是一个小模型、单卡、小预算的机制复现项目。它能够证明训练链路、算法目标、Agent 状态机、verifier 和实验方法正确可运行，但不能据此声称达到工业模型质量，也不能把部分训练的 MoE 描述成优于 Dense。

## 2. 交付物地图

| 问题 | 首选文档 | 核心代码/证据 |
|---|---|---|
| 从租服务器到最终验收怎样执行 | [端到端执行指南](./00_END_TO_END_GUIDE.md) | `scripts/run_*.sh`、`out/logs/` |
| Pretrain/SFT/LoRA/KD 的原理与项目拷问 | [训练链路报告](./01_TRAINING_PIPELINE_REPORT.md) | `trainer/train_*.py`、独立 holdout JSON |
| 多轮 Agent、工具环境、action mask、RLVR reward | [Agentic RLVR 报告](./02_AGENTIC_RLVR_FROM_ZERO_TO_MASTERY.md) | `trainer/train_agent.py`、数据审计、逐轨迹 JSONL |
| GRPO/CISPO/DAPO/GSPO 公式到张量 | [策略优化报告](./03_POLICY_OPTIMIZATION_FROM_ZERO_TO_MASTERY.md) | `trainer/policy_optimization.py`、算法单测 |
| 如何保证结果可信 | [实验规范](./04_EXPERIMENT_PROTOCOL.md) | split manifest、配置日志、checksum |
| 本机最终数字与可用口径 | [真实实验结果](./05_ACTUAL_EXPERIMENT_RESULTS.md) | `out/run_meta/experiment_evidence.json` |
| 论文、官方仓库与原创性边界 | [来源与归属](./SOURCES.md) | 文件级 provenance 映射 |

## 3. 从基座到后训练的实际工作流

### 3.1 环境冻结

首先记录 Git commit、PyTorch/CUDA/cuDNN、GPU 型号、BF16 支持、`pip freeze` 与 `nvidia-smi -q`。所有长任务放入 `screen`，日志使用 `tee` 持久化。这样 SSH 断开不会杀死训练，结果也能追溯到准确环境。

主要目录约定：

```text
dataset/       持久化训练/评测数据与固定划分
out/           权重、评测、指标与运行元数据
checkpoints/   可选断点恢复状态
scripts/       数据准备、训练矩阵、评测与审计入口
tests/         公式、架构、mask、verifier 回归测试
docs/          执行指南与面试报告
```

### 3.2 Dense Pretrain

主模型为 hidden size 768、8 层、8 个 Query heads、4 个 KV heads，参数量 63,912,192。预训练从随机权重开始，使用 causal next-token loss、BF16、梯度累积和 cosine decay。梯度累积的作用是用多个 micro-batch 模拟更大的 effective batch；混合精度降低激活显存；梯度裁剪限制异常更新。

本次在 mini 数据上训练 2 epochs。训练 loss 只能证明优化过程发生，不能代替独立测试集能力。

### 3.3 Assistant-only SFT

SFT 从 Pretrain checkpoint 初始化。label 仅保留 assistant span，system/user/tool observation 与 padding 全部写成 `-100`，交叉熵不对这些位置反传。项目提供逐 token 审计脚本，而不是只凭肉眼相信 chat template。

本次通用 SFT 训练 2 epochs。审计同时检查：

- 实际 label mask 与根据 assistant BOS/EOS 重建的期望 mask 一致；
- supervised token 的 label 等于 input token；
- pad token 不参与监督；
- 每个样本至少有一个监督 token。

### 3.4 LoRA

LoRA 冻结基础线性层，在目标层加入低秩增量 `ΔW = BA`。本次仅训练 393,216 个 adapter 参数，占完整参数 0.61%。评测先固定 90/10 数据划分，再把同一批 token 物化后分别送给 base 与 LoRA，避免随机模板增强造成输入不一致。

独立医疗 holdout 上，LoRA 相对 base 的 token-weighted loss 下降 7.62%，PPL 下降 12.15%。该结论只覆盖此医疗划分，不代表通用能力或医疗安全性提高。

### 3.5 知识蒸馏

学生模型为 hidden size 512、8 层、30,025,216 参数，比教师少 53.02%。白盒 logits KD 使用：

```text
L = α · CE(y, p_student) + (1-α) · T² · KL(p_teacher^T || p_student^T)
```

其中 `T²` 补偿温度缩放带来的梯度量级变化。实验用相同 student 初始化、相同 50k 子集、相同顺序和更新预算比较 CE control 与 KD，避免把更多数据或更多 step 当成蒸馏收益。

本次 KD 是诚实负结果：相对 CE control，holdout loss 上升 0.258%，PPL 上升 0.532%。这说明链路实现完成，但 `α=0.5,T=1.5` 与当前小预算不优；后续应在 validation 上搜索 α/T/训练长度，再一次性报告 test。

### 3.6 DPO

DPO 用 chosen/rejected 偏好对直接优化策略相对 reference 的隐式奖励 margin，无需单独训练 reward model。数据按 prompt 分组拆分，避免同一 prompt 的不同 pair 跨入 train/eval。

本次 1,000 对独立 holdout 上，base 与 DPO chosen preference accuracy 都为 45.7%；DPO mean implicit reward margin 为 -0.001024，正 margin 比例 51.0%，DPO loss 相对 base 上升 0.224%。因此不能写“DPO 提升偏好准确率”，只能写“复现 DPO 并建立公平负对照”。

## 4. 架构机制与系统结论

### 4.1 RMSNorm、RoPE/YaRN、GQA、KV Cache、SwiGLU

- RMSNorm 只按均方根缩放，不减均值；参数更少，适合 Pre-Norm Decoder；
- RoPE 将相对位置编码进 Q/K 旋转；YaRN 修改长位置下的频率与幅值，实现上下文外推；
- GQA 让多组 Query heads 共享较少的 K/V heads，直接降低 KV Cache；
- KV Cache 保存历史层的 K/V，使自回归第 t 步不再重算全部前缀；
- SwiGLU 用门控分支 `SiLU(W_gate x) ⊙ W_up x` 提升 FFN 表达能力。

本次 GQA 为 8 Query heads / 4 KV heads，理论 KV Cache 是对应 MHA 的 50%。KV Cache 还有逐 token cached/full logits 一致和 cached/uncached greedy token 一致的单测。

4096-token 功能基准中，RoPE 与 YaRN 都能完成前向和解码；YaRN 吞吐略低，因此只说明长位置机制可运行，不说明长文本任务质量提升。

### 4.2 Flash Attention 的准确表述

模型通过 PyTorch `scaled_dot_product_attention` 接入后端。验证分两层：

1. 强制 Flash backend 并查看 profiler kernel 名；
2. 通过实际 MiniMind 模型再次确认 kernel 被选择。

原语级 BF16、`[B=2,H=8,S=1024,D=96]` 实测 Flash median 0.2069 ms、峰值 12.06 MiB；math backend 1.3551 ms、190.14 MiB，约 6.55× 加速。这个结论只适用于该硬件、dtype 与 shape，不等于“手写 FlashAttention-2”或所有模型 shape 都会选择 Flash。

### 4.3 稀疏 MoE

MoE 为 4 experts、Top-1 routing。总参数约 198.417M，每 token 估算激活约 63.937M；它把参数容量扩大到 Dense 的约 3.10 倍，而每 token 激活规模接近 Dense。

部分训练 checkpoint 的固定批次审计显示 router 与 expert 梯度均 finite，聚合路由没有坍缩。系统基准则显示朴素单卡实现比 Dense 更慢、更占权重显存，因为没有 expert parallel、all-to-all、capacity factor 和 fused dispatch。

因此后训练统一使用收敛更充分的 Dense SFT。这样可隔离策略算法变量，也避免 current/reference/optimizer 同时驻留时放大 198M 总参数成本。MoE 与 LoRA/DPO/RL 在原理上兼容，只是当前硬件与 checkpoint 质量不适合作为公平主线。

## 5. Agentic RL / RLVR

### 5.1 Agentic RL 要解决的不是单轮问答

普通数学 RL 可以把一段回答直接交给答案 verifier；Tool-Use 则是一段带环境交互的轨迹。模型必须先生成结构化调用，环境执行后返回 observation，模型根据新信息继续调用工具或生成最终答案：

```text
System/User + Tool Schema
  → Assistant Action: <tool_call>{...}</tool_call>
  → Parser: 标签、JSON、工具名、参数 schema
  → Tool Environment: 确定性执行
  → Observation: 执行结果追加到上下文
  → Assistant Action: 下一次调用或 Final Answer
  → Verifier: 过程与结果联合验收
```

训练目标不能只看最终文本是否包含答案。若模型调用了错误工具、参数非法、伪造 observation，或者完全不调用工具却猜中答案，都不应被视作成功。因此 Agentic RLVR 的核心不是“给答案打一个分”，而是同时构建轨迹状态机、工具执行器、action mask、可验证成功条件与逐轨迹证据。

当前环境包含 calculator、单位换算、天气、时间、汇率、翻译和多工具组合。除 calculator 的 AST 白名单计算外，其余工具是确定性 mock，目的是保证同一输入在不同算法、seed 和 checkpoint 下可重放。它适合验证算法和 credit assignment，但不等同于联网搜索、浏览器或生产代码沙箱。

### 5.2 先审计数据：不能把普通对话冒充 Tool-Use RLVR

对上游 Agent 数据审计后发现，20,000 条可验证样本主要是 calculator，另有约 19,988 条普通对话没有 tools 和 ground truth。后者无法提供可执行轨迹和严格任务成功标签，不能直接用于 Tool-Use RLVR。

因此重新构建并冻结两套工具数据：

| 数据集 | Train/Eval | 用途 | 泄漏与执行审计 |
|---|---:|---|---|
| Basic Tool | 896/224 | Agent SFT 冷启动与 readiness | question overlap=0；1,120 条 oracle replay failures=0 |
| Challenge Tool | 384/96 | 四算法正式对照 | 与 Agent SFT oracle 零重合；480 条 replay failures=0 |

Basic Tool 覆盖 calculator、unit、weather、time、exchange、translation 和 multi-tool 七类；Challenge Tool 使用未进入冷启动标签的双工具/三工具组合。数学侧另外构建 512 条 Agent SFT 轨迹，表达式和 ground truth 在写入前重新计算。

数据切分不是随机运行时临时完成，而是持久化 manifest、问题哈希和 SHA-256。这样四算法使用完全相同的 train/eval 集，评测结果可追溯到具体样本。

### 5.3 为什么先做 Agent SFT 冷启动

严格 RLVR 的任务成功是稀疏二值信号。若一个 prompt 的 `G` 条 rollout 全错或全对，组内标准差为 0：

$$
\operatorname{std}(R_{1:G})=0
\quad\Longrightarrow\quad
A_{1:G}=0.
$$

通用 Full SFT 在数学 Agent readiness 上 Task Accuracy 为 0%，所有组几乎全错，直接运行 GRPO 家族基本没有策略梯度。于是只使用训练集 oracle 轨迹做 Agent SFT，不使用任何 eval label，再把生成能力达到最低门槛的 `agent_sft` 作为四算法共同初始化。

128 条 readiness 对照如下：

| 任务/初始化 | Task Acc | Format | Tool Valid | Execution | Required | Evidence | DAPO Effective Groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math / Full SFT | 0.00% | 81.25% | 71.09% | 69.53% | 48.44% | 10.81% | 0.00% |
| Math / Agent SFT | 59.38% | 92.97% | 100.00% | 97.27% | 100.00% | 85.68% | 15.63% |
| Basic Tool / Full SFT | 7.03% | 84.38% | 51.17% | 49.61% | 41.80% | 20.70% | 15.63% |
| Basic Tool / Agent SFT | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |

这个结果说明两个问题：

1. Agent SFT 明显提升了格式、工具协议和基础任务完成率，是后续 RL 的必要冷启动；
2. Basic Tool 已达到 100% 饱和，不能继续用它比较四种 RL 算法，否则所有算法都会被 ceiling effect 掩盖。

因此 Formal 改用更难的 Challenge Tool。其训练集前 128 条校准 Task Accuracy 只有 0.781%，DAPO effective group rate 为 3.125%，说明任务困难、严格成功稀疏，但仍偶尔产生可学习混合组。

### 5.4 多轮轨迹与精确 Token Ledger

`rollout_single` 每轮生成 assistant action，解析并执行工具，再把 observation 写回 messages。训练 loss 只覆盖模型真正采样的 assistant action 和 EOS，不覆盖 prompt、padding 和工具 observation。

初版实现把历史 assistant 文本 `decode` 后再通过 chat template `encode`。调试发现 tokenizer 并不保证任意 token 序列可逆：一个真实反例中，原始 assistant action 为 36 个 token，`decode→encode` 后变成 38 个。此时保存的 `π_old` 概率和训练时 `πθ` 看到的 action ID 不一致，importance ratio 即使数值可算，也失去统计意义。

修复方案是：

1. 保留 rollout engine 返回的精确 sampled IDs 和对应 old log-prob；
2. 只从 canonical chat template 中提取新增 tool observation suffix；
3. observation mask 设为 0，下一轮 assistant sampled IDs 继续追加；
4. 用 non-roundtrip 回归样例逐 token 验证 36 个原始 action IDs 全部保留。

这项修复使多轮轨迹满足 on-policy 目标的基本条件。训练器还会在首次 current forward 时检查 behavior-current log-prob MAE，若超过阈值直接报错。

### 5.5 可验证奖励：过程和结果必须同时正确

verifier 为每条轨迹分别输出：

- `format_valid`：tool tags 是否闭合且与解析出的调用数一致；
- `tool_call_valid`：工具名和参数 schema 是否合法；
- `tool_execution_success`：调用是否实际执行成功；
- `required_tool_coverage`：任务要求的工具是否全部执行；
- `tool_evidence_coverage`：执行结果是否支持 ground truth；
- `answer_accuracy`：最终答案是否通过 verifier；
- `task_success`：完整轨迹是否满足严格任务定义；
- `unfinished`：达到最大轮数后是否仍未结束。

Strict RLVR 成功条件可以概括为：

```python
task_success = (
    final_answer_correct
    and tool_tags_closed
    and required_tools_executed
    and every_call_valid_and_executed
    and execution_results_support_ground_truth
    and not unfinished
)
reward = +1 if task_success else -1
```

项目同时保留 shaped reward，用格式、合法调用、执行和答案覆盖提供稠密诊断，但 Formal 明确使用 strict `+1/-1`，并关闭主观 Reward Model。DAPO 动态采样也必须基于 `task_success`，不能基于 shaped reward；否则一个答案错误但格式漂亮的轨迹可能被误当成成功样本。

### 5.6 针对三类失控行为的约束

#### Reward Hacking

数字 verifier 使用边界和 final-answer region，拒绝 GT=4、回答=14 的子串投机；Tool verifier 重新执行确定性工具，要求执行证据覆盖 GT，拒绝“调错工具后猜中答案”和伪造 observation；calculator 用 AST 白名单替代 Python `eval`。

#### 格式崩溃

格式不是单个正则布尔值，而是拆成标签闭合、JSON 解析、工具名、参数、执行五层指标。格式奖励不能覆盖答案错误，未闭合轨迹不能获得严格成功。

#### 过长推理

训练记录平均/P95 assistant action length 和 unfinished rate；DAPO Soft Overlong 在硬上限前线性惩罚；重复 n-gram 另有惩罚。完整轨迹超过 `max_total_len` 时训练直接报错，不允许通过左截断继续，因为那会改变 old/current log-prob 的条件上下文。

当前 `unfinished rate` 表示达到最大轮数仍未完成，不等同于严格的 token truncation rate。若需要报告“截断率”，必须进一步记录 `eos/max_new_tokens/max_turns/max_context/tool_error` 等 stop reason。本文和当前简历应优先使用“未完成率、平均/P95 响应长度”。

### 5.7 一致性、可控性与任务完成率分别指什么

- **一致性**：相同协议下能持续输出闭合标签、合法 JSON 和可执行参数；由 Format/Tool Valid/Execution 等指标衡量；
- **可控性**：成功必须通过显式 verifier，模型不能靠格式分、猜答案或伪造工具结果绕过门禁；
- **任务完成率**：严格 `task_success`，同时要求答案、工具过程和执行证据正确。

当前有正向因果证据的是 Agent SFT 冷启动：它显著提高 Math/Basic Tool readiness。严格 verifier 证明成功判定更可控，但没有逐奖励组件 ablation。四种 RL 算法在本次小预算 fixed holdout 上没有继续提高上述指标，因此不能把 Agent SFT 的提升归因于 RL。

## 6. 四种策略优化方法

本节对应简历中的两项核心工作：一是统一复现 GRPO、CISPO、DAPO、GSPO；二是把四个目标接入多轮 Agentic RL / RLVR 环境，用相同数据、相同初始化、相同优化器更新上限和相同固定集比较 Reward、准确率、KL、响应长度和训练波动。

算法定义分别以 DeepSeekMath、MiniMax-M1、DAPO 和 GSPO 的论文/官方实现为一手依据；公式与本项目代码的逐项映射、版本和原创性边界统一列在 [SOURCES](./SOURCES.md)，避免把论文结果或上游已有代码写成本项目实测结果。

需要先明确结论边界：这里的“提升训练稳定性”不是指四种 RL 算法已经提高了固定测试集准确率。当前实验能证明的是：DAPO 动态采样把被用于更新的零方差组降为 0，使稀疏奖励下的每个 accepted group 都有非零相对优势；同时 ratio、KL、梯度和多轮 token 对齐均有保护与监控。固定集能力是否提升必须单独看第 7 节，而本次小预算下四算法的固定集任务指标没有改善。

### 6.1 统一训练框架：一次策略更新经历什么

四种算法共享 [`trainer/train_agent.py`](../../trainer/train_agent.py) 的 rollout、工具环境、verifier、action mask、旧策略概率、参考策略、优化器和日志系统，只在 [`trainer/policy_optimization.py`](../../trainer/policy_optimization.py) 中替换 policy objective；DAPO 额外启用动态采样和 Soft Overlong recipe。这样可以避免“算法 A 使用另一套数据或 rollout 框架”成为混杂变量。

一次训练更新按以下顺序执行：

```text
共同 Agent SFT checkpoint
  → 为每个 prompt 采样 G 条多轮轨迹
  → Assistant 生成 tool_call
  → 工具名/参数校验并执行工具
  → Observation 写回上下文，Assistant 继续生成
  → verifier 计算严格成功、分项指标与 Reward
  → DAPO 可丢弃全对/全错组并继续采样
  → 对 accepted groups 计算组相对优势
  → 保存 π_old 的逐 token log-prob
  → π_ref 在完全相同 token/context 上计算参考 log-prob
  → πθ 前向，按 GRPO/CISPO/DAPO/GSPO 计算目标
  → 反向传播、梯度裁剪、optimizer/scheduler step
  → 同步 rollout policy，记录指标并保存 checkpoint
```

#### 6.1.1 三个策略分别是什么

- `π_old`：产生当前 rollout 的 behavior policy。它的逐 token log-prob 在采样时保存，在同一批数据的多个 policy epochs 中保持冻结，是 importance ratio 的分母；
- `πθ`：当前正在更新的 policy，是 importance ratio 的分子；
- `π_ref`：冻结的 Agent SFT reference，用于计算策略偏移和可选 KL 惩罚。

Token importance ratio 定义为：

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}
{\pi_{old}(y_{i,t}\mid x_i,y_{i,<t})}
=\exp\left(\log\pi_\theta-\log\pi_{old}\right).
$$

如果 `π_old` 与 `πθ` 不是在完全相同的 action token 和上下文上计算，这个 ratio 就没有意义。因此多轮轨迹不能把 assistant 文本 `decode→encode` 后重新构造。项目保存精确 sampled-token ledger，只追加工具 observation；首次 policy forward 还会检查：

```text
rollout_logprob_mae ≈ 0
rollout_ratio_mean ≈ 1
```

超过阈值就停止训练，而不是让错位 ratio 静默污染梯度。

Formal 对照显式使用 `--beta 0`，所以 `π_ref` 在该组实验中主要用于报告固定集 KL 和诊断策略偏移，不对 policy loss 施加实际 KL 惩罚。框架本身支持 `beta>0`，但不能把未启用的惩罚描述成 Formal 结果的来源。

#### 6.1.2 Action mask 为什么是算法正确性的前提

多轮 Agent 轨迹包含模型动作与环境 observation：

| Token 来源 | 模型可见 | 计入 policy loss |
|---|---:|---:|
| system/user prompt | 是 | 否 |
| assistant tool call | 是 | 是 |
| tool observation | 是 | 否 |
| assistant final answer | 是 | 是 |
| assistant EOS | 是 | 是 |
| padding | 否 | 否 |

工具结果是环境给出的条件，不能被当作模型生成动作。如果 observation 进入 loss，模型会被奖励或惩罚它无法控制的 token，old/current ratio 和 credit assignment 都会失真。`shift_action_mask` 将原始 response mask 对齐到 next-token log-prob 位置，Formal 的长度、KL、ratio 和 loss 都只统计有效 assistant action token。

### 6.2 共同基础：组相对优势与零方差问题

对同一个 prompt 采样 `G` 条回答，先在组内对 Reward 标准化：

$$
A_i=\frac{R_i-\mu_G}{\sigma_G+\epsilon},\qquad
\mu_G=\frac{1}{G}\sum_{j=1}^{G}R_j.
$$

对应实现为：

```python
grouped = rewards.view(-1, group_size)
means = grouped.mean(dim=1, keepdim=True)
stds = grouped.std(dim=1, unbiased=False, keepdim=True)
advantages = ((grouped - means) / (stds + eps)).reshape(-1)
```

GRPO 家族不需要单独训练 Critic，而是用同 prompt 的其他回答作为相对基线。代价是：若一组回答全错或全对，组内标准差为 0，所有 advantage 都为 0，当前 prompt 不产生策略梯度。

这正是项目不能直接从通用 SFT 开始 RL 的原因。通用 SFT 在数学 Agent readiness 上 Task Accuracy 为 0%，几乎所有组全错；使用训练集 oracle 做 Agent SFT 后，数学 Task Accuracy 达到 59.375%，DAPO effective group rate 达到 15.625%，才产生可供相对优化使用的混合成功组。基础 Tool-Use 在 Agent SFT 后又达到 100% 饱和，因此另建零冷启动标签重合的组合挑战集。

### 6.3 GRPO：组内相对优势 + 对称 Token Clip

GRPO 使用 PPO 风格的对称裁剪：

$$
L_{GRPO}=-\frac{1}{N}\sum_i\frac{1}{|o_i|}
\sum_t m_{i,t}\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right).
$$

其中 `m_{i,t}` 是 action mask。正式配置 `ε=0.2`。当前实现先对每条序列的有效 token 求均值，再对序列求均值，因此每条轨迹在 batch 中权重相同，不随长度线性增加。

训练过程可以理解为：

1. 同一 prompt 采样 4 条轨迹；
2. 使用严格 `+1/-1` 成功奖励计算组内优势；
3. 对每个 action token 计算 `πθ/πold`；
4. 当 ratio 超过 `[0.8,1.2]` 时使用裁剪 surrogate；
5. 对每条序列内部求平均，再聚合 batch。

优点是实现简单、不需要 Critic；核心问题是稀疏任务中全错组没有梯度。Tool Formal 中 GRPO 训练组零方差率为 `90.00%±10.00%`，说明在当前挑战集和小采样组下，大多数更新候选没有相对成功信号。

### 6.4 CISPO：裁剪采样系数并直接优化 Log-Probability

CISPO 不使用 GRPO 的 clipped surrogate，而是把 importance ratio 视为采样校正系数，只裁剪其上界并停止梯度：

$$
c_{i,t}=\operatorname{stopgrad}
\left[\min\left(\rho_{i,t},1+\epsilon_{high}^{IS}\right)\right],
$$

$$
L_{CISPO}=-
\frac{\sum_{i,t}m_{i,t}c_{i,t}A_i\log\pi_\theta(y_{i,t})}
{\sum_{i,t}m_{i,t}}.
$$

核心代码语义是：

```python
cispo_upper = 1.0 + cispo_epsilon_high
coefficient = token_ratio.clamp(max=cispo_upper).detach()
per_token_policy = -(coefficient * token_advantages * current_logps)
policy_loss = masked_mean(per_token_policy, completion_mask)
```

`detach()` 很关键：ratio 只负责把 behavior distribution 下的样本校正到 current policy，不让梯度再经过 ratio 形成另一条优化路径；真正求导的是 `logπθ`。实现按论文把 CLI 参数解释为 `1+ε_high_IS`，而不是把 `ε_high_IS` 本身误当绝对 ratio 上界。

Formal 保持实现默认 `ε_high_IS=5.0`，所以 coefficient 上界为 `6.0`。该阈值远宽于 GRPO 的 `1.2`，原因是 CISPO 裁剪的是已经 `stop-gradient` 的采样校正系数，而不是把它代入 PPO 的 `min` surrogate；两者数值不能直接按“谁的 clip 更松”横向解释。

CISPO 使用全局 Token-level mean。与 GRPO 的 sequence-first mean 相比，两条长度为 10 和 100 的轨迹在 GRPO 中各占约 50%，在 Token-level mean 中每个有效 token 权重相同，长轨迹总权重更大。这样可以避免先把长短序列强行等权，但也必须结合长度指标和 Overlong 约束监控长回答偏置。

Formal CISPO 只启用上述核心目标，没有打开 DAPO dynamic sampling；因此 Tool 训练组仍有 `86.67%±11.55%` 的零方差率。在大量全错组下，即使目标函数正确，也没有足够非零 advantage 产生能力变化。

### 6.5 DAPO：为稀疏可验证奖励补齐采样与长度 Recipe

DAPO 在本项目中同时启用 Dynamic Sampling、Clip-Higher、Token-level Loss 和 Soft Overlong。只把 GRPO 的 clip 上界从 0.2 改为 0.28，不能称为完整 DAPO。

#### 6.5.1 Dynamic Sampling：只保留有学习信号的组

动态采样使用严格二值 `task_success`，而不是 shaped reward：

$$
\operatorname{keep}(g)=
\mathbb{1}\left[0<\sum_{i=1}^{G}s_i<G\right],
\qquad s_i\in\{0,1\}.
$$

```python
successes = task_success.bool().view(-1, group_size).sum(dim=1)
group_mask = (successes > 0) & (successes < group_size)
```

全错组和全对组都被丢弃，训练器继续 rollout，直到动态 buffer 填满一个 accepted batch。格式奖励不能参与这个判断，否则模型可能仅凭合法标签获得“成功”，形成 Reward Hacking。

每次 optimizer update 后都会清空多余的 accepted trajectories，避免把 `π_old` 生成的旧样本携带到下一次已经更新过的 policy，造成隐性 off-policy staleness。

动态采样改善的是“被用于更新的组是否拥有非零优势”，并没有凭空提高数据分布上的真实成功率。它还改变了 environment budget，所以对比时必须同时记录：

- `candidate_groups / accepted_groups`；
- `candidate_trajectories`；
- `candidate_action_tokens`；
- 工具调用次数；
- 动态接受率；
- wall time。

#### 6.5.2 非对称 Clip / Clip-Higher

DAPO 使用独立下界和上界：

$$
\hat\rho_{i,t}=\operatorname{clip}
(\rho_{i,t},1-\epsilon_{low},1+\epsilon_{high}),
$$

$$
L_{DAPO-token}=-\min(\rho_{i,t}A_i,\hat\rho_{i,t}A_i).
$$

Formal 配置为 `ε_low=0.2, ε_high=0.28`。更宽的上界给正优势、原本概率较低的正确 token 更大的上升空间；下界仍限制负优势 token 的概率更新幅度。非对称 clip 只是更新约束，不能解决 reward 全零问题，所以它要与 Dynamic Sampling 配合。

#### 6.5.3 Token-level Loss

DAPO 不先对每条序列求平均，而是对所有有效 action token 做一次全局平均：

$$
L_{token}=\frac{\sum_{i,t}m_{i,t}L_{i,t}}
{\sum_{i,t}m_{i,t}}.
$$

在 DDP 下，各 rank 的有效 token 数可能不同。如果每张卡只除以本地 token 数再做梯度平均，就会错误地让 token 少的 rank 权重过高。项目使用 `distributed_token_mean_scale` 根据全局有效 token 数修正 local loss，使多卡语义仍等价于全局 Token-level mean。当前正式实验只有一张 GPU，因此验证的是实现路径和单卡结果，不宣称已有多卡 scaling 数字。

#### 6.5.4 Soft Overlong：在硬上限前渐进惩罚

设整条多轮 assistant action 的最大 token 数为 `L_max`，线性缓冲区为 `L_cache`：

$$
R_{len}(L)=
\begin{cases}
0, & L\le L_{max}-L_{cache},\\
-\frac{L-(L_{max}-L_{cache})}{L_{cache}},
&L_{max}-L_{cache}<L<L_{max},\\
-1, & L\ge L_{max}.
\end{cases}
$$

Formal 中 `max_gen_len=96`、`max_turns=3`，所以 `L_max=288`；`overlong_cache_len=128`，惩罚从约 160 个 assistant action tokens 开始。数学与 Tool 固定集的平均/P95 长度分别约为 `80.19/100.08` 和 `92.34/104.85`，大多低于惩罚区间。因此本实验能证明 Soft Overlong 的公式、边界测试和训练接入正确，不能声称它已经实测显著缩短响应。

#### 6.5.5 DAPO 的真实机制结果

Tool Formal 中，普通算法的训练组零方差率为 `86.7%–93.3%`，DAPO accepted groups 为 0%。这意味着每个被送入 optimizer 的 DAPO 组都有成功和失败轨迹，优势非零。

代价同样明显：DAPO 平均使用 `103.7±17.6` 个候选组才完成 10 个 accepted updates，平均动态接受率 `14.70%±3.86%`；候选组约为普通算法的 10.4 倍，wall time 约为 9.9 倍。DAPO 训练组准确率 `48.33%` 是筛选后的条件统计，不是测试准确率。

### 6.6 GSPO：在序列层面计算 Importance Ratio

Token ratio 若沿长序列直接相乘，容易出现极大或极小值。GSPO 先对一条轨迹的 token log-ratio 求长度归一化均值，再指数化：

$$
\rho_i^{seq}=\exp\left[
\frac{1}{|o_i|}\sum_t m_{i,t}
\left(\log\pi_\theta(y_{i,t})-\log\pi_{old}(y_{i,t})\right)
\right].
$$

它等价于有效 token ratios 的几何平均，而不是算术平均，也不是未归一化乘积。之后在序列层面做裁剪和 surrogate：

$$
L_{GSPO}=-\frac{1}{N}\sum_i
\min\left(
\rho_i^{seq}A_i,
\operatorname{clip}(\rho_i^{seq},1-\epsilon_{low},1+\epsilon_{high})A_i
\right).
$$

核心实现为：

```python
seq_log_ratio = (log_ratio * mask).sum(1) / token_counts.clamp(min=1)
seq_ratio = torch.exp(seq_log_ratio)
clipped_seq_ratio = seq_ratio.clamp(1 - eps_low, 1 + eps_high)
seq_surrogate = torch.minimum(
    seq_ratio * sequence_advantages,
    clipped_seq_ratio * sequence_advantages,
)
policy_loss = -seq_surrogate[valid_rows].mean()
```

Formal 配置使用 `ε_low=3e-4, ε_high=4e-4`。当前 advantage 是每条轨迹一个标量，因此实现的是标准 sequence-level GSPO，不是带 token advantage 的 GSPO-token 变体。

GSPO 改善的是 importance sampling 的粒度和长序列比率的数值行为，不会解决 reward 组内全错。Tool Formal 中 GSPO 零方差组率仍为 `93.33%±5.77%`，说明在当前困难数据与 4-rollout 设置下，主要瓶颈仍然是成功样本不足，而不是 ratio 聚合形式。

### 6.7 Agentic RLVR：Reward、Reward Hacking 与可控性

#### 6.7.1 Strict Reward 与 Shaped Reward 必须分开

项目同时保留两种奖励模式：

- `strict`：完整可验证任务成功为 `+1`，否则为 `-1`，用于四算法正式公平对照；
- `shaped`：格式、合法调用、执行、答案覆盖等分项加权，用于课程学习或诊断，限制在 `[-3,3]`。

Formal 使用 `reward_mode=strict`、`use_reward_model=0`，没有混入主观 Reward Model。严格 `task_success` 同时要求：

```python
task_success = (
    final_answer_correct
    and tool_tags_closed
    and required_tools_executed
    and every_tool_call_valid_and_executed
    and tool_results_support_ground_truth
    and not unfinished
)
```

这使 Task Accuracy 成为比 Answer Accuracy 更严格的指标：模型即使猜中最终答案，只要没有按要求执行工具或缺少执行证据，也不能算任务成功。

#### 6.7.2 Reward Hacking 防护

项目针对以下投机路径设置了验证和单测：

| 风险 | 模型可能的投机方式 | 防护方式 |
|---|---|---|
| 数字子串攻击 | GT 为 4，回答 14 | 数值边界和 final-answer region 精确验证 |
| 格式奖励替代任务成功 | 输出合法标签但答案错误 | shaped format bonus 与二值 `task_success` 分离 |
| 调错工具后猜答案 | 工具过程错误但最终碰巧正确 | 必需工具、调用合法性、执行成功和证据覆盖联合门禁 |
| 伪造 observation | 自己编写“工具结果” | verifier 重新执行确定性工具，以实际执行结果验证 GT |
| 非法参数/任意代码 | Calculator 参数注入 | 工具 schema 检查 + AST 白名单解释器，禁用 Python `eval` |
| 未闭合轨迹 | 在 token 上限前不输出最终答案 | `unfinished=True` 时 final answer 无效、Task Success 必为 false |
| 重复套话 | 通过冗长重复积累 shaped reward | n-gram repetition penalty |

这些结果证明奖励门禁能拒绝已知攻击样例，但没有逐组件 reward ablation，因此不能量化“每个惩罚分别提高了多少准确率”。

#### 6.7.3 格式崩溃与输出一致性

格式正确被拆为多个层级：标签是否闭合、JSON 是否可解析、工具名是否合法、参数是否满足 schema、执行是否成功。只统计一个“能否 parse”的指标会掩盖参数错误和执行失败。

Agent SFT 冷启动后，数学 readiness 的格式正确率从 `81.25%` 提升到 `92.97%`、工具调用合法率从 `71.09%` 提升到 `100%`；基础 Tool readiness 的格式、工具合法、执行、必需工具和证据覆盖全部达到 100%。这些改善来自 Agent SFT 冷启动和严格数据构造，不应归因于后续四种 RL 算法，因为 Formal 固定集上四算法相对 Agent SFT 的格式指标 delta 都为 0。

#### 6.7.4 过长推理、未完成率与“截断率”

当前系统记录平均/P95 action length 与 `unfinished rate`。`unfinished` 表示达到最大 assistant/tool 轮数后仍未完成；代码发现完整轨迹超过 `max_total_len` 时直接报错，禁止 rollout 后左截断，因为左截断会改变 old/current log-prob 的条件上下文。

因此当前 `unfinished rate` 不能严格等同于 Token Truncation Rate。若简历必须写“截断率”，还需把终止原因扩展为：

```text
eos / max_new_tokens / max_turns / max_context / tool_error
```

再单独统计命中 `max_new_tokens` 或 `max_context` 的比例。基于当前产物，最准确的表述是“比较平均/P95 响应长度和未完成率，并禁止破坏 importance ratio 的静默左截断”。

### 6.8 指标如何定义、为什么需要同时看

| 指标 | 定义 | 主要回答的问题 |
|---|---|---|
| Reward | Formal 中严格成功 `+1/-1` 的均值 | 整体轨迹成功信号如何变化 |
| Task Accuracy | 满足答案、格式、工具、执行和证据条件的轨迹比例 | Agent 是否真正完成任务 |
| Answer Accuracy | 最终答案通过 verifier 的比例 | 答案是否正确，但不保证工具过程正确 |
| KL k3 | `exp(δ)-δ-1`，`δ=logπref-logπθ` | 策略偏离 SFT reference 的程度 |
| Avg/P95 Response Length | assistant action token 的均值/95 分位 | 平均长度和长尾是否增长 |
| Unfinished Rate | 达到最大轮数仍未闭合的比例 | 多轮轨迹是否正常结束 |
| Format Valid | tool tags 是否闭合且可解析 | 是否出现格式崩溃 |
| Tool Call Valid | 合法工具调用数/解析出的调用数 | 工具名和参数是否合法 |
| Execution Success | 成功执行数/解析出的调用数 | 调用是否可被环境执行 |
| Required Tool Coverage | 已执行必需工具/全部必需工具 | 是否漏掉任务要求的工具 |
| Evidence Coverage | 被执行结果支持的 GT 项/全部 GT 项 | 是否调用工具后仍凭空猜答案 |
| Zero-variance Rate | 组内 Reward 方差为 0 的比例 | 是否存在非零相对优势 |
| Dynamic Acceptance | accepted groups/candidate groups | 获取有效组的采样效率 |
| Ratio/Clip Fraction | importance ratio 分布与被裁剪比例 | 更新是否过大或 clip 是否失效 |
| Temporal Std | 同一 run 最后 N 次更新的指标标准差 | 单次训练随时间的波动 |
| Seed Std | 独立训练 seed 最终指标的样本标准差 | 结果对随机种子的敏感性 |

KL 使用逐点非负的 k3 estimator：

$$
\delta=\log\pi_{ref}-\log\pi_\theta,
\qquad k_3=\exp(\delta)-\delta-1\ge0.
$$

所有 log-softmax 和 KL 计算转为 FP32，极端 log-ratio 截到 `[-20,20]`。训练日志同时保留 KL mean、token P95 和 max，因为极少数近零概率 token 会制造大尾值。最终策略偏移应以共同 fixed holdout KL 为准，不能用某个训练 batch 的极值代替。

### 6.9 “训练稳定性”在本项目中的准确含义

本项目从四个维度判断稳定性：

1. **信号稳定性**：组内 Reward 是否有方差、advantage 是否非零；
2. **数值稳定性**：log-prob、ratio、KL、loss 和梯度是否 finite；
3. **策略稳定性**：ratio mean/std、clip fraction、fixed holdout KL 是否受控；
4. **统计稳定性**：最后 N 个更新的 temporal std 与三个训练 seed 的 sample std。

DAPO 的正向证据属于第一类：它把 accepted groups 的零方差率降为 0。BF16/FP32 分工、log-ratio 截断、梯度裁剪和 behavior-current MAE 门禁属于第二、三类保护。三 seed 和固定解码 seed 属于第四类实验设计。

不能据此笼统写“整体训练稳定性显著提升”：DAPO 的训练 Reward 跨 seed 仍有波动，而且采样成本显著上升；GRPO/CISPO/GSPO 在当前 Tool 数据上仍大量遇到全错组；固定集输出未发生变化。更准确的简历措辞是“缓解稀疏奖励下的零方差/零梯度问题，并量化跨 seed 稳定性与额外 rollout 成本”。

### 6.10 实验结果的完整解释

#### 6.10.1 先看 Agent SFT，而不是把所有提升归因于 RL

冷启动前后，数学 Task Accuracy 从 `0%` 提升到 `59.375%`，基础 Tool-Use 从 `7.031%` 提升到 `100%`。这证明训练集 oracle Agent SFT 提升了协议遵循和任务启动能力，也让数学任务出现可用于组相对优化的混合成功组。

基础 Tool 集达到 100% 后已经饱和，无法继续区分四种算法，所以项目构建了 384/96 的组合挑战 train/eval。挑战集与 Agent SFT oracle 零重合，全量 480 条 oracle replay failures=0。这一步避免在饱和数据上制造虚假算法提升。

#### 6.10.2 训练组结果：DAPO 改善有效采样，但不能直接比较准确率

数学任务中，DAPO 平均消耗 `160.3±43.7` 个候选组才获得 10 个 accepted groups，动态接受率只有 `8.23%±1.35%`；Tool 任务平均消耗 `103.7±17.6` 个候选组，接受率 `14.70%±3.86%`。

DAPO 的训练组 Reward/Accuracy 高于普通算法，原因是它有条件地选择“组内既有成功又有失败”的样本。这个筛选同时提高了组内平均成功率和 Reward，因此不能把 DAPO 的训练组 `48.33%` 与 GRPO 的随机组 `3.33%` 当成公平能力对比。训练组表只能用于解释采样机制、有效优势和成本。

#### 6.10.3 固定集结果：四算法没有产生能力增益

数学固定集共 2,496 条轨迹，Agent SFT baseline 与四算法的 Reward 均为 `-0.03125`、Task Accuracy 均为 `48.4375%`、平均响应长度均为 `80.19`；固定集 KL 仅为 `0–5.43e-9`。

Tool 固定集同样为 2,496 条轨迹，baseline 与四算法的 Reward 均为 `-0.88542`、Task Accuracy 均为 `5.7292%`、Format Valid 均为 `97.9167%`、Execution Success 均为 `71.6146%`、平均长度均为 `92.34`、Unfinished Rate 均为 `1.0417%`；固定集 KL 为 `0–1.53e-8`。

所有主要任务指标对 Agent SFT 的 delta 都为 0。结合极小 fixed holdout KL，最合理的解释是：`10 accepted updates、lr=1e-7` 使参数变化不足以改变固定随机种子的生成结果，而不是“四算法效果完全等价”或“评测证明某算法更好”。

#### 6.10.4 为什么会得到零增益

- 更新预算只有 10 个 accepted groups，每组 4 条轨迹；
- 学习率 `1e-7` 很保守，固定集 KL 接近 0；
- GRPO/CISPO/GSPO 在 Tool 训练中 `86.7%–93.3%` 的候选组没有组内方差；
- 64M 模型对组合 Tool-Use 的初始严格成功率低，奖励极稀疏；
- DAPO 虽找到有效组，但 10 次更新仍不足以迁移到固定 holdout；
- Formal 为隔离 policy objective 使用 `beta=0`、无主观 Reward Model，没有用额外模型提供稠密信号。

下一轮应在不触碰 test 的独立 validation 上增加 accepted updates，搜索学习率和 clip，加入由单工具到多工具的课程，并分别消融 Dynamic Sampling、Clip-Higher、Token-level Loss 和 Soft Overlong。锁定配置后只对 test 做一次最终报告。

### 6.11 简历两条表述应如何理解

“引入动态采样、非对称 Clip、Token-level Loss、序列级重要性比率等方法，提升强化学习阶段的训练稳定性”在本项目中的可验证含义是：

- 四种目标已在统一框架实现并通过测试；
- DAPO accepted groups 的零方差率降为 0，缓解无有效梯度问题；
- ratio、KL、行为策略对齐、梯度和跨 seed 波动都有日志与门禁；
- 代价是 DAPO 约 10.4 倍候选组和 9.9 倍 wall time；
- 当前没有固定集准确率提升，所以“稳定性提升”不能解释为“模型能力提升”。

“构建 Agentic RL 训练流程……提升模型输出的一致性、可控性与任务完成率”应拆开归因：

- 多轮轨迹、Tool execution、action mask、verifier、Reward Hacking 防护和指标体系已经实现；
- Agent SFT 冷启动确实提高了数学/基础 Tool 的任务完成率和格式/工具指标；
- 严格 verifier 提高了成功判定的可控性，防止猜答案和格式奖励冒充任务成功；
- Soft Overlong、重复惩罚和 unfinished 门禁限制异常输出，但当前未做逐组件 ablation；
- 后续 GRPO/CISPO/DAPO/GSPO 在小预算固定集上没有进一步提升，因此不能把 Agent SFT 的正结果归因于 RL。

第 7 节给出 Formal 配置、三 seed 聚合表和逐任务结果；机器证据位于 `out/metrics/algorithm_comparison_{math,tool}.csv`、`out/eval_rlvr/{math,tool}/algorithm_comparison.csv` 与对应 `trajectories.jsonl`。

## 7. Formal 对照设计与最终结果

Formal 的目标不是追求最好单次曲线，而是在控制变量后回答：四种算法是否产生不同的训练信号、策略偏移、任务能力和采样成本。

| 控制项 | Formal 配置 |
|---|---|
| 共同初始化 | `agent_sft_768` |
| 任务 | Math、Challenge Tool-Use 分开运行 |
| 算法 | GRPO、CISPO、DAPO、GSPO |
| 训练 seed | 42、43、44 |
| 每组轨迹 | 4 |
| 更新上限 | 每 run 10 个 accepted/update groups |
| Policy epochs | 每批 rollout 复用 2 次 |
| 学习率 | `1e-7`，cosine decay 到 `1e-8` |
| KL loss | `beta=0`；reference 仅用于 KL 诊断 |
| Reward | Strict RLVR `+1/-1`，关闭主观 Reward Model |
| 多轮上限 | 3 turns × 96 new tokens，`max_total_len=1024` |
| 固定评测 | 每 checkpoint 64 prompts × 3 decode seeds |

每个任务评测共同 Agent SFT baseline 加 12 个训练后 checkpoint，即：

$$
13\ checkpoints\times64\ prompts\times3\ decode\ seeds=2,496\ trajectories.
$$

这里的“同预算”只指 optimizer-update 上限相同。DAPO 为找到 10 个有效组会消耗更多候选 rollout，不能说四算法的 environment calls、action tokens 或 wall time 相同。公平报告必须同时给出 update budget 和实际 sampling budget。

训练 seed 是三个独立 run；每个 checkpoint 的三个 decode seed 先求平均，再把三个训练 checkpoint 当作独立样本计算 sample std。训练日志用于解释优化动态和采样成本，固定 holdout 才用于判断泛化能力。

### 7.1 数学推理 Formal

训练日志中，GRPO/CISPO/GSPO 每个 run 都使用 10 个候选组完成 10 个更新；DAPO 因动态采样丢弃零方差组，三个 seed 分别消耗 111、194、176 个候选组，平均接受率为 8.23%。按三个训练 seed 聚合：

| 算法 | 训练组 Reward | 训练组准确率 | 候选组 | 动态接受率 | Wall time |
|---|---:|---:|---:|---:|---:|
| GRPO | -0.5167 ± 0.0577 | 24.17% ± 2.89% | 10.0 | — | 32.41 ± 0.62 s |
| CISPO | -0.5167 ± 0.0577 | 24.17% ± 2.89% | 10.0 | — | 32.79 ± 0.32 s |
| DAPO | 0.0667 ± 0.1893 | 53.33% ± 9.46% | 160.3 ± 43.7 | 8.23% ± 1.35% | 487.04 ± 137.22 s |
| GSPO | 0.0500 ± 0.0500 | 52.50% ± 2.50% | 10.0 | — | 28.77 ± 2.58 s |

DAPO 的训练组数字更高，是因为只保留“一组内既有成功也有失败”的候选组，属于选择条件后的统计量，不能与未筛选的 GRPO/CISPO 训练组直接比较，更不能当成泛化收益。可信比较必须看共同固定集。

数学固定集共评测 `13 checkpoints × 64 prompts × 3 decode seeds = 2,496` 条轨迹。共同 Agent SFT baseline 的 Reward 为 -0.03125，task/answer accuracy 为 48.4375%，格式正确率 75.00%，工具执行率 95.3125%，响应长度 80.19 tokens。四种算法的这些指标及其余任务指标与 baseline **完全相同，delta 均为 0**；固定集 k3 KL 也仅为 `0–5.43e-9`。

因此本次数学 Formal 的结论不是“某算法提升了准确率”，而是：四种目标都按预期完成更新，DAPO 的动态采样成本被真实量化；但在 `10 updates、1e-7` 的受控小预算下，参数变化不足以改变固定随机种子的采样结果，无法据此判断算法优劣。训练过程中少量极低概率 token 会使 k3 mean 出现大尾值，所以诊断同时保留 token p95/max；最终策略偏移以固定 holdout KL 为准。

### 7.2 组合 Tool-Use Formal

Tool-Use 训练集的严格成功很稀疏。按三个训练 seed 聚合：

| 算法 | 训练组 Reward | 训练组准确率 | 零方差组比例 | 候选组 | Wall time |
|---|---:|---:|---:|---:|---:|
| GRPO | -0.9333 ± 0.0764 | 3.33% ± 3.82% | 90.00% ± 10.00% | 10.0 | 28.97 ± 0.86 s |
| CISPO | -0.8667 ± 0.1155 | 6.67% ± 5.77% | 86.67% ± 11.55% | 10.0 | 28.80 ± 0.69 s |
| DAPO | -0.0333 ± 0.0289 | 48.33% ± 1.44% | 0% | 103.7 ± 17.6 | 285.19 ± 47.06 s |
| GSPO | -0.9667 ± 0.0289 | 1.67% ± 1.44% | 93.33% ± 5.77% | 10.0 | 28.16 ± 0.44 s |

DAPO 三个 seed 分别消耗 118、109、84 个候选组，平均动态接受率 14.70% ± 3.86%。它成功避开了普通采样中 86.7%–93.3% 的零方差组，确保每个保留组有非零相对优势；代价是候选组约 10.4 倍、wall time 约 9.9 倍。再次强调，DAPO 表中 48.33% 是动态筛选后的组准确率，不是测试准确率。

Tool 固定集同样评测 2,496 条轨迹。共同 baseline 的 Reward 为 -0.88542，task/answer accuracy 为 5.7292%，格式正确率 97.9167%，工具调用合法率 99.4792%，执行成功率 71.6146%，必需工具覆盖率 56.6840%，工具证据覆盖率 35.9375%，响应长度 92.34 tokens，unfinished rate 1.0417%。四算法的上述指标都与 baseline **完全相同，delta 全为 0**，固定集 k3 KL 为 `0–1.53e-8`。

这组结果把“机制有效”和“能力改善”清楚分开：DAPO 动态采样确实把稀疏任务中的零优势组转成可更新组，但 10 个 accepted groups、`1e-7` 学习率仍不足以改变固定生成。GRPO/CISPO/GSPO 在普通抽样中又大多遇到全错组，几乎没有有效优势。正确下一步是在独立 validation 上增加训练更新与学习率、做难度课程和逐组件消融；当前结果不能宣传任何四算法的准确率提升。

机器证据分别位于 `out/metrics/algorithm_comparison_{math,tool}.csv`、`out/eval_rlvr/{math,tool}/algorithm_comparison.csv` 和两套 `trajectories.jsonl`。

### 7.3 四种算法横向结论

| 算法 | 理论上改变什么 | 训练中观察到什么 | 额外成本 | 固定集结论 |
|---|---|---|---|---|
| GRPO | 组相对优势、对称 Token Clip | Math 有部分混合组；Tool 90% 组零方差 | 基准成本 | Math/Tool delta 均为 0 |
| CISPO | 上界裁剪系数、stop-gradient、Token-level mean | 目标运行正常；Tool 86.67% 组零方差 | 接近 GRPO | Math/Tool delta 均为 0 |
| DAPO | 动态采样、非对称 Clip、Token mean、Soft Overlong | accepted groups 零方差率为 0 | Tool 候选组约 10.4×、wall time 约 9.9× | Math/Tool delta 均为 0 |
| GSPO | 序列级 ratio、clip 和 surrogate | ratio 在序列级计算；Tool 93.33% 组零方差 | 接近 GRPO | Math/Tool delta 均为 0 |

因此当前实验回答了“实现是否正确、采样机制是否生效、成本是多少”，但没有回答“哪种算法最终能力最好”。四算法固定生成完全相同，意味着更新太小，而不是四种目标在充分训练时必然等价。

### 7.4 训练波动与跨 Seed 稳定性

训练日志同时统计每个 run 最后若干更新的 Reward temporal std，以及三个训练 seed 的 Reward sample std：

| 算法 | Math Reward seed std | Math temporal std | Tool Reward seed std | Tool temporal std |
|---|---:|---:|---:|---:|
| GRPO | 0.0577 | 0.8531 | 0.0764 | 0.1652 |
| CISPO | 0.0577 | 0.8531 | 0.1155 | 0.3220 |
| DAPO | 0.1893 | 0.4119 | 0.0289 | 0.4279 |
| GSPO | 0.0500 | 0.9834 | 0.0289 | 0.1054 |

这些训练波动不能被简单排序为“越小越好”：

- DAPO 只统计动态筛选后的混合组，Reward 分布与普通随机组不同；
- 10 个更新太少，temporal std 对单个组非常敏感；
- Tool 中 GRPO/GSPO 的较低 temporal std 可能只是长期采到全错组，Reward 稳定地低，并不代表优化更健康；
- Math DAPO 的 Reward seed std 更高，说明有效组的难度和成功比例在 seed 间仍有明显差异。

固定集上四算法任务指标的训练 seed std 都为 0，是因为三个 checkpoint 都没有改变固定生成。这不是“跨 seed 稳健提升”的证据，而是“当前更新不足以形成可观测策略差异”的证据。

因此稳定性必须同时看零方差率、有效组接受率、ratio/KL、temporal std、seed std 和固定集能力；只挑一个方差最小的数字会得到错误结论。

### 7.5 Reward、准确率、KL、长度和未完成率的联合分析

#### Reward 与准确率

Formal 使用 strict `+1/-1`，因此 Reward 与 Task Accuracy 单调对应，但训练组经过动态筛选时分布会改变。固定集未经过 DAPO 筛选，才是公平能力比较。两任务固定集 Reward 和 Accuracy 都没有变化。

#### KL

数学固定集最大平均 k3 KL 约 `5.43e-9`，Tool 最大约 `1.53e-8`，说明训练后策略与 Agent SFT reference 极接近。结合 `lr=1e-7` 和 10 次更新，这是固定生成不变的直接解释之一。

训练 batch 的 KL mean 曾被极少数低概率 token 放大，所以日志还记录 token P95/max。不能把训练极值解释成整体策略已经大幅偏移；fixed holdout KL 更能反映最终 checkpoint 的平均变化。

#### 响应长度与过长推理

数学固定集平均/P95 响应长度为 `80.19/100.08`，Tool 为 `92.34/104.85`，四算法与 baseline 完全相同。DAPO Soft Overlong 从约 160 action tokens 才开始惩罚，因此当前固定集大多数轨迹没有进入惩罚区，无法由这些结果证明 Soft Overlong 缩短了推理。

#### 未完成率与截断

数学 Unfinished Rate 为 25%，Tool 为 1.0417%，四算法均无变化。数学较高的 unfinished 说明部分轨迹在最大轮数内没有闭合最终答案，但当前日志无法区分命中单轮 token 上限、最大轮数或上下文上限。因此该指标应报告为“未完成率”，不能改名为严格“截断率”。

#### 格式与工具指标

Tool 固定集 Format Valid 为 97.9167%、Tool Call Valid 为 99.4792%，但 Execution Success 只有 71.6146%、Evidence Coverage 只有 35.9375%。这说明“能输出合法 JSON”远不等于“正确完成工具任务”：主要短板已经从语法格式转向正确工具链、参数、执行证据和最终答案整合。

四算法没有进一步改变这些指标。当前输出一致性与协议遵循的主要改善来自 Agent SFT，而不是本次小预算 RL。

### 7.6 可复现性如何保证

1. train/eval 使用持久化 manifest，问题哈希零交叉；
2. 四算法从同一个 SHA-256 可追踪的 Agent SFT checkpoint 初始化；
3. 每个任务固定 3 个训练 seed 和 3 个 decode seed；
4. 每个 run 使用独立 checkpoint 与 JSONL，脚本拒绝覆盖或混写已有实验；
5. 保存逐轨迹 prompt、completion、Reward 分项、长度、unfinished 和 KL；
6. 训练指标与固定集指标分别聚合，避免把同一 checkpoint 的 decode seed 当作独立训练；
7. 数据、权重、日志、CSV 和轨迹生成 SHA-256；
8. 最终 31/31 单元测试通过，证据索引 `missing=[]`。

如果只保留聚合均值而没有逐轨迹、seed、config 和 checkpoint，就无法复查 Reward Hacking、verifier 假阳性或某个异常 KL token，本项目因此把逐样本证据作为正式产物的一部分。

### 7.7 最终结果应怎样下结论

可以下的结论：

- 完成 GRPO、CISPO、DAPO、GSPO 的统一实现和三 seed Formal；
- DAPO Dynamic Sampling 在稀疏 Tool-Use 中消除了 accepted groups 的零方差问题；
- 精确量化 DAPO 的额外 rollout、action-token 和 wall-time 成本；
- 建立覆盖 Reward、Accuracy、KL、长度、unfinished、工具执行和训练波动的评测体系；
- Agent SFT 显著改善冷启动任务完成率，严格 verifier 能拒绝已知 Reward Hacking 路径。

不能下的结论：

- DAPO 或 GSPO 已提高固定集准确率；
- 四算法在充分训练后效果相同；
- Soft Overlong 已实测降低过长推理；
- `unfinished` 就是严格 token 截断率；
- Agent SFT 的正向结果来自后续 RL；
- 三 seed 固定输出相同代表算法具有统计显著稳定性优势。

这组实验最有价值的结果，是把“代码实现正确”“训练信号有效”“模型能力改善”三个层级分开，并对负结果给出可验证原因和下一轮实验方案。

## 8. 失败、负结果与工程复盘

1. Hugging Face 网络被拒绝时，检查本地数据而不是反复下载；本次确认本地文件哈希一致后继续。
2. macOS AppleDouble `._*.py` 会令 Linux `compileall` 报 null bytes；清理这些元数据文件，不修改真实源码。
3. 首版多轮 rollout 对 assistant 文本重编码，破坏 sampled IDs；改成精确 token ledger 后通过非 round-trip 回归测试。
4. 基础工具集经 Agent SFT 后 100% 饱和；新增零冷启动标签重合的组合挑战集，而不是在饱和集上宣传算法提升。
5. BF16 current/old 与 FP32 reference 混用会放大极低概率 token 的 k3；统一精度后仍保留 mean/p95/max，识别真实长尾。
6. KD 与 DPO 都没有在当前预算下改善独立 holdout；保留负结果，后续调参必须另设 validation，不能在 test 上反复挑配置。
7. MoE 只部分训练，实测吞吐低于 Dense；项目只声称机制验证，不声称更快或更准。
8. 训练脚本原有 remainder accumulation 保存顺序可能漏持久化最后一个 update；已改为完成 remainder step 后原子保存，并如实记录旧 checkpoint 的影响比例。

## 9. 面试叙事、简历措辞与证据边界

项目名称建议写成“轻量级大语言模型全流程训练与 Agent RLVR 系统”，不要把工程基线名称当成项目核心。面试叙事应遵循“目的—问题—方法—结果—边界”的顺序，而不是罗列名词。

### 9.1 一分钟实验主线

项目首先基于轻量 Decoder-only Transformer 完成 Pretrain 和 Assistant-only SFT，获得可用于后训练的统一基座；随后用 LoRA、logits KD 和 DPO 验证监督微调、参数高效微调、知识迁移和偏好优化链路。为了研究可验证强化学习，又构造数学推理与 Tool-Use 两类任务，并实现多轮 `Assistant → Tool → Observation → Assistant` 状态机、精确 action-token 账本和严格 verifier。

初始实验发现两个关键问题：一是原始 SFT 模型缺少工具协议冷启动能力，Basic Tool 成功率仅 7.03%；二是 strict binary reward 在 Tool-Use 上极度稀疏，GRPO/CISPO/GSPO 的组内 reward 经常全错而使 advantage 为零。前者通过 Agent SFT 将 Basic Tool 提升到 100%、数学任务提升到 59.38%；后者通过 DAPO 动态采样只接收同时包含正负样本的有效组，将 Tool 训练 accepted-group 的零方差率降到 0，但付出了约 10.4 倍候选组、10.9 倍 action token 和 9.9 倍 wall time。

在统一 Agent SFT 初始化、相同训练预算和三训练 seed 下，项目进一步比较 GRPO、CISPO、DAPO、GSPO 的训练 Reward、Accuracy、KL、响应长度、未完成率、训练波动和资源成本。最终固定 holdout 上四种算法都没有产生可观测能力提升，checkpoint 与 reference 的 KL 也接近 0。该负结果说明 10 个 accepted updates、`1e-7` 学习率的微型预算足以验证训练机制，却不足以推动策略跨过贪心解码的行为边界；因此报告将“实现正确”“获得有效训练信号”和“能力提升”明确分开。

### 9.2 简历第 3 条的证据对齐版本

建议写为：

> 在统一 Agent RLVR 框架中复现 GRPO、CISPO、DAPO 与 GSPO，完成 DAPO 动态采样、非对称 Clip、Token-level Loss、Soft Overlong，以及 GSPO 序列级重要性比率；在三训练 seed 下分析稀疏奖励、有效组比例、ratio/KL 与训练波动。DAPO 将 Tool-Use accepted-group 零方差率降至 0，但候选组、action token 和训练时长分别增加约 10.4×、10.9× 和 9.9×。

这句话中每个动词的证据含义是：

- “复现”表示目标函数、采样过程、更新过程、日志和单元测试已经落地，不表示复现了论文原始模型规模或论文最终榜单；
- “动态采样”表示按组过滤全对/全错样本，直到收集足够的非零方差组，并显式记录 candidate/accepted groups；
- “非对称 Clip”表示允许负优势 token 在更宽的下界内继续降概率，以减轻 clip 对错误答案惩罚的过早截断；
- “Token-level Loss”表示按全 batch 有效 action token 数归一化，而不是先对每条序列等权平均；
- “序列级重要性比率”表示 GSPO 先聚合整条响应的 log-ratio，再以同一个 sequence ratio 约束该响应内所有 action token；
- “稳定性”表示联合分析有效组、zero-variance、ratio/KL 长尾、时间波动、跨 seed 波动和固定集能力，不能用单一 loss 曲线替代；
- “提升训练稳定性”在本实验中只能具体化为 DAPO 消除 accepted-group 的零梯度退化、Soft Overlong 提供连续长度约束、GSPO 降低 token ratio 长尾敏感性，不能表述为四算法都提升了最终准确率。

### 9.3 简历第 4 条的证据对齐版本

建议写为：

> 构建数学推理与 Tool-Use 的多轮 Agentic RL/RLVR 数据、环境和固定评测集，基于格式、参数合法性、执行状态、证据覆盖与最终答案设计确定性可验证奖励；记录 Reward、任务准确率、KL、响应长度、未完成率、工具执行率及跨 seed/时间波动。通过 Assistant-only action mask、精确 token ledger、严格终局奖励、无答案惩罚和 Soft Overlong 约束 Reward Hacking、格式崩溃与过长推理，并对 Agent SFT 的正向收益和小预算 RL 的零增益分别归因。

这句话需要按以下口径解释：

| 简历术语 | 本项目的具体实现 | 实验能支持的结论 |
|---|---|---|
| Agentic RL | 模型产生 assistant action，环境解析工具调用、执行工具、返回 observation，再由模型继续决策 | 多轮轨迹、mask、环境状态和终局 verifier 均已实现并测试 |
| RLVR | 不训练主观 Reward Model，而用确定性规则与工具执行结果产生 `+1/-1` 可验证奖励 | 奖励可重放、可审计；1600 条 oracle 轨迹重放失败为 0 |
| Reward | strict 终局奖励要求协议、执行、必要工具、证据和最终答案同时正确 | Reward 与 Accuracy 在固定集上单调对应；训练组经 DAPO 筛选后不能与固定集 reward 直接横比 |
| 准确率 | 数学最终答案 exact-match；Tool 任务需通过完整 strict verifier | Agent SFT 有明确正向结果；四算法 Formal 固定集均无增益 |
| KL | 记录 current 与 reference 的 token 级 k3 mean/P95/max，并在固定 holdout 重新评估 | 固定集 KL 近 0，解释了生成行为未变化；不能据此声称更新充分 |
| 响应长度 | 记录每条 assistant action token 总数及 mean/P95 | Formal 中四算法长度完全相同，不能声称 Soft Overlong 已缩短响应 |
| 截断率 | 当前可靠字段是 unfinished；它混合最大轮数、token/context 上限等终止原因 | 只能报告“未完成率”；要声称严格截断率，需补充 stop-reason 分类 |
| 训练波动 | 同时统计每 run 的 temporal std 和三训练 seed 的 seed std | DAPO 在 Tool accepted groups 上消除零方差，但筛选分布和额外采样成本必须同时披露 |
| Reward Hacking | 只输出格式标签、伪造 observation、调用无关工具、不给最终答案或靠冗长文本骗局部奖励 | observation 仅由环境写入；终局一票否决与工具证据链拒绝这些路径 |
| 格式崩溃 | 工具 JSON、标签闭合或角色状态不合法 | parser/schema 校验、Agent SFT 冷启动和 format gate 共同约束 |
| 过长推理 | 生成接近预算仍不闭合答案 | 无最终答案直接失败；Soft Overlong 在阈值后连续降 reward，避免只在硬截断点产生不连续惩罚 |
| 一致性与可控性 | 合法协议、可执行工具链、环境 observation 不可伪造、奖励规则确定 | Agent SFT 显著改善协议一致性；严格 verifier 提高过程可控性，但本轮 RL 未进一步改善固定集 |
| 任务完成率 | strict verifier 的最终成功率，不等同于格式正确率或单次工具执行成功率 | Tool 固定集格式 97.92%、调用合法 99.48%，但任务准确率仅 5.73%，证明必须分层报告 |

### 9.4 高频项目拷问的回答边界

**为什么不用训练 Reward Model？** 任务有唯一数学答案或可执行工具结果，规则 verifier 更便宜、可解释、可重放，也不会引入 Reward Model 的泛化误差；代价是奖励稀疏，需要 Agent SFT 冷启动和 DAPO 动态采样。

**为什么 GRPO/CISPO/GSPO 在 Tool 上容易没有梯度？** group-relative advantage 依赖组内 reward 方差。当同一 prompt 的四条 rollout 全错时，标准差为 0，归一化 advantage 全为 0。换 surrogate 或 clip 不能凭空创造偏好信号；DAPO 通过额外采样找到至少一个成功和一个失败响应。

**为什么训练 Reward 看起来变好，固定集却不变？** DAPO 的训练统计来自条件分布——只统计被接收的非零方差组；固定集来自未筛选的原始分布。再加上 10 次 update 和极小学习率使策略 KL 近 0，训练有效组的高 Reward 不能解释为总体能力提升。

**为什么 GSPO 使用序列级 ratio？** Tool/Reasoning 的好坏通常是整条轨迹属性。token ratio 的极端值会让少数 token 主导梯度，而几何平均序列 ratio 将整条响应作为一致的策略单元；但它也牺牲 token 级 credit assignment，需要用独立实验比较。

**Soft Overlong 与硬截断有什么区别？** 硬截断只在预算耗尽时给离散失败信号；Soft Overlong 在进入缓存区后按长度连续扣分，使模型在真正截断前就收到梯度。本实验响应长度大多未进入惩罚区，所以验证了代码路径，尚未证明长度改善。

**你自己实现了什么？** 应明确区分归属：上游已有模型基本模块与 Pretrain/SFT/LoRA/KD/DPO 主链路；本项目扩展工作的核心是统一四算法优化目标、DAPO/GSPO、多轮 Agent 环境与 strict verifier、精确 token ledger/action mask、挑战数据构造、固定集多指标评测、三 seed 调度、证据归档和回归测试。详细归属见 [SOURCES](./SOURCES.md)。

### 9.5 不应使用的表述

- 不说“从零原创了全部模型和训练框架”，应说“基于轻量代码基线复现并扩展”；
- 不说“DAPO/GSPO 提升了固定集准确率”，因为 Formal 的 `delta_accuracy=0`；
- 不说“Soft Overlong 降低了截断率”，因为当前只记录 unfinished，且大多数响应未进入惩罚区；
- 不说“KL 很低说明训练特别稳定”，低 KL 在这里主要说明更新幅度过小；
- 不说“Tool 格式正确率接近 100% 所以 Agent 成功率很高”，格式、执行、证据与最终任务成功是不同层级；
- 不把 Agent SFT 的提升归因给 RL，也不把 DAPO accepted-group 的训练 Reward 当成固定分布能力；
- 不以三 seed 固定输出相同声称统计显著，应报告预算、effect size、置信区间与负结果。

## 10. 最终验收

最终流水线执行以下门禁：

```text
确定性 Agent SFT mask 审计
→ MoE 路由与梯度审计
→ 实际模型 Flash kernel profiler
→ 多轮 token ledger 回归
→ 两套 Tool-RLVR 数据全量 oracle 重放
→ Python compileall
→ 全量 unit tests
→ 证据索引与 SHA-256
```

最终验收为 31/31 单元测试通过。实际模型 BF16、1024-token forward 的 profiler 命中 PyTorch Flash SDPA kernel；Agent SFT mask 审计 128 样本 mismatch=0、zero-supervision=0；两套 Tool 数据共 1,600 行 oracle 重放失败数为 0；MoE router/expert 梯度全部 finite；多轮 non-roundtrip 回归证明精确 sampled IDs 被保留。

机器可读总入口为 `out/run_meta/experiment_evidence.json`：7 个流水线 marker 全为 true，`missing=[]`。任何缺失实验都会列在 `missing`，不会静默填零。最终数字以 CSV/JSON 和 checksum 为准，本文是这些证据的解释层。
