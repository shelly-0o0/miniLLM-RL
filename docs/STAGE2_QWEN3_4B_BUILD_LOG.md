# Stage 2：Qwen3-4B Agentic RL 完整搭建、算法与执行记录

> 日期：2026-10-03
> 分支：`gsm8k-agentic-rl`
> 比较矩阵：Base / Pure GRPO / SFT only / SFT → GRPO
> 当前状态：代码、配置、数据与 tokenizer 集成已通过；本地环境没有可见 CUDA，但已在远程 8×RTX 4090D 主机的独立目录和独立 Conda 环境完成三个真实 GPU smoke。

## 1. 本次工作的目标与边界

Stage 2 不是把一个普通 GSM8K 单轮答案脚本换成更大的模型。它保留 Stage 1 的完整 Agent 环境：

```text
数学题 + 工具定义
  → 模型生成 calculate_math 调用
  → 环境解析并执行受限算术表达式
  → 工具结果作为 observation 回填上下文
  → 模型继续生成最终答案
  → verifier 同时核验答案、调用、执行结果和证据
  → 严格可验证奖励进入 GRPO
```

因此本阶段仍属于 Agentic RLVR：动作不只是最终文本，还包括工具选择、参数构造、停止工具轮次、读取观察并完成回答。当前环境只提供 calculator，所以它是“单工具、短时域、确定性环境”的 Agentic RL，不应夸大为开放世界通用智能体。

本次确定的四组实验为：

| 组别 | 初始化 | 是否 SFT | 是否 GRPO | 研究含义 |
|---|---|---:|---:|---|
| Base | Qwen3-4B-Base | 否 | 否 | 原始模型能力 |
| Pure GRPO | Base + 新建零初始化 LoRA | 否 | 是 | RL 能否从无 Agent 冷启动直接学习 |
| SFT only | Base + Agent SFT LoRA | 是 | 否 | 行为克隆本身带来的收益 |
| SFT → GRPO | 同一个 SFT adapter | 是 | 是 | 有可优化行为先验后，RL 的增量收益 |

Pure GRPO 与 SFT → GRPO 的区别只允许是初始化 adapter；二者使用相同训练题目、工具、训练奖励、组大小、采样参数、学习率和候选组预算，最终再用同一个严格 verifier 评测。

## 2. 为什么采用 QLoRA，而不是全参数训练

Qwen3-4B-Base 有约 40 亿参数。Stage 2 的 RL 同时需要：

- 可训练 policy；
- 冻结 reference policy；
- rollout 的 KV cache；
- 当前策略、旧行为策略和 reference 的 token log-probability；
- optimizer state 与梯度。

直接 BF16 全参数训练不适合目标机器的消费级 GPU。QLoRA 将冻结基座以 NF4 4-bit 形式加载，只训练低秩增量：

```text
W_effective = dequant(W_4bit) + (alpha / r) · B · A
```

当前配置：

- NF4 4-bit 基座；
- double quantization；
- BF16 计算；
- LoRA `r=16`、`alpha=32`、`dropout=0.05`；
- `target_modules=all-linear`；
- 基座参数冻结，只保存 PEFT adapter。

这保留了足够的可训练表达能力，同时显著降低 policy/reference 双模型的显存和 checkpoint 大小。

## 3. 数据边界和为什么没有“只用几百条训练”

正式训练使用预先固定的全部训练 split：

| 文件 | 行数 | 用途 |
|---|---:|---|
| `train_rl.jsonl` | 6726 | Pure GRPO 与 SFT → GRPO 的 prompt |
| `train_sft.jsonl` | 6637 | Agent SFT |
| `validation_rl.jsonl` | 747 | 调参和模型选择，不反向传播 |
| `test_rl.jsonl` | 1319 | 最终一次报告，不反向传播 |

SFT 比 RL 少 89 条，是因为只有能从 GSM8K 标注中抽取并重放出可靠 calculator oracle 的题目才进入监督轨迹；RL prompt 本身不要求预制 oracle action，所以保留 6726 条。

“候选组”不是另一个小数据集。对一个训练题目采样 `G=8` 条轨迹，就是一个候选组：

```text
1 prompt × 8 stochastic trajectories = 1 GRPO group
6726 prompts × 8 trajectories = 53808 rollout trajectories / epoch
```

DAPO 过去所说的“有效组接受率”是动态采样的计算预算概念，并不表示只训练 30 道题。本 Stage 2 矩阵只比较 GRPO，因此每个 prompt group 都被消费；零方差组会得到零相对优势，同时被日志明确记录。

四个数据文件的 SHA-256 已由 `dataset/manifests/gsm8k_agent.json` 固定。启动前审计会重新计算行数和哈希，任何静默改动都会阻止训练。

## 4. Agent SFT 的目标与掩码

SFT 样本包含：

```text
system: 任务规则 + 工具 schema       条件，不计算 loss
user:   GSM8K 问题                  条件，不计算 loss
assistant: calculator tool_call     计算 loss
tool:   calculator observation      条件，不计算 loss
assistant: Final answer             计算 loss
```

损失是 assistant token 上的标准自回归交叉熵：

```text
L_SFT = - 1 / |A| · Σ[t∈assistant actions] log πθ(x_t | x_<t)
```

工具 observation 必须作为下一轮输入，但绝不能作为模型动作监督，否则模型会学会伪造工具返回。

### 4.1 Qwen3 模板的真实边界问题

真实 tokenizer 验证发现，Qwen3 的 tool-aware 模板是上下文相关的：

- `add_generation_prompt=True` 时会为即将生成的 assistant 插入空 thinking block；
- 同一个已完成的历史 tool-call turn 在完整对话中可能不含这个空 block。

所以不能假设“逐段渲染的 token 一定是完整对话 token 的前缀”。本次实现采用两级策略：

1. 对前缀稳定的模板，直接通过逐轮前缀长度定位 assistant span；
2. 对 Qwen3 的上下文相关模板，先渲染完整对话，再使用 fast tokenizer 的字符 offset，从 `<|im_start|>assistant` 到 `<|im_end|>` 精确选择动作 token。

若 token 跨越监督边界、assistant block 数量不匹配或 tokenizer 无 offset，立即失败，不进行可能泄漏的训练。

真实 Qwen tokenizer 对 128 条样本的结果：

```text
rows_checked:            128
zero_supervision_count:  0
min_supervised_tokens:   34
max_supervised_tokens:   182
max_sequence_length:     515
```

第一条样本的监督解码只含两个 `<tool_call>`、两个 assistant 轮次的 `<|im_end|>` 和 `Final answer: 72`；不含 Natalia 问题，也不含 `{"result": ...}` 工具 observation。

## 5. 多轮 Agent rollout 的实现

一个 rollout 的状态转换为：

1. 使用 Qwen chat template 渲染 system、user、tools 和 generation prompt；
2. policy 采样 assistant token，保留每个动作 token 的行为策略 log-probability；
3. 解析 `<tool_call>{...}</tool_call>`；
4. 检查工具名和参数 schema；
5. `safe_calculate` 只执行受限算术语法，不执行任意 Python；
6. 以 `<tool_response>` 把执行结果追加到上下文；
7. observation token 的 action mask 为 0；
8. 下一轮 assistant token 的 action mask 恢复为 1；
9. 无工具调用时把该轮视为最终回答；到达长度或轮数上限则标记 unfinished。

轨迹账本记录：

```text
prompt_ids
response_ids = action_1 + observation_1 + action_2 + ...
response_mask = 1...1 + 0...0 + 1...1 + ...
old_logps = sampled logp + neutral observation slots
```

不会在 rollout 后左截断轨迹，因为截断会让行为策略概率、action mask 和模型上下文不再表示同一个过程。若轨迹超过 `max_total_length`，训练直接报错，要求调整生成预算。

无 4B 权重的协议审计已用确定性假 policy 走完真实 Qwen tokenizer 的两轮工具流程：

```text
strict_reward:      +1
turns:               2
response_tokens:    76
action_tokens:      46
observation_tokens: 30
stop_reason:         final_answer
```

## 6. 严格 RLVR 指标与训练课程奖励

最终任务指标始终采用二值严格奖励：

```text
r = +1  当且仅当整个任务可验证成功
r = -1  其他情况
```

“整个任务成功”同时要求：

- 最终数值答案正确；
- tool-call 标签完整且 JSON 可解析；
- 调用的是允许的工具；
- 参数通过 schema 检查；
- 必需工具已实际执行；
- 工具执行结果能支持 ground truth；
- 最终轨迹不是 unfinished。

因此以下 reward hacking 都不能得分：

- 不调用工具直接猜答案；
- 调用无关工具后猜答案；
- 写一个正确答案但传入错误算式；
- 把 ground truth 当作工具返回文本伪造；
- 撞满 `max_new_tokens` 后留下半截工具调用。

SFT 用 oracle action 提供行为冷启动；RL reward 不读取 oracle action，只读取题目 ground truth 和真实执行轨迹。

正式启动前的 Base 16×8 probe 发现：严格奖励 128/128 都为 -1；Base
已经输出 calculator JSON 和部分正确算式，但没有协议标签。若仍直接用严格
奖励，组内优势恒为 0。Track 1 的两条 GRPO 分支因此改用完全相同的
deterministic shaped curriculum reward 训练，最终 validation/test 仍只报告上述
strict success。

课程奖励分层衡量：标签平衡、JSON/schema、允许工具与参数、实际执行、必需工具
覆盖、工具结果证据、最终答案、重复和 unfinished。额外的
`protocol_progress` 只在没有正式 tagged call 时识别裸 JSON 中的：

```text
schema → tool name → arguments → safe execution → result evidence
```

它有固定上界，重复候选只取最大值；裸 JSON 不会被送入环境，也不会增加
`tool_call_valid`、`tool_execution_success` 或 strict task success。这样可为 Pure
冷启动提供相对优势，同时不能把“看起来像工具调用”伪装成 Agent 成功。保存的
Base 轨迹离线重评分后 strict success 仍为 0，而 shaped 非零方差组率从 0%
变为 100%，验证了这条边界。

## 7. GRPO 原理与当前实现

对同一个 prompt 采样 `G=8` 条轨迹，得到训练奖励 `r_i`。Track 1 两分支使用
同一个 shaped 定义；strict outcome 作为独立日志和最终指标。组内优势为：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-4)
```

若一组八条全错或全对，标准差为 0，优势自然为 0。这正是 Pure GRPO 冷启动可能很慢的原因，也是本矩阵要比较 SFT warm start 的核心。

行为策略比率：

```text
ratio_i,t = exp(log πθ(a_i,t|s_i,t) - log πold(a_i,t|s_i,t))
```

剪切 surrogate：

```text
min(ratio · A, clip(ratio, 1-ε, 1+ε) · A),  ε = 0.2
```

reference KL 使用非负 k3 估计。令：

```text
δ = log πref - log πθ
KL_k3 = exp(δ) - δ - 1
```

最终损失由“每条序列先做动作 token 平均，再在组内平均”的 policy loss，加上“所有动作 token 的全局平均 KL”构成：

```text
L = - mean_sequence(min(ratio·A, clipped_ratio·A))
    + β · mean_action_token(KL_k3)

β = 0.02
```

### 7.1 三个策略角色

- `policy/current`：正在反向传播的 LoRA；
- `old/behavior`：生成本批 rollout 时的 policy，通过保存的/recomputed log-probability 固定；
- `reference`：本次训练开始时的 adapter，整个运行期间冻结，用于 KL 锚定。

Pure GRPO 的 policy/reference 都从同一个 fresh LoRA 状态建立，并显式复制 PEFT state 保证字节级一致。SFT → GRPO 的二者都从同一个 SFT adapter 建立。resume 时只恢复 current policy；reference 从首次保存的 `reference_adapter` 恢复，不能随训练漂移。

## 8. 4B 模型的显存设计

Qwen3 词表约 151669。若为整条长轨迹保留 `[sequence, vocabulary]` logits，显存会非常高。实现采取两项等价变换：

1. 根据 action mask 只请求“动作 token 前一个位置”的 logits，prompt 和工具 observation 不物化完整词表 logits；
2. 一次只对一条轨迹执行 current forward/backward，将每条轨迹的贡献累加。

`trajectory_grpo_loss` 的缩放使逐轨迹求和与原本 dense `[group, sequence]` GRPO 完全相等：

- policy contribution 除以 group size；
- KL contribution 按全组动作 token 总数归一化。

安全门必须采用同一个归一化定义。若先求每条轨迹的 KL 均值再逐条与阈值比较，短轨迹或单个极低 reference-probability token 会被不等权放大，可能拒绝一个实际组级 KL 仍安全的 batch。因此实现会在任何 backward 之前，用已重算的 policy/reference log-probability 计算整组 action-token 全局 KL；只有该组均值有限且不超过 `max_group_kl_k3=10` 才进入 streaming backward。诊断仍保存最坏轨迹和最坏 token，梯度仍按 `max_grad_norm=1` 裁剪。

长程 formal 对“有限但超过阈值”的 sampled 重尾组采用有界拒绝：不执行 backward/optimizer step，保存独立诊断，并把已消费 candidate 与 RNG 原子写入 checkpoint；最多 64 个、最多连续 3 个，超限仍终止。非有限 KL 无条件立即终止。该模式只允许 `gradient_accumulation_steps=1`，否则清梯度会误丢前一组的累计贡献。最终审计把正常 train 记录与 safety-rejection 记录共同覆盖全部 candidate group，同时要求拒绝数与 final 元数据一致。

这一路径又用历史上确定会在第 190 组产生重尾 KL 的 checkpoint 做了确定性回放，而不是只靠单元测试推断：第 189 组结束时 optimizer update 为 189；第 190 组的整组 KL 为 `41.8680496`，被写为 `safety_rejection`，optimizer update 仍为 189；第 191 组 KL 回落到 `0.00162286` 并正常完成第 190 次更新。回放最终记录 191 个候选组、190 次更新、1 次拒绝，保存了独立诊断和终态 adapter，且没有 `safety_failure.json`。这证明拒绝分支同时满足“异常组不更新”“消费位置/RNG 前进”“下一正常组可恢复训练”三个要求。

单元测试同时计算 dense 目标和 streaming 目标，数值一致并能反向传播。

rollout 时显式启用 KV cache；训练时关闭 cache 并启用 non-reentrant gradient checkpointing。实测后 policy 和 reference 默认同置 `cuda:0`，但仍是两份参数独立、reference 冻结的模型视图。

同一 prompt 的 G 条首轮 action 使用一次批量生成；产生合法工具调用后，每条轨迹
再按各自 observation 独立续写。批量输出的 PAD 有独立 completion mask，绝不进入
action ledger。真实 pilot 还证明两份 4-bit policy/reference 可同放一张 24 GiB
GPU（峰值约 14.2 GiB）；由于两者本来顺序前向，单卡每组 38.35 秒与双卡
38.90 秒相当，因此正式两分支可各占一张卡并行。

## 9. 新增与修改的实现文件

### 9.1 新增文件

| 文件 | 作用 |
|---|---|
| `trainer/agent_chat.py` | 统一 MiniMind/Qwen 消息规范化、模板渲染、assistant-only mask |
| `dataset/qwen_stage2.py` | Qwen Agent SFT JSONL 数据集和动态右 padding collator |
| `trainer/qwen3_adapter.py` | Qwen3、NF4、LoRA、PEFT 保存/加载和 action-only logp |
| `trainer/train_qwen_lora_sft.py` | 配置驱动的 QLoRA Agent SFT |
| `trainer/qwen_grpo_objective.py` | 与 dense GRPO 等价的逐轨迹目标 |
| `trainer/train_qwen_grpo.py` | Pure GRPO 与 SFT → GRPO 共用训练入口 |
| `scripts/evaluate/evaluate_qwen_stage2.py` | 四组固定 validation/test 评测 |
| `scripts/audit_qwen_stage2.py` | 依赖、哈希、SFT mask、真实模板和多轮协议门禁 |
| `scripts/audit_qwen_stage2_results.py` | 完整训练预算、finite adapter、四组评测与哈希终态门禁 |
| `scripts/promote_qwen_sft_candidate.py` | 带审计/探针/哈希 provenance 的校准候选提升工具 |
| `scripts/run_qwen_stage2.sh` | smoke、正式训练与评测统一入口 |
| `scripts/summarize_qwen_grpo_run.py` | 无侵入汇总运行中 JSONL、最近窗口与数值门禁 |
| `requirements-stage2.txt` | Stage 2 固定依赖 |
| `configs/qwen3_4b/*.yaml` | smoke、正式 SFT、两条 GRPO 和评测矩阵 |
| `tests/test_qwen_stage2.py` | Stage 2 数据、模板、目标与配置测试 |

### 9.2 修改的共享文件

- `trainer/train_agent.py`：通过统一 helper 渲染不同模型的工具模板；
- `trainer/rollout_engine.py`：推理生成启用 KV cache；
- `.gitignore`：忽略 Hugging Face 缓存和本地 tokenizer 副本；
- `README.md`、`PROJECT_PLAN.md`、`EXPERIMENT_PROTOCOL.md`：固定新的四组矩阵。

## 10. 配置与训练预算

正式 SFT 分为两段。第一段用全部 oracle 建立内容和工具行为先验；第二段从第一段 adapter 扩展 `lm_head` LoRA，并再次完整遍历同一训练集，以权重 8 强化三个 Qwen 协议边界 token：

```text
data rows:                  6637
micro batch:                1
gradient accumulation:     16
effective batch:            16
epochs:                     1
learning rate:              1e-4
scheduler:                  cosine, warmup 3%
max sequence length:        1536
protocol repair epochs:     1
protocol token weight:      8
LoRA targets after repair:  7 projections + lm_head
```

正式 GRPO（两组相同）：

```text
train prompts:              6726
candidate groups:           6726
generations per prompt:     8
max turns:                  3
max new tokens per turn:    384
max total trajectory:       2500
temperature/top-k/top-p:    1.0 / 0 / 1.0
learning rate:              1e-6
epsilon/beta:               0.2 / 0.02
policy update epochs:       1
training reward:            shared shaped curriculum
final task metric:          strict executable success
```

这里的 full-softmax 采样是算法门禁，不只是解码偏好。rollout ledger、old/current
importance ratio 和 reference k3 都从模型未加温度、未截断的完整 softmax 重算；若生成
使用 temperature、top-k 或 top-p 的另一分布，样本就不是来自日志中所称的 policy。
训练入口因此拒绝任何非 `1.0 / 0 / 1.0` 的采样配置，probe 与最终评测也保持一致。

smoke 配置有独立输出目录，绝不会覆盖正式 adapter：SFT 只跑 1 step；两条 GRPO 各跑 2 个 prompt group、每组 4 条轨迹、每轮最多 128 tokens。

## 11. 断点、原子保存和日志

SFT 使用 Transformers Trainer checkpoint；`--resume` 自动从最近 checkpoint 恢复，也可直接调用 Python 入口传具体路径。

GRPO checkpoint 同时保存：

- current policy adapter；
- frozen reference adapter；
- optimizer；
- scheduler；
- Python、CPU Torch 和全部 CUDA RNG state；
- 当前 epoch 和 candidate group；
- 实际使用的 YAML。

adapter 通过“临时目录 → 原子替换”保存，避免中断后留下半个 checkpoint。

训练 JSONL 重点字段：

- `reward`、`task_accuracy`、`answer_accuracy`；
- `format_valid_rate`、`tool_call_valid_rate`；
- `tool_execution_success_rate`、`required_tool_coverage_rate`；
- `tool_evidence_coverage_rate`；
- `group_reward_std`、`zero_variance_group`、`advantages_std`；
- `policy_loss`、`kl_k3`、`clip_fraction`、`ratio_mean`；
- `rollout_logprob_mae`；
- `action_tokens`、`tool_calls`、`unfinished_rate`；
- `optimizer_updates`、`learning_rate`、`wall_time_seconds`。

`rollout_logprob_mae` 比较 rollout 保存的行为概率和训练前重算概率。超过 0.1 会停止训练，因为这通常说明采样 legal set、精度、模板或 token 账本不一致。

## 12. 本次实际执行记录

### 12.1 环境依赖

执行：

```bash
/home/user/miniconda3/envs/mini-rl/bin/python -m pip install \
  -r requirements-stage2.txt
```

结果：安装 `peft 0.17.1`、`bitsandbytes 0.48.1`，并按锁定文件将 `accelerate` 调整为 `1.10.1`、`PyYAML` 调整为 `6.0.2`。已有 `torch 2.6.0+cu124`、`transformers 4.57.6`、`datasets 3.6.0`。

### 12.2 tokenizer/config 集成

只下载并保存 Qwen3-4B-Base 的 tokenizer 和 config，没有下载 4B 权重：

```text
tokenizer class:     Qwen2TokenizerFast
vocabulary size:     151669
chat template:       present, tool-aware
base EOS:            <|endoftext|> / 151643
agent turn EOS:      <|im_end|> / 151645
model type:          qwen3
hidden size/layers:  2560 / 36
```

统一加载器把 `<|im_end|>` 设为生成停止 token，并启用 Transformers 4.57 建议的 `fix_mistral_regex=True`，避免旧 pre-tokenizer regex 产生错误 token 边界。

本地副本位于 `models/qwen3-4b-base-tokenizer/`，被 `.gitignore` 忽略；正式机器也可以直接从 `Qwen/Qwen3-4B-Base` 加载。

### 12.3 回归测试

第一次全量测试的唯一错误是 Hugging Face 默认缓存目录在沙箱中只读，不是代码错误。把缓存指向 `/tmp` 后重跑：

```bash
export HF_HOME=/tmp/mini_rl_hf_test
export HF_DATASETS_CACHE=/tmp/mini_rl_hf_test/datasets
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
```

最终结果：

```text
Ran 50 tests
OK
```

新增测试覆盖：

- tool schema 与 JSON tool_calls 规范化；
- 前缀稳定模板的 assistant-only mask；
- Qwen 类上下文相关模板的 offset mask；
- input/attention/label 使用不同 padding 值；
- action-indexed logits 与完整 logits 一致并可反向传播；
- streaming GRPO 与 dense GRPO 数值相等；
- 四组评测矩阵不漂移。

### 12.4 readiness 审计

执行：

```bash
PYTHONPATH=. python scripts/audit_qwen_stage2.py \
  --tokenizer models/qwen3-4b-base-tokenizer \
  --samples 128
```

结果为 `PASS`，报告写入 `out/run_meta/qwen3_stage2_readiness.json`。数据四个哈希全部与 manifest 匹配，SFT mask 和两轮 Agent 协议通过。

本机审计同时报告：

```text
cuda_available: false
cuda_devices:   []
```

因此没有把本机 CPU readiness 冒充成 GPU 验证。GPU 验证随后在远程目标机完成，详见下一节。

### 12.5 远程隔离部署与真实 GPU smoke

远程主机为 8 张 NVIDIA GeForce RTX 4090D，每张 23.64 GiB、支持 BF16；地址不写入公开仓库。检查时物理 GPU 5、6、7 空闲，原 `/root/Mini-RL` 中的 Stage 1 CISPO/DAPO/GSPO 仍在运行。

为避免修改 Stage 1 的代码或环境，执行了两层隔离：

```text
Stage 2 code:  /root/Mini-RL-stage2
Stage 2 env:   /root/miniconda3/envs/mini-rl-stage2
```

同步时排除 `.git`、`.cache`、`out`、`checkpoints` 和旧模型产物，只同步源码、固定 processed data 和已验证 tokenizer。新环境由 `mini-rl` clone；clone 后首次 pip 出现内部版本不一致，错误为 `cannot import name get_runnable_pip`。只在新环境执行 `conda install --force-reinstall pip` 后恢复，再安装 Stage 2 requirements；原环境未修改。

远程重复执行 50 个测试与 readiness audit，全部通过，CUDA 报告 8/8 可用。Hugging Face 主站不可达，`hf-mirror.com` 可达，因此以：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

完成首次权重下载。缓存完成后用 `HF_HUB_OFFLINE=1` 验证后续 smoke 完全复用本地权重。

SFT smoke（物理 GPU 5）：

```text
loaded shards:             3/3
trainable LoRA parameters: 33,030,144
optimizer steps:           1
loss:                      2.281991
gradient norm:             4.547428
runtime:                   2.262 s（不含首次下载/加载）
non-finite error:          none
adapter saved:             yes
```

Pure GRPO smoke（物理 GPU 5/6）：

```text
candidate groups:          2
trajectories:              8（4/group）
optimizer updates:         2
rollout_logprob_mae:       0.006145, 0.005558
zero-variance groups:      2/2
tool calls:                0
adapter saved:             yes
```

SFT → GRPO smoke（物理 GPU 5/6；仅从 1-step smoke SFT 初始化）：

```text
candidate groups:          2
trajectories:              8（4/group）
optimizer updates:         2
rollout_logprob_mae:       0.005725, 0.007790
zero-variance groups:      2/2
tool calls:                0
adapter saved:             yes
```

smoke 的 SFT 只有一个 step，不足以教会稳定工具行为，所以两个 GRPO smoke 都无工具调用、严格奖励全为 -1。这是预期的工程测试结果，不能用来得出 warm start 无效的算法结论。

产物级 safetensors 验收：

```text
Pure GRPO: changed parameters 25,270,833; relative L2 1.0837e-4; max |Δ| 1.5481e-6
SFT→GRPO:  changed parameters 33,022,995; relative L2 1.5209e-4; max |Δ| 1.5504e-6
non-finite tensors: none
```

Pure GRPO 即使优势为 0 仍有极小更新，是因为 policy/reference 在不同 GPU 上各自加载 4-bit 基座会产生微小数值差，非负 KL 项并非精确为零；`rollout_logprob_mae` 和 KL 都在日志中保留，正式实验需要继续监控。

最后将 Transformers 已弃用的 `torch_dtype` 加载参数改为 `dtype`，并在物理 GPU 7 以离线缓存重新加载全部 3 个 checkpoint shard；`model_type=qwen3`、`is_quantized=True`、device 为 `cuda:0`（对应物理 7），兼容性检查通过且不再出现该弃用提示。

## 13. 从零开始的正确执行顺序

以下命令都在仓库根目录执行，且先激活目标 Conda 环境。

### 步骤 1：安装 Stage 2 依赖

```bash
python -m pip install -r requirements-stage2.txt
```

原理：补齐 QLoRA 所需的 PEFT 和 bitsandbytes，并锁定已验证的 Transformers/Accelerate 组合。PyTorch 应按目标机 CUDA 单独安装，不能盲目由 requirements 覆盖。

### 步骤 2：执行无权重 readiness audit

```bash
python scripts/audit_qwen_stage2.py
```

原理：在消耗数小时 GPU 前先验证依赖、数据哈希、真实 Qwen 模板、assistant-only mask、四组矩阵和多轮工具协议。

### 步骤 3：检查 GPU

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30)
PY
```

原理：SFT 需要一张 GPU。实测 4-bit base 加 policy/reference 两个 LoRA 视图的峰值约 14.2 GiB，因此每条 GRPO 分支也只需一张 24 GiB GPU；`CUDA_VISIBLE_DEVICES=5` 后配置里的 `cuda:0` 就对应物理 GPU 5。

### 步骤 4：跑三个独立 smoke

```bash
bash scripts/run_qwen_stage2.sh smoke_sft
bash scripts/run_qwen_stage2.sh smoke_pure_grpo
bash scripts/run_qwen_stage2.sh smoke_sft_grpo
```

原理：

- `smoke_sft` 验证 4-bit 加载、LoRA 注入、梯度、optimizer 和 adapter 保存；
- `smoke_pure_grpo` 验证 fresh adapter、rollout、工具执行、冻结 reference、KL 和 GRPO backward；
- `smoke_sft_grpo` 验证 SFT adapter 能被同时加载为 policy/reference 并继续 RL。

smoke 通过标准：进程正常退出；无 NaN/Inf/OOM；adapter 文件存在；`rollout_logprob_mae ≤ 0.1`；至少有 action token；日志能看到工具与轨迹字段。smoke 准确率不用于判断模型效果。

这三个 smoke 已在上述远程环境通过；在另一台机器复现时仍应重新执行，因为 CUDA、驱动和显存条件可能不同。

若同名 smoke 输出已存在，训练会拒绝静默覆盖。要继续中断运行使用 `--resume`；要重新做实验，应人工归档旧目录后再启动。

### 步骤 5：正式 SFT

```bash
bash scripts/run_qwen_stage2.sh sft
bash scripts/run_qwen_stage2.sh lm_head_sft
bash scripts/run_qwen_stage2.sh audit_lm_head_sft
bash scripts/run_qwen_stage2.sh probe_sft_smoke
```

原理：先用 6637 条可靠 oracle 轨迹建立可优化的工具行为先验；再加入 `lm_head` LoRA 并完整训练一轮，使 `<tool_call>`、`</tool_call>`、`<|im_end|>` 不只在 teacher forcing 下可见，也能在自由生成中真正触发。审计检查协议 token 概率与普通位置误触发，探针检查实际工具闭环。产物：

```text
out/stage2/qwen3_4b/sft_adapter/
checkpoints/stage2/qwen3_4b/sft/
out/metrics/qwen3_4b_sft_s42.jsonl
out/logs/qwen3_4b_sft_s42.log
out/stage2/qwen3_4b/sft_lm_head_w8_s42_adapter/
checkpoints/stage2/qwen3_4b/sft_lm_head_w8_s42/
out/run_meta/qwen3_4b_sft_lm_head_w8_s42_audit.json
```

### 步骤 6：正式 Pure GRPO 与 SFT → GRPO

正式作业前先用完整 rollout 参数各跑 4 个候选组，测量真实每组耗时与显存峰值：

```bash
bash scripts/run_qwen_stage2.sh pilot_pure_grpo
bash scripts/run_qwen_stage2.sh pilot_sft_grpo
```

pilot 通过 `--output-tag pilot4` 写入独立 checkpoint、adapter、metrics 和日志，不会污染正式目录；`runtime_overrides.json` 记录预算覆盖。根据 pilot 的 wall time 估算 6726 组总时长，并确认持续零方差、撞长度上限或吞吐过低时是否需要先优化 rollout backend。

确认预算可接受后，两组互相独立；有两张空闲 GPU 时可各占一张并行，只有一张时顺序运行：

```bash
bash scripts/run_qwen_stage2.sh pure_grpo
bash scripts/run_qwen_stage2.sh sft_grpo
```

原理：两者各遍历全部 6726 个训练 prompt，各采样 8 条轨迹。Pure GRPO 测无 warm start 的探索难度；SFT → GRPO 测行为先验是否降低零方差组并提高样本效率。

断点恢复：

```bash
bash scripts/run_qwen_stage2.sh pure_grpo --resume
bash scripts/run_qwen_stage2.sh sft_grpo --resume
```

### 步骤 7：固定 validation 评测

```bash
bash scripts/run_qwen_stage2.sh eval_validation
```

原理：依次加载 Base、SFT only、Pure GRPO、SFT → GRPO，在同一个 validation 前缀 128 题、同一 decode seed 和同一采样参数下比较。评测只生成轨迹和统计，不更新参数。

### 步骤 8：完整 test 评测

```bash
bash scripts/run_qwen_stage2.sh eval_test
bash scripts/run_qwen_stage2.sh audit_results
```

原理：`--limit 0` 明确定义为使用完整 1319 条 official test。test 用于最终报告，不用于反复调参。评测记录配置、数据和 adapter SHA-256，拒绝覆盖既有正式 summary；最终 audit 只有在两条 GRPO 都消费完整预算、四组 validation/test 齐全、权重 finite 且哈希一致时才通过。

## 14. 结果应如何解读

重点比较：

```text
SFT gain        = Accuracy(SFT only) - Accuracy(Base)
Pure RL gain    = Accuracy(Pure GRPO) - Accuracy(Base)
Post-SFT RL gain= Accuracy(SFT→GRPO) - Accuracy(SFT only)
Warm-start gap  = Accuracy(SFT→GRPO) - Accuracy(Pure GRPO)
```

同时报告训练代价：rollout 数、动作 token、optimizer update、wall time 和 GPU 型号。若 Pure GRPO 的 `zero_variance_group` 长期接近 1，应解释为组内奖励没有形成相对信号，而不是简单写成“GRPO 公式无效”。

validation/test 必须同时报告 task accuracy 与过程指标。仅有 format/tool-call rate 提升但 evidence/answer accuracy 不提升，表示模型学会了外形而没有学会任务。

## 15. 从参考仓库采用了什么、没有采用什么

参考项目：<https://github.com/jjyaoao/qwen-grpo-gsm8k>

采用的工程思想：

- Hugging Face Qwen 基座；
- LoRA/QLoRA 降低资源门槛；
- 先短 SFT warm start、再做 GRPO 的对照；
- 配置化超参数、JSONL 日志、smoke 后再全量训练。

没有直接照搬：

- 单轮 XML 格式奖励；
- 只根据答案字符串打分的 reward；
- TRL 默认 trainer 的单轮 rollout；
- 与 Stage 1 不同的数据切分或 verifier。

原因是这些替换会把当前项目从多轮 Agentic RL 变成普通格式化数学 RL，也会破坏 Stage 1/Stage 2 的可比性。因此 Stage 2 只借鉴其模型搭建和实验组织方式，环境、轨迹、奖励和评测继续复用本项目的严格实现。

## 16. 当前执行状态

截至 2026-10-05，执行状态为：

1. 正式 SFT 已完成并通过结构审计；SFT→GRPO 的 4-group pilot 已通过并启动全量训练。
2. Pure GRPO 的 off-policy、逐轨迹门禁和第 190 组重尾 KL 三类失败均保留式归档；当前 formal 从 fresh LoRA 采用有上限的危险组拒绝机制重启。
3. 四组 validation/test 必须等两条 formal GRPO adapter 完成后执行，结果尚未产生。
4. 本轮先完成 seed 42 的四臂闭环；多训练 seed 属于扩展统计，不把尚未执行的重复实验伪装成现有结果。

正式 SFT 的 128×8 on-policy validation probe 已得到 trajectory task accuracy 34.6680%、pass@1 34.375%、pass@8 72.65625%、有效组率 69.53125%，工具执行成功率 99.6094%、evidence coverage 36.3281%，平均响应 71.80 token 且 unfinished 为 0。它证明 warm start 的行为可优化性，但最终结论仍以两条 GRPO 完成后的同一四臂 evaluator 为准。

代码 readiness、真实 tokenizer、多轮协议和 GPU smoke 通过，不等价于正式训练结果已经产生。后续每完成一项，应把命令、机器、Git commit、配置哈希、开始/结束时间、退出状态和结果路径追加到 `docs/WORKLOG.md`，最终结果再写入实验报告。
