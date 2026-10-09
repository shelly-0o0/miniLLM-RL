# Track 2：互斥 A/B Agent-SFT 与 GRPO 完整搭建记录

> 日期：2026-10-04
> 模型：Qwen3-4B-Base + NF4 QLoRA
> 环境：GSM8K + `calculate_math` 多轮工具调用
> 当前状态：代码、数据、配置、56 项回归、readiness 与远程真实 Qwen3-4B GPU smoke 均完成；正式训练与最终结果尚未执行。

## 1. Track 2 要回答什么问题

Track 1 比较 Base、Pure GRPO、SFT only 和 SFT → GRPO，重点是“Agent-SFT 作为 RL 系统的冷启动训练，是否让稀疏奖励 RL 变得可优化”。Track 2 不重复这个问题，而是在所有 RL 分支共享同一个 Agent-SFT(A) 起点后，进一步区分训练题目的作用：

1. GRPO 在 SFT 已见过的 A 题上继续训练，得到多少收益？
2. GRPO 在 SFT 没见过的 B 题上训练，能否利用已有工具行为迁移到新题？
3. 面对同一批 B 题，继续做监督学习与做在线 GRPO 的结果有什么差异？
4. SFT 后的 policy 在真正开始 GRPO 前，能否以采样方式产生正确且可验证的轨迹？

实验分支如下：

```text
Qwen3-4B-Base
  └── Agent-SFT(A)                         共享起点
       ├── 不再训练                         Agent-SFT(A) baseline
       ├── GRPO(A)                         SFT 已见题上的在线 RL
       ├── GRPO(B)                         SFT 未见题上的在线 RL
       └── Additional Agent-SFT(B)         同一 B 题上的继续监督学习
```

最终官方 test 同时评测 Base 和以上四个 adapter，共五个模型。所有分支仍执行真正的工具调用，而不是只奖励一种文本格式。

## 2. 数据划分与因果边界

### 2.1 一次性划分

`scripts/prepare/prepare_gsm8k_track2.py` 用 seed 42 对官方 7,473 条 train 的索引洗牌，再做近似二等分；写文件时恢复原始顺序。因此成员选择随机且可复现，文件顺序稳定：

| 集合 | 条数 | 用途 |
|---|---:|---|
| A source / RL | 3,736 | Agent-SFT(A) 的来源、GRPO(A) 候选池 |
| B source / RL | 3,737 | GRPO(B) 与 Additional-SFT(B) 的来源 |
| official test | 1,319 | 只做最终评测 |

准备脚本调用题目级 `assert_disjoint`，审计再次验证 A∩B、A∩test、B∩test 均为空。官方 test 不参与 SFT、probe、RL、超参数选择或 checkpoint 选择。

### 2.2 为什么 SFT 条数比 source 少

GSM8K 原始解答中的 `<<expression=result>>` 标注被转换为 calculator oracle。只有当所有表达式都能用受限算术解释器执行，且执行结果与标注及最终答案一致时，样本才能进入 Agent-SFT：

```text
A SFT: 3692 / 3736
B SFT: 3686 / 3737
```

被过滤的题不是“验证集”，而是没有足够可靠的离线 action oracle。在线 RL 原理上可以使用这些题，因为 RL 只需要题目最终答案和模型真实执行轨迹；但本 Track 2 的核心是严格比较 A/B 以及 GRPO(B)/Additional-SFT(B)，所以建立等量可验证比较集：

```text
A compare RL: 3686（从 3692 条可验证 A 中按固定 seed 选取）
B compare RL: 3686（与 B SFT 的题目 ID 完全一致）
```

这样两条 GRPO 都消费 3,686 个 prompt group，且 B 上的 RL 与追加 SFT 面对同一组问题。完整的 3,736/3,737 条 RL 文件仍被生成和写入 manifest，便于以后开展“使用所有可用 RL prompt”的扩展实验，但不混入本轮主比较。

### 2.3 Probe 不是额外训练集

A/B 各固定选 128 条 probe，它们仍是各自训练池成员。Probe 只在 GRPO 之前读取模型，不做反向传播，所以它是可优化性诊断，不是模型选择用的 holdout：

```text
A compare ──固定抽样──> A probe 128
B compare ──固定抽样──> B probe 128
```

所有文件的条数和 SHA-256 固定在 `dataset/manifests/gsm8k_track2.json`。重复运行准备脚本会产生相同内容；任何文件被静默修改，readiness audit 都会失败。

## 3. Agent-SFT 究竟学习什么

Agent-SFT 是离线行为克隆，不会在训练时重新调用 calculator。训练样本已经包含由可信 oracle 构造并验证过的完整轨迹：

```text
system: 工具规则和 schema                  条件 token
user: GSM8K 问题                           条件 token
assistant: <tool_call>...</tool_call>       监督 token
tool: <tool_response>...</tool_response>    条件 token
assistant: 最终答案                         监督 token
```

因此它不只是学习“答案格式”，还在 token 层面学习：选择 calculator、构造合法 JSON 参数、结束工具调用轮、在 observation 后继续生成、最后停止并回答。但工具返回由环境产生，不能作为模型动作监督。

损失只覆盖 assistant action 集合 `A`：

```text
L_SFT = -(1 / |A|) Σ[t∈A] log πθ(x_t | x_<t)
```

本次分别抽查 A/B 各 128 条真实 Qwen tokenizer 样本：

| 项目 | A | B |
|---|---:|---:|
| zero-supervision | 0 | 0 |
| 最少监督 token | 32 | 34 |
| 最多监督 token | 206 | 182 |
| 最长完整序列 | 555 | 527 |

这证明 user 问题和 tool observation 只作为上下文，assistant 的工具动作与最终回答才进入交叉熵。

## 4. 在线 Agent rollout 与严格奖励

GRPO 阶段每条轨迹真实执行以下状态转换：

```text
prompt + tool schema
  → policy 生成 tool_call
  → parser 检查标签、JSON、工具名和参数
  → safe_calculate 执行受限算术文法
  → 环境回填 tool observation
  → policy 继续生成下一轮 assistant action
  → verifier 检查最终答案和执行证据
```

`response_mask` 对 assistant 动作为 1，对环境 observation 为 0。因此 observation 会影响后续预测，但既不作为 action 计算 policy ratio，也不反向奖励模型去伪造工具输出。

严格 RLVR reward 为：

```text
r = +1  当且仅当：答案正确、调用合法、工具执行成功、
                  必需工具被覆盖、执行证据支持答案、轨迹完整
r = -1  其他情况
```

只猜对答案、写错算式后猜答案、伪造工具结果、调用无关工具或撞满长度上限均不能成功。

## 5. GRPO 的数学与实现

对一个 prompt 采样 `G=8` 条随机轨迹。第 `i` 条的组内标准化优势为：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-4)
```

若八条全错或全对，优势均为 0；该组不会提供相对排序梯度。因此真正重要的不是 loss 看起来是否波动，而是模型是否能生成“部分成功、部分失败”的 effective group。

对动作 token 的行为策略比率：

```text
ratio_i,t = exp(log πθ(a_i,t|s_i,t) - log πold(a_i,t|s_i,t))
```

GRPO 使用 PPO 风格剪切目标：

```text
surrogate = min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
ε = 0.2
```

reference policy 是训练开始时 Agent-SFT(A) adapter 的冻结副本。非负 k3 KL 估计为：

```text
δ = log πref - log πθ
KL_k3 = exp(δ) - δ - 1
L = -mean(surrogate) + β·mean(KL_k3), β = 0.02
```

current policy、rollout 时的 old behavior 和 frozen reference 三个角色彼此分离。训练支持 checkpoint/resume；reference 从第一次运行保存的副本恢复，不随 current policy 漂移。

## 6. Pre-GRPO probe 的原理

`scripts/evaluate/probe_qwen_track2.py` 对 A/B 各 128 个 prompt 分别采样 G=8，不更新参数。对每个问题记录：

```text
pass@1       = 第一条采样是否成功
pass@8       = 八条中是否至少一条成功
effective    = 0 < 成功条数 < 8
zero variance= 成功条数为 0 或 8
```

解释方式：

- `pass@8 > pass@1`：正确行为已经可达，但单次输出不稳定，RL 有排序空间；
- effective-group rate 高：组内奖励存在方差，GRPO 能获得相对优势；
- A 好、B 差：SFT 更可能在记忆 A 的轨迹，而不是把工具行为迁移到新题；
- A/B 都接近全错：应先改进 Agent-SFT 冷启动训练、采样或奖励，不应直接烧完整 GRPO 预算；
- A/B 都接近全对：继续 GRPO 的边际信息也很少。

Probe 输出 group、trajectory 和 aggregate 三层记录，便于从汇总指标追溯到原始生成。

## 7. 五组最终结果分别能说明什么

设官方 test 准确率为 `Acc(·)`：

| 对比 | 解释 |
|---|---|
| Agent-SFT(A) − Base | 一次 Agent 行为克隆的总体收益 |
| GRPO(A) − Agent-SFT(A) | 在 SFT 已见题上进行在线 RL 的增量 |
| GRPO(B) − Agent-SFT(A) | SFT 行为先验迁移到新 RL 题后的增量 |
| GRPO(B) − GRPO(A) | RL 数据新颖性是否影响泛化；不是单纯“哪个算法更强” |
| Additional-SFT(B) − Agent-SFT(A) | 增加 B oracle 监督的收益 |
| GRPO(B) − Additional-SFT(B) | 同一 B 题上在线验证奖励与继续行为克隆的差异 |

单 seed 只能作为主流程运行结果，不足以声称稳定因果结论。正式报告应补至少 3 个训练 seed、相同 decode seeds、GPU 时间、rollout token、optimizer update 和置信区间。

## 8. 实现文件及职责

| 文件 | 职责 |
|---|---|
| `scripts/prepare/prepare_gsm8k_track2.py` | A/B/test 划分、oracle 过滤、等量比较集、probe 与 manifest |
| `scripts/audit_qwen_track2.py` | 哈希、交集、subset、mask、协议和配置门禁 |
| `scripts/evaluate/probe_qwen_track2.py` | G=8 的 A/B 可达性与有效组诊断 |
| `trainer/train_qwen_lora_sft.py` | fresh Agent-SFT(A) 和从 A adapter 继续 Additional-SFT(B) |
| `trainer/train_qwen_grpo.py` | GRPO(A)/GRPO(B) 的共用在线 Agentic GRPO 实现 |
| `scripts/evaluate/evaluate_qwen_stage2.py` | 五个模型在官方 test 上的统一评测 |
| `scripts/run_qwen_track2.sh` | 所有阶段的单一入口、日志路径和前置依赖检查 |
| `configs/qwen3_4b/track2/*.yaml` | 固定模型、数据、rollout、训练预算和输出路径 |
| `tests/test_qwen_track2.py` | 划分、配置与 probe 聚合回归测试 |
| `dataset/manifests/gsm8k_track2.json` | 数据行数与内容哈希的事实来源 |

SFT 与 GRPO trainer 增加 `--adapter-path` 运行时覆盖，只用于把短程 smoke adapter 串成完整 smoke 链；正式运行仍从 YAML 固定路径加载。所有覆盖都会写入 `runtime_overrides.json`。SFT smoke 使用 2 steps，因为 1 step 配合 warmup/cosine 时唯一一步的学习率为 0，只能验证梯度而不能证明参数真的更新。

## 9. 已实际执行的本地操作

### 9.1 生成数据

```bash
/home/user/miniconda3/envs/mini-rl/bin/python \
  scripts/prepare/prepare_gsm8k_track2.py
```

原理：从官方 train 一次性产生 A/B 成员，构造 RL/SFT/compare/probe 表示，写入 SHA-256 manifest。

### 9.2 静态门禁

```bash
/home/user/miniconda3/envs/mini-rl/bin/python -m py_compile \
  scripts/prepare/prepare_gsm8k_track2.py \
  scripts/audit_qwen_track2.py \
  scripts/evaluate/probe_qwen_track2.py \
  scripts/evaluate/evaluate_qwen_stage2.py \
  trainer/train_qwen_lora_sft.py \
  trainer/train_qwen_grpo.py

bash -n scripts/run_qwen_track2.sh
```

原理：在下载模型或占用 GPU 前排除 Python 语法和 shell 分支语法错误。

### 9.3 Track 2 定向测试与 readiness

```bash
/home/user/miniconda3/envs/mini-rl/bin/python \
  -m unittest tests.test_qwen_track2 -v

/home/user/miniconda3/envs/mini-rl/bin/python \
  scripts/audit_qwen_track2.py
```

结果：4/4 定向测试通过；readiness `PASS`。审计报告位于：

```text
out/run_meta/qwen3_track2_readiness.json
```

### 9.4 全仓回归

```bash
HF_HOME=/tmp/minirl-track2-hf-cache \
  /home/user/miniconda3/envs/mini-rl/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

结果：54/54 通过。`HF_HOME` 指向 `/tmp` 是因为当前受限执行环境不允许测试在用户级 Hugging Face cache 创建 lock；它不改变测试数据或算法。

## 10. 正式执行顺序和每条命令的原理

所有命令在仓库根目录执行，并选择明确的物理 GPU。以下示例仅展示单机环境变量；GRPO 配置内部使用两个逻辑设备 `cuda:0/cuda:1`，所以 `CUDA_VISIBLE_DEVICES` 应暴露两张 GPU。

### 10.1 再生数据与门禁

```bash
bash scripts/run_qwen_track2.sh prepare
bash scripts/run_qwen_track2.sh audit
```

第一条重建确定性数据和 manifest；第二条在任何昂贵训练之前验证输入、模板、mask、工具协议和配置不变量。

### 10.2 端到端 smoke

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh smoke_sft_a
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh probe_smoke
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh smoke_additional_sft_b

CUDA_VISIBLE_DEVICES=6,7 bash scripts/run_qwen_track2.sh smoke_grpo_a
CUDA_VISIBLE_DEVICES=6,7 bash scripts/run_qwen_track2.sh smoke_grpo_b
```

Smoke 的目标不是测准确率，而是证明真实 4-bit 权重可加载、LoRA 可训练、adapter 可保存/重载、两轮工具 rollout 能运行、GRPO 能反向传播并保存状态。smoke 输出带独立后缀，不会覆盖正式 adapter。

### 10.3 完整 Agent-SFT(A)

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh sft_a
```

若被中断：

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh sft_a --resume
```

这一步用 A 的 3,692 条可靠 oracle 轨迹学习工具行为，是后续四个分支中的三个训练分支和一个 baseline 的共同起点。

### 10.4 正式 Pre-GRPO probe

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh pre_grpo_probe
```

先观察 A/B 的 pass@1、pass@8 与 effective-group rate，再决定完整 GRPO 是否具有合理学习信号。脚本不允许静默覆盖已有正式 probe；需要重跑时应先保存旧结果并选择新 tag。

### 10.5 追加监督分支

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh additional_sft_b
```

它从 Agent-SFT(A) adapter 加载可训练 LoRA，再对 B 的 3,686 条 oracle 轨迹做一个 epoch，而不是重新从 Base 开始。

### 10.6 GRPO 小预算 pilot

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/run_qwen_track2.sh pilot_grpo_a
CUDA_VISIBLE_DEVICES=6,7 bash scripts/run_qwen_track2.sh pilot_grpo_b
```

每条只消费 4 个 group，用于估计速度、显存、KL、clip fraction、rollout log-probability 一致性和参数更新。pilot 不能当作实验结果。

### 10.7 完整 GRPO(A/B)

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/run_qwen_track2.sh grpo_a
CUDA_VISIBLE_DEVICES=6,7 bash scripts/run_qwen_track2.sh grpo_b
```

中断后分别在同一命令尾部加 `--resume`。两条分支都从原始 Agent-SFT(A) 起点独立开始，不能让 B 继承 A 的 GRPO checkpoint，也不能串行训练。

### 10.8 官方 test 统一评测

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh eval_test
```

配置启用 `require_all_runs: true`，五个模型只要缺一个 adapter 就立即失败，避免把不完整表格误当正式结果。test 只能在训练与决策冻结后执行。

## 11. 训练监控与异常判据

SFT 重点检查：

- loss 是否有限并总体下降；
- grad norm 是否有限；
- learning rate 是否按 warmup/cosine 变化；
- adapter 是否保存且全部 tensor 有限；
- mask audit 的 zero-supervision 必须为 0。

GRPO 重点检查：

- reward / task accuracy；
- group reward std 与 zero-variance rate；
- policy loss、KL_k3、clip fraction；
- rollout logprob MAE 与 ratio mean；
- unfinished rate、平均长度和生成 token 数；
- candidate groups、optimizer updates 与 stop reason。

典型异常：

- reward 永远 -1 且 zero-variance≈1：没有可排序轨迹，loss 接近 0 是数学结果；
- rollout logprob MAE 超阈值：采样概率与重算概率不一致，训练应终止；
- KL 突增、clip fraction 持续很高：更新过猛，应降低学习率或增加 KL 约束；
- unfinished/长度上限高：模型没有学会收束，先排查模板、stop token 和 Agent-SFT 冷启动训练；
- tool-call valid 高但 evidence coverage 低：会写格式，但计算或证据链错误。

GRPO 的 current-policy forward 保持训练模式以启用 gradient checkpointing，但所有 Dropout 子模块固定为 eval。否则 behavior log-probability 在无 dropout 条件下重算，而 current log-probability 带随机 dropout，会把随机失配错误解释为策略/KL 变化。optimizer step 前还按整组所有 action token 计算全局平均 `kl_k3 <= 10` 的硬门禁；这与 streaming loss 中 KL 项的归一化口径一致。超过阈值会在 backward 前保存最坏轨迹/token 诊断并终止，不把异常更新写入 checkpoint。

## 12. 输出、恢复与防覆盖

正式输出彼此隔离在：

```text
checkpoints/stage2_track2/qwen3_4b/
out/stage2_track2/qwen3_4b/
out/metrics/track2_*.jsonl
out/logs/track2_*.log
```

安全规则：

1. SFT 发现目标 adapter 已存在且未传 `--resume` 时拒绝启动；
2. GRPO 发现 run checkpoint 已存在且未传 `--resume` 时拒绝启动；
3. smoke/pilot 使用独立 tag，不能覆盖正式运行；
4. probe 发现目标 summary 已存在时拒绝覆盖；
5. 运行时覆盖和实际 config 都复制到输出目录；
6. 最终评测必须集齐五个模型。

## 13. 当前已经证明与尚未证明的内容

已证明：

- Track 2 数据可确定性重建；
- A/B/test 题目级互斥；
- A/B 比较预算相同，B 的 RL/SFT 题目完全匹配；
- Qwen assistant-only mask 正确且无空监督；
- 确定性假 policy 能完成真实两轮工具协议并得到严格 +1；
- Track 2 定向测试和全仓 56 项回归均通过；
- 代码具备 smoke、正式训练、resume、日志和防覆盖路径。
- 远程 `/root/Mini-RL-stage2-track2` 已完成 SFT(A)、A/B probe、Additional-SFT(B)、GRPO(A/B) 工程 smoke；修复版 B 第 2 组 KL 为 0.00803。

尚未证明：

- Agent-SFT(A) 的正式训练质量；
- A/B probe 的真实 pass@k 和 effective-group rate；
- GRPO(A)、GRPO(B) 或 Additional-SFT(B) 哪个最终更好；
- 单 seed 结果能否跨 seed 稳定复现。

因此在生成实际 artifact 前，不应写任何准确率提升或算法优劣结论。

## 14. 首次 GPU smoke 暴露并修正的 KL 问题

远程首次 GRPO(B) smoke 的第 2 组出现 `kl_k3=5090.64`，而 rollout ratio 与 log-probability MAE 仍正常。追踪执行模式后发现：rollout 和 old/reference 重算使用 eval mode，但 current-policy 的可微 forward 被恢复到 train mode，LoRA dropout=0.05 因而只作用于 current 概率。特定低概率 token 会让 k3 的指数项放大这种随机差异。

修正包括：

1. policy 主体保持 train mode，让 gradient checkpointing 生效；
2. 单独把所有 Dropout module 固定为 eval，使 old/current/reference 概率可比较；
3. 在整组 backward 前检查有限性与 `max_group_kl_k3=10`；组均值按所有 action token 全局加权，而不是对不等长轨迹均值再等权；
4. 新增单元测试，验证 dropout 已关闭而梯度仍能反向传播；
5. 首次异常 smoke 只保留为诊断证据，不作为成功训练或实验结果。

实际复测保持同一 B 数据、seed、G=8、最大生成长度和初始 adapter：第 2 组 `kl_k3` 从 5090.64 降至 0.00803，`clip_fraction=0`、`ratio_mean=1.0`、rollout log-probability MAE=0.00576，并正常完成更新和 adapter 保存。由此确认 dropout 执行模式是异常主因，修复有效；KL=10 的组级硬门禁继续作为后续防线。

## 15. SFT smoke 的零学习率检查

首次 SFT smoke 只覆盖 1 step。参数审计发现 Additional-SFT(B) 的 504 个 adapter tensor 与输入 A adapter 完全相同；Trainer 日志同时显示该步 learning rate 为 0。原因是 `max_steps=1` 与 warmup/cosine 调度组合使唯一一步没有实际更新。

runner 因而改为 SFT smoke 运行 2 steps。smoke 的验收不再只看 loss/grad norm/文件存在：trainer 会在短程运行前保存 trainable tensor 的 CPU 快照，结束后要求 tensor 全部有限，并且至少一个 tensor 相对初始化发生变化，否则在保存 adapter 前报错。实际复测中，fresh Agent-SFT(A) 有 252/504 个 tensor 改变，Additional-SFT(B) 相对 A adapter 有 504/504 个 tensor 改变。正式训练原本包含数百步，不受该 smoke 特例影响。

## 16. 正式 probe 暴露的 generation-prefix 漂移

正式 Agent-SFT(A) 完成后，旧 prompt 在 A 的 128 题、1,024 条轨迹上得到：`pass@1=0`、`pass@8=0`、有效组率 0、合法工具调用率 0。模型经常生成裸 calculator JSON，却没有 parser 要求的 `<tool_call>...</tool_call>`，平均响应达到 371 tokens。

对照真实 Qwen chat template 后确认，问题不是简单的“少训一轮”：

```text
SFT 第一轮目标： <|im_start|>assistant\n<tool_call>...
旧 inference：   <|im_start|>assistant\n<think>\n\n</think>\n\n...
```

Qwen3 模板在 `add_generation_prompt=True` 且禁用 thinking 时预填空 thinking block，但完整历史中的首个 tool-call assistant 不包含该 block；工具 observation 后的最终 assistant 则包含它。修复因此必须按状态选择：首轮工具调用不预填 thinking，工具 observation 后恢复 disabled-thinking 的空 block。

新增 `resolve_agent_open_thinking` 在每轮根据最后一个 user 之后是否已有 tool observation 决定模板模式，并在 observation 回填后重新计算下一轮模式。`audit_agent_generation_prefixes` 会把每一轮真实 inference prompt 与完整 SFT serialization 做严格前缀比较。A/B 各 128 条审计结果均为：两个 assistant turn、模式 `[true, false]`、mismatch 0。

旧正式 probe 被停止并保存在远程：

```text
out/stage2_track2/qwen3_4b/pre_grpo_probe_s42_prompt_mismatch/
out/logs/track2_pre_grpo_probe_s42_prompt_mismatch.log
```

修复后的第一步不是重训，而是复用相同正式 adapter 做 A 池 16 题、G=4 的短 probe；只有合法工具调用恢复后，才决定是否需要增加 SFT epoch。

## 17. 对齐后短 probe 与参考仓库的批判性核对

修复 generation prefix 后，复用同一个正式 Agent-SFT(A) adapter，在 A 池执行 16 题、每题 4 条轨迹。该 probe 确认提示词修复已真正进入 rollout：首轮 trace 的 `open_thinking=true`，不再预填空 thinking block；但结果仍为 0 次合法工具调用、0 个成功轨迹、0 个有效组，64 条轨迹的平均 action token 数恰好为上限 384。输出通常包含重复的裸 calculator JSON，却没有 `<tool_call>...</tool_call>`。

这组证据把问题进一步定位为 SFT 行为可达性，而不是继续归咎于模板：普通 assistant-token 交叉熵会把少数协议边界 token 与大量 JSON/解释 token 等权平均。第一轮 SFT 的总 loss 可以下降到约 1.0，同时 `<tool_call>`、`</tool_call>` 和 `<|im_end|>` 仍具有很低概率；模型因而“学会了内容形状”，但没有学会进入工具协议和及时结束。

同时逐项阅读参考仓库 [jjyaoao/qwen-grpo-gsm8k](https://github.com/jjyaoao/qwen-grpo-gsm8k) 的实际实现，而不是直接套用其项目总结。可迁移的思想是：

1. 先用 SFT 建立目标行为的可达性，再运行 GRPO；
2. 将格式奖励拆成标签计数、宽松格式、严格格式和重复标签惩罚；
3. 同时记录 completion、reward、loss 与最新指标；
4. SFT 使用右 padding，生成使用左 padding，prompt label 与 pad label 均为 `-100`；
5. 4-bit NF4、BF16 和 LoRA 只是资源方案，必须用实际更新与有限值审计验证。

不能直接移植之处是：参考项目只有单轮 XML 文本输出，没有工具执行、observation 回填、required-tool coverage 或执行证据验证；它的“格式正确”不能等价于本项目的 agent task success。当前项目保留严格成功条件：答案正确、标签平衡、调用 schema 合法、工具可执行、所需工具齐全，且执行结果支持最终答案。

核对时还发现一个本项目自己的诊断缺陷：原逻辑会把 `open_tags=close_tags=parsed_calls=0` 判为格式有效。这不会绕过严格 task success，但会虚高 `format_valid_rate`，也会在 shaped reward 中给完全未调用工具的轨迹格式分。现已改为：当任务要求工具时，至少存在一个成功解析的工具调用才可记为格式有效，并增加回归测试。

后续 probe 会在同一批 rollout 上同时计算：

- strict reward 方差：回答“是否已出现真正成功/失败差异”；
- shaped reward 方差：回答“即使尚无完整成功，分层信号是否足以产生非零优势”。

因此是否把 Qwen GRPO 从 strict 切到 shaped，不凭经验决定：先看同一批轨迹的两套方差。如果两者都为零，直接 RL 仍没有梯度；如果 strict 为零但 shaped 有稳定方差，才进行小预算 shaped-GRPO pilot，并继续用 strict task success 作为最终验收指标。

## 18. 结构 token 加权的 Agent-SFT 修复

为保留原始实验产物并隔离变量，修复不是覆盖原 adapter，也不是放宽 parser，而是从正式 Agent-SFT(A) adapter 继续一个 epoch，并只改变监督损失的 token 权重：

```text
普通 assistant token                       weight = 1
<tool_call>, </tool_call>, <|im_end|>      weight = 8
prompt、system、user、tool observation     ignore = -100
```

实现使用 causal shift 后的逐 token FP32 cross entropy，再按有效 token 权重归一化。这样不会把 batch 长度或 padding 计入分母，也不会简单地把总 loss 乘 8。tokenizer 启动时必须证明三个 marker 都是单独且互异的 token；远程 Qwen3-4B-Base 实测 ID 分别为 151657、151658、151645。

新增阶段：

```bash
# 两步工程验证
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh smoke_repair_sft_a

# 从原正式 adapter 增量训练，输出独立 adapter_structure_w8
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh repair_sft_a

# A 池 16 题、G=4 的修复后决策 probe
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh probe_repair
```

两步 smoke 的验收结果：504/504 个可训练 LoRA tensor 发生变化，最大绝对变化约 `1.001e-4`，非有限 tensor 为 0。初始结构加权 loss 约 59，远高于普通 SFT 的整体均值，这不是数值爆炸的直接证据；它说明正式 adapter 对少数控制 token 仍赋予极低概率，也是本次修复的目标。训练继续使用 gradient clipping，并以 loss、grad norm、参数有限性和修复后 rollout 行为共同判定，不用单个 loss 数值宣称成功。

## 19. 结构加权但不训练输出头：负结果

结构加权增量 SFT 完成 231 optimizer step、3,692 条 A 轨迹和一个 epoch，耗时 1,636 秒，adapter 独立保存。训练执行本身没有 NaN，但 batch loss 在整个 epoch 没有明显下降。随后没有直接运行 rollout，而是先在固定 128 条训练轨迹上对原 adapter 与修复 adapter 做 teacher-forced 对照：

| 指标 | 原 Agent-SFT(A) | structure-w8 | 变化 |
|---|---:|---:|---:|
| assistant mean NLL | 0.95265 | 0.94307 | -0.00957 |
| ordinary-token mean NLL | 0.24741 | 0.24096 | -0.00644 |
| structure-token mean NLL | 7.79652 | 7.75660 | -0.03992 |
| structure-token Top-1 | 0% | 0% | 0 pp |

结构 NLL 只改善约 0.5%，三类 marker 的 Top-1 仍全部为 0，故判定这条路线没有解决协议可达性，停止在该 adapter 上做 rollout 或继续加 epoch。

审计还解释了 loss 日志约 59 的来源。自定义 loss 已在 micro-batch 内求平均，但 Transformers 4.57 看到 `compute_loss` 接受 `num_items_in_batch` 后假定它自行处理梯度累积，导致日志和 backward loss 被累积步数 16 放大。Trainer 现显式声明 `model_accepts_loss_kwargs=false`，由框架按 gradient accumulation 正确归一化；相同目标的真实加权 loss 约为 3.7。旧 run 的梯度多数被 `max_grad_norm=1` 截断，因此没有非有限更新，但其 loss 数值不能与修复后日志直接横向比较。

## 20. 扩展 lm_head LoRA 的第二次修复

进一步检查 PEFT 的实际 adapter 配置发现，`target_modules=all-linear` 会刻意排除语言模型输出层；正式 adapter 只包含 `q/k/v/o/gate/up/down_proj` 七类 target，`lm_head` 不在其中。Qwen3-4B-Base 又不像 Instruct 模型那样已经建立 chat control token 的强先验，这与结构 token NLL≈7.8、Top-1=0 的证据一致。

第二次修复采用以下约束：

1. 原正式 Agent-SFT(A) 的 504 个 LoRA tensor 必须逐 tensor 原样复制；
2. 目标模块扩展为原七类线性层加 `lm_head`；
3. 只增加 rank-16 `lm_head` LoRA，新增可训练参数约 247 万，不解冻完整输出矩阵；
4. 新增 LoRA-B 为 0，因此扩展瞬间与原 adapter 函数等价；
5. 保存时强制 `save_embedding_layers=false`，避免 PEFT 因 Qwen tied embeddings 自动写入完整 embedding/lm_head；
6. 结构 token 继续使用 weight 8，普通 assistant token 继续 weight 1。

10-step pilot 先于完整训练执行。其真实加权 loss 为 3.72，506/506 个可训练 tensor 更新且全部有限。固定 32 条对照显示：

| 指标 | 原 Agent-SFT(A) | lm_head 10-step | 变化 |
|---|---:|---:|---:|
| ordinary-token mean NLL | 0.25570 | 0.25312 | -0.00258 |
| structure-token mean NLL | 7.80959 | 7.67970 | -0.12988 |
| structure-token Top-1 | 0% | 2.43% | +2.43 pp |
| `<|im_end|>` Top-1 | 0% | 10.94% | +10.94 pp |
| `<tool_call>` Top-1 | 0% | 0% | 0 pp |

仅 10 step 的结构 NLL 改善已是前一条 231-step 隐层-only run 的 3.25 倍，同时普通 token 没有退化，故允许进入完整 lm_head 低秩 epoch。它仍不是最终成功：完整训练后必须重复 128 条概率审计，并且只有 `<tool_call>` 可达后才运行 16×4 rollout probe。

## 21. 完整修复的分级验收与奖励模式隔离

完整 lm-head LoRA 训练不能只靠 train loss 验收，后续固定使用三层门槛：

1. **产物层**：adapter 原子保存、可离线重载、只含 LoRA delta、参数全部有限；
2. **教师强制层**：固定 128 条轨迹比较普通/结构 token NLL，并检查 `<tool_call>`、`</tool_call>`、`<|im_end|>` 的 Top-1 与 Top-20；Top-20 对应正式 rollout 的 `top_k=20`，若 marker 不在候选集合内，自由生成就不可能采到它；
3. **自由生成层**：先运行 A 池 2 题×2 轨迹的 `probe_lm_head_smoke`，确认真实 parser 能解析工具调用且生成可停止，再扩大为 16×4 probe。所有 probe 同时报告 strict 与 shaped reward 方差。

Qwen GRPO 新增显式 `reward_mode` 配置和 `--reward-mode` 覆盖，但正式 `grpo_a/grpo_b` 配置仍固定为 `strict`。只有当同一批轨迹满足“strict 无方差、shaped 有方差”时，才允许单独命名、单独输出的小预算 shaped curriculum pilot；其结果不能替代 strict task success，也不能与正式严格 RLVR 曲线混写。若两种奖励都无方差，则继续 RL 没有可学习的组内优势，应返回 SFT/生成策略诊断，而不是盲目增加训练时长。

## 22. 完整 lm-head 修复、自由生成验收与课程 pilot

### 22.1 完整 SFT 修复产物

完整修复在物理 GPU 6 上训练 3,692 条 A 轨迹、231 optimizer step、一个 epoch，耗时 1,660 秒，最终平均加权 loss 为 0.88589。训练过程中 loss 从约 3.72 下降到末段约 0.35；所有梯度范数日志都是裁剪前数值，实际使用 `max_grad_norm=1`。

最终 adapter 位于：

```text
out/stage2_track2/qwen3_4b/agent_sft_a_lm_head_w8_s42_adapter
```

目录 151 MB，`adapter_model.safetensors` 136 MB。保存器显式设置 `save_embedding_layers=false`，因此只包含 506 个 LoRA delta tensor，没有重复写入完整 tied embedding。其 SHA-256 为 `69e36f0aed18c1866ac29a66c74f65130b9e05604d20e83a4a4732d9d6f0ba04`，且已通过离线重载。

### 22.2 固定 128 条 teacher-forced 审计

| 指标 | 原 Agent-SFT(A) | lm-head structure-w8 | 变化 |
|---|---:|---:|---:|
| assistant mean NLL | 0.95265 | 0.18180 | -0.77084 |
| ordinary mean NLL | 0.24741 | 0.14207 | -0.10534 |
| structure mean NLL | 7.79652 | 0.56740 | -7.22911 |
| structure Top-1 | 0% | 98.09% | +98.09 pp |
| structure Top-20 | 0% | 100% | +100 pp |
| `<tool_call>` Top-1 | 0% | 99.78% | +99.78 pp |
| `</tool_call>` Top-1 | 0% | 97.09% | +97.09 pp |
| `<|im_end|>` Top-1 | 0% | 96.88% | +96.88 pp |

审计同时新增普通目标位置上的控制 token 误报测量。修复 adapter 的结构标记 Top-1 误报率为 2.455%，进入 Top-20 的比例为 10.538%；原 adapter 两者都是 0。它解释了为何控制 token 已可达、生成能及时停止，但采样时仍会出现重复或嵌套标签。只看目标 marker 的 Top-1 会遗漏这种副作用。

### 22.3 自由生成 probe

先运行 A 池 2 题×2 轨迹 smoke。4 条轨迹均能停止，平均 78.75 tokens，不再撞 384 上限；合法且可执行工具调用率为 50%，答案准确率为 50%。但 strict success 为 0，其中一条轨迹工具证据与最终答案都正确，却因多余/不平衡标签被严格 verifier 拒绝。这个结果证明不能因为 teacher-forced marker Top-1 接近 100% 就宣布 agent 行为已正确。

随后固定 A 池 16 题×4 轨迹得到：

| 指标 | 结果 |
|---|---:|
| strict trajectory task accuracy | 3.125%（2/64） |
| pass@4 | 12.5% |
| strict effective-group rate | 12.5% |
| strict zero-variance group rate | 87.5% |
| answer accuracy | 28.125% |
| format valid rate | 7.8125% |
| tool-call / execution success rate | 59.375% |
| required-tool coverage | 59.375% |
| tool-evidence coverage | 14.0625% |
| unfinished rate | 0% |
| average response tokens | 63.125 |

因此问题已经从“完全不会进入工具协议”变成“严格成功可达，但格式校准与证据覆盖仍弱”。strict GRPO 现在有真实信号，但大多数小组仍浪费为零优势；不能把 3.125% 当作最终效果，只能作为训练可达性证据。

### 22.4 shaped reward 的实际缺陷与修复

原 shaped reward 只惩罚 `abs(open_tags-close_tags)`。当开闭标签数量相同、但嵌套/空标签导致只有部分 JSON 能解析时，格式奖励不会扣分；同时 `[-3,3]` 截断会让“严格完整成功”和“正确但格式损坏”共同饱和到 3。

修复后：

```text
format_error = |open-close| + |open-parsed| + |close-parsed|
invalid format penalty = -0.5 * max(1, format_error)
strictly successful trajectory bonus = +2
shaped clamp = [-6, 6]
```

严格 task-success 定义与 strict `+1/-1` 完全不变。对同一批已保存的 64 条轨迹进行无生成、无参数更新的离线重评分后，两个严格成功样本均为 6.0；最高的格式损坏近似成功为 4.5，其余常见近似成功为 3.5、2.5。16/16 组仍具有 shaped 方差，说明新课程奖励同时保留可学习信号与正确排序。新增回归测试专门覆盖“标签数平衡但存在不可解析额外标签”的情形。

### 22.5 两组 shaped-GRPO 工程 pilot

课程 pilot 从完整修复 adapter 出发，只消费 2 组×8=16 条轨迹，独立输出，不计入 formal strict 结果：

```text
out/stage2_track2/qwen3_4b/grpo_a_s42_adapter_lm_head_shaped_pilot2
```

两组 reward std 分别为 2.5927 和 2.9161，advantage std 均约 1；KL k3 为 `8.36e-5`、`2.58e-4`，rollout log-probability MAE 为 0.0100、0.00670，clip fraction 为 0，均通过安全门禁。相对输入 adapter，506/506 个 tensor、99.887% 的参数元素发生变化；相对 L2 变化 `1.776e-4`，最大绝对变化 `1.552e-6`，非有限 tensor 为 0。

同一 2×2 smoke 的更新后复测保持工具调用/执行 50%、答案 50%、证据覆盖 25%、未完成 0%，平均长度 73.5。它只证明两步课程更新没有立即破坏行为，不能证明泛化或准确率提升。

### 22.6 正式入口与历史产物隔离

启动前远程检查确认没有正式 GRPO(A/B) 或正式 Additional-SFT(B) 产物，只有 smoke，因此正式配置已安全切换到修复 adapter：

```text
additional_sft_b
grpo_a
grpo_b
pre_grpo_probe
```

四者共享 `agent_sft_a_lm_head_w8_s42_adapter`，并显式保留八类 target module；formal GRPO 的 `reward_mode` 仍固定为 `strict`。原 `agent_sft_a_s42_adapter` 保留为失败诊断基线，`agent_sft_a_lm_head.yaml` 仍从它执行无损 target expansion。最终评测中的 `agent_sft_a` 标签也指向修复 adapter。

远程定向测试 31/31、A/B 各 128 条 generation-prefix 审计、确定性两轮工具协议以及 readiness 均通过。readiness 报告确认 A/B 仍为 G=8、各 3,686 个候选组，数据预算和共享起点没有因修复而失去可比性。

## 23. 对参考项目经验的最终判断

本项目确实出现了参考项目总结中的三类现象：生成撞长度上限、严格 reward 稀疏、模型缺少目标格式先验。但不能把所有清单项都当作当前原因：

- padding 已按 SFT 右侧、rollout 左侧配置，并有 mask/前缀审计；不是本次主因；
- NF4 + BF16 训练与所有保存参数均有限，没有发现精度或量化故障；
- 当前日志已经覆盖 reward 方差、工具调用、证据、KL、ratio、长度和参数变化，不再是不可观测状态；
- 真正的根因是 Qwen3-4B-Base 的 chat control-token 先验弱、PEFT `all-linear` 排除 `lm_head`、旧 inference prefix 与 SFT 不一致，以及 shaped reward 对重复标签排序失真；
- 权重 8 的完整修复又引入控制 token 过生成，说明“把格式奖励/损失加大”本身不是单调更优，必须同时检查目标命中率与非目标误报率。

参考仓库可迁移的是 SFT 初始化、分层奖励、padding 区分和细粒度监控；不能直接复制其 XML formatter，因为本项目的成功还要求 schema 合法、真实执行、required-tool coverage、执行证据与最终答案共同成立。

## 24. 正式运行顺序

正式实验按以下顺序执行。实测 policy/reference 可以在一张 24 GiB GPU 上安全共存，因此每条 GRPO 分支只占一张物理卡；配置内两者都写 `cuda:0`，由 `CUDA_VISIBLE_DEVICES` 映射到指定物理卡。

```bash
cd /root/Mini-RL-stage2-track2
export PATH=/root/miniconda3/envs/mini-rl-stage2/bin:$PATH
export HF_HOME=/root/Mini-RL-stage2/.cache/huggingface
export HF_HUB_OFFLINE=1

# 1. 无更新地测 A/B 各 128 题、G=8 的正式起点可达性。
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh pre_grpo_probe

# 2. Additional-SFT(B) 对照；与两条 GRPO 分支共享修复后的 A 起点。
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh additional_sft_b

# 3. 两条正式 strict-GRPO 可在两张 GPU 上并行。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_qwen_track2.sh grpo_a
CUDA_VISIBLE_DEVICES=1 bash scripts/run_qwen_track2.sh grpo_b

# 4. 五个模型全部存在后，统一在官方 test 1,319 题上评测。
CUDA_VISIBLE_DEVICES=6 bash scripts/run_qwen_track2.sh eval_test
```

旧 `pre_grpo_probe` 给出的 G=8 effective-group rate 为 A `25.78125%`、B `17.96875%`，但它来自旧的 top-k/top-p 截断采样，只能证明行为可达，不能外推为 full-softmax on-policy 奖励密度。两条 4-group on-policy pilot 通过有限性、KL 和 log-probability 一致性门禁后，正式 GRPO(A/B) 已分别在 GPU 0/1 启动；观察到早期组持续零方差后，又在 GPU 2/5 启动 A/B 各 128×8 的 on-policy probe 作纠偏判断。Additional-SFT(B) 已在 GPU 4 启动。正式训练放在独立 tmux 中，以 `out/metrics/*.jsonl`、checkpoint 和终态 adapter 判定进度，不能用终端是否刷屏判定。

## 25. on-policy pilot 与正式启动证据

为避免把旧的截断采样 ledger 当成正式结果，A/B 都使用 `temperature=1.0, top_k=0, top_p=1.0` 重新做隔离 4-group pilot。训练前 old policy、可微 current policy 与 frozen reference 使用同一合法 token 集；ratio 因而应接近 1，而 reference KL 只反映初始量化重算差异和后续真实参数变化。

| 指标 | GRPO(A) pilot | GRPO(B) pilot |
|---|---:|---:|
| candidate groups / trajectories | 4 / 32 | 4 / 32 |
| max group KL k3 | 0.0005200 | 0.0004606 |
| max rollout logprob MAE | 0.01175 | 0.03138 |
| numeric finite | 是 | 是 |
| safety failure | 无 | 无 |

两条 pilot 的 4 组 strict reward 都是零方差；这是小样本观测，不与 128 题 probe 的有效组率矛盾，也不授权把正式目标换成 shaped reward。pilot adapter 只作工程证据，正式 A/B 都重新从同一个 SFT(A) adapter 开始，确保不会把 pilot 更新偷偷带入正式比较。

## 26. on-policy 奖励密度与正式中期证据

旧截断采样 probe 不能代表 formal full-softmax behavior policy，因此用完全一致的 `temperature=1, top_k=0, top_p=1` 重新生成 A/B 各 128×8 条轨迹：

| 指标 | A | B |
|---|---:|---:|
| trajectory task accuracy | 1.3672% | 0.8789% |
| pass@8 / effective-group rate | 9.375% | 7.03125% |
| tool execution success | 34.7656% | 36.7188% |
| evidence coverage | 6.3477% | 4.4922% |
| shaped nonzero-variance groups | 100% | 100% |

严格信号比旧探针低，但不为零，所以正式 A/B 不改 reward 定义。运行到约 1.3k 组时，A/B 的观测有效组率为 12.67%/9.19%，累计严格成功轨迹 198/143；最大 KL 0.04055/0.01389，最大 rollout logprob MAE 0.04226/0.04760，均在门限内且无非有限值。

Additional-SFT(B) 已完成 3,686 条、231 step、一个完整 epoch，`train_loss=0.092773`、耗时 1,658.43 秒。它与 GRPO(B) 使用相同 B 题 ID、相同 A 起点，但一个做离线 oracle 监督、一个做在线 strict RLVR，最终差异才能解释为训练目标差异，而不是数据内容差异。

终态 teacher-forced 审计显示 Additional-SFT(B) 的结构 token Top-1 为 99.1071%，两个 tool-call 边界均为 100%，普通位置结构 token Top-1 误触发仅 0.0280%，没有出现协议遗忘。B seen-pool 的 16×4 无更新行为探针得到 trajectory task accuracy 59.375%、pass@4 93.75%、工具执行 100%、evidence coverage 59.375%。由于样本仅 16 题且属于 B seen 数据，这些数值只能作为行为验收，不能替代 1,319 题官方 test 或证明泛化优势。

最终使用 `bash scripts/run_qwen_track2.sh audit_results` 进行 fail-closed artifact audit。审计要求 Agent-SFT(A)、Additional-SFT(B)、GRPO(A)、GRPO(B) 均有终态记录，五臂 official-test manifest 状态为 COMPLETE，每臂轨迹数和数据哈希一致，adapter hash 与评测记录一致，所有 tensor/指标有限，且 GRPO 正常组与安全拒绝组共同覆盖全部候选题。正式训练进行中时，审计已实测会因缺少 terminal metrics 而失败，不能误报完成。

验收原则：课程 shaped pilot 只回答“是否存在可优化信号”；正式论文/报告比较继续使用 strict task accuracy、pass@k、工具执行与证据覆盖。官方 test 在所有训练决策完成前不得用于调参。

## 27. 五臂官方 test 并行评测

GRPO(A/B) 均完成全部 3,686 个候选组、29,488 条 rollout 和 3,686 次 optimizer update，终态 adapter 已原子保存。A/B 全程分别耗时 54,748.84/57,185.88 秒，无 safety rejection、无非有限指标；最大组级 KL 为 0.04055/0.04109，最大 rollout log-probability MAE 为 0.061996/0.047598，均通过既定门限。

为在固定截止时间内完成 1,319 题五臂评测，评测器增加单臂 shard 模式，但没有改变模型、数据、seed 或采样定义。五臂使用同一 canonical YAML、同一 test SHA-256、seed 42、full-softmax sampling，分别写入隔离目录；Base、Agent-SFT(A)、GRPO(A)、GRPO(B)、Additional-SFT(B) 分配到 GPU 0/1/2/4/6。正式启动前用 Base 的 1 题隔离 tag 完成真实权重预检，不进入正式目录。

合并器不是简单拼接 JSON：它要求五个 shard 均为 COMPLETE，配置/数据/adapter 哈希一致，标签顺序等于预注册矩阵，每臂恰有 1,319 个唯一 `(seed, index)`，所有 summary 数值有限；随后以硬链接或复制方式在临时目录建立 canonical trajectories，写入带轨迹哈希的 summary/manifest，再原子重命名。任一检查失败都不会生成正式 `evaluation/test`，因此后续 `audit_qwen_track2_results.py` 仍保持 fail-closed。

## 28. 终态结果与审计

五个 shard 全部完成后，合并器验证配置/数据/adapter 哈希、每臂 1,319 个唯一 `(seed,index)` 和有限 summary，原子生成 canonical test 目录。`scripts/audit_qwen_track2_results.py` 随后对两个完整 SFT、两个完整 GRPO、五臂官方 test 和全部 adapter 再次独立检查，最终状态为 `PASS`。

官方 test 的严格任务成功率为：Base 0%、Agent-SFT(A) 0.910%、GRPO(A) 2.578%、GRPO(B) 2.654%、Additional-SFT(B) 42.532%。GRPO(A/B) 相对 Agent-SFT(A) 的同题配对增量为 +1.668/+1.744 个百分点，95% paired bootstrap 区间均不跨 0；GRPO(B) 相对 GRPO(A) 仅 +0.076 个百分点，区间 `[−1.061,+1.213]`，不能声称 A/B 有差异。Additional-SFT(B) 相对 GRPO(B) 高 39.879 个百分点。

结果支持“strict-GRPO 在可靠 Agent-SFT 冷启动训练产物上有小幅有效改进”，但不支持“RL 比继续使用可靠 oracle 的 SFT 更强”。正式末 200 组 A/B 的零方差率仍为 82%/84.5%，说明稀疏奖励是效果上限的主要原因。完整结果、算法解释与限制见 `docs/STAGE2_TRACK2_FINAL_REPORT.md`。
