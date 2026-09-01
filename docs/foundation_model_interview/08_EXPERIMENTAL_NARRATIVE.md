# 从基座到 Agentic RLVR：完整实验叙事

> 本文按照真实实验发生的顺序讲述项目，不按代码文件罗列功能。每个阶段都回答：为什么做、怎么做、发现了什么、问题在哪里、如何修复、修复后得到了什么。

> 本文叙事截至初始四算法 Formal；其后对 FP16 checkpoint 舍入、行为 log-prob 对齐和稀疏奖励冷启动的二次诊断与优化见 [09_RL_OPTIMIZATION_FOLLOWUP.md](./09_RL_OPTIMIZATION_FOLLOWUP.md)。

## 1. 实验目的与可验证假设

项目面向大模型基座岗位，目标不是训练一个达到工业规模的模型，而是在一张 RTX 4090 上完成一条可审计的小模型闭环：

```text
Decoder-only 基座
  → Pretrain
  → Assistant-only SFT
  → LoRA / KD / DPO 支线
  → Agent SFT 冷启动
  → GRPO / CISPO / DAPO / GSPO
  → 数学推理与多轮 Tool-Use 固定集
```

实验开始前先把简历中的宽泛表述拆成可验证假设：

1. RMSNorm、RoPE/YaRN、GQA、KV Cache、SwiGLU、Flash SDPA 和稀疏 MoE 能否在统一 Decoder-only 模型中正确运行？
2. Pretrain、SFT、LoRA、KD 是否能使用相同 tokenizer、mask 和 checkpoint 顺利衔接？
3. DAPO 的动态采样能否解决 GRPO 在稀疏奖励下的零方差组问题？GSPO 的序列级 ratio 是否与 token-level 目标在张量上区分开？
4. Agent 的工具格式、执行结果和最终答案能否形成严格可验证奖励，并且防止格式作弊、答案猜测和过长输出？
5. 在共同初始化、共同固定集和三训练 seed 下，四种策略是否真的改善准确率、Reward、KL、长度或稳定性？

最后一个问题很重要：如果没有固定集提升，也必须如实报告，而不是用训练组筛选后的高 Reward 代替泛化结果。

## 2. 实验环境与控制变量

实验基线为官方仓库 commit `393e387e9ad99f0f04c296e4c5e7353f4444629f`，单张 NVIDIA RTX 4090（23.52 GiB），PyTorch 2.6.0+cu124，BF16。所有长任务用 `screen` 后台运行，日志和权重分别保存，避免 SSH 断开造成结果丢失。

正式对照遵守四个控制原则：

- 四种 RL 算法从同一个 `agent_sft` checkpoint 出发；
- train/eval 按问题哈希固定切分，评测集不参与冷启动；
- 每种算法使用 3 个独立训练 seed；
- 固定集使用相同 prompt、相同解码 seed 和相同评测脚本。

证据目录为 `out/remote_final/evidence/out`，包括配置、日志、checkpoint 哈希、CSV、逐轨迹 JSONL 和最终验收索引。

## 3. 第一阶段：先跑通 Dense 基座

### 3.1 为什么先做 Dense

后续 RL 需要同时驻留 current policy、reference policy、rollout 激活和 optimizer state。若一开始就使用未经充分训练的 MoE，算法收益会同时混入 Router 漂移、专家负载和基座质量差异。于是先固定 64M Dense 主线，把后训练的变量隔离出来；MoE 作为并行架构验证分支。

### 3.2 Pretrain 实际过程

Dense 模型配置为 hidden size 768、8 层、8 Query heads、4 KV heads，共 63,912,192 参数。预训练使用 `pretrain_t2t_mini.jsonl`，sequence length 768、micro-batch 16、梯度累积 4、BF16、学习率 `5e-4`、2 epochs。

训练核心是 next-token CE。padding 的 label 设置为 `-100`，不进入损失；每次前向后执行 BF16 autocast、反向传播、梯度累积、梯度裁剪和 optimizer step。

结果：2 epochs 正常完成，最后一批日志 loss 为 1.8342，产生可用于 SFT 的 `pretrain_768` checkpoint。

### 3.3 训练过程中遇到的问题

第一，远程 Hugging Face 连接曾被拒绝。没有重复下载，而是检查本地 `dataset/`，对文件做 SHA-256 校验，确认数据与目标版本一致后继续。

第二，容器没有 `/usr/bin/time`。训练本身没有失败，改用 shell `SECONDS`、`PIPESTATUS` 和日志时间戳记录退出码与耗时。

第三，Linux `compileall` 报告 `trainer/._*.py` 含 null bytes。定位后发现这是 macOS AppleDouble 元数据，不是真实 Python 源码；清理 `._*` 后重新执行 31 项测试。

第四，审计断点续训时发现最后一个梯度累积窗口可能在 checkpoint 保存之后才完成。修复顺序为：先对 remainder 梯度做比例校正，再 optimizer step，最后原子保存 checkpoint。该修复避免“日志显示处理完，但权重少一次更新”。

## 4. 第二阶段：SFT 与 Assistant-only Mask

### 4.1 为什么需要 SFT

Pretrain 只学习文本分布，不保证模型遵守 system/user/assistant 对话协议。SFT 将多轮对话套入 chat template，但不能把 prompt 也作为监督目标，否则模型会学习复述用户输入。

真实 `SFTDataset.generate_labels` 的策略是：默认所有 label 为 `-100`，扫描 assistant 起止标记，只打开 assistant span，包括 assistant EOS。

```python
labels = [-100] * len(input_ids)       # system/user/tool 默认不监督
if input_ids[i:i + len(self.bos_id)] == self.bos_id:
    start = i + len(self.bos_id)
    end = find_assistant_eos(input_ids, start)
    for j in range(start, end + len(self.eos_id)):
        labels[j] = input_ids[j]
```

### 4.2 结果与审计

Dense SFT 使用同一架构，2 epochs 正常退出，最后一批 loss 为 1.4291。Mask 审计不是只看监督 token 数量，而是重新根据 chat template 找出期望 assistant span，再逐 token 比较：

- 通用 SFT 固定 128 样本：mismatch=0，zero-supervision=0；
- Agent SFT 固定 128 样本：9,407 supervised tokens，mismatch=0，zero-supervision=0。

这一步证明后续 RL 的 action mask 有可靠的监督数据基础。

## 5. 第三阶段：架构机制核验与 MoE 分支

### 5.1 RMSNorm、RoPE/YaRN、SwiGLU

源码中 RMSNorm 在 FP32 计算 `x * rsqrt(mean(x²)+eps)`，然后恢复 BF16；SwiGLU 使用 `down(SiLU(gate(x)) * up(x))`；RoPE 预计算 cos/sin 并旋转 Q/K；YaRN 对超出原始训练长度的频率做 ramp 插值。

4096-token prompt 的 RoPE 与 YaRN 都能完成前向和解码：

| 方案 | Cached tok/s | Uncached tok/s |
|---|---:|---:|
| RoPE | 130.97 | 106.81 |
| YaRN | 122.15 | 103.53 |

这里出现的现象是 YaRN 吞吐略低，但这不是失败：YaRN 的目标是位置外推，不是单纯提速。由于没有 Needle/Passkey 或长文本 PPL 结果，最终结论只能是“长位置路径可运行”，不能写成“长文本能力提升”。

### 5.2 GQA 与 KV Cache

模型使用 8 个 Query heads、4 个 KV heads。历史 K/V 在 `repeat_kv` 之前缓存，因此保存的是紧凑的 4-KV-head 表示；只有进入 attention 计算时才逻辑扩展到 8 个 heads。

理论上 KV Cache 与 KV head 数成正比，所以相对 8-KV-head MHA 降低 50%。两个回归测试确认：

1. cached 单步 logits 与 full forward 对应位置一致；
2. cached 与 uncached greedy decode token 完全一致。

### 5.3 Flash SDPA

最初不能只凭源码中的 `scaled_dot_product_attention` 宣称使用 Flash。于是做了两层验证：先强制 Flash/math backend 比较原语，再对真实模型运行 profiler。

RTX 4090、BF16、`[B=2,H=8,S=1024,D=96]` 的原语结果：

| Backend | Median | Peak memory |
|---|---:|---:|
| Forced Flash | 0.2069 ms | 12.06 MiB |
| Forced Math | 1.3551 ms | 190.14 MiB |

该 shape 下约加速 6.55×；真实 `full_sft` forward 也命中 `aten::_scaled_dot_product_flash_attention` 和 CUDA Flash kernel。严谨表述是“接入 PyTorch SDPA，并在目标 GPU/shape 上验证 Flash backend”，不是“手写了 FlashAttention-2”。

### 5.4 MoE 为什么没有成为后训练主线

MoE 配置为 4 experts、Top-1 routing。它将总参数扩大到 198.417M，每 token 激活参数约 63.937M，聚合路由负载为 `[21.09%,24.02%,29.59%,25.29%]`，CV=12.23%，8 个 Router 与 96 个 Expert 梯度全部 finite。

但实际 MoE checkpoint 只训练到 `8350/79390`，约为第一轮的 10.52%，只适合验证路由和梯度，不适合拿来与 Dense 做质量结论。1024-token prompt + 128-token decode 的系统结果为：

| 架构 | Cached tok/s | Cached peak MiB |
|---|---:|---:|
| Dense | 130.87 | 448.46 |
| Partial MoE | 88.06 | 1212.42 |

原因是当前实现没有 expert parallel、all-to-all、capacity factor 和 fused dispatch。于是后训练统一使用收敛更充分的 Dense SFT；MoE 只回答“容量扩展、稀疏激活、路由均衡和系统成本”四个问题。

## 6. 第四阶段：LoRA、KD 与 DPO 支线

### 6.1 LoRA：低参数适配得到正结果

LoRA 冻结原始权重，只训练低秩增量 `ΔW=BA`。`A` 随机初始化，`B` 零初始化，保证初始 `ΔW=0`；部署时可将 `W+B@A` 合并回原权重。

本次只训练 393,216 个 adapter 参数，占完整参数 0.61%。固定医疗 holdout 上，base 与 LoRA 使用同一批物化 assistant tokens：

| 模型 | Loss | PPL |
|---|---:|---:|
| Full SFT base | 1.698209 | 5.464150 |
| + Medical LoRA | 1.568724 | 4.800518 |

loss 下降 7.62%，PPL 下降 12.15%。这是真实正结果，但只适用于该医疗划分，不能外推为通用或医疗安全能力提升。

### 6.2 KD：完成链路但得到负结果

Student 为 30.025M 参数，Teacher 为 63.912M 参数，学生参数减少 53.02%。KD 损失为：

$$L=\alpha L_{CE}+(1-\alpha)T^2KL(p_T^T\|p_S^T)$$

CE control 与 KD 使用同一 Student 初始化、同一 50k 子集、相同顺序和更新预算；独立 holdout 为 2,000 条、383,926 个 assistant tokens。

| 模型 | Loss | PPL |
|---|---:|---:|
| CE control | 2.056352 | 7.817403 |
| KD student | 2.061657 | 7.858981 |

KD 比 CE 的 PPL 高 0.532%。这不是需要隐藏的失败，而是说明当前 `α=0.5,T=1.5` 与小预算组合没有带来收益。可能原因包括软目标正则过强、teacher 与域分布不匹配、训练步数不足；下一步应在 validation 上调参，再锁定 test。

### 6.3 DPO：偏好链路的独立负对照

DPO 按 prompt 分组切分 chosen/rejected，避免同 prompt 泄漏到 train/eval。固定 1,000 个 eval pair 上，Full SFT 与 DPO 的 chosen preference accuracy 都为 45.70%，DPO loss 从 0.693147 变为 0.694702。结论是“链路和评测完成，当前小预算无偏好能力提升”。

## 7. 第五阶段：先解决 RL 没有学习信号的问题

### 7.1 第一次直接做 RL 为什么失败

上游 `agent_rl.jsonl` 中只有一部分样本真正带有 tools 和 GT，后半部分是普通对话。若直接让通用 SFT 模型进行多轮 Tool-Use rollout，许多 prompt 的 G 条回答会全部失败：

$$\operatorname{std}(r_{1:G})=0\Rightarrow A_{1:G}=0$$

这不是“算法不收敛”，而是策略没有产生可比较的成功/失败样本。

### 7.1.1 这次“直接 RL”到底执行了什么

严格说，第一次没有让 `full_sft` 完成正式 policy optimizer 更新，而是先做了一个 RLVR readiness/preflight。原因是如果连 rollout 的成功/失败分布都没有确认，直接训练会把“无梯度”误判成“算法失效”。实际步骤如下：

```bash
# 由 scripts/run_rlvr_readiness_after_kd.sh 调用
python scripts/eval_agent_rlvr.py \
  --weights full_sft \
  --reference_weight full_sft \
  --data_path dataset/agent_rl_math_eval.jsonl \
  --seeds 101 --limit 32 --batch_size 2 \
  --num_generations 4 --max_turns 3 \
  --max_gen_len 128 --max_total_len 1536 \
  --reward_mode strict \
  --require_tool_call_for_success 1 \
  --hidden_size 768 --num_hidden_layers 8 \
  --use_moe 0 --device cuda:0
```

这一步对 `full_sft` 做的事情是：

1. 对每个 prompt 采样 4 条 completion；
2. 解析 `<tool_call>`，检查 JSON 和参数 Schema；
3. 执行确定性 mock tool，把 observation 拼回下一轮上下文；
4. 根据格式、执行、必需工具、证据和最终答案计算 strict `+1/-1`；
5. 统计 `task_success`、`reward_std` 和 DAPO 可接受组比例；
6. 不保存 policy update，只判断是否具备开始 RL 的条件。

第一次运行还遇到了一次保护性失败：多轮 chat template 重新渲染后，在 sampled token 位置 419 改写了之前的 token，程序主动抛出：

```text
Multi-turn assistant/tool template changed previously sampled tokens at position 419
```

如果忽略这个错误，`π_old` 与当前策略会在不同条件上下文上计算 log-prob，importance ratio、clip 和 KL 都不可信。修复方式是保存精确 sampled-token ledger，只追加环境 observation suffix；修复后重新执行 readiness。

### 7.1.2 readiness 的真实输出

修复后的 `full_sft` 结果为：

| 任务 | Reward | Task Accuracy | Zero-variance groups | DAPO effective groups |
|---|---:|---:|---:|---:|
| Math | -1.0000 | 0% | 100% | 0% |
| Basic Tool | -0.8594 | 7.031% | 84.375% | 15.625% |

数学任务的 4 条 rollout 全部无法形成严格成功/失败混合组，因此 `A=0`，任何 GRPO/CISPO/DAPO/GSPO 的 policy loss 都没有有效相对优势。形式上即使调用 `optimizer.step()`，也不会得到有意义的任务梯度。

Basic Tool 有少量混合组，但 Agent SFT 后该集合直接达到 100%，继续在这个集合上训练会得到饱和的“伪提升”。所以没有把这次 readiness 结果冒充正式 RL 训练，也没有在数学 0% 的 `full_sft` 上硬跑四算法。

`scripts/run_rlvr_pilot_after_dpo.sh` 和正式脚本随后加入了 gate：读取 readiness 的 `dapo_effective_group_rate`，若为 0 就跳过该任务，并提示先做冷启动或课程学习。正式 policy 训练的命令形式仍然是：

```bash
cd trainer
torchrun --standalone --nproc_per_node 1 train_agent.py \
  --loss_type grpo \
  --from_weight agent_sft \
  --data_path ../dataset/agent_rl_tool_challenge_train.jsonl \
  --num_generations 4 --max_updates 30 \
  --policy_update_epochs 2 --learning_rate 1e-7 \
  --reward_mode strict --require_tool_call_for_success 1
```

这里的关键变化不是把 `grpo` 改成更复杂的算法，而是先把初始化从 `full_sft` 换成经过训练集 oracle 冷启动的 `agent_sft`，并把基础 Tool 集换成零题目重合的组合挑战集。

### 7.2 数据和冷启动修复

于是先做三件事：

1. 从确定性 mock 工具构建 896/224 的 Tool train/eval，问题交集为 0；
2. 构建 512 条数学 Agent SFT 轨迹，并在写入前重新计算表达式 GT；
3. 只使用训练集 oracle 做 Agent SFT，eval label 不进入训练。

128 条 readiness 结果：

| 任务 | Full SFT | Agent SFT |
|---|---:|---:|
| Math Task Accuracy | 0% | 59.375% |
| Basic Tool Task Accuracy | 7.031% | 100% |

数学任务从 0% 提升到 59.375%，基础 Tool-Use 从 7.031% 提升到 100%。这一步是后续 RL 能够运行的关键新模型：`agent_sft` 冷启动 checkpoint。

基础 Tool-Use 很快饱和，继续在上面比较 RL 会得到虚假的“全算法 100%”。因此又构建了与 Agent SFT oracle 零重合的 384/96 组合挑战集，包含双工具和三工具链。

## 8. 第六阶段：多轮 Agentic RLVR 与奖励约束

### 8.1 轨迹采集

每条轨迹按以下状态机执行：

```text
Assistant <tool_call>
       → parse + schema validation
       → deterministic tool execution
       → Observation
       → Assistant next action / final answer
       → verifier
```

工具 observation 可以进入下一轮上下文，但不计算 policy gradient。action mask 只打开 assistant tool call、assistant final answer 和 assistant EOS；system/user/tool/padding 均为 0。

项目曾找到真实 tokenizer non-roundtrip token：sampled action 为 36 tokens，canonical decode→encode 后变成 38 tokens。修复方式不是强行 round-trip，而是保存精确 sampled-token ledger，只追加环境 observation suffix，使 `π_old`、`π_θ` 在相同 action IDs 上计算 log-prob。

### 8.2 Reward verifier

verifier 分开统计格式、工具调用合法性、执行成功、必需工具覆盖、工具证据覆盖和最终答案。严格成功相当于：

```python
task_success = (
    answer_correct
    and format_valid
    and calls_satisfied
    and calls_correct
    and evidence_satisfied
    and not unfinished
)
```

因此以下攻击都会被拒绝：

- GT=4 时输出 14 的数字子串攻击；
- 格式合法但答案错误；
- 调错工具后猜中答案；
- 伪造 observation；
- 未闭合的多轮轨迹；
- 非法 JSON、非法参数和重复套话。

Calculator 使用 AST 白名单，不执行任意 Python 表达式。Soft Overlong 在长度上限前设置线性惩罚，避免模型等到硬截断才受到惩罚。

### 8.3 “截断率”的口径修正

当前正式日志记录的是 `unfinished`、平均/P95 响应长度，并拒绝 rollout 后左截断；它没有区分命中 `max_new_tokens`、`max_turns`、`max_context` 还是工具错误。因此本报告使用“未完成率”，不把它冒充 token truncation rate。

## 9. 第七阶段：四算法 Pilot 与 Formal

### 9.1 统一目标

四种算法共享 rollout、reward、mask、old/reference log-prob、optimizer 和评测集，只替换 policy objective：

- GRPO：对称 token ratio clip，sequence-first reduction；
- CISPO：单侧上界 coefficient，stop-gradient，global token mean；
- DAPO：dynamic sampling、asymmetric clip、token mean、soft overlong；
- GSPO：长度归一化 sequence importance ratio 和 sequence-level clip。

KL 使用 FP32 k3 estimator `exp(δ)-δ-1`，并记录 mean/P95/max；第一次 policy forward 检查 `rollout_logprob_mae` 与 `rollout_ratio_mean`，防止 trajectory 拼接错位。

### 9.2 Pilot 阶段

Pilot 每算法只跑一个 seed、少量 updates，目的不是写入简历，而是检查：

1. ratio 在 current=old 时是否接近 1；
2. 第二个 policy epoch 后 ratio 是否发生变化；
3. DAPO 是否真的丢弃零方差组；
4. GSPO 是否使用几何平均而不是 token ratio 算术平均；
5. tool parser、observation mask 和 reward verifier 是否一致。

Pilot 发现一个重要现象：模型生成能力和参数变化都很弱，固定评测结果几乎不动。因此没有把 pilot 训练组的偶然高分写成能力提升，而是增加了固定集、三 seed 和成本统计。

### 9.3 Formal 训练设置

- Math 与组合 Tool-Use 分开运行；
- 每个任务 4 算法 × 3 训练 seed；
- 每 run 最多 10 个 accepted/update groups；
- 每组 4 trajectories，policy epochs=2，learning rate=`1e-7`；
- 每个 checkpoint 在固定 64 prompts × 3 decode seeds 上评测；
- 每个任务共 2,496 条固定集轨迹。

这里的“同预算”只指 optimizer updates，不指总 rollout。DAPO 为填满有效组会额外采样候选轨迹。

## 10. Formal 结果：机制有效，但当前预算没有泛化提升

### 10.1 Math

共同 Agent SFT baseline 的固定集 Reward=-0.03125、Task Accuracy=48.4375%、响应长度=80.19 tokens。GRPO、CISPO、DAPO、GSPO 的固定集这些指标全部相同，delta=0；固定集 k3 KL 为 `0–5.43e-9`。

训练组中 DAPO 的 Reward/Accuracy 较高，是因为只保留混合成功组的条件统计，不能与普通随机组直接比较。它证明了采样机制生效，不证明泛化提升。

### 10.2 Tool-Use

固定集 baseline Reward=-0.88542、Task Accuracy=5.7292%、Format=97.9167%、Execution=71.6146%、平均长度=92.34、Unfinished=1.0417%。四算法固定集全部保持这些数值，delta=0，k3 KL 为 `0–1.53e-8`。

但训练机制层面出现了清晰差异：

| 算法 | 零方差组比例 | 候选组 | Wall time |
|---|---:|---:|---:|
| GRPO | 90.00% | 10.0 | 28.97 s |
| CISPO | 86.67% | 10.0 | 28.80 s |
| DAPO | 0% | 103.7 | 285.19 s |
| GSPO | 93.33% | 10.0 | 28.16 s |

DAPO 将被用于更新的零方差组降到 0%，但候选组约增加 10.4×，wall time 约增加 9.9×。因此最终结论是：DAPO 改善了稀疏奖励下“获得有效相对优势”的机会，代价是显著的 rollout 成本；当前训练预算不足以让模型在固定集上产生可观测能力变化。

## 11. 最终问题复盘与解决方案

| 问题 | 定位依据 | 解决方法 | 修复后结论 |
|---|---|---|---|
| HF 下载失败 | 连接错误，但本地数据存在 | 校验本地文件哈希后继续 | 数据来源可追溯 |
| `._*.py` null bytes | `compileall` 只命中 AppleDouble 文件 | 清理元数据文件 | 31/31 测试通过 |
| `/usr/bin/time` 不存在 | shell 命令缺失 | 使用 `SECONDS/PIPESTATUS` | 训练与退出码仍可记录 |
| 梯度累积尾 batch 保存顺序 | checkpoint 少最后 remainder update | 先校正梯度、step 后原子保存 | 后续运行不丢最后更新 |
| Full SFT 直接 RL 无有效优势 | Math=0%，大量全错组 | Agent SFT 冷启动 | Math=59.375%，产生混合组 |
| 基础 Tool 集饱和 | Agent SFT 后达到 100% | 构建零重合组合挑战集 | 保留可学习难度 |
| token decode/encode 不可逆 | 36→38 token 反例 | 精确 sampled-token ledger | old/current action IDs 对齐 |
| Reward Hacking | 子串、猜答案、伪造 observation | strict verifier、执行证据、AST 白名单 | 攻击样例被拒绝 |
| 普通 RL 大量零方差组 | Tool 86.7%–93.3% | DAPO dynamic sampling | 保留组零方差率 0% |
| RL 固定集无提升 | fixed holdout delta=0，KL≈0 | 保留负结果，建议增加 validation budget | 结论不夸大 |
| Warmup 未实现 | `get_lr` 只有 cosine decay | 简历改写为 cosine；若必须写 Warmup，另补代码和 A/B | 当前报告不冒充 Warmup |
| 没有严格 token 截断字段 | 只有 unfinished/length | 新增 stop_reason 后再报告 truncation | 当前使用“未完成率” |

## 12. 最终可复现实验入口

在远程仓库根目录执行以下顺序，能够复核关键链路：

```bash
# 1) 核心回归
python -m unittest discover -s tests -v

# 2) mask、MoE、Flash 与架构审计
python scripts/audit_sft_mask.py --data_path dataset/agent_sft_coldstart.jsonl --limit 128
python scripts/audit_moe_routing.py --weight pretrain_partial --hidden_size 768 --num_hidden_layers 8
python scripts/verify_flash_sdpa.py
python scripts/verify_model_flash_sdpa.py --weight full_sft

# 3) Agent 数据与冷启动
python scripts/audit_tool_rlvr_dataset.py --data_path dataset/agent_rl_tool_verified_train.jsonl
bash scripts/run_agent_sft.sh

# 4) 四算法 Formal（每个 seed 使用独立 screen/log/checkpoint）
bash scripts/run_rlvr_ablation.sh

# 5) 固定集评测与聚合
bash scripts/run_rlvr_eval.sh
python scripts/summarize_eval_results.py out/eval_rlvr/summary.csv \
  --output out/eval_rlvr/algorithm_comparison.csv
python scripts/summarize_rl_metrics.py 'out/metrics/*.jsonl' \
  --last_n 20 --output out/metrics/algorithm_comparison.csv
```

实际完整服务器命令、screen 管理、数据下载和故障处理见 [`00_END_TO_END_GUIDE.md`](./00_END_TO_END_GUIDE.md)。

## 13. 最终结论与面试叙事

最流畅的面试讲法是：

> 我先用 Dense Decoder-only 模型跑通 Pretrain 和 Assistant-only SFT，再验证 GQA/KV Cache、Flash SDPA、RoPE/YaRN 和 MoE 的结构与系统行为。过程中发现未经 Agent 冷启动的模型在 Tool-Use 上会产生大量全错组，GRPO 没有优势信号，于是用训练集 oracle 做 Agent SFT，并新建零题目重合的组合挑战集。随后在同一 Agent SFT 初始化、同一 fixed holdout 和三个训练 seed 下实现 GRPO、CISPO、DAPO、GSPO。DAPO 确实把 Tool 训练的保留组零方差率降到 0%，但付出约 10.4 倍候选采样成本；在当前 10-update、1e-7 小预算下，四算法固定集 delta 都为 0。因此我把结论拆成机制结论和能力结论：机制正确、训练信号改善、成本可量化，但不能把训练组筛选统计包装成泛化提升。与此同时，LoRA 在固定医疗 holdout 上取得 12.15% PPL 改善，KD 和 DPO 则保留为公平负对照。

这条叙事同时覆盖四条简历要求，并且能解释为什么某些结果没有提升：不是回避问题，而是先定位到数据、奖励稀疏、预算和评测设计，再用新模型、验证器和固定实验协议把问题变成可测量结论。
