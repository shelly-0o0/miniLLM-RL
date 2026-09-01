# 第三阶段：GRPO、CISPO、DAPO、GSPO 从零到代码精通

本文假设已经完成上一阶段：能够产生可信的多轮轨迹、action mask、behavior log-prob 和可验证 reward。现在解决第二个问题：如何用这些轨迹稳定更新语言模型策略。

全文使用统一链条：

```text
训练中出现的问题 → 根因 → 方法 → 公式 → tensor 形状
→ MiniMind 代码 → 运行参数 → 指标 → 如何验证结果改善
```

配套实验：[09_GRPO_CISPO_DAPO_GSPO从公式到代码.ipynb](../../learning_notebooks/09_GRPO_CISPO_DAPO_GSPO从公式到代码.ipynb)。

---

## 0. 学完后应该具备什么能力

读完并完成练习后，应当能够：

1. 从 next-token probability 推导策略梯度的 log-prob 形式；
2. 解释 reward、return、baseline、advantage、old policy 和 reference policy；
3. 手算一组 GRPO group-relative advantage；
4. 根据 advantage 正负解释 PPO/GRPO clip 到底裁掉了什么；
5. 从公式和代码区分 GRPO、CISPO、DAPO、GSPO；
6. 解释 sequence-first 与 global token-mean reduction 的差异；
7. 读懂 `compute_policy_loss` 的每一个 tensor、mask 和返回指标；
8. 运行单元实验检查梯度、ratio、clip、KL、动态采样和超长惩罚；
9. 设计不会混淆算法、reward 和 rollout 成本的公平对照；
10. 根据日志定位无更新、策略漂移、reward hacking、policy lag 和 MoE 不稳定。

---

## 1. 从语言模型概率开始

### 1.1 Decoder-only 模型在做什么

给定 prompt `x` 和响应 token：

```text
y = (y₁, y₂, ..., y_T)
```

自回归语言模型将序列概率分解为：

```text
πθ(y|x) = Π_t πθ(y_t | x, y_<t)
```

直接连乘很多小概率容易数值下溢，所以代码使用 log-prob：

```text
log πθ(y|x) = Σ_t log πθ(y_t | x, y_<t)
```

模型 forward 输出 logits `[N, L, V]`。对 vocabulary 维做 `log_softmax`，再用真实 next token 的 id `gather`，得到每个已采 action 的 log-prob。

MiniMind 中的核心写法：

```python
per_token_logps = F.log_softmax(
    res.logits[:, :-1, :].float(), dim=-1
).gather(
    2, input_ids[:, 1:].unsqueeze(-1)
).squeeze(-1)
```

- `[:, :-1]`：位置 `t` 的 logits 预测 `t+1` token；
- `input_ids[:, 1:]`：对应的目标 token；
- `.float()`：FP32 log-softmax，降低 ratio 对精度差异的放大；
- 得到形状 `[N, L-1]`，再由 completion/action mask 选中策略动作。

### 1.2 SFT 和 RL 更新方向的区别

SFT 对数据中每个 assistant token 最小化：

```text
L_SFT = -log πθ(y_t|context)
```

无论整条回答最终是否完成任务，都把数据 token 概率提高。

RL 根据 reward 决定方向：高于 baseline 的回答提高概率，低于 baseline 的回答降低概率。它不需要一条唯一标准措辞，而需要可靠的任务结果。

---

## 2. 从期望奖励推导策略梯度

目标是最大化当前策略产生轨迹的期望奖励：

```text
J(θ) = E_{τ~πθ}[R(τ)]
```

对离散轨迹直接对采样过程求导困难。利用 log-derivative trick：

```text
∇θ πθ(τ) = πθ(τ) ∇θ log πθ(τ)
```

得到：

```text
∇θ J(θ) = E_{τ~πθ}[R(τ) ∇θ log πθ(τ)]
```

自回归展开：

```text
∇θ log πθ(τ) = Σ_t ∇θ log πθ(a_t|s_t)
```

因此最基础的 Monte Carlo loss 可以写成：

```text
L = -R(τ) Σ_t log πθ(a_t|s_t)
```

负号是因为 optimizer 默认最小化 loss。

### 2.1 为什么需要 baseline

如果所有 reward 都为正，所有采样都会被提高概率，只是程度不同；reward 数值尺度也随题目难度变化。减去不依赖当前 action 的 baseline 不改变期望梯度，却能降低方差：

```text
A = R - b
```

- `A > 0`：回答比 baseline 好，应提高概率；
- `A < 0`：回答比 baseline 差，应降低概率；
- `A = 0`：当前样本没有相对训练信号。

PPO 常用 value/critic 估计状态价值作为 baseline。GRPO 用同一 prompt 多条回答的 reward 统计量替代 critic。

---

## 3. 三个策略：current、old、reference

这是整个实现中最重要的概念区分。

### 3.1 `π_old`：behavior policy

生成本批 trajectory 的策略。它回答：

```text
“这条 action 是在什么采样分布下得到的？”
```

rollout 时保存 `old_per_token_logps`，本批多次 policy update 期间保持冻结。

### 3.2 `π_θ`：current policy

正在被 optimizer 更新的模型。每次 forward 重新得到 `current_logps`。

### 3.3 `π_ref`：reference policy

通常是冻结的 SFT checkpoint，用来测量或约束当前模型偏离初始行为的程度。它不是 importance ratio 的采样分母。

### 3.4 为什么不能混用

训练开始时 `old` 和 `ref` 可能都来自 `full_sft`；第一次更新后：

```text
old = 最近生成本批轨迹的 policy
ref = 始终冻结的 SFT policy
```

将 ref 当 old 会让 importance correction 与真实采样分布不符；将 old 当 ref 会让 KL 锚点随着训练移动。

---

## 4. Importance Ratio 从哪里来

旧策略采样的数据被当前策略复用时，用重要性比率校正分布变化：

```text
r_t(θ) = πθ(a_t|s_t) / π_old(a_t|s_t)
       = exp(log πθ(a_t|s_t) - log π_old(a_t|s_t))
```

解释：

- `r=1`：当前和旧策略对该 action 概率相同；
- `r=1.2`：当前概率是旧策略的 1.2 倍；
- `r=0.8`：当前概率下降到 0.8 倍。

在 `policy_optimization.py` 中：

```python
log_ratio = (safe_current - safe_old).clamp(min=-20.0, max=20.0)
token_ratio = torch.exp(log_ratio)
```

对 log-ratio 做数值裁剪是为了防止 `exp` 溢出，不等同算法中的 PPO clip。算法 clip 的范围通常在 1 附近，而数值保护范围约是 `exp(-20)` 到 `exp(20)`。

### 4.1 第一次 forward 应该是什么

Torch rollout 由当前模型生成，本批尚未更新时：

```text
current ≈ old → ratio ≈ 1
```

项目记录：

```text
rollout_logprob_mae
rollout_ratio_mean
```

应分别接近 0 和 1。否则先修 tokenizer、padding、mask、上下文截断或推理/训练 kernel 精度，不能继续相信 clip/KL 指标。

本次 pilot 实际捕获了一个精度混用反例：策略与行为策略用 BF16 autocast，而冻结 reference 曾以 FP32 前向；即使三者权重尚未发生有效更新，极低概率 token 的舍入差异也会被 `exp(logπ_ref-logπ_current)` 放大，令 k3 尾部达到 `10^5` 量级。修复是让 current、old、reference 在同一 autocast 精度下计算 log-prob，并保留 FP32 `log_softmax`。这类异常不能简单解释成“策略已经远离 SFT”。

统一精度后，未发生非零更新的组恢复为 `KL_k3≈0`；但一旦策略完成有效更新，极少数低概率 action token 仍可能形成真实长尾。例如 Formal 的一个 batch 中 `KL_k3 mean=243680.8`，而 token p95 仅约 `5.0e-5`，最大单 token 约 `7.06e7`。因此实现同时记录 `local_reference_log_ratio_abs_mean`、`local_kl_k3_p95` 和 `local_kl_k3_max`。面试时应解释“均值被单点主导”，并优先用固定 holdout KL 判断整体策略漂移；不能只报一个巨大的 train mean，也不能把异常值静默删掉。

---

## 5. PPO Clip：为什么要限制更新

### 5.1 观察到的问题

同一批有限样本上做大步更新，会让已采 action 概率剧烈改变。Importance ratio 方差变大，策略可能为了少数高 reward 样本迅速坍缩，旧轨迹也越来越不代表当前策略。

### 5.2 方法

PPO clipped surrogate：

```text
min(rA, clip(r, 1-ε, 1+ε)A)
```

不要只背公式，要按 advantage 符号分析：

| Advantage | 策略想做什么 | 被限制的危险方向 |
|---:|---|---|
| `A > 0` | 提高好 action 概率 | ratio 高于上界后不再继续奖励 |
| `A < 0` | 降低差 action 概率 | ratio 低于下界后不再继续惩罚 |

### 5.3 一个手算例子

令 `ε=0.2`：

- `A=+1, r=1.5`：原 surrogate=1.5，clip=1.2，取 min=1.2；
- `A=-1, r=0.5`：原 surrogate=-0.5，clip 分支=-0.8，取 min=-0.8；
- `A=+1, r=0.5`：取 0.5，不阻止概率变差，保留恢复梯度；
- `A=-1, r=1.5`：取 -1.5，不阻止坏 action 概率变高，保留强纠正梯度。

所以 clip 不是简单把所有 ratio 截断后乘 advantage，而是通过 `min` 做悲观 surrogate。

---

## 6. 问题一：PPO Critic 成本高 → GRPO

### 6.1 观察到的问题

对大语言模型训练 PPO，通常还需要 value/critic：

- 额外模型参数或 value head；
- optimizer state 和激活显存；
- value regression 的训练不稳定；
- 长推理终局 reward 的 token-level value 很难准确拟合。

### 6.2 GRPO 方法

对同一个 prompt `x` 采样 `G` 条回答：

```text
R = [R₁, R₂, ..., R_G]
A_i = (R_i - mean(R)) / (std(R) + ε)
```

用组内均值作为 prompt-specific baseline，不训练 critic。

当前函数：

```python
group_relative_advantages(rewards, group_size)
```

输入 flat `[B*G]`，内部 `view(-1, G)`，逐组标准化后重新展平。

### 6.3 数值例子

严格 reward：

```text
R = [-1, -1, +1, +1]
mean = 0
population std = 1
A ≈ [-1, -1, +1, +1]
```

如果：

```text
R = [-1, -1, -1, -1]
```

则所有 advantage 为 0。这不是数值 bug，而是组内没有相对排序信号。

### 6.4 GRPO 目标

项目中的 GRPO 使用 token ratio、对称 clip、sequence-first reduction：

```text
L_i,t = -min(r_i,t A_i, clip(r_i,t,1-ε,1+ε) A_i)
L_GRPO = mean_i [(Σ_t mask_i,t L_i,t) / (Σ_t mask_i,t)]
```

每条序列先按自身 action token 数平均，再对序列平均，因此一条 20-token 和一条 200-token 回答在最终样本层权重相同。

### 6.5 代码分支

`compute_policy_loss`：

```python
low, high = grpo_epsilon, grpo_epsilon
clipped_ratio = token_ratio.clamp(1.0 - low, 1.0 + high)
surrogate = torch.minimum(
    token_ratio * token_advantages,
    clipped_ratio * token_advantages,
)
per_token_policy = -surrogate
```

随后计算每序列 mean，再对有效序列 mean。

### 6.6 结果改善与代价

预期改善：省去 critic，简化显存和系统组件，组内比较自动适配 prompt 难度。

代价：每个 prompt 要生成 `G` 条轨迹；全对/全错组无梯度；所有 token 共享终局 advantage；长序列 token ratio 噪声仍存在。

验证时必须同时报告模型显存、rollout 数、zero-variance group rate 和固定集 Accuracy，不能只说“少了 critic 所以更高效”。

---

## 7. 问题二：GRPO Clip 后部分 token 梯度消失 → CISPO

### 7.1 观察到的问题

对正 advantage token，如果 ratio 超过上界，PPO/GRPO 的 `min` 选择常数 clip 分支，该 token 的 policy surrogate 对当前 log-prob 的局部梯度为 0。长推理中的稀有 reflection/fork token 可能正是需要持续学习的关键动作。

### 7.2 CISPO 方法

CISPO 把裁剪后的重要性比率当作停止梯度的权重，直接乘 `log πθ`：

```text
r̂_i,t = min(r_i,t, 1 + ε_high^IS)
L_CISPO = - Σ_i,t mask_i,t · sg(r̂_i,t) · A_i · log πθ(a_i,t|s_i,t)
          / Σ_i,t mask_i,t
```

论文式只设上界；本项目默认 `--epsilon_high 5.0` 表示 `ε_high^IS=5`，实际上界是 6，而不是 5。

### 7.3 为什么 `detach` 很重要

代码：

```python
coefficient = token_ratio.clamp(max=1.0 + cispo_epsilon_high).detach()
per_token_policy = -(coefficient * token_advantages * safe_current)
```

`detach` 后 ratio 只负责对采样校正加权，梯度从 `log πθ` 直接传播：

```text
∂L/∂logπθ = -sg(r̂)A
```

若不 detach，梯度还会穿过 `exp(current-old)`，目标不再是论文的 clipped IS-weight policy gradient。

### 7.4 为什么采用 global token mean

CISPO 的分母是所有有效 response token 数：

```text
Σ token loss / Σ valid tokens
```

因此长序列按 token 数贡献更多梯度，而不是每序列等权。

### 7.5 DDP 中容易犯的错误

假设 rank 0 有 100 个有效 token，rank 1 有 300 个。每个 rank 都做 local mean，再让 DDP 等权平均，会变成：

```text
0.5 * mean(rank0) + 0.5 * mean(rank1)
```

正确 global token mean 应为：

```text
(sum(rank0) + sum(rank1)) / 400
```

项目的 `distributed_token_mean_scale` 返回：

```text
world_size × local_token_count / global_token_count
```

在 DDP 自动平均梯度后，结果严格等价于全局 token mean。

### 7.6 预期改善和风险

预期：减少 clip 饱和后零梯度 token，提高样本利用率和多 policy epoch 下的有效更新。

风险：裁剪 IS 权重本身引入 bias；没有下界和持续有梯度不代表可以无限提高学习率；CISPO loss 含加权 `logπ` 的绝对尺度，不能与 GRPO policy loss 数值横向比较。

验证：联合看 Accuracy、Reward、ratio、KL、clip fraction、gradient norm 和达到同等验证准确率需要的 optimizer/rollout 成本。

### 7.7 CISPO objective 与完整 recipe

MiniMax-M1 的完整训练还使用 dynamic sampling 和 length penalty。公平研究应区分：

```text
CISPO objective only
CISPO + Dynamic Sampling + Length
```

否则不能把组合收益全部归因于 CISPO loss。

---

## 8. 问题三：有效样本少、探索受限、长度不稳 → DAPO

DAPO 是四项相互配合的 recipe，不只是“上界从 0.2 改到 0.28”。

### 8.1 子问题 A：全对/全错组浪费 rollout

#### 方法：Dynamic Sampling

对 strict 二值 task success，只保留：

```text
0 < success_count < G
```

这样的 prompt 同时包含正负 advantage，位于模型当前能力边界。

代码：

```python
group_mask = effective_group_mask(task_success, num_generations)
```

`train_agent.py` 把有效 prompt group 放入 `dynamic_buffer`，继续采候选，直到填满和 baseline 相同的有效 batch。DDP 用 `MIN all_reduce` 保证所有 rank 同时 ready；达到 `dynamic_sampling_rounds` 仍填不满则明确失败。

为什么不能用 shaped reward 方差筛选：格式分可能让所有最终答案都错的组产生 reward 方差，看似有效，实际优化的不是 correctness。

预期改善：降低 `zero_variance_group_rate`、提高每个 optimizer step 的有效梯度密度。

代价：改变 prompt 训练分布，并消耗额外 candidate prompts、trajectory tokens、工具调用和墙钟。必须报告 `dynamic_acceptance_rate` 及环境预算。

### 8.2 子问题 B：低概率优质 token 上升空间太小

#### 方法：Clip-Higher

```text
clip(r, 1-ε_low, 1+ε_high)
ε_low=0.2, ε_high=0.28
```

对正 advantage token，右侧更宽，允许探索得到的低概率优质 action 增长更多；左侧仍保持较强限制，避免负 advantage action 概率被一步压得过低。

代码与 GRPO 相同，只是：

```python
low, high = dapo_epsilon_low, dapo_epsilon_high
```

预期改善：更好保留探索和策略熵。验证应看低概率 token 分桶、entropy、ratio upper-tail、clip fraction，而不只是总 Reward。

### 8.3 子问题 C：每序列等权忽略长推理中的 token 信息

#### 方法：Token-level Policy Gradient Loss

GRPO：

```text
(1/N) Σ_i [(1/|y_i|) Σ_t L_i,t]
```

DAPO：

```text
Σ_i,t L_i,t / Σ_i |y_i|
```

区别可以用两条序列说明：第一条 10 tokens，第二条 100 tokens。

- sequence-first：两条各占 50%；
- token-mean：第一条约 9.1%，第二条约 90.9%。

预期：长的有效推理提供更多 token 级学习信号。

风险：可能鼓励长回答，也会放大长错误轨迹，因此必须与长度惩罚、平均/P95 长度和长度分桶准确率一起分析。

### 8.4 子问题 D：硬截断产生突变奖励

#### 方法：Soft Overlong Punishment

设最大 action 长度 `L_max`，线性缓冲 `L_cache`：

```text
R_len(y) = 0                                           if |y| ≤ L_max-L_cache
         = -(|y|-(L_max-L_cache))/L_cache             in buffer
         = -1                                          if |y| ≥ L_max
```

代码：

```python
soft_overlong_penalty(lengths, max_length, cache_length)
```

它在即将超限时平滑给出信号，减少“只差一个 token 就从无惩罚跳到失败”的 reward noise。长度只统计 assistant action，tool observation 不计入。

预期：降低 P95/unfinished 和边界处 reward 波动，同时尽量保留必要推理。

风险：系数过大会造成过早停止和 Accuracy 下降。

### 8.5 什么才叫完整 DAPO

在本项目中需要同时具备：

```text
loss_type=dapo
非对称 clip
global token-mean
dynamic_sampling
soft overlong punishment
```

只使用 `loss_type=dapo` 而不打开动态采样和 overlong 参数，只能称为 DAPO objective/部分消融。

### 8.6 怎样归因结果改善

逐项消融：

```text
GRPO
→ + asymmetric clip
→ + token-level loss
→ + soft overlong
→ + dynamic sampling
```

每次只改一项。主表可比较完整 recipe，但必须用消融表说明收益来自哪里。

---

## 9. 问题四：Token ratio 与序列 reward 粒度不一致 → GSPO

### 9.1 观察到的问题

RLVR 通常在完整答案结束后给一个序列 reward，但 GRPO 在每个 token 上分别计算 ratio 和 clip。长序列可能有少量 token ratio 剧烈波动，导致大量 token 被独立裁剪；MoE 中路由变化还会进一步增加单 token likelihood 波动。

### 9.2 方法：序列级重要性比率

先平均整条 action 序列的 log-ratio，再取指数：

```text
s_i(θ) = exp(
    (1/|y_i|) Σ_t [logπθ(a_i,t|s_i,t) - logπold(a_i,t|s_i,t)]
)
```

这等于 token ratio 的几何平均，而不是算术平均。

### 9.3 为什么要长度归一化

若直接使用完整 sequence probability ratio：

```text
exp(Σ_t log-ratio_t)
```

长度越长，乘积越容易指数爆炸或衰减，不同长度无法共享 clip 范围。除以 action token 数后，ratio 表示平均每 token 的概率变化尺度。

### 9.4 数值例子

token ratios 为 `[0.5, 2.0]`：

```text
算术平均 = 1.25
几何平均 = sqrt(0.5 × 2.0) = 1.0
```

它们的含义完全不同。GSPO 使用几何平均，因为 sequence likelihood 来自 token probability 的乘积。

### 9.5 GSPO 目标

```text
L_GSPO = -mean_i min(
    s_i A_i,
    clip(s_i, 1-ε_low, 1+ε_high) A_i
)
```

reward、advantage、ratio、clip 和 loss 都统一在序列粒度。

### 9.6 代码分支

```python
seq_log_ratio = (log_ratio * mask).sum(1) / token_counts.clamp(min=1)
seq_ratio = torch.exp(seq_log_ratio)
clipped_seq_ratio = seq_ratio.clamp(
    1.0 - gspo_epsilon_low,
    1.0 + gspo_epsilon_high,
)
```

之后只在有效序列上求 mean。返回的 `clip_fraction` 也是“被裁剪序列比例”，不能与 GRPO 的“被裁剪 token 比例”按完全相同统计单位解释。

### 9.7 为什么默认 clip 很小

项目默认 `3e-4/4e-4` 来自论文的大规模 MoE 实验。Sequence ratio 是平均 log-ratio 的指数，分布尺度与单 token ratio 不同，因此不能机械沿用 GRPO 的 0.2。

对 64M MiniMind 这只是复现起点，不是已证明最优值。应在验证集做 clip grid，并同时看 clip fraction、KL、Accuracy 和 seed 方差。

### 9.8 预期改善和边界

预期：减少个别 token/路由波动造成的训练不稳定，使优化粒度与序列 reward 对齐，尤其可能有利于长推理和 MoE。

边界：当前每条轨迹只有一个 advantage，不能解决细粒度 credit assignment；小模型未必出现大规模 MoE 相同问题；sequence clip 过窄可能让几乎所有序列被裁剪。

---

## 10. KL：约束谁偏离谁

### 10.1 为什么监控 KL

策略可能为了 verifier 的局部漏洞快速偏离 SFT 分布，表现为语言退化、格式异常、响应变长或熵坍缩。冻结 reference policy 提供行为锚点。

### 10.2 为什么旧的 signed 差值不好解释

单样本 `logπ_ref - logπ_current` 可正可负，有限采样均值也可能为负，不能直接当作非负 KL 数字。

本项目使用 k3 estimator：

```text
δ = clip(logπ_ref - logπ_current, -20, 20)
KL_k3 = exp(δ) - δ - 1 ≥ 0
```

数学原因是对任意 `z`，`exp(z) ≥ 1+z`。

这里的 ±20 是数值保护，不是 PPO/GSPO 的策略裁剪；`exp(20)≈4.85×10^8`，仍足以暴露严重尾部偏移，同时避免 FP32 溢出以及 `beta=0 × inf → NaN`。因此报告中应称为“截断的 sampled k3 诊断”，不要冒充解析 KL。

代码：

```python
positive_kl_estimate(current_logps, reference_logps)
```

只在 action mask 上平均。

### 10.3 监控和惩罚的区别

- `beta=0`：KL 只记录，不进入训练目标；
- `beta>0`：`beta * masked_mean(k3)` 加入 loss。

DAPO、CISPO 论文式复现通常以 `beta=0` 为主；加 KL 应标为独立消融。

固定集 `kl_k3` 也是截断的 sampled estimator，不是对完整 vocabulary 分布精确求和的解析 KL。

---

## 11. 为什么需要多个 Policy Update Epoch

### 11.1 问题

Torch rollout 刚由当前模型生成，第一次 forward：

```text
current ≈ old → ratio≈1 → clip 几乎不触发
```

如果每批只做一次 backward，四种 clip 的差异可能看不出来。

### 11.2 方法

```text
--policy_update_epochs 2~4
```

固定本批 old log-prob，重复使用同一轨迹更新。第二次起 current 与 old 分离，importance ratio、clip 和不同目标开始产生实际差异。

### 11.3 风险

复用太多次会增加 policy lag：旧轨迹越来越不代表当前策略，clip fraction 和 bias 上升。必须把 policy epoch 作为消融，并报告 ratio/KL。

SGLang 训推分离还必须按 `rollout_sync_interval` 同步更新后的权重；主算法对照建议设 1。

---

## 12. `compute_policy_loss` 输入输出逐项解释

文件：`trainer/policy_optimization.py`。

### 12.1 输入 tensor

假设 `N=B×G`，序列对齐后 next-token 长度为 `T`：

| 参数 | 形状 | 含义 | 是否梯度 |
|---|---|---|---:|
| `current_logps` | `[N,T]` | 当前模型对已采 token 的 log-prob | 是 |
| `old_logps` | `[N,T]` | behavior policy rollout 时的 log-prob | 否 |
| `reference_logps` | `[N,T]` | 冻结 SFT reference log-prob | 否 |
| `advantages` | `[N]` 或 `[N,T]` | 轨迹或 token advantage | 否 |
| `completion_mask` | `[N,T]` | assistant action=1，其余=0 | 否 |

### 12.2 公共预处理

```python
mask = completion_mask.to(current_logps.dtype)
token_counts = mask.sum(dim=1)
valid_rows = token_counts > 0
log_ratio = current.float() - old.float()
token_ratio = exp(log_ratio)
```

轨迹 advantage `[N]` 会 broadcast 到 `[N,T]`；若传 token advantage，则 GSPO 当前实现会对有效 token 平均得到 sequence advantage。真正使用细粒度 token advantage 时应进一步实现 GSPO-token，而不是简单平均。

### 12.3 四个分支

```text
cispo  → detached clipped IS coefficient × advantage × current logp
grpo   → token PPO symmetric clip → sequence-first mean
dapo   → token PPO asymmetric clip → global token mean
gspo   → sequence geometric ratio/clip → sequence mean
```

### 12.4 返回诊断

`PolicyLossOutput`：

- `loss`：policy + beta·KL；
- `policy_loss`：纯策略项；
- `kl_loss`：进入优化的 KL 项；
- `approx_kl`：用于记录的 sampled k3；
- `clip_fraction`：有效 token 或序列的裁剪比例；
- `ratio_mean/std`：有效 ratio 分布；
- `token_count`：有效 action token 数。

注意不同算法的 `policy_loss` 公式和 reduction 不同，绝对数值不可直接横比；最终比较 Accuracy、Reward、KL、长度和稳定性。

---

## 13. 从 `rl_train_epoch` 到参数更新

### 13.1 得到 reward 和 advantage

```python
reward_output = calculate_rewards(..., return_details=True)
rewards = reward_output.rewards
advantages = group_relative_advantages(rewards, args.num_generations)
```

DAPO/CISPO 完整 recipe 可在此处用二值 `task_success` 动态筛组。

### 13.2 打包序列

每条样本构成：

```text
input_ids = prompt_ids + response_ids
action mask = zeros(prompt) + response_mask
old logps = zeros(prompt next-token slots) + response_old_logps
```

右侧 PAD 统一 batch 长度。每个 assistant turn 的 EOS 都留在 action mask；tool observation mask=0，不能在第一轮 EOS 后截掉后续 assistant action。

### 13.3 Reference forward

冻结 `ref_model` 计算 reference log-prob。它不需要梯度，但需要和当前模型看到完全相同的 `input_ids` 与 attention mask。

### 13.4 Policy forward 和 loss

```python
for policy_epoch in range(args.policy_update_epochs):
    res = model(input_ids, attention_mask=full_mask)
    per_token_logps = ...
    policy_output = compute_policy_loss(...)
    loss.backward()
```

MoE 模型另加 `aux_loss`；CISPO/DAPO 在 DDP 下应用 global token-mean scale。

### 13.5 Optimizer 与 accumulation

```python
loss = total_loss / accumulation_steps
loss.backward()

if micro_step % accumulation_steps == 0:
    clip_grad_norm_
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

RL 默认 `weight_decay=0`。原因是零 advantage 时 policy gradient 为零，但 AdamW 的 decoupled weight decay 仍可能改变参数，让策略在没有 reward 信号时漂移。

### 13.6 同步 rollout policy

更新完成后：

```python
rollout_engine.update_policy(model)
```

Torch engine 只是更新引用；SGLang 会保存并通知远程服务加载权重。同步失败必须中止，不应继续混用未知 policy version。

---

## 14. 如何运行四种算法

先进入 trainer：

```bash
cd /path/to/minimind/trainer
```

共同参数：

```bash
COMMON="--from_weight full_sft \
--data_path ../dataset/agent_rl_math_train.jsonl \
--epochs 1 --batch_size 2 --num_generations 8 \
--policy_update_epochs 4 --rollout_sync_interval 1 \
--beta 0 --weight_decay 0 --dtype bfloat16 \
--reward_mode strict --use_reward_model 0 \
--max_gen_len 768 --max_total_len 2500"
```

GRPO：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type grpo \
  --save_weight agent_grpo_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/grpo_s42.jsonl
```

CISPO objective：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type cispo \
  --epsilon_high 5.0 \
  --save_weight agent_cispo_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/cispo_s42.jsonl
```

DAPO 完整 recipe：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type dapo \
  --dapo_epsilon_low 0.2 \
  --dapo_epsilon_high 0.28 \
  --dynamic_sampling \
  --dynamic_sampling_rounds 10 \
  --overlong_cache_len 128 \
  --overlong_penalty_coef 1.0 \
  --save_weight agent_dapo_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/dapo_s42.jsonl
```

GSPO：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type gspo \
  --gspo_epsilon_low 0.0003 \
  --gspo_epsilon_high 0.0004 \
  --save_weight agent_gspo_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/gspo_s42.jsonl
```

完整实验直接从仓库根目录运行：

```bash
MM_GPUS=1 MM_SEEDS="42 43 44" bash scripts/run_rlvr_ablation.sh
```

---

## 15. 公平对比应该怎样设计

### 15.1 固定项

- 同一个 `agent_sft` 冷启动 checkpoint；
- 同一 train/eval manifest；
- 同一 model、dtype、temperature、`G`、batch 和 max length；
- 同一 strict verifier；
- 同一 optimizer update 和 policy epoch 预算；
- 42/43/44 等独立训练 seed；
- 固定评测解码设置。

固定集还要在完全相同的题目和解码 seed 上评测未做 RL 的 `agent_sft`，否则四算法只能相互排序，不能声称相对共同初始化取得了提升。

### 15.2 同时报告两种预算

```text
训练预算：optimizer steps、训练 action tokens、GPU-hours
环境预算：candidate prompts、rollout trajectories/tokens、tool calls、wall time
```

DAPO dynamic sampling 为填满有效 batch 会额外 rollout。若只按 optimizer step 比较，会隐藏环境成本。

当前 3/10-update 的 pilot 与小预算正式消融使用 `--save_resume 0`：最终半精度模型、逐步 metrics 和评测轨迹都会保留，但不为每个短 run 持久化约数百 MiB 的 AdamW 一、二阶矩。模型文件采用临时文件加原子替换；若短 run 中断则整次重跑。Pretrain、SFT、KD、DPO 等长任务仍保留完整 resume。

### 15.3 数学与 Tool-Use 分开

数学：

- exact/equivalence accuracy；
- pass@k；
- Reward；
- sampled KL；
- 平均/P95 响应长度。

Tool-Use：

- trajectory task success；
- format/schema/argument valid；
- execution success；
- final answer；
- 平均轮数、工具调用、unfinished；
- 单成功任务环境成本。

不要把两类数据混成一个 Reward 后得出“推理和工具能力都提升”。

---

## 16. 消融矩阵：区分方法收益

### 16.1 DAPO

```text
A0 GRPO
A1 A0 + asymmetric clip
A2 A1 + global token mean
A3 A2 + soft overlong
A4 A3 + dynamic sampling
```

### 16.2 CISPO

```text
B0 GRPO
B1 CISPO objective
B2 B1 + dynamic sampling
B3 B2 + length penalty
B4 upper clip grid
B5 beta 0 vs KL penalty
```

### 16.3 GSPO

```text
C0 GRPO token ratio
C1 sequence ratio/clip only
C2 policy epochs 1/2/4
C3 sequence clip grid
C4 dense vs MoE
```

每次只改变一项，验证集选超参数，测试集只在最终方案上使用一次。

---

## 17. 训练稳定性到底是什么

不能只用“loss 更平滑”定义稳定。

### 17.1 数值稳定性

- NaN/Inf；
- gradient norm spike；
- optimizer step 是否成功；
- FP32 log-softmax；
- 首次 rollout log-prob invariant。

### 17.2 策略稳定性

- sampled `kl_k3`；
- ratio mean/std；
- clip fraction；
- entropy 或概率集中程度；
- 平均/P95 长度；
- unfinished。

### 17.3 优化信号稳定性

- group reward std；
- zero-variance group rate；
- DAPO dynamic acceptance；
- advantage std；
- CISPO clipped coefficient 分布。

### 17.4 泛化稳定性

- 最后 N 次固定验证的时间波动；
- 三个独立训练 seed 的均值和 sample std；
- 不同长度/难度/工具类型分桶；
- 失败案例类型是否一致。

一个算法 Reward 更高但三 seed 中只有一个成功，不能称为更稳定。

---

## 18. 日志诊断决策表

| 现象 | 可能根因 | 检查 | 处理 |
|---|---|---|---|
| 首次 ratio 不为 1 | old/current 不对齐 | `rollout_logprob_mae` | 停训，修 tokenizer/mask/context |
| ratio 永远为 1 | 只有一次 on-policy update | policy epochs | 设 2～4，检查 old 未被覆盖 |
| clip fraction≈1 | LR/epoch 太大或 clip 太窄 | ratio 分布 | 降 LR/epoch，验证集调 clip |
| clip fraction≈0 | 步长过小或阈值太宽 | KL/梯度 | 增 policy epoch 或重新审计公式 |
| KL 爆炸 | 策略偏移 | ref、LR、reward | 降步长，检查 reward hacking |
| KL=0 且 Accuracy 不变 | 没有有效更新 | advantage/grad | 检查全零组、mask、optimizer |
| zero-variance 高 | 题目全会/全不会 | success 分布 | 调难度或动态采样 |
| DAPO acceptance 低 | 能力边界不匹配 | candidate trace | 课程化任务，审计 verifier |
| CISPO loss 数值很大 | 公式尺度不同 | Accuracy/KL | 不横比绝对 loss |
| GSPO 全被 clip | clip 沿用论文但不适配小模型 | seq ratio | 验证集做 grid |
| 长度持续上涨 | token mean/奖励诱导 | P95/overlong | 加软长度消融，检查准确率 |
| 零 advantage 仍漂移 | AdamW weight decay | weight_decay | RL 主对照设 0 |

---

## 19. 如何证明“结果改善”

### 19.1 不合格证据

- 单次训练 Reward 上升；
- 上游论文的模型结果；
- CPU 随机小模型 smoke test；
- 最好 seed；
- 不同算法用了不同初始 checkpoint 或 rollout 预算；
- 测试集调 clip 后再报告同一个测试集。

### 19.2 合格证据链

```text
问题：GRPO 存在零方差组、clip 梯度饱和或 token ratio 波动。
方法：按诊断引入 DAPO DS、CISPO 或 GSPO，保持其余变量不变。
机制指标：zero variance、clip fraction、ratio、KL 是否按预期变化。
能力指标：固定测试集 Math Accuracy / Tool Success 相对 baseline 的变化，包括无变化。
稳定指标：三训练 seed 均值±标准差、失败率和长度尾部改善。
成本指标：rollout tokens、工具调用、GPU-hours 没有被隐藏。
```

### 19.3 本项目可用的真实简历口径

> 在相同 Agent SFT 初始化、10 个 accepted groups 和 3 个训练随机种子下，统一复现 GRPO、CISPO、DAPO、GSPO；DAPO 在组合 Tool-Use 中将保留组的零方差率降至 0%，但平均消耗 103.7 个候选组和 39,039 个 action tokens，约为普通算法的 10.4/10.9 倍。固定集数学准确率为 48.4375%、工具任务成功率为 5.7292%，四算法相对 Agent SFT 的 delta 均为 0，定位到当前 `10 updates / 1e-7` 小预算只完成机制和成本验证，未产生可见泛化收益。

这段表述刻意同时报告有效机制、额外成本和负结果。若简历空间不足，可删小数位，但不能删除“相对 baseline 无提升”的结论，也不能把 DAPO 筛选后 48.33% 的训练组准确率写成测试准确率。

---

## 20. 论文结果与本项目结果的边界

论文结果只能用于解释为什么选择方法：

- DAPO 作者报告了 Clip-Higher、soft overlong、token-level loss 和 dynamic sampling 的逐项消融；
- MiniMax-M1 作者报告 CISPO 在其大模型设置中的训练效率；
- GSPO 作者报告其在大规模 MoE 推理训练中的稳定性。

这些结果不能外推成 64M 模型的个人实验结果。本项目已经真实验证：四种 tensor objective 可前后向、31 项回归测试、多轮训练循环、old/current 对齐、超长轨迹保护，以及数学/Tool-Use 各 4 算法 × 3 训练 seed × 3 解码 seed 的固定集评测。两套固定集四算法对共同 Agent SFT 的任务指标 delta 都为 0；DAPO 在 Tool 训练中用约 10.4 倍候选组消除了保留组的零方差，但没有转化为当前预算下的 holdout 提升。最终结论只取机器生成的聚合 CSV，不用论文或训练批次数字替代。

---

## 21. 高频面试深挖

### 21.1 GRPO 为什么不需要 critic？

用同 prompt 的 `G` 个 reward 做组内 baseline。优势是去掉 value model，代价是更多 rollout、组内依赖和全对/全错零梯度。

### 21.2 为什么 old policy 和 reference policy 不能共用？

old 描述数据采样分布，随 rollout 批次变化；reference 是固定行为锚点。前者用于 ratio，后者用于 KL。

### 21.3 为什么 advantage 标准化用 population std？

代码使用 `unbiased=False`，对组内实际 `G` 个样本直接计算 population std，避免小组 sample correction 改变尺度；加 `eps` 处理零方差。

### 21.4 CISPO 为什么裁上界但仍有梯度？

裁的是 detached sampling coefficient，梯度通过 `logπθ` 传播；PPO clip 分支对已饱和正 advantage token 可能是常数。

### 21.5 Dynamic sampling 有 bias 吗？

有。它改变 prompt 训练分布，聚焦能力边界。必须保留原始分布固定测试，并报告 acceptance 和额外采样成本。

### 21.6 Token-level loss 是否一定更好？

不一定。它让长序列贡献更多 token 梯度，可能强化有效长推理，也可能鼓励冗长；必须联合 soft overlong 和长度分层评测。

### 21.7 GSPO 为什么用几何平均？

序列概率是 token 概率乘积；对 log-ratio 求平均再 exp 等价于 token ratio 几何平均，并消除总长度的指数尺度。

### 21.8 GSPO 为什么可能适合 MoE？

MoE 路由切换会放大个别 token likelihood 波动；序列聚合对局部波动更不敏感。但 MiniMind 小 MoE 必须实测，不能引用大模型结论替代。

### 21.9 sampled k3 是精确 KL 吗？

不是。它是在采样 action 上的非负 Monte Carlo estimator。精确 token KL 需要对完整 vocabulary 分布求和。

### 21.10 Reward 上升但 Accuracy 不升怎么查？

先拆 reward components 和 verifier false positive，再检查长度、格式、数据泄漏和训练/测试分布。训练 Reward 不是最终能力。

---

## 22. 从入门到精通的练习路线

### 入门

1. 手算两组 strict reward 的 advantage；
2. 在 Notebook 中改变 current log-prob，观察四算法 ratio 和梯度；
3. 验证 current=old 时 ratio=1；
4. 对比 sequence-first 和 token-mean 权重。

### 熟练

1. 画出正负 advantage 下 clip surrogate 曲线；
2. 添加 CISPO coefficient histogram；
3. 跑 DAPO 五阶段消融；
4. 对 GSPO 做 clip grid，并按响应长度分桶；
5. 从 JSONL 聚合最后 N step 时间波动。

### 精通

1. 为 token-level advantage 实现并测试 GSPO-token；
2. 在 DDP 两 rank 不同 token 数下验证 global token-mean 梯度；
3. 增加 MoE expert load/router entropy/routing switch 指标；
4. 实现严格 policy version 与 trajectory version 校验；
5. 对 Math/Tool-Use 各跑三训练 seed，做 paired bootstrap 和失败案例归因。

完成后，读者应能从一条 trajectory 一直追踪到某个 parameter gradient，并能解释最终指标变化究竟来自 reward、采样、loss、clip、长度还是系统成本。
