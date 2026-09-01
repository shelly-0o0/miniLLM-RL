# 轻量语言模型全阶段训练与 Agentic RLVR：简历对齐技术报告

> 实验日期：2026-08-29 至 2026-08-30  
> 实验基线：`393e387e9ad99f0f04c296e4c5e7353f4444629f`  
> 硬件：单张 NVIDIA RTX 4090 23.52 GiB  
> 软件：PyTorch 2.6.0+cu124、CUDA 12.4、BF16  
> 验收：31/31 单元测试通过，7 个 pipeline markers 全部为 true，证据索引 `missing=[]`

> 2026-09-01 后续优化：RL 不增长的根因、核心代码修复、Tool 冷启动和新增严格 DAPO 结果见 [09_RL_OPTIMIZATION_FOLLOWUP.md](./09_RL_OPTIMIZATION_FOLLOWUP.md)。本文保留初始 Formal 的负结果作为对照。

本报告以简历中的四条项目描述为目录，将每一项拆成：**问题、原理、核心代码、实验设计、真实结果、结论边界和面试追问**。报告只引用本项目实际产生的日志、checkpoint、逐轨迹记录、CSV/JSON 与固定集评测；论文数字和上游仓库已有能力不冒充个人实验结果。

---

## 0. 先给结论：四条简历要求支持到什么程度

| 简历主张 | 当前状态 | 能说的结论 | 不能说的结论 |
|---|---|---|---|
| Decoder-only、RMSNorm、RoPE/YaRN、GQA、KV Cache、SwiGLU、Flash、MoE | 已完成代码和系统验证 | 64M Dense 跑通；4096-token RoPE/YaRN 可运行；GQA 理论 KV Cache 减少 50%；指定 BF16 shape 下 Flash SDPA 加速 6.55×；MoE 路由与梯度正常 | YaRN 提升了长文本任务准确率；部分训练 MoE 比 Dense 更快或更准 |
| Pretrain、SFT、LoRA、KD、AMP、梯度累积、Assistant-only mask | 已完成训练与评测 | Pretrain/SFT 完成；LoRA 0.61% 参数使医疗 holdout PPL 下降 12.15%；KD 做到同初始化同预算对照；mask 审计零错位 | 当前 KD 有正收益；BF16/累积带来多少加速；已经实测多卡扩展 |
| DPO 偏好优化 | 已完成训练、单测和独立 holdout | prompt-grouped 切分零重合；策略/reference 目标正确；形成公平负对照 | 当前 DPO 提高了偏好准确率；DPO 能替代在线工具环境 |
| Learning-rate Warmup | **尚未实现** | 当前真实实现是 cosine decay | 不能把 cosine decay 写成 Warmup |
| GRPO/CISPO/DAPO/GSPO 与核心机制 | 已完成实现、单测和三 seed 训练 | DAPO 动态采样在 Tool 训练中把保留组零方差率降为 0%，并量化额外 rollout 成本 | 当前小预算 RL 提升了固定集准确率、Reward 或总体跨 seed 稳定性 |
| Agentic RL、数学/Tool-Use 评测、可验证奖励 | 已完成 | 多轮轨迹、精确 action mask、verifier、固定集、逐轨迹证据和指标聚合均可复现；Agent SFT 冷启动有显著 readiness 提升 | 不能把 Agent SFT 提升归因于后续 RL；当前没有独立的 token 截断率字段 |

最重要的面试原则是区分三层事实：

1. **实现正确**：代码、张量和单元测试符合算法定义；
2. **机制生效**：例如 DAPO 确实过滤零方差组；
3. **泛化改善**：必须在共同未见固定集上优于 baseline。

本项目的四算法已经达到前两层，但在当前 `10 accepted groups / lr=1e-7` 预算下没有达到第三层。

---

# 第一部分：构建轻量 Decoder-Only 语言模型

## 1. Decoder-Only Transformer 总体结构

### 1.1 为什么选择 Decoder-Only

Decoder-Only 模型使用因果注意力，第 `t` 个 token 只能访问位置 `≤t` 的上下文，天然对应自回归语言建模：

$$
\mathcal{L}_{LM}=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
$$

一个 Pre-Norm Block 的主干是：

```text
token ids
  → Embedding
  → [RMSNorm → Causal Self-Attention → Residual
     RMSNorm → SwiGLU / Sparse MoE → Residual] × L
  → RMSNorm
  → LM Head
  → next-token logits
```

模型配置为 hidden size 768、8 层、8 个 Query heads、4 个 KV heads，Dense 总参数为 63,912,192。

### 1.2 核心 Block 代码

代码位置：[`model/model_minimind.py`](../../model/model_minimind.py)

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Dense 使用 SwiGLU；MoE 分支把 FFN 替换为稀疏专家层。
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None):
        # Pre-Norm：先归一化再进入子层，比 Post-Norm 更利于深层梯度传播。
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = hidden_states + residual

        # 第二个残差分支可以选择 Dense SwiGLU 或 Sparse MoE。
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
```

面试时应说明：模型的基本模块来自上游轻量模型实现；本项目的新增工作主要是机制审计、KV Cache 修复/一致性验证、Flash kernel profiler、MoE 路由审计、系统基准以及后训练框架。

---

## 2. RMSNorm

### 2.1 原理

RMSNorm 不减去均值，只使用均方根归一化：

$$
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
$$

相对 LayerNorm，它去除了中心化步骤。这里在 FP32 中计算平方均值和倒平方根，再转换回输入 dtype，降低 BF16/FP16 下的数值误差。

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 归一化计算临时提升到 FP32；输出恢复为原始混合精度 dtype。
        normed = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (self.weight * normed).type_as(x)
```

**实验能证明什么：** RMSNorm 数值单测通过，Dense Pretrain/SFT 与 MoE 短程训练均未出现非有限梯度。这里没有 RMSNorm 与 LayerNorm 的独立质量/吞吐消融，因此不报告加速百分比。

---

## 3. RoPE 与 YaRN

### 3.1 RoPE 原理

RoPE 把每一对通道视作二维向量，按位置旋转 Query 和 Key。位置 `m` 与 `n` 的内积自然包含相对位置信息 `m-n`：

$$
q_m'=R_mq_m,\quad k_n'=R_nk_n,\quad
(q_m')^Tk_n'=q_m^TR_{n-m}k_n
$$

核心实现先预计算所有位置的 `cos/sin`，前向时对 Q/K 应用旋转：

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        # [x1, x2] → [-x2, x1]，对应二维旋转的正交分量。
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)
```

### 3.2 YaRN 原理

训练长度之外直接使用原 RoPE 高频分量可能导致相位变化过快。YaRN 对不同频率采用插值/外推混合，通过 ramp 函数平滑调整频率：

$$
f_i'=f_i\left[(1-\gamma_i)+\frac{\gamma_i}{s}\right]
$$

其中 `s` 是扩展倍率，`γ_i` 在低频和高频区间平滑变化。项目代码根据 `original_max_position_embeddings`、`factor`、`beta_fast` 和 `beta_slow` 计算 ramp。

### 3.3 真实结果

4096-token prompt + 32-token decode：

| 位置方案 | Cached tok/s | Uncached tok/s | Cached peak MiB | Uncached peak MiB |
|---|---:|---:|---:|---:|
| RoPE | 130.97 | 106.81 | 571.04 | 542.54 |
| YaRN | 122.15 | 103.53 | 571.04 | 542.54 |

**结论：** 两种路径都能完成 4096-token 前向和生成，证明长位置机制可运行；YaRN 在该基准没有吞吐优势，而且没有 Needle/Passkey/长文 PPL 结果，所以不能声称长文本任务质量提高。简历应写“验证 4096-token 长序列执行路径”，不应写“YaRN 显著提高长文准确率”。

---

## 4. GQA 与 KV Cache

### 4.1 GQA 原理

Multi-Head Attention 为每个 Query head 保存独立 K/V；GQA 让多组 Query head 共享较少的 KV heads。本模型配置为：

```text
num_attention_heads = 8
num_key_value_heads = 4
n_rep = 8 / 4 = 2
```

每个 KV head 逻辑上服务两个 Query heads。KV Cache 理论内存与 KV head 数近似成正比，因此相对 8-KV-head MHA 降低 50%。

### 4.2 为什么 `expand + reshape` 不是物理复制缓存

```python
def repeat_kv(x, n_rep):
    # 输入缓存保持 [B, S, H_kv, D]，这是实际保存的紧凑表示。
    if n_rep == 1:
        return x
    # expand 创建 stride view；reshape 后提供注意力所需的逻辑 head 形状。
    # 关键点：past_key_value 保存发生在 repeat_kv 之前。
    return (
        x[:, :, :, None, :]
        .expand(x.size(0), x.size(1), x.size(2), n_rep, x.size(3))
        .reshape(x.size(0), x.size(1), x.size(2) * n_rep, x.size(3))
    )
```

### 4.3 KV Cache 核心路径

```python
# 新 token 的 K/V 与历史缓存按序列维拼接。
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)

# 缓存仍保持紧凑 GQA 形状；repeat 只用于本次 attention 计算。
present_kv = (xk, xv) if use_cache else None
xk = repeat_kv(xk, self.n_rep)
xv = repeat_kv(xv, self.n_rep)
```

增量生成第 `t` 步只计算新 token 的 Q/K/V，不再重新计算全部前缀。项目用两个测试验证：

1. cached 单步 logits 与 full-forward 对应位置 logits 一致；
2. cached 与 uncached greedy decode token 完全一致。

在 4096+32 tokens 的 BF16 理论计算中，GQA KV Cache 为 48.375 MiB，MHA 为 96.75 MiB，降低 50%。这是结构推导值，不应伪装成 CUDA allocator 的独立实测差值。

---

## 5. PyTorch SDPA Flash Backend

### 5.1 为什么不能只看函数名

`scaled_dot_product_attention` 是统一调度接口；是否进入 Flash kernel 取决于 GPU、dtype、head dimension、mask 和 PyTorch 版本。调用 SDPA 不等于实现了 FlashAttention-2。

```python
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=sdpa_mask,
    dropout_p=self.dropout if self.training else 0.0,
    is_causal=False if cached_single_token else True,
)
```

项目采用两层验证：

1. 原语级强制 Flash/math backend 并比较时间、显存；
2. 对真实 64M `full_sft` 模型强制 Flash，确认 forward 成功、logits finite，并在 profiler 中命中 `aten::_scaled_dot_product_flash_attention` 与 CUDA `flash_fwd_kernel`。

### 5.2 真实结果

RTX 4090、BF16、`[B=2,H=8,S=1024,D=96]`：

| Backend | Median latency | Peak memory |
|---|---:|---:|
| Forced Flash | 0.2069 ms | 12.06 MiB |
| Forced Math | 1.3551 ms | 190.14 MiB |

该 shape 下 Flash 约加速 6.55×。这个数字只适用于记录的硬件、dtype 与 shape。

---

## 6. SwiGLU

SwiGLU 用一条分支产生门控，另一条分支产生特征：

$$
\operatorname{SwiGLU}(x)=W_{down}\left[\operatorname{SiLU}(W_{gate}x)\odot(W_{up}x)\right]
$$

```python
class FeedForward(nn.Module):
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))  # SiLU 门控
        value = self.up_proj(x)                # 候选特征
        return self.down_proj(gate * value)    # 逐元素调制后投影回 hidden size
```

它替代传统 `ReLU(Wx)` FFN。本项目验证其训练/推理路径，但没有 ReLU/GELU 对照，因此不宣称单独的质量收益。

---

## 7. Top-1 稀疏 MoE

### 7.1 原理

每个 token 先通过 Router 得到专家概率，然后只发送给 Top-1 专家：

$$
p(e\mid x)=\operatorname{softmax}(W_rx),\qquad
y=\sum_{e\in\operatorname{TopK}(p)}p_eE_e(x)
$$

项目配置为 4 experts、Top-1。负载均衡辅助损失近似为：

$$
L_{aux}=\lambda E\sum_{e=1}^{E}f_ep_e
$$

其中 `f_e` 是实际路由到专家 `e` 的 token 比例，`p_e` 是平均路由概率。

```python
scores = F.softmax(self.gate(x_flat), dim=-1)      # [tokens, experts]
topk_weight, topk_idx = torch.topk(scores, k=1)   # 每 token 只选一个专家

y = torch.zeros_like(x_flat)
for expert_id, expert in enumerate(self.experts):
    mask = topk_idx == expert_id
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        # 专家只处理被路由到自己的 token，再按原位置累加回输出。
        y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)

load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
self.aux_loss = (
    load * scores.mean(0)
).sum() * self.config.num_experts * self.config.router_aux_loss_coef
```

### 7.2 真实结果与为什么后训练使用 Dense

- Dense：63.912M 总参数、63.912M active/token；
- MoE：198.417M 总参数、约 63.937M active/token；
- 总容量约为 Dense 的 3.10×，每 token 激活参数量接近 Dense；
- 固定批聚合路由 `[21.09%, 24.02%, 29.59%, 25.29%]`，CV=12.23%；
- 8 个 Router 和 96 个 Expert 梯度张量全部 finite；
- MoE checkpoint 只训练到 `8350/79390`，约为一轮的 10.52%，只能作机制验证。

1024-token prompt + 128-token decode：

| 架构 | Cached tok/s | Cached peak MiB |
|---|---:|---:|
| Dense | 130.87 | 448.46 |
| Partial MoE | 88.06 | 1212.42 |

朴素单卡 MoE 没有 expert parallel、all-to-all、capacity factor 与 fused dispatch，因此比 Dense 更慢、权重显存更高。后训练统一基于收敛更充分的 Dense SFT，目的是隔离策略算法变量，并控制 current/reference/optimizer 同驻留成本。

---

# 第二部分：全阶段训练链路

## 8. 阶段关系

```text
随机初始化
   │
   ├─ Pretrain：学习语言分布与通用表示
   │      ↓
   └─ Full SFT：学习指令与对话行为
          ├─ LoRA：低成本领域适配分支
          ├─ KD：小 Student 压缩分支
          ├─ DPO：离线 chosen/rejected 偏好优化分支
          └─ Agent SFT → GRPO/CISPO/DAPO/GSPO
```

LoRA 与 KD 是从 SFT checkpoint 分出的两条支线，不应表述成必须依次执行的单链路。

---

## 9. Pretrain

### 9.1 数据与目标

Pretrain 对纯文本添加 BOS/EOS，padding label 设为 `-100`：

```python
tokens = [bos_id] + tokenizer(text)[:max_length - 2] + [eos_id]
input_ids = tokens + [pad_id] * (max_length - len(tokens))
labels = input_ids.clone()
labels[input_ids == pad_id] = -100  # padding 不参与交叉熵
```

模型内部执行 next-token shift，即位置 `t` 的 logits 预测位置 `t+1` 的 label。

### 9.2 实验

- hidden size 768，8 层，63.912M 参数；
- sequence length 768；
- micro-batch 16，gradient accumulation 4，有效 batch 64；
- BF16；
- learning rate `5e-4`，cosine decay；
- 2 epochs，每轮 79,390 micro-batches；
- 最后一批记录 loss 1.8342。

训练 loss 只证明优化过程运行和权重可作为 SFT 初始化，不能替代独立 validation PPL。

---

## 10. SFT 与 Assistant-only Loss Mask

### 10.1 为什么只监督 Assistant

若把 system/user prompt 也放入 loss，模型会学习复述输入，而且容易记住 prompt 表达。SFT 的目标应该是：在给定上下文条件下拟合 assistant action。

$$
L_{SFT}=-\frac{1}{\sum_tm_t}\sum_tm_t\log p_\theta(y_t\mid x,y_{<t})
$$

其中 `m_t=1` 只覆盖 assistant token，其他位置 label 为 `-100`。

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)       # 默认全部忽略
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)   # assistant 内容起点
            end = find_assistant_eos(input_ids, start)
            for j in range(start, end + len(self.eos_id)):
                labels[j] = input_ids[j]   # 只打开 assistant span
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels
```

实际源码使用循环查找 assistant BOS/EOS，上述 `find_assistant_eos` 是为了展示逻辑而做的等价简写。

### 10.2 审计与结果

- Dense SFT：2 epochs，退出码 0，最后一批 loss 1.4291；
- 通用 SFT mask：128 个固定样本 mismatch=0、zero-supervision=0；
- Agent SFT mask：128 样本，9,407 supervised tokens，mismatch=0、zero-supervision=0。

审计不是只统计 label 数量，而是重新根据 chat template 计算期望 assistant spans，再逐 token 与实际 label 比较。

---

## 11. BF16 混合精度、梯度累积与稳定性处理

```python
with autocast_ctx:
    result = model(input_ids, labels=labels)
    # 除以 K，使累计 K 个 micro-batch 后的梯度近似大 batch 的均值。
    loss = (result.loss + result.aux_loss) / accumulation_steps

scaler.scale(loss).backward()

if should_update:
    # 先 unscale，再裁剪；否则裁剪的是被放大后的梯度。
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

BF16 主线中 `GradScaler` 实际关闭，因为 BF16 指数范围较大；FP16 时才需要动态 loss scaling。KD 和 RL 的 `log_softmax`、重要性比率与 KL 在 FP32 中计算，以避免低概率 token 导致溢出。

当最后一个 accumulation window 不完整时，代码把梯度乘以 `accumulation_steps/remainder` 后再更新，避免最后一个小窗口被错误缩小；checkpoint 在该更新之后原子保存。

**结果边界：** 主线以 BF16 和累积稳定完成，但没有 FP32 或无累积的同预算 A/B，不能报告“BF16 加速多少”或“梯度累积提升准确率”。

---

## 12. 学习率调度：当前是 Cosine Decay，不是 Warmup

当前真实代码：

```python
def get_lr(current_step, total_steps, lr):
    # step=0 时约为 lr；结束时为 0.1*lr。
    return lr * (
        0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps))
    )
```

所以当前实验只能写“余弦学习率衰减”。如果简历必须保留 Warmup，需要先补以下调度器，并重新产生至少一个可审计 run：

```python
def get_lr_with_warmup(step, total_steps, peak_lr, warmup_steps, min_lr_ratio=0.1):
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if warmup_steps and step <= warmup_steps:
        # 从接近 0 线性升到 peak_lr，减少随机初始化阶段的大梯度冲击。
        return peak_lr * step / warmup_steps

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
```

这段是**闭环建议，不是本次最终实验实际使用的代码**。正确补实验应比较相同初始化、数据顺序和预算下的：前 100/500 step loss 方差、gradient norm、非有限 step 数与固定 holdout loss。

---

## 13. LoRA

### 13.1 原理与实现

冻结原权重 `W`，只训练低秩增量：

$$
W'=W+\Delta W,\qquad \Delta W=BA,\quad r\ll\min(d_{in},d_{out})
$$

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()  # 初始 ΔW=0，不扰动 base 输出

    def forward(self, x):
        return self.B(self.A(x))

# 运行时：original(x) + LoRA(x)
# 部署合并：W_merged = W + B @ A
```

### 13.2 真实结果

- LoRA 参数：393,216，占完整参数 0.61%；
- 固定医疗划分：22,792 train / 2,484 holdout；
- 两模型评测使用同一批物化 token，共 785,657 assistant tokens。

| 模型 | Token-weighted loss | PPL |
|---|---:|---:|
| Full SFT base | 1.698209 | 5.464150 |
| + Medical LoRA | 1.568724 | 4.800518 |

holdout loss 下降 7.62%，PPL 下降 12.15%。它只证明该医疗文本划分上的语言建模适配，不代表医疗事实性、安全性或通用能力提高。

---

## 14. 知识蒸馏

### 14.1 目标函数

学生同时学习 hard label 和教师软分布：

$$
L=\alpha L_{CE}+(1-\alpha)T^2
\operatorname{KL}\left(p_T^T\parallel p_S^T\right)
$$

`T>1` 使分布更平滑，暴露类别之间的暗知识；`T²` 补偿温度缩放造成的梯度量级下降。

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    # 概率计算强制 FP32；teacher 完全 stop-gradient。
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(
            teacher_logits.float() / temperature, dim=-1
        ).detach()
    student_log_probs = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    kl = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="batchmean",
        log_target=True,
    )
    return temperature ** 2 * kl

loss = alpha * ce_loss + (1 - alpha) * distill_loss
```

### 14.2 公平对照

- Teacher：768/8，63.912M 参数；
- Student：512/8，30.025M 参数，参数减少 53.02%；
- CE 与 KD 使用同一个 Student 初始化、同一 50k 训练子集、相同顺序、seed 和更新预算；
- KD 配置：`α=0.5, T=1.5`；
- 独立 2,000 条 RLAIF holdout，383,926 个相同 assistant tokens。

| 模型 | Loss | PPL |
|---|---:|---:|
| Teacher 768 | 1.822442 | 6.186949 |
| Student init | 2.057247 | 7.824396 |
| CE control | 2.056352 | 7.817403 |
| KD student | 2.061657 | 7.858981 |

KD 相对 CE 的 loss 上升 0.258%、PPL 上升 0.532%，当前配置是负结果。它证明蒸馏链路和公平评测完成，不证明蒸馏带来质量收益。可能原因包括训练预算太小、`α/T` 未调优、teacher 与训练域不匹配；这些只能作为待验证假设。

---

## 15. DPO：离线偏好优化、实际问题与后续在线 RLVR

### 15.1 DPO 在训练链路中的位置

DPO（Direct Preference Optimization）接收同一个 prompt 下的优选回答 `y_w` 和劣选回答 `y_l`，直接优化策略相对冻结 reference 的偏好 margin。它位于 SFT 之后，是一条独立的离线偏好对齐支线：

```text
Full SFT checkpoint
    ├─ DPO：离线 chosen/rejected pairs，不执行环境
    └─ Agent SFT → GRPO/CISPO/DAPO/GSPO：在线 rollout + verifier
```

因此不能说 GRPO、DAPO 等“建立在 DPO checkpoint 上”。本项目四算法统一从 `agent_sft` 初始化；DPO 是用于验证另一类后训练范式的独立对照。

### 15.2 从 KL 约束 RLHF 到 DPO 分类目标

标准 KL 约束奖励最大化可写为（这里把 `β>0` 定义为 **KL 惩罚系数**）：

$$
\max_{\pi_\theta}\ \mathbb{E}_{y\sim\pi_\theta}[r(x,y)]
-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})
$$

固定一个 prompt `x`，并加入概率归一化约束 `Σ_y π(y|x)=1`。拉格朗日函数为：

$$
\mathcal{J}(\pi,\lambda)=
\sum_y\pi(y|x)r(x,y)
-\beta\sum_y\pi(y|x)\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}
+\lambda\left(\sum_y\pi(y|x)-1\right)
$$

对每个 `π(y|x)` 求偏导并令其为 0：

$$
r(x,y)-\beta\left(\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+1\right)+\lambda=0
$$

整理并归一化得到 Boltzmann 形式的最优策略：

$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)
\exp\left(\frac{r(x,y)}{\beta}\right)
$$

其中：

$$
Z(x)=\sum_y\pi_{ref}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)
$$

再取对数并反解奖励：

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}
+\beta\log Z(x)
$$

记 `C(x)=β log Z(x)`，并用参数化策略 `πθ` 表示待学习的最优策略，就得到 DPO 的隐式奖励参数化：

$$
r_\theta(x,y)=\beta\left[
\log\pi_\theta(y\mid x)-\log\pi_{ref}(y\mid x)
\right]+C(x)
$$

这里的 `r(x,y)` 是对完整回答 `y` 的**标量质量/偏好分数**：在传统 RLHF 中可由人工偏好训练的 Reward Model 给出；在数学或工具 RLVR 中也可以由答案 verifier、工具执行状态等直接给出。DPO 本身不训练或调用一个显式 Reward Model，而是用 `β(logπθ-logπref)` 表示与偏好一致的隐式奖励。绝对奖励中的 `C(x)` 无法从同一 prompt 内的排序唯一确定，但它在 chosen/rejected 奖励差中会抵消。

把这个隐式奖励代入 Bradley–Terry 偏好模型，得到 DPO loss：

$$
L_{DPO}=-\mathbb{E}_{(x,y_w,y_l)}\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]\right)
$$

直观上，它要求策略相对 reference **更多地提高 chosen，而不是只同时提高 chosen/rejected 的绝对概率**。`β` 控制偏好更新相对 reference 的尺度。

### 15.3 数据与 Assistant-only Mask

每条数据包含两条共享 prompt 的完整对话：

```json
{
  "chosen":  [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "更优回答"}],
  "rejected":[{"role": "user", "content": "..."},
              {"role": "assistant", "content": "较差回答"}]
}
```

核心数据代码：[`dataset/lm_dataset.py`](../../dataset/lm_dataset.py)

```python
chosen_ids = tokenize(apply_chat_template(chosen))
rejected_ids = tokenize(apply_chat_template(rejected))

# 与 SFT 相同：prompt 只是条件，不进入偏好序列 log-prob。
chosen_mask = generate_assistant_loss_mask(chosen_ids)
rejected_mask = generate_assistant_loss_mask(rejected_ids)

# next-token shift；mask 也同步右移。
x_chosen, y_chosen = chosen_ids[:-1], chosen_ids[1:]
x_rejected, y_rejected = rejected_ids[:-1], rejected_ids[1:]
mask_chosen = chosen_mask[1:]
mask_rejected = rejected_mask[1:]
```

实际源码中的 `generate_loss_mask` 查找每个 assistant BOS/EOS span，并把 assistant EOS 纳入统计；user/system/tool prompt 不贡献 pair log-prob。

### 15.4 核心目标代码

代码位置：[`trainer/train_dpo.py`](../../trainer/train_dpo.py)

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # [B,T] → [B]：只累计 assistant action token 的序列 log-prob。
    ref_sequence_logp = (ref_log_probs * mask).sum(dim=1)
    policy_sequence_logp = (policy_log_probs * mask).sum(dim=1)

    # DataLoader 先拼 chosen，再拼 rejected，因此按 batch 前后半切分。
    half = ref_sequence_logp.shape[0] // 2
    chosen_ref, rejected_ref = ref_sequence_logp[:half], ref_sequence_logp[half:]
    chosen_pi, rejected_pi = policy_sequence_logp[:half], policy_sequence_logp[half:]

    # 策略偏好 margin 减 reference 偏好 margin，得到隐式奖励差。
    pi_margin = chosen_pi - rejected_pi
    ref_margin = chosen_ref - rejected_ref
    implicit_reward_margin = beta * (pi_margin - ref_margin)

    # 最大化 chosen 胜过 rejected 的对数概率。
    return -F.logsigmoid(implicit_reward_margin).mean()
```

训练时一次拼接 chosen/rejected，减少前向调用次数；reference 冻结且 `no_grad`：

```python
x = torch.cat([x_chosen, x_rejected], dim=0)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)

with torch.no_grad():
    ref_logps = token_logps(ref_model(x), y)
policy_logps = token_logps(policy_model(x), y)
loss = dpo_loss(ref_logps, policy_logps, mask, beta=0.15)
```

当 `π_θ=π_ref` 时，隐式 margin 为 0，loss 应为 `-log σ(0)=log 2`。单元测试还验证：增加 chosen 相对 rejected 的 policy margin 会严格降低 loss，且梯度 finite。

### 15.5 数据泄漏与公平评测

同一个 prompt 可能有多条 preference pairs。如果逐行随机切分，同 prompt 的不同 pair 会跨入 train/eval，形成泄漏。项目先对去掉最终 assistant 的对话前缀做规范化和 SHA-256 分组，再以 prompt group 为单位切分：

- unique prompt groups：17,048；
- train pairs：15,456；
- eval pairs：1,710；
- exact prompt overlap：0；
- split seed：42。

评测固定取 1,000 个 holdout pairs。由于数据模板存在随机 empty-think 后处理，评测先物化一次 token pairs，再让 reference 和 DPO checkpoint 读取完全相同的输入，避免随机模板差异伪造提升。

评测指标：

```text
Preference Accuracy = mean(logπ(y_chosen|x) > logπ(y_rejected|x))
Implicit Reward Margin = β[(logπ_chosen-logπ_rejected)
                           -(logπref_chosen-logπref_rejected)]
DPO Holdout Loss = mean[-log sigmoid(Implicit Reward Margin)]
```

### 15.6 真实训练配置与结果

- 初始化：`full_sft_768`；
- policy/reference：均为 63.912M Dense；reference 完全冻结；
- BF16，sequence length 768；
- batch size 8，gradient accumulation 2；
- `lr=4e-8`，`β=0.15`，1 epoch，1,932 micro-batches；
- 训练退出码 0，耗时 238 秒。

| 模型 | Chosen Preference Accuracy | Mean policy pair margin | Mean implicit reward margin | Positive implicit margin | DPO loss |
|---|---:|---:|---:|---:|---:|
| Full SFT reference | 45.70% | -59.1558 | 0 | 0% | 0.693147 |
| DPO checkpoint | 45.70% | -59.1626 | -0.001024 | 51.00% | 0.694702 |

结果分析：

1. chosen preference accuracy 没有变化，说明 DPO 没有翻转更多 pair 的排序；
2. 51% pair 的隐式 margin 为正，但均值为负，说明少量幅度更大的负 margin 抵消了正样本；
3. holdout DPO loss 相对 reference 上升 0.224%，当前配置没有泛化改善；
4. 训练 batch loss 多次低于 `log 2` 不能替代未见 holdout，不能用训练曲线宣传偏好能力提升。

### 15.7 DPO 暴露出的实际问题

#### 问题一：离线偏好数据与当前策略分布脱节

DPO 只学习数据集已有的 chosen/rejected。策略训练后可能生成数据集中从未出现的新错误，但 DPO 不会在线执行环境并获得新反馈。这是典型的 offline distribution shift。

#### 问题二：一个 Pair 标签压缩了完整过程信息

chosen/rejected 只告诉模型“哪个整体更好”，无法直接区分：工具名错误、参数错误、环境执行失败、证据不足、最终答案错误分别发生在哪里。多轮 Tool-Use 尤其需要结构化 verifier，而不是单个 pair label。

#### 问题三：静态偏好不等于可验证正确性

偏好数据可能受标注噪声、风格偏好或长度偏好影响。回答写得更长、更像人类，并不保证数学答案或工具结果正确。

#### 问题四：当前序列 log-prob 求和存在长度敏感性

实现对 assistant token log-prob 直接求和。chosen/rejected 长度差异会影响序列 margin。项目没有完成 length-normalized DPO 消融，因此不能断言长度影响已经消除。

#### 问题五：超参数和数据质量敏感

`β`、learning rate、训练长度以及 preference pair 的难度都会改变 reference 约束与偏好强度。本次只有一个正式配置，不能从负结果判断 DPO 算法本身无效；也不能在 1,000 对 test 上反复调参后继续称其为独立 holdout。

#### 问题六：虽然无需 Critic，仍需驻留 Reference

DPO 比 PPO 少了 Reward Model、Value/Critic 和在线 rollout，但训练仍需 policy 与冻结 reference 前向。它降低了系统复杂度，不是零额外成本。

### 15.8 后续如何处理这些问题

这些问题没有由某一个算法“一次性解决”，而是通过训练阶段和反馈形式分工处理：

| DPO 局限 | 后续处理 | 本项目证据 |
|---|---|---|
| 模型不会稳定输出工具协议 | 先做 Agent SFT 冷启动 | Math `0→59.375%`；Basic Tool `7.031→100%` |
| 静态 pair 无法反馈当前策略的新错误 | GRPO 家族在线采样当前策略 | 已完成真实 rollout/current/reference 闭环 |
| Pair 标签不能验证工具执行过程 | 多轮 verifier 分别检查格式、参数、执行、证据和答案 | Reward Hacking 单测与全量 oracle replay 通过 |
| PPO 的 Critic/Value 成本较高 | GRPO 用同 prompt 组内相对 Reward 代替 Critic | 四算法统一无 Critic 训练框架完成 |
| GRPO 全错/全对组无梯度 | DAPO Dynamic Sampling 只保留混合成功组 | Tool 保留组零方差率降为 0% |
| 对称 Clip、长短序列 reduction | DAPO 非对称 Clip、Token-level Loss、Soft Overlong | 代码和单测完成；固定集尚无收益 |
| Token ratio 与序列级 Reward 粒度不一致 | GSPO 使用长度归一化 sequence ratio/clip | 代码和单测完成；当前小预算尚无固定集收益 |

这里必须区分“解决反馈/优化机制问题”和“解决最终任务性能问题”：

- **已解决到机制层：** 当前策略可以在线产生轨迹，verifier 可以给出可验证反馈，DAPO 可以恢复非零优势组；
- **已解决到冷启动能力层：** Agent SFT 显著提高协议和基础任务成功率；
- **尚未解决到泛化层：** DPO 没有改善 preference holdout，四种 RL 算法也没有改善共同 Math/Tool 固定集。

DPO 的正确面试定位是：

> 我复现并审计了离线偏好优化链路，通过 prompt-grouped split、相同 token 物化和独立 holdout 发现当前 DPO 配置没有提升偏好准确率。这个负结果说明静态 pair optimization 不能替代面向数学与工具执行的在线 verifier，因此主线进一步构建 Agent SFT 冷启动和无 Critic RLVR；后续算法解决了在线反馈与有效梯度问题，但当前小预算仍未形成固定集增益。

---

# 第三部分：GRPO、CISPO、DAPO 与 GSPO

## 16. 统一符号与训练闭环

对每个 prompt 采样 `G` 条轨迹。训练涉及三套策略：

- `π_old`：产生 rollout 的 behavior policy，log-prob 必须冻结；
- `π_θ`：当前待更新策略；
- `π_ref`：冻结的 Agent SFT 参考策略，用于 KL 约束。

所有算法共享相同 rollout、reward、action mask、old/reference log-prob、optimizer 和评测集，只替换 policy objective 与 DAPO sampling recipe。

组相对优势：

$$
A_i=\frac{r_i-\operatorname{mean}(r_{1:G})}
{\operatorname{std}(r_{1:G})+\epsilon}
$$

```python
def group_relative_advantages(rewards, group_size, eps=1e-4):
    grouped = rewards.view(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, unbiased=False, keepdim=True)
    # 全对/全错组 std=0，分子也为 0，因此 advantage 全为 0。
    return ((grouped - means) / (stds + eps)).reshape(-1)
```

---

## 17. GRPO 基线

Token importance ratio：

$$
r_{i,t}(\theta)=\exp\left(\log\pi_\theta(y_{i,t})-
\log\pi_{old}(y_{i,t})\right)
$$

对称裁剪目标：

$$
L_{GRPO}=-\mathbb{E}_i\left[\frac{1}{|o_i|}\sum_t
\min(r_{i,t}A_i,\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i)
\right]+\beta KL
$$

GRPO 不需要单独训练 Critic，通过同 prompt 多次采样的组内比较估计优势。局限是稀疏奖励下全错组会产生全零优势。

---

## 18. CISPO

CISPO 裁剪 importance coefficient，并停止 coefficient 的梯度，然后直接优化当前策略 log-prob：

$$
c_{i,t}=\operatorname{stopgrad}\left(
\min(r_{i,t},1+\epsilon_{high}^{IS})
\right)
$$

$$
L_{CISPO}=-\frac{1}{\sum_{i,t}m_{i,t}}
\sum_{i,t}m_{i,t}c_{i,t}A_i\log\pi_\theta(y_{i,t})
$$

```python
cispo_upper = 1.0 + cispo_epsilon_high
coefficient = token_ratio.clamp(max=cispo_upper).detach()
per_token_policy = -(coefficient * token_advantages * current_logps.float())
policy_loss = masked_mean(per_token_policy, completion_mask)
```

`detach()` 很关键：importance coefficient 只负责校正采样分布，不让梯度通过 ratio 再形成另一条路径。实现按论文把 CLI 参数解释为 `1+ε_high_IS`，修正了把参数直接当绝对 ratio 上界的歧义。

---

## 19. DAPO 的四个核心部件

### 18.1 Dynamic Sampling

仅保留组内既有成功又有失败的 prompt：

```python
def effective_group_mask(task_success, group_size):
    successes = task_success.bool().view(-1, group_size).sum(dim=1)
    # success count 不能为 0，也不能等于 G。
    return (successes > 0) & (successes < group_size)
```

必须使用严格二值 `task_success`，不能使用带格式分的 shaped reward；否则格式奖励可能让错误答案被误判为“有效成功”。

### 18.2 非对称 Clip / Clip-Higher

$$
\hat r=\operatorname{clip}(r,1-\epsilon_{low},1+\epsilon_{high}),
\quad \epsilon_{high}>\epsilon_{low}
$$

```python
clipped_ratio = token_ratio.clamp(
    1.0 - dapo_epsilon_low,
    1.0 + dapo_epsilon_high,
)
surrogate = torch.minimum(
    token_ratio * token_advantages,
    clipped_ratio * token_advantages,
)
```

正式配置 `ε_low=0.2, ε_high=0.28`。上界更宽，目的是降低低概率正确 token 被过早限制的风险；并不意味着 ratio 可以无限增大。

### 18.3 Token-level Loss

GRPO 常先对每条序列求均值，再对 batch 求均值，因此长短序列先被等权。DAPO 对所有有效 action token 做全局均值：

$$
L_{token}=\frac{\sum_{i,t}m_{i,t}l_{i,t}}
{\sum_{i,t}m_{i,t}}
$$

```python
policy_loss = (per_token_policy * completion_mask).sum() \
              / completion_mask.sum().clamp(min=1)
```

DDP 下分母必须是全局 token 数。本项目额外计算 `distributed_token_mean_scale`，修正各 rank 有效 token 数不同导致的梯度权重偏差。

### 18.4 Soft Overlong

在最大长度前设置线性缓冲区，而不是只在硬截断时突然惩罚：

$$
R_{len}(L)=
\begin{cases}
0, & L\le L_{max}-L_{cache}\\
-\frac{L-(L_{max}-L_{cache})}{L_{cache}}, & L_{max}-L_{cache}<L<L_{max}\\
-1, & L\ge L_{max}
\end{cases}
$$

```python
start = max_length - cache_length
penalty = -((lengths.float() - start) / cache_length).clamp(0.0, 1.0)
```

该函数和边界值已通过单测，但正式固定集响应大多没有进入惩罚区间，因此当前不能声称它实测降低了平均长度。

---

## 20. GSPO：序列级重要性比率

GSPO 对一条轨迹的 log-ratio 求长度归一化均值，再指数化：

$$
r_i^{seq}=\exp\left[
\frac{1}{|o_i|}\sum_t
\left(\log\pi_\theta(y_{i,t})-log\pi_{old}(y_{i,t})\right)
\right]
$$

它等价于 token ratios 的几何平均，而不是乘积或算术平均。裁剪和 surrogate 都在序列级执行：

```python
seq_log_ratio = (log_ratio * mask).sum(1) / token_counts.clamp(min=1)
seq_ratio = torch.exp(seq_log_ratio)
clipped_seq_ratio = seq_ratio.clamp(
    1.0 - gspo_epsilon_low,
    1.0 + gspo_epsilon_high,
)
seq_surrogate = torch.minimum(
    seq_ratio * sequence_advantages,
    clipped_seq_ratio * sequence_advantages,
)
policy_loss = -seq_surrogate[valid_rows].mean()
```

当前 advantage 是轨迹级标量，因此这是标准 GSPO，不是 GSPO-token 变体。

---

## 21. KL 与数值稳定性

项目使用逐点非负的 k3 estimator：

$$
\delta=\log\pi_{ref}-\log\pi_\theta,\qquad
k_3=\exp(\delta)-\delta-1\ge0
$$

```python
delta = (reference_logps - current_logps).float().clamp(-20.0, 20.0)
kl_k3 = torch.exp(delta) - delta - 1.0
```

安全措施包括：

- rollout/current/reference 统一 BF16 forward，log-softmax 转 FP32；
- 第一次 policy forward 检查 `behavior-current log-prob MAE`；
- ratio/log-ratio 截断防止 `exp` 溢出；
- 同时记录 KL mean、token P95 和 max，避免均值被极少数低概率 token 支配；
- 记录 ratio mean/std、clip fraction、gradient clipping、response length 与采样成本。

训练批次 KL 曾出现大尾值，因此最终策略偏移以共同固定集 KL 为准，而不是用单个训练 batch 的 KL 宣传稳定性。

---

## 22. 四算法正式实验设计

- 共同初始化：同一个 `agent_sft`；
- 两个任务：数学推理、组合 Tool-Use；
- 四算法：GRPO、CISPO、DAPO、GSPO；
- 三个独立训练 seed；
- 每 run 最多 10 个 accepted/update groups；
- 每组 4 trajectories；
- 每批复用 2 个 policy epochs；
- learning rate `1e-7`；
- 每个 checkpoint 使用相同的 64 个 holdout prompt 与 3 个 decode seeds；
- 每个任务共 `13 checkpoints × 64 prompts × 3 seeds = 2,496` 条固定集轨迹。

“同预算”只指 optimizer-update 上限相同。DAPO 为填满 10 个有效组会采样更多候选轨迹，所以不能声称总 token、environment call 或 wall time 相同。

---

## 23. 数学推理结果

### 22.1 训练组统计

| 算法 | Reward | 训练组准确率 | 候选组 | 动态接受率 | Wall time |
|---|---:|---:|---:|---:|---:|
| GRPO | -0.5167 ± 0.0577 | 24.17% ± 2.89% | 10.0 | — | 32.41 ± 0.62 s |
| CISPO | -0.5167 ± 0.0577 | 24.17% ± 2.89% | 10.0 | — | 32.79 ± 0.32 s |
| DAPO | 0.0667 ± 0.1893 | 53.33% ± 9.46% | 160.3 ± 43.7 | 8.23% ± 1.35% | 487.04 ± 137.22 s |
| GSPO | 0.0500 ± 0.0500 | 52.50% ± 2.50% | 10.0 | — | 28.77 ± 2.58 s |

DAPO 的训练准确率高是动态采样“只保留混合成功组”产生的条件统计，不能与普通随机组直接比较，更不能当作测试集提升。

### 22.2 固定集结果

共同 baseline 和四算法均为：

- Reward：-0.03125；
- Task/Answer Accuracy：48.4375%；
- Format：75.00%；
- Tool Execution：95.3125%；
- 平均响应长度：80.19 action tokens；
- Unfinished：25.00%；
- 四算法相对 baseline 的任务指标 delta 全为 0；
- 固定集 k3 KL：`0–5.43e-9`。

结论是：四个目标都完成更新，DAPO 的采样机制与成本被量化，但当前参数变化不足以改变固定随机种子的生成结果。

---

## 24. 组合 Tool-Use 结果

### 23.1 训练组统计

| 算法 | Reward | 训练组准确率 | 零方差组比例 | 候选组 | Wall time |
|---|---:|---:|---:|---:|---:|
| GRPO | -0.9333 ± 0.0764 | 3.33% ± 3.82% | 90.00% ± 10.00% | 10.0 | 28.97 ± 0.86 s |
| CISPO | -0.8667 ± 0.1155 | 6.67% ± 5.77% | 86.67% ± 11.55% | 10.0 | 28.80 ± 0.69 s |
| DAPO | -0.0333 ± 0.0289 | 48.33% ± 1.44% | 0% | 103.7 ± 17.6 | 285.19 ± 47.06 s |
| GSPO | -0.9667 ± 0.0289 | 1.67% ± 1.44% | 93.33% ± 5.77% | 10.0 | 28.16 ± 0.44 s |

DAPO 确实把保留组零方差率降为 0%，保证被用于更新的组有非零优势；代价是候选组约为普通算法的 10.4×、action tokens 约 10.9×、wall time 约 9.9×。

### 23.2 固定集结果

| 指标 | Agent SFT baseline | GRPO/CISPO/DAPO/GSPO |
|---|---:|---:|
| Reward | -0.88542 | -0.88542 |
| Task/Answer Accuracy | 5.7292% | 5.7292% |
| Format Valid | 97.9167% | 97.9167% |
| Tool Call Valid | 99.4792% | 99.4792% |
| Execution Success | 71.6146% | 71.6146% |
| Required Tool Coverage | 56.6840% | 56.6840% |
| Tool Evidence Coverage | 35.9375% | 35.9375% |
| Avg Response Length | 92.34 | 92.34 |
| P95 Response Length | 104.85 | 104.85 |
| Unfinished Rate | 1.0417% | 1.0417% |
| k3 KL | 0 | `0–1.53e-8` |

所有主要 delta 均为 0。因此当前可以说“动态采样缓解零优势组并量化稳定性/成本”，不能说“RL 已提升固定集准确率或任务完成率”。

---

# 第四部分：Agentic RL / RLVR

## 25. 多轮轨迹状态机

```text
System/User Prompt
      ↓
Assistant Action: <tool_call>{...}</tool_call>
      ↓ parse + schema validation
Tool Execution
      ↓
Observation: deterministic tool result
      ↓
Assistant Action: next tool call or final answer
      ↓
Verifier: format + execution + evidence + final answer
```

训练 loss 只能覆盖模型自己采样的 assistant action；tool observation 是环境给出的上下文，模型可以读取，但不能当作自己的 action 计算 policy gradient。

```text
system/user prompt       action_mask=0
assistant tool call      action_mask=1
tool observation         action_mask=0
assistant final answer   action_mask=1
assistant EOS            action_mask=1
padding                   action_mask=0
```

项目曾发现真实 tokenizer non-roundtrip 反例：36 个 sampled action IDs 经过 `decode→encode` 会变成 38 个。修复后保存精确 sampled-token ledger，只把环境 observation suffix 追加到上下文，保证 `π_old` 和 `π_θ` 对同一 action IDs 计算概率。

---

## 26. Tool-Use 数据集与泄漏控制

基础 Tool 数据覆盖 calculator、unit、weather、time、exchange、translation 和 multi-tool：

- 896 train / 224 eval；
- train/eval question overlap=0；
- 1,120 条 deterministic oracle 全量重放，failures=0。

Agent SFT 后基础集达到 100%，已经饱和，因此新增组合挑战集：

- 384 train / 96 eval；
- 6 类双工具/三工具组合；
- 不使用任何 Agent SFT eval label；
- 与 Agent SFT oracle rows overlap=0；
- 480 条 oracle 全量重放 failures=0。

数学冷启动另外构建 512 条训练轨迹，表达式和 ground truth 在写入前重新计算校验。

---

## 27. 可验证奖励与 Reward Hacking 防护

### 26.1 为什么分开 Shaped Reward 与 Task Success

Shaped reward 可以提供格式、工具合法性等稠密信号；但 DAPO 动态采样和最终准确率必须使用严格的二值任务成功。否则模型可能只靠格式分进入“成功组”。

严格任务成功条件：

```python
task_success = bool(
    is_answer_correct          # 最终答案精确通过 verifier
    and tags_valid            # tool tags 成对且 JSON 可解析
    and calls_satisfied        # 必需工具确实被调用
    and calls_correct          # 工具名、参数合法且执行成功
    and evidence_satisfied     # 执行结果能够支持 GT
)

if reward_mode == "strict":
    reward = 1.0 if task_success else -1.0
else:
    reward = clamp(shaped_reward, -3.0, 3.0)
```

### 26.2 具体防护

| 风险 | 防护机制 | 已有验证 |
|---|---|---|
| 数字子串攻击，例如 GT=4、回答=14 | 数值边界和 final-answer region verifier | 单测拒绝 |
| 只输出漂亮格式但答案错误 | 格式分与 `task_success` 分离 | 单测拒绝 |
| 调错工具后猜中答案 | 检查必需工具与执行证据 | 单测拒绝 |
| 伪造 observation | verifier 对确定性工具重新执行 | 数据全量 replay |
| 非法 JSON/参数 | parser + per-tool schema checker | 单测覆盖 |
| 未闭合多轮轨迹 | `unfinished=True` 时最终答案无效 | 单测拒绝 |
| 重复套话 | n-gram repetition penalty | 训练中启用 |
| 过长推理 | Soft Overlong + P95/unfinished 监控 | 函数和边界单测通过 |
| 左截断导致 old/current 上下文不同 | 超过 `max_total_len` 直接报错 | 回归测试覆盖 |
| 计算器任意代码执行 | AST 白名单解释器替代 `eval` | 工具审计通过 |

这些机制证明“约束被实现并能拒绝已知攻击样例”，但没有独立 reward-ablation 证明每个约束分别提升了多少任务准确率。

---

## 28. 指标定义

| 指标 | 定义 | 解释 |
|---|---|---|
| Reward | strict `+1/-1` 或单列 shaped reward | 主对照使用 strict |
| Task Accuracy | 成功轨迹数 / 总轨迹数 | 同时要求答案、格式、工具、证据 |
| Answer Accuracy | 最终答案通过 verifier 的比例 | 不一定代表工具过程正确 |
| Tool Execution Rate | 成功执行 calls / parsed calls | 区分格式正确和环境成功 |
| Evidence Coverage | 被工具执行结果支持的 GT 项 / 全部 GT 项 | 防止“调用工具后仍凭空猜答案” |
| KL k3 | sampled `exp(δ)-δ-1` | 衡量 current 与 reference 偏移 |
| Response Length | assistant action token 数 | 不包含 prompt 和 observation |
| P95 Length | 响应长度 95 分位 | 比均值更敏感于长尾 |
| Unfinished Rate | 达到最大轮数仍未闭合的轨迹比例 | 当前最接近“截断”的指标 |
| Zero-variance Rate | 组内 reward 方差为 0 的比例 | 衡量是否存在有效相对优势 |
| Temporal Stability | 最后 N 个更新中指标的时间标准差 | 训练波动 |
| Seed Stability | 独立训练 seed 的样本标准差 | 可重复性 |

### 27.1 为什么当前不能直接写“截断率”

当前 `unfinished` 表示达到最大轮数仍未完成；代码又禁止 rollout 后左截断。它没有区分：

- 命中单轮 `max_new_tokens`；
- 命中 `max_turns`；
- 命中 `max_total_len`；
- 正常 EOS；
- 工具/环境错误。

因此简历应暂写“未完成率、平均/P95 响应长度”。若必须报告截断率，应把 rollout 返回值扩展为：

```python
stop_reason in {"eos", "max_new_tokens", "max_turns", "max_context", "tool_error"}
truncated = stop_reason in {"max_new_tokens", "max_context"}
truncated_rate = mean(truncated)
```

在没有新增日志前，不能用 `unfinished rate` 冒充 token truncation rate。

---

## 29. Agent SFT 冷启动结果

通用 SFT 在严格 Agent 任务上几乎没有有效成功组，直接做 GRPO 类训练会得到全零优势。因此先只用训练集 oracle 做 Agent SFT，评测 label 不参与训练。

| 任务/初始化 | Task acc | Format | Tool valid | Execution | Required | Evidence | DAPO effective groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math / Full SFT | 0.00% | 81.25% | 71.09% | 69.53% | 48.44% | 10.81% | 0.00% |
| Math / Agent SFT | 59.38% | 92.97% | 100.00% | 97.27% | 100.00% | 85.68% | 15.63% |
| Basic Tool / Full SFT | 7.03% | 84.38% | 51.17% | 49.61% | 41.80% | 20.70% | 15.63% |
| Basic Tool / Agent SFT | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |

可归因结论：Agent SFT 冷启动显著提升协议遵循和基础任务完成率，并为数学 RL 提供部分混合成功组。基础 Tool 集饱和后主动更换零标签重合的挑战集，避免在饱和集上宣传 RL 收益。

---

# 第五部分：可复现性、证据与面试结论

## 30. 机器证据入口

关键文件：

```text
out/remote_final/evidence/out/
├── flash_sdpa_verification.json
├── model_flash_sdpa_verification.json
├── architecture_benchmark_long_clean.csv
├── architecture_yarn_4096.csv
├── eval/
│   ├── lora_medical_holdout_metrics_fixed.json
│   └── kd_comparison.json
├── metrics/
│   ├── algorithm_comparison_math.csv
│   └── algorithm_comparison_tool.csv
├── eval_rlvr/
│   ├── math/{summary.csv,algorithm_comparison.csv,trajectories.jsonl}
│   └── tool/{summary.csv,algorithm_comparison.csv,trajectories.jsonl}
└── run_meta/
    ├── experiment_evidence.json
    ├── dataset_sha256.txt
    ├── moe_routing_audit.json
    ├── agent_sft_mask_audit_128.json
    └── final_validation.sha256
```

快速验收：

```bash
cd /root/minimind

# 核心算法、架构、mask、verifier 回归
python -m unittest discover -s tests -v

# 查看四算法固定集主表
column -s, -t out/eval_rlvr/math/algorithm_comparison.csv | less -S
column -s, -t out/eval_rlvr/tool/algorithm_comparison.csv | less -S

# 验证最终证据索引没有缺失
jq '.missing' out/run_meta/experiment_evidence.json
```

---

## 31. 面试版结果归因

### 31.1 哪些是真正的正结果

1. GQA 的 KV Cache 理论规模相对对应 MHA 降低 50%，cache 一致性测试通过；
2. 指定 BF16 shape 下 PyTorch Flash SDPA 相对 math backend 加速 6.55×，实际模型命中 Flash kernel；
3. LoRA 以 0.61% 可训练参数使固定医疗 holdout PPL 下降 12.15%；
4. Agent SFT 将数学 Task Accuracy 从 0% 提升到 59.375%，基础 Tool 从 7.031% 提升到 100%；
5. DAPO 在稀疏 Tool 训练中把保留组零方差率从普通算法的 86.7%–93.3% 降到 0%。

### 31.2 哪些是负结果但仍有价值

1. 当前 KD 比 CE control 的 PPL 高 0.532%；
2. DPO 的偏好准确率仍为 45.70%，隐式奖励 margin 均值为负，holdout loss 上升 0.224%；
3. Partial MoE 比 Dense 慢且更占单卡权重显存；
4. YaRN 4096-token 路径可运行，但没有任务质量提升证据；
5. 四种 RL 算法在固定 Math/Tool 测试集上相对 Agent SFT 的主要 delta 全为 0；
6. DAPO 恢复有效组的代价是显著增加 rollout 和 wall time。

负结果说明实验做了共同初始化、固定集和成本核算，而不是只展示训练曲线中最好看的点。

---

## 32. 推荐简历表述

### 32.1 架构

> 复现并扩展 64M Decoder-Only 轻量语言模型，集成 RMSNorm、RoPE/YaRN、GQA、KV Cache、SwiGLU、PyTorch SDPA Flash Backend 及 Top-1 稀疏 MoE；完成 4096-token 长序列与 Cache 一致性验证，GQA 理论 KV Cache 较 MHA 降低 50%，指定 BF16 shape 下 Flash SDPA 加速 6.55×；构建 198M-A64M MoE 分支并量化路由均衡、激活参数与单卡系统开销。

### 32.2 全阶段训练

> 打通 Pretrain→SFT 主线及 LoRA、知识蒸馏支线，支持 BF16 混合精度、梯度累积、余弦学习率衰减和 Assistant-only Loss Mask；Mask 审计 128 个样本无错位、无零监督样本，LoRA 仅训练 0.61% 参数并使固定医疗 holdout PPL 下降 12.15%，同时完成 30M Student 与 64M Teacher 的同初始化、同预算 KD/CE 对照。

在真正加入并运行 Warmup 前，把原简历中的“学习率 Warmup”改为“余弦学习率衰减”。

若希望把 DPO 作为补充项目点，可增加：

> 完成 DPO 离线偏好优化及 prompt-grouped 独立评测，实现 policy/reference 隐式奖励 margin、Assistant-only pair mask 与相同 token 物化；1,000 对 holdout 上偏好准确率未提升，据此将复杂数学与 Tool-Use 主线转向在线可验证 RL，并保留 DPO 作为公平负对照。

### 32.3 后训练算法

> 在统一 RLVR 训练框架中实现 GRPO、CISPO、DAPO 与 GSPO，复现动态采样、非对称 Clip、Token-level Loss、Soft Overlong 及序列级重要性比率；4 算法×3 随机种子实验中，DAPO 将稀疏 Tool-Use 训练的保留组零方差率降至 0%，并量化约 10.4× Rollout 与 9.9×训练时间开销，定位当前小更新预算下固定集收益不显著的问题。

### 32.4 Agentic RL

> 构建 Assistant→Tool→Observation→Assistant 多轮轨迹采集、精确 sampled-token action mask 与可验证奖励体系，覆盖数学推理和组合 Tool-Use；统一评测 Reward、准确率、KL、平均/P95 响应长度、未完成率及跨 seed 波动，并针对数字子串攻击、格式作弊、无效工具调用、伪造证据和过长输出实现奖励门禁。Agent SFT 冷启动将数学准确率由 0% 提升至 59.375%、基础 Tool-Use 由 7.031% 提升至 100%。

---

## 33. 高频项目拷问

### DPO 与 GRPO 是前后继承关系吗？

不是。DPO 是离线 chosen/rejected 分类式偏好优化，不与环境交互；GRPO 家族是在线采样当前策略、由 Reward/verifier 给分的无 Critic policy gradient。本项目两条分支都从 SFT 系列 checkpoint 出发，四算法并不从 DPO checkpoint 初始化。

### 为什么 DPO 没有提升，后面还值得做在线 RL？

DPO 负结果只说明当前数据、`β=0.15`、`lr=4e-8` 和一轮训练配置没有改善该 holdout，不能证明算法普遍无效。更关键的是，离线 pair 无法执行工具、核对 observation 或持续发现当前策略的新错误；在线 RLVR 可以对每条新轨迹运行 verifier。这是反馈能力的差异，不是为了掩盖 DPO 负结果。

### 为什么 DAPO 的训练准确率高，却不能说泛化更好？

因为动态采样只保留同时含成功和失败的组。筛选后的准确率是条件分布统计，天然高于包含大量全错组的普通采样；只有共同未见固定集能判断泛化，而当前固定集 delta 为 0。

### DAPO 是否提升了训练稳定性？

准确说法是：它把被用于更新的零方差组降为 0，改善了“有没有有效相对优势”这一稳定性维度；但 rollout 成本和 wall time 大幅上升，固定集跨 seed 指标没有改善，所以不能泛化成“总体训练更稳定”。

### Token-level Loss 和 sequence-level ratio 是否矛盾？

不矛盾。Token-level Loss 是 DAPO/CISPO 的 reduction 方式；GSPO 先把 token log-ratio 聚合成 sequence ratio，再在序列级裁剪和优化。它们属于不同算法设计。

### 为什么 Tool-Use 不能只看答案准确率？

模型可能猜中答案、调用错误工具或伪造 observation。严格成功还必须验证工具名、参数、执行状态、必需工具覆盖与执行证据。Answer Accuracy 是必要但不充分条件。

### 为什么训练 KL 很大而固定集 KL 很小？

训练 batch 中少数极低概率 token 会让 `exp(logπref-logπθ)` 形成大尾值；固定集使用共同数据和解码配置，结果显示实际 checkpoint 偏移非常小。应同时报告 mean、P95、max，并把固定 holdout KL 作为最终策略偏移依据。

### 为什么不继续在 MoE 上做 RL？

Partial MoE 没有完成同质量 SFT，且单卡朴素 dispatch 更慢、更占权重显存。把它用于 RL 会把基座质量、Router 漂移和策略算法混在一起，还需同时驻留 current/reference/optimizer。Dense 主线能更公平地隔离算法变量。

### 当前最优先补哪三个实验？

1. 实现线性 Warmup + cosine decay，并用固定初始化做早期稳定性 A/B；
2. 为 rollout 增加明确 `stop_reason`，独立统计 token truncation rate；
3. 在独立 validation 上增加 accepted updates、学习率和课程难度搜索，锁定配置后一次性评测未见 test，验证 RL 是否产生真实增益。

---

## 34. 来源与代码归属

- Decoder-only Transformer：Attention Is All You Need；
- RMSNorm：Root Mean Square Layer Normalization；
- RoPE：RoFormer；YaRN：YaRN: Efficient Context Window Extension；
- GQA：Training Generalized Multi-Query Transformer Models；
- Flash：FlashAttention 与 PyTorch SDPA 文档；
- SwiGLU：GLU Variants Improve Transformer；
- Sparse MoE：Switch Transformers；
- LoRA：Low-Rank Adaptation of Large Language Models；
- KD：Distilling the Knowledge in a Neural Network；
- DPO：Direct Preference Optimization: Your Language Model is Secretly a Reward Model；
- GRPO：DeepSeekMath；
- CISPO：MiniMax-M1；
- DAPO：DAPO 论文与作者公开 recipe；
- GSPO：GSPO 论文；
- 多轮工具 RL：ReTool 作为范式参考。

具体论文链接、仓库版本和逐文件代码归属见 [`SOURCES.md`](./SOURCES.md)。上游已有模型、Pretrain/SFT/LoRA/KD/DPO 基本链路；本次主要新增统一四算法目标、DAPO/GSPO、多轮 verifier/token ledger、数据构造、固定集评测、实验日志、审计脚本和单元测试。

---

## 35. 最终一句话总结

本项目最可信的定位不是“64M 小模型已经通过 RL 获得工业级能力”，而是：**在单卡可复现环境中，把轻量 Decoder-only 架构、全阶段训练、离线 DPO 偏好优化、四种在线策略目标和多轮可验证 Agent 连接成一条可审计链路，并用正结果、负结果、成本和结论边界证明自己真正理解了每个模块。**
