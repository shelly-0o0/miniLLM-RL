import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)

    RMSNorm 是 LayerNorm 的简化版本，只对输入进行缩放，不进行中心化（不减去均值）。
    相比 LayerNorm，RMSNorm 计算更高效，且在实际应用中效果相当。

    公式：
        RMSNorm(x) = weight * (x / sqrt(mean(x^2) + eps))

    与 LayerNorm 的区别：
        - LayerNorm: (x - mean(x)) / sqrt(var(x) + eps)
        - RMSNorm: x / sqrt(mean(x^2) + eps)
        - RMSNorm 不减去均值，只进行缩放，计算更快
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        初始化 RMSNorm 层
        Args:
            dim: 输入特征的维度
            eps: 防止除零的小常数（默认 1e-5）
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # 可学习的缩放参数，初始化为全 1

    def norm(self, x):
        """
        计算 RMS 归一化
        公式：x / sqrt(mean(x^2) + eps)
        Args:
            x: 输入张量 [..., dim]
        Returns:
            归一化后的张量，形状与输入相同
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # x.pow(2): 计算 x 的平方
        # .mean(-1, keepdim=True): 在最后一个维度上求均值，保持维度
        # torch.rsqrt: 计算 1/sqrt，比先 sqrt 再除更快

    def forward(self, x):
        """
        前向传播：
        Args:
            x: 输入张量，可以是任意精度（float16, bfloat16, float32）
        Returns:
            归一化并缩放后的张量，保持原始精度
        """
        return (self.weight * self.norm(x.float())).type_as(x)
        # 先转换为 float32 进行归一化计算（提高数值稳定性）
        # 然后转换回原始精度（type_as(x)）
        # 最后乘以可学习的权重

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算 RoPE (Rotary Position Embedding) 的频率矩阵

    RoPE 通过旋转矩阵将位置信息编码到 Query 和 Key 中，使模型能够理解 token 的相对位置。
    本函数预计算所有位置的 cos 和 sin 值，避免在每次前向传播时重复计算。

    支持 YaRN (Yet another RoPE extensioN) 外推方法，可以处理超过训练时最大长度的序列。

    Args:
        dim: 每个注意力头的维度（head_dim）
        end: 最大序列长度（默认 32768）
        rope_base: RoPE 的基频率参数（默认 1e6）
        rope_scaling: RoPE 外推配置字典（YaRN 方法），如果为 None 则不使用外推

    Returns:
        freqs_cos: 预计算的 cos 值 [end, dim]
        freqs_sin: 预计算的 sin 值 [end, dim]
    """
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # During cached autoregressive decoding the query length is one and
        # the query represents the newest (right-most) token.  It may attend
        # to every cached key, so no causal triangle is needed.  Sending this
        # case through SDPA avoids the previous per-token fallback to the
        # explicit q@k/softmax path while preserving padding semantics.
        cached_single_token = past_key_value is not None and seq_len == 1
        if self.flash and cached_single_token:
            sdpa_mask = None
            if attention_mask is not None:
                sdpa_mask = attention_mask[:, None, None, :].to(torch.bool)
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=sdpa_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        elif self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络

    实现了 SwiGLU (Swish-Gated Linear Unit) 激活函数的前馈网络。
    SwiGLU 是 GLU (Gated Linear Unit) 的变体，使用 Swish/SiLU 作为门控激活函数。

    公式：
        FFN(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))

    其中：
        - gate_proj: 门控投影，用于生成门控信号
        - up_proj: 上投影，用于生成特征
        - Swish(x) = x * sigmoid(x) = x * silu(x)
        - down_proj: 下投影，将中间维度映射回 hidden_size

    相比标准 FFN (ReLU(xW1)W2)，SwiGLU 通常有更好的性能。
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # gate_proj: 门控投影，hidden_size -> intermediate_size
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        # down_proj: 下投影，intermediate_size -> hidden_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # up_proj: 上投影，hidden_size -> intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        # 激活函数：通常是 'silu' (Swish)

    def forward(self, x):
        """
        前向传播
        SwiGLU 公式：FFN(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))

        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
        Returns:
            输出张量 [batch, seq_len, hidden_size]
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config # 保存 MoE 需要的配置参数
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        #  gate: 路由/门控投影，hidden_size -> num_experts 为每个 token 选择专家并计算权重
        self.experts = nn.ModuleList([
            FeedForward(
                config,
                intermediate_size=config.moe_intermediate_size
            )
            for _ in range(config.num_experts)
        ]) # 创建 num_experts 个相互独立的专家
        self.act_fn = ACT2FN[config.hidden_act]
        # 根据配置取得激活函数

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim) # [B,S,H]-->[B * S,H]记为[N,H]
        # 将x展开，所有token数=batch_size * seq_len，等价于x.reshape(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 计算所有专家的路由概率
        # 先用self.gate(x_flat)得到每个token对每个专家的logits
        # 然后用softmax得到概率分布，[N,H]->[N,E]

        topk_weight, topk_idx = torch.topk(
            scores,
            k=self.config.num_experts_per_tok,
            dim=-1,
            sorted=False
        )# 对每个 Token 选择概率最大的K个专家[N,K]

        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            ) # 对选中的K个专家的权重进行归一化，确保它们的和为1 [N,K]

        y = torch.zeros_like(x_flat)
        #创建与 x_flat 相同形状的零张量：y([N,H])，后续每个专家的加权结果都会累加到这里。

        #逐专家执行稀疏计算
        #代码虽然遍历了所有专家，但每个专家只计算分配给自己的 Token，因此专家内部计算是稀疏的
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            #mask是一个布尔张量，形状为[N,K]，表示每个Token选择的K个专家中是否有当前专家i

            if mask.any():
            # 如果至少有一个 True，说明当前专家收到了至少一个 Token，需要执行计算
                token_idx = mask.any(dim=-1).nonzero().flatten()
                #mask.any(dim=-1)：[N,K]->[N]，判断每个 Token 是否选择了当前专家
                #.nonzero():找到为真的位置; .flatten():将1为真的位置整理为一维索引[N]->[N_i]
                weight = topk_weight[mask].view(-1, 1)
                #使用同一个布尔掩码提取专家i对应所有选择它的token的权重[N_i]

                #执行专家并累加输出
                y.index_add_(
                    0,
                    token_idx,
                    (expert(x_flat[token_idx]) * weight).to(y.dtype)
                )#x_flat[token_idx]:取出当前专家i负责的 Token :[N_i,H]
                 #expert(.):送入对应的 SwiGLU Expert_i
                 #乘路由权重weight,累加回原 Token 位置
                 # (使用“加法”而不是直接赋值，是因为一个 Token 可能被多个专家处理)
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
            #当某个专家没有收到任何 Token 时，进行0计算，生成网络连接

        #负载均衡辅助损失
        #只在训练时计算：
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(
                topk_idx,
                self.config.num_experts
            ).float().mean(0)
            #对topk_idx的token维求平均
            #得到laod，laod_i：(选择专家i的token数量)/N
            self.aux_loss = (
                load * scores.mean(0)
            ).sum() * self.config.num_experts * self.config.router_aux_loss_coef
            #scores.mean(0):计算所有 Token 对每个专家的平均路由概率，得到：importance_i
            #router_aux_loss_coef: 负载均衡损失 = [sum_i (load_i * importance_i)] * num_experts * router_aux_loss_coef
            #路由负载均衡辅助损失的权重系数
        else:
        #推理时不需要负载均衡损失，因为推理不进行反向传播。
            self.aux_loss = scores.new_zeros(1).squeeze()
            #推理时设为零

        #将展开的 Token 重新恢复为序列结构
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        # An all-valid mask carries no information but can prevent SDPA from
        # selecting its fastest kernel during single-token cached decoding.
        # Check it once before the loop instead of once per generated token.
        if attention_mask is not None and bool(torch.all(attention_mask == 1)):
            attention_mask = None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
